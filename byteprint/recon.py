"""AEROBLADE: training-free detection by autoencoder reconstruction error.

A latent diffusion model's images are, by construction, close to the output
manifold of its VAE decoder -- so that VAE can round-trip them with much less
perceptual damage than it does a real photograph. Reconstruct, measure the
perceptual distance, and a small distance is itself evidence of synthesis.
No training data, no training run.

The paper's robustness comes from reconstructing through *several* autoencoders
and taking the minimum distance: an image only needs to be near one generator's
manifold to be caught. That minimum is applied at scoring time
(:func:`aeroblade_score`), not at extraction, so which autoencoders count can be
ablated without recomputing anything.

Ricker et al., "AEROBLADE: Training-Free Detection of Latent Diffusion Images
Using Autoencoder Reconstruction Error", CVPR 2024. arXiv:2401.17879
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)

# Hugging Face ids for the autoencoders we know how to load, as
# (repo, subfolder). The stabilityai/stable-diffusion-2-* repos are no longer
# resolvable on the Hub, so they are deliberately absent rather than listed and
# broken. Verified reachable 2026-08-29.
AUTOENCODERS: dict[str, tuple[str, str | None]] = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"),
    "sd14": ("CompVis/stable-diffusion-v1-4", "vae"),
    "vae-mse": ("stabilityai/sd-vae-ft-mse", None),
    "vae-ema": ("stabilityai/sd-vae-ft-ema", None),
    "sdxl": ("stabilityai/sdxl-vae", None),
}

# Three decoders that differ from one another; sd14 and vae-ema are near
# duplicates of sd15 and vae-mse, so they add cost without adding coverage.
DEFAULT_AUTOENCODERS = ("sd15", "vae-mse", "sdxl")


@dataclass(frozen=True, slots=True)
class Autoencoder:
    """A named round-trip function on tensors in [-1, 1]."""

    name: str
    reconstruct: Callable[[torch.Tensor], torch.Tensor]


class PerceptualDistance:
    """LPIPS when the learned weights are installed, otherwise plain VGG features.

    The distinction matters and is reported rather than hidden: real LPIPS
    applies linear weights fitted to human judgements on top of VGG activations.
    The fallback is an unweighted, channel-normalised feature distance, which is
    the same idea with a weaker calibration.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        try:
            import lpips

            self._model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
            self.backend = "lpips"
        except Exception as exc:  # pragma: no cover - exercised only without lpips
            log.warning("lpips unavailable (%s); falling back to unweighted VGG distance", exc)
            self._model = _UnweightedVGGDistance(device)
            self.backend = "vgg-unweighted"

        for parameter in getattr(self._model, "parameters", lambda: [])():
            parameter.requires_grad_(False)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(a, b).reshape(a.shape[0])


class _UnweightedVGGDistance(torch.nn.Module):
    """Channel-normalised VGG16 feature distance, averaged over early layers."""

    LAYERS = (3, 8, 15)

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.device = device

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        distance = torch.zeros(a.shape[0], device=a.device)
        x, y = a, b
        for index, layer in enumerate(self.features):
            x, y = layer(x), layer(y)
            if index in self.LAYERS:
                nx = torch.nn.functional.normalize(x, dim=1)
                ny = torch.nn.functional.normalize(y, dim=1)
                distance = distance + ((nx - ny) ** 2).sum(dim=1).mean(dim=(1, 2))
            if index > max(self.LAYERS):
                break
        return distance / len(self.LAYERS)


class ReconExpert:
    """Reconstruction distances through a bank of autoencoders.

    Implements the same interface as :class:`byteprint.backbone.TorchBackbone`
    (``name``/``dim``/``embed``), so extraction, caching, resume and the
    laundering ladder all apply to it unchanged.
    """

    def __init__(
        self,
        autoencoders: Sequence[Autoencoder],
        *,
        distance,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        if not autoencoders:
            raise ValueError("a reconstruction expert needs at least one autoencoder")
        self.autoencoders = list(autoencoders)
        self.distance = distance
        self.device = device
        self.batch_size = batch_size
        self.dim = len(self.autoencoders)

    @property
    def ae_names(self) -> list[str]:
        return [ae.name for ae in self.autoencoders]

    @property
    def name(self) -> str:
        return "aeroblade:" + "+".join(self.ae_names)

    def _to_tensor(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        stacked = np.stack([np.asarray(c, dtype=np.float32) for c in crops])
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2) / 255.0
        return (tensor * 2.0 - 1.0).to(self.device)  # autoencoders expect [-1, 1]

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(n_crops, n_autoencoders)`` matrix of reconstruction distances."""
        if len(crops) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        columns = []
        for autoencoder in self.autoencoders:
            distances = []
            for start in range(0, len(crops), self.batch_size):
                batch = self._to_tensor(crops[start : start + self.batch_size])
                with torch.inference_mode():
                    rebuilt = autoencoder.reconstruct(batch)
                distances.append(self.distance(batch, rebuilt).float().cpu().numpy())
            columns.append(np.concatenate(distances))

        return np.stack(columns, axis=1).astype(np.float32)


def aeroblade_score(distances: np.ndarray) -> np.ndarray:
    """Turn reconstruction distances into a synthetic-is-high score.

    The minimum across autoencoders is the paper's aggregation: an image only
    has to sit near *one* generator's manifold. Negated so that, like every
    other score in this codebase, higher means more likely synthetic.
    """
    distances = np.atleast_2d(np.asarray(distances, dtype=np.float64))
    if distances.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return -distances.min(axis=1)


def _load_diffusers_autoencoder(key: str, device: str):
    from diffusers import AutoencoderKL

    repo, subfolder = AUTOENCODERS[key]
    kwargs = {"subfolder": subfolder} if subfolder else {}
    vae = AutoencoderKL.from_pretrained(repo, **kwargs).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    def reconstruct(batch: torch.Tensor) -> torch.Tensor:
        latent = vae.encode(batch).latent_dist.mode()
        return vae.decode(latent).sample

    return Autoencoder(name=key, reconstruct=reconstruct)


def load_recon_expert(
    ae_ids: Sequence[str] = DEFAULT_AUTOENCODERS,
    *,
    device: str = "auto",
    batch_size: int = 4,
) -> ReconExpert:
    """Load the requested autoencoders and wrap them in a reconstruction expert."""
    from byteprint.backbone import resolve_device

    unknown = [key for key in ae_ids if key not in AUTOENCODERS]
    if unknown:
        raise ValueError(
            f"unknown autoencoder(s) {unknown}; expected any of {sorted(AUTOENCODERS)}"
        )

    resolved = resolve_device(device)
    autoencoders = [_load_diffusers_autoencoder(key, resolved) for key in ae_ids]
    return ReconExpert(
        autoencoders,
        distance=PerceptualDistance(resolved),
        device=resolved,
        batch_size=batch_size,
    )
