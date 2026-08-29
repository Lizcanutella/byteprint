"""Frozen feature extraction.

The backbone is never fine-tuned. That is the whole design: a general-purpose
self-supervised representation transfers across generators far better than a
network trained end-to-end on one of them, which reliably descends onto
generator-specific shortcuts. It also means features can be computed once and
cached, so the probe on top retrains in seconds.

DINOv2 rather than CLIP because CLIP's image-text objective encourages
semantic shortcuts, while DINOv2's purely visual pretraining keeps more of the
low-level structure a forensic head needs. That is a hypothesis, not a law,
which is why backbones live in a registry: adding SigLIP2 or EVA02 and
measuring the difference should cost one function, not a refactor.

    @register_backbone("my_vit", dim=768, patch_size=16, mean=(.5,)*3, std=(.5,)*3)
    def _build():
        return some_pretrained_model_with_pooled_output()

Mind the competition's <2B parameter budget when you add one: DINOv2-giant is
already ~1.1B of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from byteprint.registry import Registry

# ImageNet statistics, which is what DINOv2 was trained with. Backbones with a
# different preprocessing convention (SigLIP uses 0.5/0.5) declare their own.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """Everything needed to build and feed one frozen feature extractor."""

    name: str
    dim: int
    build: Callable[[], nn.Module]
    patch_size: int = 14
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


BACKBONES: Registry[BackboneSpec] = Registry("backbone")

DEFAULT_BACKBONE = "dinov2_vits14"


def register_backbone(
    name: str,
    *,
    dim: int,
    patch_size: int = 14,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    replace: bool = False,
):
    """Decorator: register a no-argument function returning a frozen torch model.

    The model must map ``(n, 3, h, w)`` to ``(n, dim)`` -- pooled features, no
    classification head.
    """

    def decorator(build: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
        BACKBONES.register(
            name,
            BackboneSpec(
                name=name, dim=dim, build=build, patch_size=patch_size, mean=mean, std=std
            ),
            replace=replace,
        )
        return build

    return decorator


def _dinov2(hub_name: str) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        return torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)

    return build


# Widths from the DINOv2 paper. vitg14 is ~1.1B parameters -- over half the
# competition's <2B budget on its own, so pair it with a small second expert.
for _name, _dim in (
    ("dinov2_vits14", 384),
    ("dinov2_vitb14", 768),
    ("dinov2_vitl14", 1024),
    ("dinov2_vitg14", 1536),
):
    register_backbone(_name, dim=_dim)(_dinov2(_name))


def resolve_device(preferred: str = "auto") -> str:
    """Pick a torch device, falling back through cuda -> mps -> cpu."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preferred not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device {preferred!r}; use auto, cuda, mps or cpu")
    return preferred


class TorchBackbone:
    """Turns uint8 crops into embeddings with a frozen torch model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        name: str,
        dim: int,
        device: str = "cpu",
        batch_size: int = 16,
        patch_size: int = 14,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.name = name
        self.dim = dim
        self.device = device
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self._model = model.to(device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _to_tensor(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        stacked = np.stack([np.asarray(c, dtype=np.float32) for c in crops])
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed every crop, returning an ``(n_crops, dim)`` float32 matrix."""
        if len(crops) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        for crop in crops:
            height, width = np.asarray(crop).shape[:2]
            if height % self.patch_size or width % self.patch_size:
                raise ValueError(
                    f"crop size {height}x{width} must be a multiple of {self.patch_size} "
                    f"for {self.name}"
                )

        outputs = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = self._to_tensor(crops[start : start + self.batch_size])
                features = self._model(batch.to(self.device))
                outputs.append(features.float().cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)


def load_backbone(
    name: str = DEFAULT_BACKBONE, *, device: str = "auto", batch_size: int = 16
) -> TorchBackbone:
    """Build a registered backbone. Pretrained weights are cached by torch/HF."""
    spec = BACKBONES.resolve(name)
    return TorchBackbone(
        spec.build(),
        name=name,
        dim=spec.dim,
        device=resolve_device(device),
        batch_size=batch_size,
        patch_size=spec.patch_size,
        mean=spec.mean,
        std=spec.std,
    )


# Registered last so `register_backbone` above is defined. These read weights
# from a local HuggingFace cache instead of torch.hub, which is what an offline
# compute node needs; importing the module is what performs the registration.
from byteprint import backbone_hf as _backbone_hf  # noqa: E402,F401  (side effect)
