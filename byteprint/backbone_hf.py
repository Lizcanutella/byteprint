"""Backbones loaded from a HuggingFace cache rather than ``torch.hub``.

``torch.hub.load("facebookresearch/dinov2", ...)`` contacts GitHub on every
call, which a cluster compute node with no internet cannot do. The same weights
are published on the Hub, so these registrations read them from a local
``HF_HOME`` -- stage the snapshot on a machine that has internet, point
``HF_HOME`` at it, and the compute node needs no network at all.

The widths and normalisation match the ``torch.hub`` variants exactly, so a
probe trained on one is comparable to a probe trained on the other. Only the
transport differs.

``transformers`` is imported inside the builder, never at module import: a
laptop without it must still be able to run ``byteprint list``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from byteprint.backbone import register_backbone


class PooledHFVision(nn.Module):
    """Adapt a HuggingFace vision model to byteprint's ``(n, 3, h, w) -> (n, dim)``.

    DINOv2's ``torch.hub`` entry point returns the class token, so this returns
    the same thing -- the model's own ``pooler_output`` when it publishes one,
    and token 0 of the final hidden state otherwise.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state[:, 0, :]


class NaflexVision(nn.Module):
    """Adapt SigLIP2's variable-resolution vision tower to a fixed square crop.

    The naflex checkpoint does not accept an image tensor. It takes a *sequence*
    of flattened patches plus the grid they were cut from, which is how it
    supports native aspect ratios without resizing. A byteprint crop is always a
    square whose side is a multiple of the patch size, so the sequence is a pure
    reshape, the attention mask is all ones, and every image reports the same
    grid.

    The flattening order is load-bearing and matches transformers' own
    ``convert_image_to_patches``: a patch is laid out pixel-major and
    channel-minor, ``(patch_h, patch_w, channels)``. The other ordering yields
    embeddings of exactly the right shape and no meaning at all, because the
    patch-embedding projection was fitted to this one.
    """

    def __init__(self, tower: nn.Module, patch_size: int) -> None:
        super().__init__()
        self.tower = tower
        self.patch_size = patch_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = pixel_values.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"a naflex crop must be a whole multiple of the patch size {patch}, "
                f"got {height}x{width}"
            )
        rows, columns = height // patch, width // patch
        patches = (
            pixel_values.reshape(n, channels, rows, patch, columns, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, rows * columns, patch * patch * channels)
        )
        outputs = self.tower(
            pixel_values=patches,
            pixel_attention_mask=torch.ones(
                n, rows * columns, dtype=torch.long, device=pixel_values.device
            ),
            spatial_shapes=torch.tensor(
                [[rows, columns]] * n, dtype=torch.long, device=pixel_values.device
            ),
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        # SigLIP has no class token, so the mean over patches is the right
        # fallback -- token 0 would be an arbitrary corner of the image.
        return outputs.last_hidden_state.mean(dim=1)


def _from_hub(repo_id: str) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        from transformers import AutoModel

        # local_files_only: fail loudly on a compute node with an incomplete
        # cache rather than hanging on a network call that cannot succeed.
        model = AutoModel.from_pretrained(repo_id, local_files_only=True)
        return PooledHFVision(model)

    return build


# Both are well inside the competition's <2B budget: ViT-L/14 is ~0.3B and
# ViT-g/14 ~1.1B. Staged on the cluster as facebook/dinov2-{large,giant}.
for _name, _repo, _dim in (
    ("dinov2_large_hf", "facebook/dinov2-large", 1024),
    ("dinov2_giant_hf", "facebook/dinov2-giant", 1536),
):
    register_backbone(_name, dim=_dim)(_from_hub(_repo))


def _siglip2_tower(repo_id: str, patch_size: int) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(repo_id, local_files_only=True)
        # Only the vision tower is a feature extractor; the text tower is dead
        # weight here and its embeddings would never be asked for.
        return NaflexVision(model.vision_model, patch_size=patch_size)

    return build


def _timm_model(model_id: str, image_size: int) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        import timm

        # num_classes=0 drops the classifier and returns pooled features.
        # img_size resizes the checkpoint's position embeddings on load, which
        # is what lets a 448-pretrained model take byteprint's 224 crops.
        return timm.create_model(
            model_id, pretrained=True, num_classes=0, img_size=image_size
        )

    return build


# The two remaining staged checkpoints, both frozen and both inside the <2B
# budget: EVA02-L is ~0.3B and SigLIP2-so400m's vision tower ~0.43B.
#
# Normalisation is per-backbone and is not a detail: EVA02's in22k_in1k
# checkpoint was fine-tuned through CLIP's statistics, and SigLIP2 through
# symmetric [-1, 1] scaling. Either one fed ImageNet statistics still produces
# embeddings, just worse ones, with nothing to signal it.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

register_backbone("eva02_large_timm", dim=1024, patch_size=14, mean=CLIP_MEAN, std=CLIP_STD)(
    _timm_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", image_size=224)
)

register_backbone(
    "siglip2_so400m_hf", dim=1152, patch_size=16, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)(_siglip2_tower("google/siglip2-so400m-patch16-naflex", patch_size=16))
