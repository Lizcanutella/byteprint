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
