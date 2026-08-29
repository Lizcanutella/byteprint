"""HuggingFace-hosted backbones.

The cluster's compute nodes are offline and the staged weights are in the
HuggingFace cache layout, so ``torch.hub.load`` -- which reaches for GitHub on
every call -- cannot be the only way into a backbone.
"""
from __future__ import annotations

import sys

import torch

from byteprint.backbone import BACKBONES
from byteprint import backbone_hf


def test_the_staged_dinov2_checkpoints_are_registered_with_their_widths() -> None:
    assert BACKBONES["dinov2_large_hf"].dim == 1024
    assert BACKBONES["dinov2_giant_hf"].dim == 1536


def test_the_hf_backbones_keep_dinov2_patch_size_and_normalisation() -> None:
    spec = BACKBONES["dinov2_large_hf"]
    assert spec.patch_size == 14
    assert spec.mean == (0.485, 0.456, 0.406)


def test_registering_them_does_not_import_transformers() -> None:
    # Registration stores a callable; the heavy dependency must stay inside it,
    # so a machine without transformers can still run `byteprint list`.
    assert "transformers" not in sys.modules


def test_the_wrapper_returns_the_pooled_class_token() -> None:
    class Output:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.last_hidden_state = tensor

    class FakeModel(torch.nn.Module):
        def forward(self, pixel_values: torch.Tensor):  # noqa: D102
            n = pixel_values.shape[0]
            tokens = torch.arange(n * 5 * 4, dtype=torch.float32).reshape(n, 5, 4)
            return Output(tokens)

    wrapped = backbone_hf.PooledHFVision(FakeModel())
    result = wrapped(torch.zeros(2, 3, 28, 28))
    assert result.shape == (2, 4)
    assert torch.equal(result[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_the_wrapper_prefers_an_explicit_pooler_output_when_the_model_has_one() -> None:
    class Output:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.last_hidden_state = torch.zeros(tensor.shape[0], 5, tensor.shape[1])
            self.pooler_output = tensor

    class FakeModel(torch.nn.Module):
        def forward(self, pixel_values: torch.Tensor):  # noqa: D102
            return Output(torch.full((pixel_values.shape[0], 4), 3.0))

    wrapped = backbone_hf.PooledHFVision(FakeModel())
    assert torch.equal(wrapped(torch.zeros(1, 3, 28, 28)), torch.full((1, 4), 3.0))
