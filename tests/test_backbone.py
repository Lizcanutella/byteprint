from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from torch import nn

from byteprint.backbone import BACKBONES, TorchBackbone, resolve_device


class RecordingModel(nn.Module):
    """Stand-in for DINOv2: records what it was handed, returns a fixed-width embedding."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.seen: list[torch.Tensor] = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.seen.append(batch.detach().clone())
        pooled = batch.mean(dim=(2, 3))  # (n, 3)
        return pooled[:, :1].expand(-1, self.dim).contiguous()


def crops(n: int, size: int = 28) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_cpu_is_selected_when_explicitly_requested() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_resolves_to_an_available_device() -> None:
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_an_unsupported_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="tpu"):
        resolve_device("tpu")


def test_the_registry_exposes_the_three_dinov2_sizes_with_their_widths() -> None:
    assert BACKBONES["dinov2_vits14"].dim == 384
    assert BACKBONES["dinov2_vitb14"].dim == 768
    assert BACKBONES["dinov2_vitl14"].dim == 1024


def test_every_registered_backbone_uses_the_patch_size_in_its_name() -> None:
    # dinov2_vitl14 -> 14. The HF and timm entries encode no patch size in their
    # names and are exempt; SigLIP2 is genuinely a /16 model, so the old blanket
    # "everything is 14" assertion was a property of the roster, not a rule.
    for name, spec in BACKBONES.items():
        encoded = re.search(r"(\d\d)$", name)
        if encoded is not None:
            assert spec.patch_size == int(encoded.group(1)), name


def test_every_registered_backbone_tiles_the_crop_size_every_run_uses() -> None:
    # A patch size that does not divide 224 fails at the first forward pass --
    # an hour into a job, after the weights and the split are already staged.
    for name, spec in BACKBONES.items():
        assert 224 % spec.patch_size == 0, name


def test_embedding_returns_one_row_per_crop() -> None:
    backbone = TorchBackbone(RecordingModel(), name="stub", dim=8)

    embeddings = backbone.embed(crops(5))

    assert embeddings.shape == (5, 8)


def test_embeddings_are_float32() -> None:
    backbone = TorchBackbone(RecordingModel(), name="stub", dim=8)

    assert backbone.embed(crops(2)).dtype == np.float32


def test_crops_are_processed_in_batches_of_the_configured_size() -> None:
    model = RecordingModel()
    backbone = TorchBackbone(model, name="stub", dim=8, batch_size=2)

    backbone.embed(crops(5))

    assert [batch.shape[0] for batch in model.seen] == [2, 2, 1]


def test_pixels_are_scaled_and_normalised_before_reaching_the_model() -> None:
    model = RecordingModel()
    backbone = TorchBackbone(model, name="stub", dim=8)

    backbone.embed([np.full((28, 28, 3), 255, dtype=np.uint8)])

    seen = model.seen[0]
    # White maps to (1 - mean) / std per channel, so it must land well above 1.0.
    assert seen.shape == (1, 3, 28, 28)
    assert seen.min() > 1.0


def test_a_crop_size_that_is_not_a_multiple_of_the_patch_size_is_rejected() -> None:
    backbone = TorchBackbone(RecordingModel(), name="stub", dim=8, patch_size=14)

    with pytest.raises(ValueError, match="multiple of 14"):
        backbone.embed(crops(1, size=30))


def test_embedding_no_crops_yields_an_empty_matrix_of_the_right_width() -> None:
    backbone = TorchBackbone(RecordingModel(), name="stub", dim=8)

    assert backbone.embed([]).shape == (0, 8)


def test_the_model_is_never_asked_for_gradients() -> None:
    model = RecordingModel()
    backbone = TorchBackbone(model, name="stub", dim=8)

    backbone.embed(crops(2))

    assert all(not batch.requires_grad for batch in model.seen)


def test_loading_an_unregistered_backbone_name_is_rejected() -> None:
    from byteprint.backbone import load_backbone

    with pytest.raises(ValueError, match="resnet50"):
        load_backbone("resnet50", device="cpu")
