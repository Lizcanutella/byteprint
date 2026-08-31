"""HuggingFace-hosted backbones.

The cluster's compute nodes are offline and the staged weights are in the
HuggingFace cache layout, so ``torch.hub.load`` -- which reaches for GitHub on
every call -- cannot be the only way into a backbone.
"""
from __future__ import annotations

import sys

import pytest
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


def test_the_sweep_checkpoints_are_registered_with_their_widths() -> None:
    assert BACKBONES["eva02_large_timm"].dim == 1024
    assert BACKBONES["siglip2_so400m_hf"].dim == 1152


def test_eva02_normalises_with_clip_statistics_rather_than_imagenet() -> None:
    # EVA02's ft_in22k_in1k checkpoint was trained through CLIP's preprocessing.
    # Feeding it ImageNet statistics is a silent accuracy leak, not an error.
    spec = BACKBONES["eva02_large_timm"]
    assert spec.patch_size == 14
    assert spec.mean == pytest.approx((0.48145466, 0.4578275, 0.40821073))
    assert spec.std == pytest.approx((0.26862954, 0.26130258, 0.27577711))


def test_siglip2_carries_its_own_patch_size_and_symmetric_normalisation() -> None:
    spec = BACKBONES["siglip2_so400m_hf"]
    assert spec.patch_size == 16
    assert spec.mean == (0.5, 0.5, 0.5)
    assert spec.std == (0.5, 0.5, 0.5)


def test_registering_them_does_not_import_timm_or_transformers() -> None:
    # Same contract as the DINOv2 entries: the heavy import lives inside the
    # build closure, so `byteprint list` works on a machine without either.
    assert "timm" not in sys.modules
    assert "transformers" not in sys.modules


class _RecordingTower(torch.nn.Module):
    """Stands in for Siglip2VisionModel, capturing the naflex triple."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.seen: dict[str, torch.Tensor] = {}

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):  # noqa: D102
        self.seen = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        }
        dim_ = self.dim

        class Output:
            pooler_output = torch.full((pixel_values.shape[0], dim_), 7.0)
            last_hidden_state = torch.zeros(pixel_values.shape[0], pixel_values.shape[1], dim_)

        return Output()


def test_the_naflex_wrapper_sends_one_flattened_patch_per_grid_cell() -> None:
    tower = _RecordingTower()
    wrapped = backbone_hf.NaflexVision(tower, patch_size=16)

    result = wrapped(torch.zeros(2, 3, 224, 224))

    # 224/16 = a 14x14 grid, each patch flattened to 16*16*3.
    assert tower.seen["pixel_values"].shape == (2, 196, 768)
    assert torch.equal(tower.seen["pixel_attention_mask"], torch.ones(2, 196, dtype=torch.long))
    assert torch.equal(tower.seen["spatial_shapes"], torch.tensor([[14, 14], [14, 14]]))
    assert torch.equal(result, torch.full((2, 4), 7.0))


def test_the_naflex_wrapper_flattens_each_patch_channels_last() -> None:
    # transformers' own convert_image_to_patches permutes to (gh, gw, p, p, c),
    # so a patch flattens as pixel-major and channel-minor. Getting this
    # backwards still yields embeddings of the right shape -- they are simply
    # meaningless, because the patch-embedding Linear was fitted to the other
    # ordering. Pinned deliberately.
    image = torch.zeros(1, 3, 2, 2)
    image[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])  # one pixel, three channels
    image[0, :, 1, 1] = torch.tensor([4.0, 5.0, 6.0])

    tower = _RecordingTower()
    backbone_hf.NaflexVision(tower, patch_size=2)(image)

    patch = tower.seen["pixel_values"][0, 0]
    assert torch.equal(patch, torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 4.0, 5.0, 6.0]))


def test_the_naflex_wrapper_matches_the_reference_patchifier() -> None:
    # Cross-check the whole permutation against a transcription of
    # transformers.models.siglip2.image_processing_siglip2.convert_image_to_patches.
    def reference(image: torch.Tensor, patch_size: int) -> torch.Tensor:
        c, h, w = image.shape
        gh, gw = h // patch_size, w // patch_size
        patched = image.reshape(c, gh, patch_size, gw, patch_size)
        return patched.permute(1, 3, 2, 4, 0).reshape(gh * gw, -1)

    images = torch.randn(2, 3, 32, 32)
    tower = _RecordingTower()
    backbone_hf.NaflexVision(tower, patch_size=16)(images)

    expected = torch.stack([reference(image, 16) for image in images])
    assert torch.equal(tower.seen["pixel_values"], expected)


def test_the_naflex_wrapper_rejects_a_crop_that_is_not_a_whole_number_of_patches() -> None:
    wrapped = backbone_hf.NaflexVision(_RecordingTower(), patch_size=16)
    with pytest.raises(ValueError, match="multiple of the patch size"):
        wrapped(torch.zeros(1, 3, 220, 220))


def test_the_naflex_wrapper_mean_pools_when_the_tower_has_no_pooler() -> None:
    class Unpooled(torch.nn.Module):
        def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):  # noqa: D102
            class Output:
                last_hidden_state = torch.arange(
                    pixel_values.shape[0] * 4 * 2, dtype=torch.float32
                ).reshape(pixel_values.shape[0], 4, 2)

            return Output()

    wrapped = backbone_hf.NaflexVision(Unpooled(), patch_size=16)
    result = wrapped(torch.zeros(1, 3, 32, 32))
    assert result.shape == (1, 2)
    assert torch.equal(result, torch.tensor([[3.0, 4.0]]))


def test_the_clip_checkpoint_is_registered_at_both_of_its_widths() -> None:
    # jiahui's detector reads the 768-d pre-projection pooled output; the 512-d
    # post-projection embedding is the one CLIP is usually used through. The two
    # differ by a single matrix, so both are registered and the sweep can say
    # which one the ladder prefers rather than assuming.
    assert BACKBONES["clip_b32_hf"].dim == 768
    assert BACKBONES["clip_b32_proj_hf"].dim == 512


def test_clip_carries_its_own_patch_size_and_clip_normalisation() -> None:
    for name in ("clip_b32_hf", "clip_b32_proj_hf"):
        spec = BACKBONES[name]
        assert spec.patch_size == 32, name
        assert spec.mean == backbone_hf.CLIP_MEAN, name
        assert spec.std == backbone_hf.CLIP_STD, name


def test_registering_clip_does_not_import_transformers() -> None:
    assert "transformers" not in sys.modules


class _StubVisionTower(torch.nn.Module):
    """Stands in for CLIPVisionTransformer, which publishes a pooler_output."""

    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, pixel_values):  # noqa: D102
        n = pixel_values.shape[0]
        dim_ = self.dim

        class Output:
            pooler_output = torch.arange(n * dim_, dtype=torch.float32).reshape(n, dim_)
            last_hidden_state = torch.zeros(n, 2, dim_)

        return Output()


def test_the_projection_wrapper_applies_the_projection_to_the_pooled_output() -> None:
    tower = _StubVisionTower(dim=6)
    projection = torch.nn.Linear(6, 3, bias=False)
    wrapped = backbone_hf.ProjectedVision(tower, projection)

    pixels = torch.zeros(2, 3, 32, 32)
    got = wrapped(pixels)

    pooled = tower(pixels).pooler_output
    assert got.shape == (2, 3)
    assert torch.allclose(got, projection(pooled))


def test_the_projection_wrapper_falls_back_to_the_class_token() -> None:
    # A tower without a pooler must still project token 0 rather than crash --
    # the same contract PooledHFVision keeps for the un-projected arm.
    class _NoPooler(torch.nn.Module):
        def forward(self, pixel_values):  # noqa: D102
            n = pixel_values.shape[0]

            class Output:
                pooler_output = None
                last_hidden_state = torch.ones(n, 4, 6)

            return Output()

    projection = torch.nn.Linear(6, 3, bias=False)
    wrapped = backbone_hf.ProjectedVision(_NoPooler(), projection)

    got = wrapped(torch.zeros(2, 3, 32, 32))

    assert got.shape == (2, 3)
    assert torch.allclose(got, projection(torch.ones(2, 6)))
