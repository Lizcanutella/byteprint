"""The depth-frontier plugin: where the tap layers land, and what a row contains.

None of this needs weights or `transformers`. The tap schedule is arithmetic,
the block layout is a reshape, and the tower is stubbed -- so the parts that
decide whether an hour of GPU time produces a meaningful cache are all pinned
on a laptop, before the job is submitted.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from byteprint_depth import (
    N_BLOCKS,
    TAP_PERCENTS,
    MultiDepthNaflex,
    block_slice,
    tap_layers,
)

WIDTH = 6
PATCH = 2


class StubTower(nn.Module):
    """Stands in for SigLIP2's vision tower: known hidden states, known pooling.

    Layer i's hidden state is filled with the value i, so a pooled block is
    recognisable on sight and a mis-indexed tap cannot pass.
    """

    def __init__(self, num_layers: int = 27, width: int = WIDTH) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=width)
        self.width = width
        self.num_layers = num_layers
        self.seen: list[dict] = []

    def forward(self, **kwargs):
        self.seen.append(kwargs)
        patches = kwargs["pixel_values"]
        n, tokens = patches.shape[0], patches.shape[1]
        # hidden_states[i] is the output of layer i; index 0 is the embedding.
        hidden = tuple(
            torch.full((n, tokens, self.width), float(i)) for i in range(self.num_layers + 1)
        )
        return SimpleNamespace(
            hidden_states=hidden,
            last_hidden_state=hidden[-1],
            # Deliberately NOT the mean of the last hidden state: the attention
            # pooler is a different function, which is the whole reason it gets
            # its own block.
            pooler_output=torch.full((n, self.width), -1.0),
        )


def pixels(n: int = 2, size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=generator)


# -- the tap schedule -----------------------------------------------------


def test_the_taps_for_a_27_layer_tower_are_denser_early() -> None:
    assert tap_layers(27) == [1, 3, 4, 5, 7, 9, 12, 15, 19, 23, 27]


def test_the_final_layer_is_always_tapped() -> None:
    for depth in range(20, 60):
        assert tap_layers(depth)[-1] == depth


def test_taps_are_strictly_increasing_and_inside_the_tower() -> None:
    for depth in range(20, 60):
        taps = tap_layers(depth)
        assert taps == sorted(set(taps))
        assert taps[0] >= 1 and taps[-1] <= depth


def test_every_supported_depth_yields_one_tap_per_percent() -> None:
    # A collision would silently shrink the row and make `dim` a lie.
    for depth in range(20, 60):
        assert len(tap_layers(depth)) == len(TAP_PERCENTS)


def test_a_tower_too_shallow_to_separate_the_taps_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        tap_layers(19)


def test_the_block_count_is_the_taps_plus_the_pooler() -> None:
    assert N_BLOCKS == len(TAP_PERCENTS) + 1


# -- the row layout -------------------------------------------------------


def test_a_row_is_one_block_per_tap_plus_one_for_the_pooler() -> None:
    tower = StubTower()
    model = MultiDepthNaflex(tower, patch_size=PATCH)
    out = model(pixels())
    assert out.shape == (2, N_BLOCKS * WIDTH)


def test_each_block_holds_the_mean_over_that_layers_patch_tokens() -> None:
    tower = StubTower()
    model = MultiDepthNaflex(tower, patch_size=PATCH)
    out = model(pixels()).detach().numpy()

    for position, layer in enumerate(tap_layers(27)):
        block = out[:, block_slice(position, width=WIDTH)]
        assert np.allclose(block, float(layer)), f"block {position} should hold layer {layer}"


def test_the_last_block_is_the_towers_own_pooler_not_a_mean() -> None:
    # This block is what makes the run comparable to the published SigLIP2
    # number; if it silently became a mean the control would be worthless.
    tower = StubTower()
    model = MultiDepthNaflex(tower, patch_size=PATCH)
    out = model(pixels()).detach().numpy()
    assert np.allclose(out[:, block_slice(N_BLOCKS - 1, width=WIDTH)], -1.0)


def test_hidden_states_are_requested_or_there_would_be_nothing_to_tap() -> None:
    tower = StubTower()
    MultiDepthNaflex(tower, patch_size=PATCH)(pixels())
    assert tower.seen[0]["output_hidden_states"] is True


def test_a_tower_shallower_than_the_schedule_fails_at_build_not_mid_run() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        MultiDepthNaflex(StubTower(num_layers=12), patch_size=PATCH)


def test_a_crop_that_does_not_tile_the_patch_grid_is_rejected() -> None:
    model = MultiDepthNaflex(StubTower(), patch_size=PATCH)
    with pytest.raises(ValueError, match="multiple of the patch size"):
        model(pixels(size=5))


# -- the property the whole design rests on -------------------------------


def test_slicing_a_block_out_of_a_pooled_row_equals_pooling_that_block_alone() -> None:
    """Mean-over-crops commutes with column slicing.

    `EmbeddingStore.add` averages an image's crops into one row before anything
    downstream sees it. The depth frontier reads a single tap back out of that
    row afterwards, and is only honest if doing so gives exactly what a
    single-tap extraction would have cached. That is this identity.
    """
    rng = np.random.default_rng(0)
    crops = rng.normal(size=(2, N_BLOCKS * WIDTH)).astype(np.float32)

    pooled_then_sliced = crops.mean(axis=0)[block_slice(4, width=WIDTH)]
    sliced_then_pooled = crops[:, block_slice(4, width=WIDTH)].mean(axis=0)

    assert np.array_equal(pooled_then_sliced, sliced_then_pooled)


def test_the_blocks_tile_the_row_without_overlap_or_gap() -> None:
    covered = np.zeros(N_BLOCKS * WIDTH, dtype=int)
    for position in range(N_BLOCKS):
        covered[block_slice(position, width=WIDTH)] += 1
    assert np.array_equal(covered, np.ones_like(covered))


# -- the duplicated patch flattening --------------------------------------


def test_the_patch_flattening_matches_the_shipped_naflex_adapter() -> None:
    """The plugin re-implements NaflexVision's flattening; pin them together.

    That ordering is load-bearing -- the wrong one yields embeddings of exactly
    the right shape and no meaning at all -- and this plugin cannot import the
    loop out of a shared module without editing it. So it is duplicated, and
    duplicated code gets an equality test rather than a comment.
    """
    from byteprint.backbone_hf import NaflexVision

    shipped_tower, plugin_tower = StubTower(), StubTower()
    NaflexVision(shipped_tower, patch_size=PATCH)(pixels())
    MultiDepthNaflex(plugin_tower, patch_size=PATCH)(pixels())

    assert torch.equal(
        shipped_tower.seen[0]["pixel_values"], plugin_tower.seen[0]["pixel_values"]
    )
    assert torch.equal(
        shipped_tower.seen[0]["spatial_shapes"], plugin_tower.seen[0]["spatial_shapes"]
    )
