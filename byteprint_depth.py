"""A SigLIP2 backbone that reports every depth at once, for the depth frontier.

The shipped adapters keep only the last thing the tower produces --
``PooledHFVision`` returns ``pooler_output``, ``NaflexVision`` the same -- so
every number this project has published describes the *final* layer. Nobody
checked whether that is the right layer to read.

There is reason to doubt it. SigLIP2's late blocks are trained to align an image
with a caption, which is a semantic objective; the cues a forensic head needs
are low-level statistics of texture and grain. If those are strongest somewhere
in the middle of the tower, the last third of the network is not merely
redundant for this task, it is discarding signal -- and truncating there is a
smaller detector *and* a better one.

Measuring that costs almost nothing extra, which is the point of this module.
``output_hidden_states=True`` returns every layer from the **same forward
pass**, so tapping eleven depths costs what tapping one costs. Each tap is
mean-pooled over patch tokens and the taps are concatenated on the feature axis,
which keeps the ``(n, 3, h, w) -> (n, dim)`` contract intact: no change to
extraction, the cache, or the CLI. Afterwards a probe per tap is a column slice
and a few seconds of CPU.

    BYTEPRINT_PLUGINS=byteprint_depth byteprint extract --backbone siglip2_depth_hf ...

**The last block is the tower's own ``pooler_output``, not a mean.** SigLIP2
pools with an attention head, so mean-over-patches at the final layer is a
different function from what the published runs used. Carrying the real pooled
output as its own block is what makes a probe fitted on it reproduce the
published 0.9497 exactly -- the control that says the plumbing is right -- and it
prices attention pooling against mean pooling at the same depth for free.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from byteprint.backbone import register_backbone

# Fractions of the tower's depth to tap, denser early. The hypothesis lives in
# the first half, so that is where the curve needs resolution; the late blocks
# are expected to be flat and are sampled just densely enough to show it. Round
# percentages rather than a tuned schedule -- there is nothing to tune yet, and
# a schedule that looked fitted would invite the question of what it was fitted
# to.
TAP_PERCENTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00)

# Every tap, plus the tower's own pooled output.
N_BLOCKS = len(TAP_PERCENTS) + 1

# SigLIP2-so400m's width. Declared here because `register_backbone` needs `dim`
# before any weights are loaded; the builder checks the loaded tower agrees.
SIGLIP2_SO400M_WIDTH = 1152

# Below this depth the percentages above collide and the row would quietly hold
# fewer blocks than `dim` promises. so400m is 27 layers, comfortably clear.
MIN_DEPTH = 20


def tap_layers(num_layers: int) -> list[int]:
    """Which layer indices to tap in a tower of ``num_layers`` blocks.

    One-indexed, so layer ``k`` is ``hidden_states[k]`` -- index 0 of that tuple
    is the embedding output, before any block has run. The final layer is always
    included, so the frontier always contains the configuration in production.
    """
    if num_layers < MIN_DEPTH:
        raise ValueError(
            f"the tap schedule needs a tower of at least {MIN_DEPTH} layers to keep "
            f"{len(TAP_PERCENTS)} distinct taps, got {num_layers}"
        )
    # floor(x + 0.5) rather than round(), which is banker's rounding in Python
    # and would land a tap on a different layer for some depths than for others.
    taps = sorted({max(1, math.floor(p * num_layers + 0.5)) for p in TAP_PERCENTS})
    if len(taps) != len(TAP_PERCENTS):  # pragma: no cover -- MIN_DEPTH prevents it
        raise ValueError(f"tap percentages collide at depth {num_layers}: {taps}")
    return taps


def block_slice(position: int, *, width: int = SIGLIP2_SO400M_WIDTH) -> slice:
    """The columns holding block ``position`` of a cached row.

    Blocks are laid out in ascending tap order with the pooled output last, so
    ``position`` indexes ``tap_layers(...)`` directly and ``N_BLOCKS - 1`` is the
    pooler. A mean over an image's crops commutes with a slice over columns, so
    the block read back here is exactly what a single-tap extraction would have
    cached -- whichever side of the cache that mean happens on. It used to
    happen on write, when ``EmbeddingStore`` stored one pooled row per image;
    since the crop-pooling work it happens on read, and the identity is what
    lets the depth curve be swept at any crop count from one extraction.
    """
    if not 0 <= position < N_BLOCKS:
        raise ValueError(f"block {position} is outside the {N_BLOCKS} stored blocks")
    return slice(position * width, (position + 1) * width)


class MultiDepthNaflex(nn.Module):
    """SigLIP2's naflex tower, returning every tapped depth concatenated.

    The patch flattening is copied from ``byteprint.backbone_hf.NaflexVision``
    rather than imported, because that loop lives inside its ``forward`` and
    lifting it out would mean editing a shared module. The ordering is
    load-bearing -- pixel-major, channel-minor, matching transformers' own
    ``convert_image_to_patches`` -- and the other ordering produces embeddings of
    exactly the right shape and no meaning at all. A test pins this copy against
    the shipped one; duplicated code earns an equality test, not a comment.
    """

    def __init__(self, tower: nn.Module, patch_size: int) -> None:
        super().__init__()
        self.tower = tower
        self.patch_size = patch_size
        self.taps = tap_layers(int(tower.config.num_hidden_layers))

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
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states
        # SigLIP has no class token, so the mean over patches is the right
        # summary of a layer -- token 0 would be an arbitrary corner of the crop.
        blocks = [hidden[layer].mean(dim=1) for layer in self.taps]

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:  # pragma: no cover -- so400m always publishes one
            pooled = outputs.last_hidden_state.mean(dim=1)
        blocks.append(pooled)

        return torch.cat(blocks, dim=1)


@register_backbone(
    "siglip2_depth_hf",
    dim=N_BLOCKS * SIGLIP2_SO400M_WIDTH,
    patch_size=16,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
def _build() -> nn.Module:
    from transformers import AutoModel

    # local_files_only: fail loudly on a compute node with an incomplete cache
    # rather than hanging on a network call that cannot succeed.
    model = AutoModel.from_pretrained(
        "google/siglip2-so400m-patch16-naflex", local_files_only=True
    )
    tower = model.vision_model
    hidden_size = int(tower.config.hidden_size)
    if hidden_size != SIGLIP2_SO400M_WIDTH:
        raise ValueError(
            f"this plugin registers a width of {N_BLOCKS} x {SIGLIP2_SO400M_WIDTH}, but the "
            f"staged tower is {hidden_size} wide; the cache would refuse the rows"
        )
    return MultiDepthNaflex(tower, patch_size=16)
