# Extending BYTEPRINT

Everything worth swapping is a **registry entry**, so adding one means writing a
function in your own module and decorating it. You never edit a file another
branch is also editing, and `--plugin` makes your module's registrations
available to every command:

```bash
byteprint train --plugin myteam.heads --head my-head --cache runs/cache/train --out runs/probe.joblib
```

`BYTEPRINT_PLUGINS=myteam.heads,myteam.backbones` does the same thing for a whole
batch job. `byteprint list` shows what is currently registered.

Registering a name that already exists is an **error**, not a silent override —
two branches both registering `"probe"` and letting import order decide is the
kind of bug that costs a day. Pass `replace=True` if shadowing is deliberate.

---

## 1. A backbone — the model

Contract: a no-argument function returning a frozen `nn.Module` that maps
`(n, 3, h, w)` to `(n, dim)`. Pooled features, no classification head.

```python
# myteam/backbones.py
from byteprint.backbone import register_backbone

@register_backbone("siglip2_so400m", dim=1152, patch_size=14, mean=(0.5,) * 3, std=(0.5,) * 3)
def _build():
    import timm
    return timm.create_model(
        "vit_so400m_patch14_siglip_384", pretrained=True, num_classes=0
    )
```

- `patch_size` is validated against `--crop-size`, which must be a multiple of it.
- `mean`/`std` default to ImageNet. SigLIP and CLIP do not use ImageNet
  statistics; getting this wrong costs a few AUC points and is invisible.
- Mind the competition's **<2B parameter** budget, summed across the ensemble.
  DINOv2-giant is ~1.1B of it on its own.

A new backbone means a new cache: embeddings are keyed by extraction config, and
`EmbeddingStore` will refuse to mix widths.

## 2. A head — the training objective

This is where the learning happens, since the backbone is frozen. Contract: a
factory taking a `ProbeConfig` and returning a scikit-learn estimator with
`fit(X, y)` and `predict_proba(X)`.

```python
# myteam/heads.py
from byteprint.heads import register_head
from sklearn.ensemble import HistGradientBoostingClassifier

@register_head("gbt")
def _build(config):
    return HistGradientBoostingClassifier(random_state=config.seed)
```

`predict_proba` is not optional: a head without probabilities cannot be
thresholded at a false-positive budget, which is the only operating point this
project reports.

Shipped: `logreg` (log loss, the default and the baseline to beat),
`linear-svm` (hinge loss, calibrated), `mlp` (one hidden layer, cross-entropy).

## 3. A crop strategy — where you look

Contract: `sample(image, *, crop_size, top_k, candidates, rng)` returning uint8
HWC arrays of `crop_size` square. The image arrives RGB and already upscaled to
at least `crop_size` unless you pass `pad=False`.

```python
# myteam/crops.py
from byteprint.crops import register_crop_mode, texture_score

@register_crop_mode("high-freq-corners")
def _sample(image, *, crop_size, top_k, candidates, rng):
    h, w = image.shape[:2]
    corners = [(0, 0), (0, w - crop_size), (h - crop_size, 0), (h - crop_size, w - crop_size)]
    crops = [image[t:t + crop_size, l:l + crop_size] for t, l in corners]
    crops.sort(key=texture_score, reverse=True)
    return crops[:top_k]
```

## 4. An autoencoder for the reconstruction expert

`AUTOENCODERS` in `byteprint/recon.py` maps a short id to `(hf_repo, subfolder)`.
Add a line. Any diffusers-loadable `AutoencoderKL` works. Then
`--aes sd15,vae-mse,yours`.

The minimum over the bank is taken at **scoring** time, so which autoencoders
count can be ablated without re-extracting anything.

---

## What is deliberately *not* pluggable

`OFFICIAL_LADDER` in `byteprint/launder.py`. It is section 5.2 of the competition
brief transcribed verbatim, and it is pinned by a test. Changing it invalidates
every robustness number anyone has quoted, including the ones already in the
report. Exploratory transform chains go in `STRESS_LADDER`, which is reported
separately and labelled as beyond the brief.

Note the unit convention it enforces: `noise` sigma is a **fraction of full
scale**, so `noise:0.10` is sigma = 25.5 of 255 levels. Values above 1.0 are
rejected rather than reinterpreted, because the previous ladder used 0–255 units
and a silently misread `noise:3` would have destroyed the image.
