"""
Frozen CLIP image-embedding extractor. This is the backbone for the
detector: CLIP:ViT-B/32 (~151M params, well under the hackathon's <2B
limit) pretrained image features, with a small classifier head trained
on top (see train_classifier.py). Standard PyTorch device handling -
uses CUDA automatically if available, falls back to CPU otherwise, so
the same code runs unmodified on this CPU-only proof-of-concept machine
and on the GPU environment used for the "real" run.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP resizes to 224x224 internally anyway; some source images (notably
# itsLeen/deepfake_vs_real_image, native-resolution, up to ~5700px on a
# side) OOM'd this machine when decoded and held in memory at full size.
# Every script that loads images from disk should go through this.
MAX_SIDE = 512


def load_image_capped(path, max_side=MAX_SIDE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return img


def load_image_native(path):
    """No resize cap at all - use for GPU crop-based extraction
    (embed_images_preproj_crops), where select_crops needs the true
    native resolution to find genuinely high-frequency patches. Do NOT
    use this for whole-image embedding on a memory-constrained machine -
    that's exactly what load_image_capped exists to prevent."""
    return Image.open(path).convert("RGB")

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        n_params = sum(p.numel() for p in _model.parameters())
        print(f"Loaded {MODEL_ID} on {DEVICE} ({n_params/1e6:.1f}M params)")
    return _model, _processor


@torch.no_grad()
def embed_images(pil_images, batch_size=16):
    """pil_images: list of PIL RGB images. Returns (N, 512) float32 numpy
    array of L2-normalized CLIP image embeddings (POST-projection - the
    space CLIP was contrastively trained to align with text in)."""
    model, processor = get_model()
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        # newer `transformers` returns a BaseModelOutputWithPooling whose
        # .pooler_output has been overwritten with the projected image
        # embedding (see CLIPModel.get_image_features source).
        feats = model.get_image_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


@torch.no_grad()
def embed_images_preproj(pil_images, batch_size=16):
    """Same as embed_images, but returns the (N, 768) PRE-projection
    vision-encoder pooled features (before CLIP's visual_projection layer).
    The projection is trained for image-text alignment and may discard
    purely-visual information (e.g. subtle generation artifacts) that
    isn't relevant to matching captions - the raw vision features may
    retain more of it. Not L2-normalized by CLIP itself, so we normalize
    here for consistency with embed_images.

    NOTE: CLIPProcessor resizes whatever it's given down to 224x224
    internally, so feeding this a larger image than 224px does NOT
    preserve extra detail - it's downsampled either way. For genuine
    native-resolution signal, use embed_images_preproj_crops instead."""
    model, processor = get_model()
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        feats = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _select_crops_batch(pil_images, top_k, candidates, seed):
    from crops import select_crops
    rng = np.random.default_rng(seed)
    return [select_crops(img, top_k=top_k, candidates=candidates, rng=rng) for img in pil_images]


def _pool_and_normalize(flat_embs, crop_counts):
    pooled, start = [], 0
    for n in crop_counts:
        pooled.append(flat_embs[start:start + n].mean(axis=0))
        start += n
    pooled = np.stack(pooled).astype(np.float32)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


@torch.no_grad()
def embed_images_preproj_crops(pil_images, top_k=3, candidates=12, batch_size=16, seed=0):
    """GPU-oriented variant: for each image, take `top_k` native-resolution
    224x224 texture-rich crops (crops.select_crops) instead of letting
    CLIPProcessor squash the whole image down to 224x224 - the crops are
    already 224x224 native pixels, so no resize/detail-loss happens for
    them. Each image's embedding is the mean of its crops' pre-projection
    embeddings. Multiplies backbone compute by ~top_k over embed_images_
    preproj, hence "GPU-oriented" - impractical on this project's
    CPU-only sandbox, intended for the free-GPU (Colab/Kaggle) retrain.

    Images should be loaded with clip_features.load_image_native (no
    resize cap), not load_image_capped, or the crops won't be genuinely
    native-resolution."""
    crops_per_image = _select_crops_batch(pil_images, top_k, candidates, seed)
    flat = [crop for crops in crops_per_image for crop in crops]
    flat_embs = embed_images_preproj(flat, batch_size=batch_size)
    return _pool_and_normalize(flat_embs, [len(c) for c in crops_per_image])


@torch.no_grad()
def embed_images_preproj_crops_with_delta(pil_images, probe_fn, top_k=3, candidates=12, batch_size=16, seed=0):
    """Like embed_images_preproj_crops, but also returns the reactivity-
    delta feature (see production_pipeline.py) computed on the SAME crop
    locations - the probe is applied to each selected crop directly, not
    to the whole image followed by a fresh crop selection, so the delta
    isolates the probe's effect rather than being confounded by a
    different set of crop locations (probing can shift texture scores
    and therefore which crops get selected).

    Returns (base_embs, delta_embs), each (N, 768), both pooled+normalized."""
    crops_per_image = _select_crops_batch(pil_images, top_k, candidates, seed)
    flat = [crop for crops in crops_per_image for crop in crops]
    probed_flat = [probe_fn(crop) for crop in flat]

    flat_base = embed_images_preproj(flat, batch_size=batch_size)
    flat_probed = embed_images_preproj(probed_flat, batch_size=batch_size)

    counts = [len(c) for c in crops_per_image]
    base = _pool_and_normalize(flat_base, counts)
    probed = _pool_and_normalize(flat_probed, counts)
    return base, probed - base
