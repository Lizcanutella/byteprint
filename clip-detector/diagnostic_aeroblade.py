"""
Quick, cheap diagnostic: does BYTEPRINT's AEROBLADE (training-free
reconstruction-error expert) carry any real/AI signal on OUR held-out
data, and does it add anything on top of our own CLIP-based pipeline's
P(AI) score when fused? Unlike the native-crop idea (proven not to
transfer to CLIP, 3 independent tests), AEROBLADE is a genuinely
different signal source - proximity to a latent-diffusion decoder's
output manifold - so it's not competing with the same information our
pipeline already extracts.

Uses byteprint's own recon.py module directly (imported from the
sibling repo at ~/byteprint) rather than reimplementing AEROBLADE -
reuses their tested code as-is. Starts with ONE autoencoder (vae-mse,
the smallest/fastest to download and run) on a small (~40 real + 40 AI)
held-out sample, since AEROBLADE is expensive on CPU (BYTEPRINT's own
README reports ~2.9-13.7s/crop) - only adds more autoencoders or a
bigger sample if this shows real promise.

Usage:
    python diagnostic_aeroblade.py
"""

import json
import sys
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/jh/byteprint")
from byteprint.recon import load_recon_expert, aeroblade_score  # noqa: E402

from clip_features import load_image_capped  # noqa: E402
from production_pipeline import predict_proba  # noqa: E402
from train_classifier import compute_split  # noqa: E402

N_PER_CLASS = 40
CROP_SIZE = 224


def to_224_crop(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = CROP_SIZE / min(w, h)
    img = img.resize((max(CROP_SIZE, round(w * scale)), max(CROP_SIZE, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - CROP_SIZE) // 2, (h - CROP_SIZE) // 2
    return np.asarray(img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)))


def main():
    paths, labels, sources, train_idx, test_idx = compute_split()
    labels_arr = np.array(labels)

    real_idx = [i for i in test_idx if labels_arr[i] == 0][:N_PER_CLASS]
    ai_idx = [i for i in test_idx if labels_arr[i] == 1][:N_PER_CLASS]
    sample_idx = real_idx + ai_idx
    y = np.array([labels_arr[i] for i in sample_idx])
    print(f"Sample: {len(real_idx)} real + {len(ai_idx)} ai = {len(sample_idx)} images")

    print("Loading images + center-cropping to 224x224...")
    pil_imgs = [load_image_capped(paths[i]) for i in sample_idx]
    crops = [to_224_crop(img) for img in pil_imgs]

    print("Loading AEROBLADE (1 autoencoder: vae-mse) - this downloads VAE weights, may take a bit...")
    t0 = time.time()
    recon_expert = load_recon_expert(["vae-mse"], device="cpu")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Computing reconstruction distances (this is the slow part on CPU)...")
    t0 = time.time()
    distances = recon_expert.embed(crops)  # (N, 1)
    print(f"  done in {time.time() - t0:.1f}s ({(time.time() - t0) / len(crops):.2f}s/image)")

    aero_scores = aeroblade_score(distances)  # higher = more likely synthetic
    auc_aero = float(roc_auc_score(y, aero_scores))
    print(f"\nAEROBLADE alone AUROC: {auc_aero:.4f}")

    print("\nComputing our own CLIP pipeline's P(AI) score on the same images...")
    our_probs, _ = predict_proba(pil_imgs, batch_size=32)
    auc_ours = float(roc_auc_score(y, our_probs))
    print(f"Our pipeline alone AUROC (on this small clean sample): {auc_ours:.4f}")

    print("\nFusing [our_score, aeroblade_score] via cross-validated logistic regression...")
    X_fused = np.column_stack([our_probs, aero_scores])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    fused_preds = cross_val_predict(clf, X_fused, y, cv=cv, method="predict_proba")[:, 1]
    auc_fused = float(roc_auc_score(y, fused_preds))
    print(f"Fused (5-fold CV) AUROC: {auc_fused:.4f}")

    result = {
        "n_samples": len(sample_idx),
        "aeroblade_alone_auroc": auc_aero,
        "our_pipeline_alone_auroc": auc_ours,
        "fused_5fold_cv_auroc": auc_fused,
        "aeroblade_advantage_over_ours": auc_fused - auc_ours,
    }
    with open("results_detector/aeroblade_diagnostic.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to results_detector/aeroblade_diagnostic.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
