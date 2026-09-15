"""Frame embeddings.

Two backends:

  dinov2  — what you should actually use. DINOv2 features are self-supervised
            and instance-discriminative, which is what we need. CLIP is the
            wrong tool here: it embeds *semantics*, so two different navy
            crewnecks land almost on top of each other. That is precisely the
            collision this pipeline cannot afford.

  hist    — colour histogram + coarse gradient signature. No downloads, no
            torch. Good enough to exercise the pipeline end to end and to
            smoke-test parameter sweeps, and genuinely decent when garments
            differ in colour. Useless for two black blazers.

Both return L2-normalised vectors so cosine similarity is a dot product.
"""

from __future__ import annotations

import numpy as np
import cv2


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


# --------------------------------------------------------------------------
# histogram backend
# --------------------------------------------------------------------------

def _hist_embed_one(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - small.mean()) / max(small.std(), 1e-6)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (9 / (2 * np.pi))
    gh, _ = np.histogram(ang, bins=9, range=(0, 9), weights=mag)
    gh = gh / max(gh.sum(), 1.0)

    return _l2(np.concatenate([hist * 2.0, small.ravel() * 0.15, gh]))


def embed_hist(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([_hist_embed_one(im) for im in images])


# --------------------------------------------------------------------------
# dinov2 backend
# --------------------------------------------------------------------------

def embed_dinov2(images: list[np.ndarray], model_name: str = "facebook/dinov2-base",
                 batch_size: int = 16) -> np.ndarray:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "dinov2 backend needs torch + transformers:\n"
            "  pip install torch transformers\n"
            "Or run with --embedder hist to exercise the pipeline offline."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[i:i + batch_size]]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            feats = model(**inputs).last_hidden_state[:, 0]  # CLS token
            out.append(feats.cpu().numpy())

    return _l2(np.concatenate(out))


EMBEDDERS = {"hist": embed_hist, "dinov2": embed_dinov2}


def embed(images: list[np.ndarray], backend: str = "dinov2") -> np.ndarray:
    if backend not in EMBEDDERS:
        raise ValueError(f"unknown embedder {backend!r}; choose from {list(EMBEDDERS)}")
    return EMBEDDERS[backend](images)
