"""Video -> candidate frames, with quality metrics attached.

Handles multiple clips as one logical walkthrough: the glasses cap recording
at 3 minutes, so a 10-minute closet tour arrives as 4 files. We concatenate
them in filename order into one ordered frame list, so a garment that
spans a clip boundary still clusters correctly.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config


@dataclass
class Frame:
    source: str
    """Which clip this came from."""
    timestamp: float
    """Seconds into that clip."""
    image: np.ndarray = field(repr=False)
    """Downscaled BGR image used for embedding and scoring."""

    sharpness: float = 0.0
    exposure: float = 0.0
    centrality: float = 0.0
    area: float = 1.0

    def quality(self, cfg: Config) -> float:
        return (
            cfg.w_sharpness * self.sharpness
            + cfg.w_area * self.area
            + cfg.w_exposure * self.exposure
            + cfg.w_centrality * self.centrality
        )


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_score(gray: np.ndarray) -> float:
    """1.0 when the histogram is well spread, falling off with clipping.

    Closets are badly lit and often backlit by a window or a bulb directly
    overhead, so blown highlights are common.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    clipped = float(hist[:4].sum() + hist[-4:].sum())
    return float(np.clip(1.0 - clipped * 4.0, 0.0, 1.0))


def _centrality_score(gray: np.ndarray) -> float:
    """How much of the frame's detail sits away from the lens edges.

    Ultra-wide optics are softest and most distorted at the periphery, so a
    garment centred in frame is worth more than the same garment at the edge
    even at identical sharpness.
    """
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    weight = np.clip(1.0 - r / 1.6, 0.0, 1.0)

    detail = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    total = detail.sum()
    if total <= 0:
        return 0.0
    return float((detail * weight).sum() / total)


def _resize(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def extract_clip(path: str, cfg: Config) -> list[Frame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / cfg.sample_fps)), 1)

    frames: list[Frame] = []
    pos = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if pos % step == 0:
            small = _resize(img, cfg.max_frame_width)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            frames.append(
                Frame(
                    source=os.path.basename(path),
                    timestamp=pos / src_fps,
                    image=small,
                    sharpness=_laplacian_variance(gray),
                    exposure=_exposure_score(gray),
                    centrality=_centrality_score(gray),
                )
            )
        pos += 1

    cap.release()
    return frames


def filter_blurry(frames: list[Frame], cfg: Config) -> list[Frame]:
    """Drop motion-blurred frames.

    Uses a percentile rather than a fixed threshold because absolute
    Laplacian variance shifts a lot with lighting and garment texture — a
    cable knit scores far higher than flat black wool at equal focus.
    """
    if not frames:
        return []
    sharps = np.array([f.sharpness for f in frames])
    cut = max(
        float(np.percentile(sharps, cfg.blur_percentile)),
        float(np.median(sharps)) * cfg.blur_rel_floor,
    )
    kept = [f for f in frames if f.sharpness >= cut]
    return kept or frames  # never return empty; better blurry than nothing


def normalize_quality(frames: list[Frame]) -> None:
    """Rescale sharpness to 0..1 in place, so the weighted score is meaningful.

    Exposure and centrality are already bounded; sharpness is unbounded and
    would otherwise dominate the sum.
    """
    if not frames:
        return
    s = np.array([f.sharpness for f in frames], dtype=float)
    lo, hi = float(s.min()), float(s.max())
    span = max(hi - lo, 1e-6)
    for f in frames:
        f.sharpness = (f.sharpness - lo) / span


def load_walkthrough(paths: list[str], cfg: Config) -> tuple[list[Frame], list[str]]:
    """Extract every clip in a walkthrough into one ordered frame list.

    Returns the frames and the resolved clip paths, so downstream stages
    (audio boundary detection) can find the source media again.
    """
    expanded: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("mp4", "mov", "MP4", "MOV"):
                expanded.extend(sorted(glob.glob(os.path.join(p, f"*.{ext}"))))
        else:
            expanded.append(p)

    if not expanded:
        raise RuntimeError(f"no video files found in {paths}")

    frames: list[Frame] = []
    for path in expanded:
        try:
            clip = extract_clip(path, cfg)
        except RuntimeError as e:
            print(f"  skipping {os.path.basename(path)}: {e}")
            continue
        if not clip:
            # Zero-length files happen: a recording stopped the instant it
            # started, or the glasses hit the clip cap on a frame boundary.
            print(f"  skipping {os.path.basename(path)}: no frames")
            continue
        frames.extend(clip)

    if not frames:
        raise RuntimeError("no readable frames in any clip")

    frames = filter_blurry(frames, cfg)
    normalize_quality(frames)
    return frames, expanded
