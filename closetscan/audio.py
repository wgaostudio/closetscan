"""Garment boundaries from narration.

In a real walkthrough the strongest segmentation signal turns out not to be
visual at all. The closet background is constant — same wall, same rail, same
shelf — so frame-to-frame visual change is dominated by hand motion rather
than by which garment is being held. But people narrate: "here's a sweater",
"this one", "this, uh, interesting". Each garment gets an utterance, and the
pauses between utterances land almost exactly on the handoffs.

This module finds those pauses from the audio energy envelope. No ASR, no
model download — just ffmpeg and numpy.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import os

import numpy as np


def _decode_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Decode a clip's audio to mono float32 at `sr` via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path,
             "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", out],
            check=True, capture_output=True,
        )
        data = np.fromfile(out, dtype=np.float32)
    finally:
        os.unlink(out)
    return data


def has_audio(path: str) -> bool:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "json", path],
            check=True, capture_output=True, text=True,
        )
        return bool(json.loads(probe.stdout).get("streams"))
    except Exception:
        return False


def speech_boundaries(
    path: str,
    sr: int = 16000,
    frame_ms: float = 25.0,
    min_pause_s: float = 0.45,
    energy_percentile: float = 40.0,
) -> list[float]:
    """Return timestamps (seconds) where a new utterance begins.

    A boundary is the *onset* after a silence of at least `min_pause_s`. We
    deliberately do not treat every silence as a boundary — brief gaps inside
    a sentence are common, and over-segmenting costs more than the occasional
    missed handoff.

    The silence threshold is a percentile of the clip's own energy rather than
    an absolute dB figure, because closet recordings vary wildly in gain and
    room noise.
    """
    audio = _decode_audio(path, sr)
    if audio.size == 0:
        return []

    hop = max(int(sr * frame_ms / 1000.0), 1)
    n = audio.size // hop
    if n < 2:
        return []

    energy = np.sqrt((audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    thresh = float(np.percentile(energy, energy_percentile))
    voiced = energy > thresh

    min_pause_frames = max(int(min_pause_s * 1000.0 / frame_ms), 1)

    boundaries: list[float] = []
    silence_run = 0
    for i, v in enumerate(voiced):
        if not v:
            silence_run += 1
            continue
        if silence_run >= min_pause_frames:
            boundaries.append(i * hop / sr)
        silence_run = 0

    return boundaries


def boundaries_to_frame_cuts(boundaries: list[float], timestamps: list[float],
                             tolerance_s: float = 0.30) -> set[int]:
    """Map speech-onset times onto indices in an extracted frame list.

    `timestamps` must be the per-frame time within the *same* clip. Each
    boundary claims the nearest frame, provided it is within `tolerance_s` —
    otherwise the boundary fell in a stretch that the blur filter removed and
    we drop it rather than cutting in the wrong place.
    """
    if not boundaries or not timestamps:
        return set()

    ts = np.asarray(timestamps)
    cuts: set[int] = set()
    for b in boundaries:
        i = int(np.argmin(np.abs(ts - b)))
        if abs(ts[i] - b) <= tolerance_s:
            cuts.add(i)
    return cuts
