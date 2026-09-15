"""Tracklets, then garments.

The insight that makes this tractable: time is the strongest signal you have.
A garment you are looking at now is the same one you were looking at 250ms
ago. So we do NOT cluster thousands of frames pairwise. Instead:

  1. Segment the frame sequence temporally into *tracklets* — runs of
     consecutive frames that look like the same thing.
  2. Cluster the tracklets. There are maybe a hundred of those, and each one
     has many frames voting on its identity, so the comparison is far more
     robust than frame-to-frame.

Step 2 is what catches revisits: you walked past the same rail twice, or
doubled back. Those become two tracklets that must merge into one garment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .config import Config
from .extract import Frame


@dataclass
class Tracklet:
    id: int
    frame_indices: list[int]
    centroid: np.ndarray
    hard_start: bool = False
    """This tracklet began at a forced cut — a narration onset or a clip
    boundary. Those are external evidence of a handoff, so the merge pass is
    never allowed to stitch across one."""

    @property
    def size(self) -> int:
        return len(self.frame_indices)


@dataclass
class Garment:
    id: int
    tracklet_ids: list[int]
    frame_indices: list[int]
    best_frame_index: int
    best_quality: float
    n_observations: int
    view_indices: list[int] = field(default_factory=list)
    """Alternate frames of this garment, *excluding* `best_frame_index`. A
    walkthrough that pulls each item out and turns it captures front and back;
    picking argmax(quality) discards exactly that. The full image set for a
    garment is `best_frame_index` followed by these."""

    @property
    def revisited(self) -> bool:
        """Seen in more than one pass. A useful confidence signal: garments
        observed twice independently are far less likely to be a segmentation
        artifact."""
        return len(self.tracklet_ids) > 1


def segment(frames: list[Frame], embeddings: np.ndarray, cfg: Config,
            extra_cuts: set[int] | None = None) -> list[Tracklet]:
    """Split the frame sequence where consecutive similarity drops.

    `extra_cuts` are frame indices forced to be boundaries regardless of
    visual evidence — in practice, narration onsets. In a fixed-background
    closet these are far more reliable than visual change, because the wall
    and rail are constant while the hands move continuously.
    """
    if len(frames) == 0:
        return []

    extra_cuts = extra_cuts or set()

    sims = np.array([float(embeddings[i] @ embeddings[i - 1])
                     for i in range(1, len(frames))])

    if cfg.segment_similarity is not None:
        is_cut = sims < cfg.segment_similarity
    else:
        # Robust change-point detection on dissimilarity. Median + MAD rather
        # than mean + std, because the spikes we are hunting for are exactly
        # the outliers that would inflate a non-robust estimate and hide
        # themselves.
        diss = 1.0 - sims
        med = float(np.median(diss))
        mad = float(np.median(np.abs(diss - med)))
        scale = mad * 1.4826 if mad > 1e-9 else float(diss.std()) or 1e-9
        is_cut = diss > med + cfg.segment_sensitivity * scale

    boundaries = [0]
    for i in range(1, len(frames)):
        # A clip boundary is always a cut: different file, discontinuous time.
        if is_cut[i - 1] or i in extra_cuts or frames[i].source != frames[i - 1].source:
            boundaries.append(i)
    boundaries.append(len(frames))

    tracklets: list[Tracklet] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < cfg.min_segment_frames:
            continue  # a transition, not an observation
        idx = list(range(a, b))
        centroid = embeddings[a:b].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        hard = a in extra_cuts or (a > 0 and frames[a].source != frames[a - 1].source)
        tracklets.append(Tracklet(id=len(tracklets), frame_indices=idx,
                                  centroid=centroid, hard_start=hard))

    return tracklets


def merge_adjacent(tracklets: list[Tracklet], embeddings: np.ndarray,
                   cfg: Config, frames: list[Frame] | None = None) -> list[Tracklet]:
    """Stitch consecutive tracklets that belong to one continuous presentation.

    Frame-level change-point detection over-fires on handheld footage: the
    hands never stop moving, so a single 13-second presentation of one sweater
    gets chopped into eight tracklets. Raising the frame-level threshold to
    compensate would then miss real handoffs between similar garments.

    The fix is hierarchical. Tracklet centroids average out the motion noise
    that caused the over-firing, so the same robust change-point test applied
    one level up is far cleaner: a genuine handoff is an outlier-high distance
    between consecutive centroids, and everything else is the same garment
    still being turned over.
    """
    if len(tracklets) < 2:
        return tracklets

    dists = np.array([
        1.0 - float(tracklets[i].centroid @ tracklets[i - 1].centroid)
        for i in range(1, len(tracklets))
    ])

    if dists.size >= cfg.merge_min_samples:
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med)))
        scale = mad * 1.4826 if mad > 1e-9 else float(dists.std()) or 1e-9
        is_handoff = dists > med + cfg.merge_sensitivity * scale
    else:
        # Too few tracklets for a robust spread estimate — MAD on a handful of
        # samples collapses everything into one. Cut at the largest relative
        # jump in the sorted distances instead, the same trick used on the
        # dendrogram.
        order = np.sort(dists)
        ratios = order[1:] / np.maximum(order[:-1], 1e-6)
        if ratios.size and ratios.max() >= cfg.gap_min_ratio:
            k = int(np.argmax(ratios))
            is_handoff = dists > float(np.sqrt(order[k] * order[k + 1]))
        else:
            is_handoff = np.ones_like(dists, dtype=bool)

    # Forced cuts usually win: narration onsets are stronger evidence of a
    # handoff than any visual similarity measure.
    for i, t in enumerate(tracklets[1:]):
        if t.hard_start:
            is_handoff[i] = True

    # A new source file is always a handoff, whatever the centroids say.
    # segment() records this as hard_start, but only on a tracklet that starts
    # exactly at the seam — and the segment straddling a clip boundary is
    # usually shorter than min_segment_frames and gets dropped, taking the
    # evidence with it. On a two-clip walkthrough that left hard_start set on
    # none of 90 tracklets. Reading the frames is the reliable test.
    if frames is not None:
        for i, t in enumerate(tracklets[1:]):
            if (frames[t.frame_indices[0]].source
                    != frames[tracklets[i].frame_indices[-1]].source):
                is_handoff[i] = True

    # ...but not over a gap too short to have been a handoff. People narrate
    # *through* a garment as readily as between garments ("and the back"), and
    # a forced cut there permanently blocks the two sides of one item from ever
    # being stitched. A handoff costs real time: you put one thing down, turn,
    # and pick the next up. Turning an item over does not.
    if cfg.stitch_max_seconds > 0 and frames is not None:
        for i, t in enumerate(tracklets[1:]):
            first, last = t.frame_indices[0], tracklets[i].frame_indices[-1]
            # A timestamp is seconds into its own clip, so the difference across
            # a clip boundary is not a duration at all — it comes out hugely
            # negative (-173.8s on a two-clip walkthrough) and clears the very
            # handoff the boundary is evidence for, welding the end of one file
            # to the start of the next. A new file is external evidence of a
            # cut, which this pass is never allowed to override.
            if frames[first].source != frames[last].source:
                continue
            gap = frames[first].timestamp - frames[last].timestamp
            if gap <= cfg.stitch_max_seconds:
                is_handoff[i] = False

    merged: list[Tracklet] = []
    current = [tracklets[0]]
    for i, t in enumerate(tracklets[1:]):
        if is_handoff[i]:
            merged.append(_fuse(current, embeddings, len(merged)))
            current = [t]
        else:
            current.append(t)
    merged.append(_fuse(current, embeddings, len(merged)))
    return merged


def _fuse(group: list[Tracklet], embeddings: np.ndarray, new_id: int) -> Tracklet:
    idx = [i for t in group for i in t.frame_indices]
    centroid = embeddings[idx].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-8)
    # The fused run begins where its first member began, forced cut included.
    # Nothing downstream reads this today, but a field that lies is worse than
    # no field.
    return Tracklet(id=new_id, frame_indices=idx, centroid=centroid,
                    hard_start=group[0].hard_start)


def _select_views(frame_idx: list[int], frames: list[Frame],
                  embeddings: np.ndarray, cfg: Config,
                  exclude: int | None = None) -> list[int]:
    """Alternate views of one garment: temporal spread under a quality floor.

    These are *additional* frames, never the best shot itself — `exclude` is
    the frame already published as `best_shot`, and it seeds the spread so the
    alternates are chosen to differ from it.

    Ranking is by distance in time, not in embedding space. Farthest-point
    sampling on embeddings picks whatever is least like what has been chosen
    already, and on real walkthrough footage that is reliably the *worst*
    frame available: a motion-blurred handoff, the empty rail between items,
    or the next garment already entering shot. Time is the better proxy for
    what we actually want — the same garment after it has been turned over —
    because turning an item over takes about a second, while the junk frames
    cluster at the moment of the handoff.

    Embedding distance is kept, but only as a rejection filter against near
    duplicates, and the quality floor is a percentile of this garment's own
    frames so it adapts to how well the item was presented.
    """
    pool = [i for i in frame_idx if i != exclude]
    if not pool:
        return []

    quals = {i: frames[i].quality(cfg) for i in pool}
    floor = float(np.percentile(list(quals.values()), cfg.view_quality_percentile))
    pool = [i for i in pool if quals[i] >= floor] or pool

    def too_similar(i: int, refs: list[int]) -> bool:
        return any(1.0 - float(embeddings[i] @ embeddings[c]) < cfg.view_min_distance
                   for c in refs)

    chosen: list[int] = []
    while len(chosen) < min(cfg.views_per_garment, len(pool)):
        refs = ([exclude] if exclude is not None else []) + chosen
        if not refs:
            # No anchor at all: start from the best frame in the pool.
            chosen.append(max(pool, key=lambda i: quals[i]))
            continue
        # Nearest frame that is far enough away, not the farthest one. A
        # tracklet is not guaranteed to hold a single garment, and its far end
        # is where the handoff to the next item lives — so maximising any
        # distance, in time or in embedding space, walks straight into the
        # neighbouring garment. Minimum sufficient separation keeps the view
        # inside the same presentation while still clearing the turn.
        eligible = [
            (min(abs(frames[i].timestamp - frames[c].timestamp) for c in refs), i)
            for i in pool if i not in chosen and not too_similar(i, refs)
        ]
        eligible = [(d, i) for d, i in eligible if d >= cfg.view_min_seconds]
        if not eligible:
            break  # nothing left that shows a different moment
        chosen.append(max(eligible)[1] if cfg.view_prefer_spread else min(eligible)[1])
    return chosen


def _gap_threshold(mat: np.ndarray, cfg: Config) -> float:
    """Pick the dendrogram cut at the largest *relative* jump in merge height.

    Why relative and not absolute: embedding backends occupy wildly different
    similarity ranges. On the histogram backend a revisit of the same garment
    sits at cosine distance ~0.0001 while distinct garments sit at ~0.03; on
    DINOv2 those numbers are far larger. What is stable across both is the
    *shape*: same-garment merges happen at heights an order of magnitude
    below different-garment merges, so there is a visible gap to cut in.

    If the largest jump is at the very bottom, nothing merges and every
    tracklet becomes its own garment — which is the right answer for a
    walkthrough with no revisits.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    heights = np.sort(linkage(pdist(mat, metric="cosine"), method="average")[:, 2])
    heights = heights[heights > 0]
    if heights.size == 0:
        return 1e-6

    # Candidate cuts: below the smallest merge (all singletons), or between
    # any two consecutive merge heights.
    candidates = np.concatenate([[cfg.gap_floor], heights])
    ratios = candidates[1:] / np.maximum(candidates[:-1], cfg.gap_floor)
    if ratios.size == 0:
        return float(heights[0]) * 0.5

    i = int(np.argmax(ratios))
    lo, hi = candidates[i], candidates[i + 1]
    if ratios[i] < cfg.gap_min_ratio:
        # No clean gap. An earlier version merged nothing here, reasoning that
        # a false merge silently deletes a garment. On real footage that was
        # the wrong call: it produced a 5x overcount, which is a worse and
        # more obvious failure than the occasional lost item. Fall back to a
        # low percentile of merge heights — conservative, but not inert.
        return float(np.percentile(heights, cfg.fallback_percentile))

    return float(np.sqrt(max(lo, cfg.gap_floor) * hi))  # geometric midpoint


def group(tracklets: list[Tracklet], frames: list[Frame], cfg: Config,
          embeddings: np.ndarray | None = None) -> list[Garment]:
    """Merge tracklets that show the same garment seen at different times."""
    if not tracklets:
        return []

    if len(tracklets) == 1:
        labels = np.array([0])
    else:
        mat = np.stack([t.centroid for t in tracklets])
        threshold = (cfg.cluster_distance if cfg.cluster_distance is not None
                     else _gap_threshold(mat, cfg))
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        ).fit_predict(mat)

    garments: list[Garment] = []
    for label in sorted(set(labels.tolist())):
        members = [t for t, l in zip(tracklets, labels) if l == label]
        frame_idx = [i for t in members for i in t.frame_indices]

        best = max(frame_idx, key=lambda i: frames[i].quality(cfg))
        views = (_select_views(frame_idx, frames, embeddings, cfg, exclude=best)
                 if embeddings is not None else [])
        garments.append(
            Garment(
                id=len(garments),
                tracklet_ids=[t.id for t in members],
                frame_indices=frame_idx,
                best_frame_index=best,
                best_quality=frames[best].quality(cfg),
                n_observations=len(frame_idx),
                view_indices=views,
            )
        )

    # Most-observed first: the garments you lingered on are the ones you were
    # most deliberate about, and they make the best top-of-report examples.
    garments.sort(key=lambda g: g.n_observations, reverse=True)
    for new_id, g in enumerate(garments):
        g.id = new_id
    return garments
