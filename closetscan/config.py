"""Tunable parameters for the closet-scan pipeline.

Every number here is a guess until you have real footage. The whole point of
keeping them in one place is that you can sweep them against a fixed corpus
without touching pipeline code.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # ---- frame extraction -------------------------------------------------
    sample_fps: float = 4.0
    """Frames per second pulled out of the source video.

    The glasses record at 30fps, but adjacent frames are near-duplicates.
    4fps is enough to catch a garment you paused on for ~250ms. Raise it if
    you walk fast; lower it if extraction is your bottleneck.
    """

    max_frame_width: int = 1280
    """Downscale extracted frames to this width. 3K frames are lovely and
    also 8x slower to embed. Keep the originals, work on the downscales."""

    blur_percentile: float = 25.0
    """Drop the blurriest N% of frames before anything else. Motion blur from
    head movement is the dominant failure mode in first-person footage."""

    blur_rel_floor: float = 0.35
    """Also drop frames below this fraction of the walkthrough's *median*
    sharpness. Relative, not absolute: a cable knit scores an order of
    magnitude higher than flat black wool at identical focus, so any fixed
    Laplacian threshold is wrong for one of them."""

    # ---- temporal segmentation (tracklets) --------------------------------
    segment_sensitivity: float = 1.2
    """Change-point sensitivity, in robust standard deviations.

    A boundary is declared where consecutive-frame dissimilarity spikes this
    far above the walkthrough's own baseline. Adaptive on purpose: absolute
    cosine thresholds are not portable between embedders (DINOv2 and the
    histogram backend occupy completely different similarity ranges), and
    they are not portable between a cluttered closet and a sparse one.
    """

    segment_similarity: float | None = None
    """Optional hard override. Set a float to use a fixed cosine threshold
    instead of adaptive change-point detection. Mostly useful for ablations."""

    min_segment_frames: int = 3
    """Segments shorter than this are transitions (you panning between
    garments), not observations. Dropped."""

    # ---- cross-segment clustering ----------------------------------------
    cluster_distance: float | None = None
    """Hard cosine-distance threshold for merging tracklets into one garment.

    Leave as None to pick the cut adaptively from the largest relative gap in
    the dendrogram (see cluster._gap_threshold). Set a float only for
    ablations — the right absolute value differs per embedder and per closet.
    """

    gap_floor: float = 1e-4
    """Distances below this are treated as zero when computing gap ratios.
    Stops a single near-identical pair from producing an infinite ratio."""

    merge_sensitivity: float = 0.0
    """Handoff sensitivity at the TRACKLET level, in robust standard
    deviations. Frame-level detection is deliberately left trigger-happy and
    this pass stitches the pieces back together; tracklet centroids have
    averaged out the hand-motion noise, so the test is much cleaner here.

    Measured against a hand-labelled walkthrough, recall falls monotonically as
    this rises — 10/11 garments at 0.0, 9/11 at 1.0, 8/11 at the old default of
    1.5 — because each increment stitches another genuine handoff shut. It buys
    only a shorter contact sheet, and a shorter sheet is worthless if the item
    you wanted is the one that got absorbed. Split cheaply and let the reader
    (a vision model, usually) collapse the duplicates: it can merge two rows,
    but it cannot recover a garment that was never emitted.

    Note that 0.0 is not "no merging" — the threshold is median + this many
    scaled deviations, so 0.0 still merges every below-median boundary. Pass a
    negative value to disable the pass outright."""

    merge_min_samples: int = 8
    """Below this many consecutive-tracklet distances, use gap detection
    instead of MAD — robust spread estimates need samples to be robust."""

    fallback_percentile: float = 20.0
    """When no clean dendrogram gap exists, cut at this percentile of merge
    heights rather than merging nothing."""

    gap_min_ratio: float = 4.0
    """Minimum jump (as a multiple) required to accept a gap as a real cut.
    Below this the tracklets are a continuum and we merge nothing: a false
    split costs one duplicate row, a false merge silently deletes a garment."""

    # ---- view selection ---------------------------------------------------
    views_per_garment: int = 3
    """Max distinct views kept per garment. Front, back, and a detail is
    usually the useful set."""

    view_min_distance: float = 0.08
    """Reject a candidate view this close in embedding space to one already
    chosen — it would be a near duplicate. This is a rejection filter only;
    views are *ranked* by temporal spread, not by embedding distance."""

    stitch_max_seconds: float = 0.4
    """Suppress a handoff between consecutive tracklets separated by less than
    this many seconds of dropped footage, overriding both the distance test and
    a narration cut. 0 disables it.

    Rationale, measured on the first real walkthrough: consecutive tracklets are
    never two different garments. A real handoff always passes through a stretch
    of nothing — you put one item down, turn, and pick the next up — so it
    reaches the next garment via junk frames, not directly. What *is* directly
    adjacent is one garment being turned over, and the frames lost between those
    two tracklets are the motion blur of the turn itself. Bounding the stitch by
    that gap is therefore safe against fusing garments; what it risks absorbing
    is junk, not a neighbour."""

    view_quality_percentile: float = 60.0
    """Quality floor for alternate views, as a percentile of the garment's own
    frames. Views are the frames a downstream model reads for attributes, so
    they are held to a higher bar than mere membership in the garment."""

    view_prefer_spread: bool = False
    """Rank alternate views by *maximum* time from those already chosen rather
    than minimum sufficient separation. Only safe when tracklets are reliably
    one garment each: the far end of a mixed tracklet is the next garment, so
    maximising spread inside one walks straight into it. With stitching on, rows
    are clean enough that the far end is the other side of the same item, which
    is exactly the frame worth publishing."""

    view_min_seconds: float = 1.0
    """Stop adding views once the most temporally distant remaining frame is
    this close in time to one already chosen. Turning a garment over takes
    about a second; anything tighter is the same pose twice."""

    # ---- best-shot selection ---------------------------------------------
    w_sharpness: float = 0.5
    w_area: float = 0.2
    w_exposure: float = 0.15
    w_centrality: float = 0.15
    """Weights for the per-frame quality score. Centrality matters more than
    you'd expect on an ultra-wide lens: the edges are where distortion and
    softness live."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
