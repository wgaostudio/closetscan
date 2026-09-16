"""CLI: walkthrough video(s) -> deduplicated garment catalogue.

    python -m closetscan.run ./clips --out ./out --embedder dinov2

Writes one best-shot JPEG per detected garment, plus a manifest.json you can
feed to the downstream image/attribute stage, plus a contact sheet for eyeballing
where clustering went wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from .config import Config
from .extract import load_walkthrough
from .embed import embed
from .cluster import segment, merge_adjacent, group


def build_contact_sheet(frames, garments, out_path: str, cols: int = 5, cell: int = 400,
                        max_rows: int = 5):
    """One tile per garment. This is the artifact you actually stare at when
    tuning cluster_distance — false merges and false splits are obvious by eye
    and nearly invisible in the numbers.

    Sheets are paged and tiles kept large on purpose: these are meant to be
    fed to a vision model for grouping and junk rejection, and a model that
    downsamples a 56-tile sheet cannot tell two navy shirts apart.
    """
    if not garments:
        return []
    per_sheet = cols * max_rows
    pages = [garments[i:i + per_sheet] for i in range(0, len(garments), per_sheet)]
    written = []
    for pn, page in enumerate(pages):
        written.append(_draw_sheet(frames, page, out_path, pn, len(pages), cols, cell))
    return written


def _draw_sheet(frames, garments, out_path, page_no, n_pages, cols, cell):
    rows = (len(garments) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 24, dtype=np.uint8)

    for n, g in enumerate(garments):
        img = frames[g.best_frame_index].image
        h, w = img.shape[:2]
        scale = cell / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
        r, c = divmod(n, cols)
        y, x = r * cell, c * cell
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        label = f"#{g.id}"
        cv2.rectangle(sheet, (x + 4, y + cell - 34), (x + 4 + 22 * len(label), y + cell - 6),
                      (0, 0, 0), -1)
        cv2.putText(sheet, label, (x + 8, y + cell - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    path = out_path if n_pages == 1 else out_path.replace(".jpg", f"_{page_no:02d}.jpg")
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster a closet walkthrough into garments.")
    ap.add_argument("inputs", nargs="+", help="video file(s) or a directory of clips")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--embedder", default="dinov2", choices=["dinov2", "hist"])
    ap.add_argument("--sample-fps", type=float)
    ap.add_argument("--segment-similarity", type=float)
    ap.add_argument("--cluster-distance", type=float)
    ap.add_argument("--segment-sensitivity", type=float)
    ap.add_argument("--merge-sensitivity", type=float)
    ap.add_argument("--stitch-max-seconds", type=float,
                    help="stitch consecutive tracklets separated by less than this "
                         "many seconds, overriding narration cuts (0 disables)")
    ap.add_argument("--view-prefer-spread", action="store_true",
                    help="rank views by max time spread (needs clean tracklets)")
    ap.add_argument("--html", action="store_true",
                    help="also write a self-contained catalogue.html next to "
                         "manifest.json, openable by double-click")
    ap.add_argument("--no-audio", action="store_true",
                    help="ignore narration; use visual change points only")
    args = ap.parse_args()

    cfg = Config()
    for key in ("sample_fps", "segment_similarity", "cluster_distance", "segment_sensitivity",
                "merge_sensitivity", "stitch_max_seconds"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    if args.view_prefer_spread:
        cfg.view_prefer_spread = True

    os.makedirs(args.out, exist_ok=True)
    shots_dir = os.path.join(args.out, "shots")
    os.makedirs(shots_dir, exist_ok=True)

    if args.embedder == "hist":
        print("WARNING: --embedder hist is a smoke-test backend. In a real closet "
              "the background is constant, so colour histograms track the room "
              "rather than the garment and segmentation will badly under-fire. "
              "Use --embedder dinov2 for real footage.")

    t0 = time.time()
    frames, resolved = load_walkthrough(args.inputs, cfg)
    print(f"extracted {len(frames)} frames  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    embeddings = embed([f.image for f in frames], backend=args.embedder)
    print(f"embedded with {args.embedder}  ({time.time() - t1:.1f}s)")

    extra_cuts: set[int] = set()
    if not args.no_audio:
        from .audio import has_audio, speech_boundaries, boundaries_to_frame_cuts
        silent = []
        for clip in sorted({f.source for f in frames}):
            src = next((p for p in resolved if os.path.basename(p) == clip), None)
            if src is None:
                continue
            if not has_audio(src):
                silent.append(clip)
                continue
            local = [(i, f.timestamp) for i, f in enumerate(frames) if f.source == clip]
            if not local:
                continue
            idxs, ts = zip(*local)
            rel = boundaries_to_frame_cuts(speech_boundaries(src), list(ts))
            extra_cuts |= {idxs[r] for r in rel}
        print(f"narration boundaries: {len(extra_cuts)}")
        if silent:
            print("WARNING: no audio track in " + ", ".join(silent))
        if not extra_cuts:
            print("WARNING: no narration boundaries found. In a fixed-background "
                  "closet, speech is the strongest evidence of a handoff from one "
                  "garment to the next; without it segmentation runs on visual "
                  "change alone and will under-fire, merging neighbouring items. "
                  "Narrate the walkthrough as you film it — a few words per "
                  "garment, with a pause between them — and keep the audio track "
                  "when trimming. Pass --no-audio to suppress this warning.")

    raw = segment(frames, embeddings, cfg, extra_cuts=extra_cuts)
    tracklets = merge_adjacent(raw, embeddings, cfg, frames=frames)
    garments = group(tracklets, frames, cfg, embeddings=embeddings)
    print(f"{len(raw)} segments -> {len(tracklets)} tracklets -> {len(garments)} garments")

    manifest = []
    for g in garments:
        f = frames[g.best_frame_index]
        name = f"garment_{g.id:03d}.jpg"
        cv2.imwrite(os.path.join(shots_dir, name), f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        view_files = []
        for vn, vi in enumerate(g.view_indices):
            vname = f"garment_{g.id:03d}_view{vn}.jpg"
            cv2.imwrite(os.path.join(shots_dir, vname), frames[vi].image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            view_files.append({"file": f"shots/{vname}",
                               "timestamp_s": round(frames[vi].timestamp, 2)})

        manifest.append({
            "id": g.id,
            "views": view_files,
            "best_shot": f"shots/{name}",
            "source_clip": f.source,
            "timestamp_s": round(f.timestamp, 2),
            "quality": round(g.best_quality, 4),
            "observations": g.n_observations,
            "tracklets": g.tracklet_ids,
            "revisited": g.revisited,
        })

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"config": json.loads(cfg.to_json()),
                   "embedder": args.embedder,
                   "n_frames": len(frames),
                   "garments": manifest}, fh, indent=2)

    sheets = build_contact_sheet(frames, garments,
                                 os.path.join(args.out, "contact_sheet.jpg"))
    written = [os.path.join(args.out, "manifest.json")] + sheets
    print(f"wrote {', '.join(written)}, {len(garments)} shots")

    if args.html:
        from .html_export import write_html
        write_html(args.out)


if __name__ == "__main__":
    main()
