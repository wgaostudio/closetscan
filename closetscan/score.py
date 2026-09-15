"""Score a run against a hand-labelled list of the garments in the footage.

The contact sheet shows you what came out. It cannot show you what did not,
and that is the failure mode that matters — a garment absent from the output
looks exactly like a garment that was never in the closet. So label the truth
once, by scrubbing the video, and score against it.

A ground-truth file is a list of [name, start_seconds, end_seconds], one entry
per garment, covering the span in which that garment is being presented. Gaps
between entries are junk time: walking, rummaging, putting things away.

    python -m closetscan.score ./out groundtruth/IMG_1027.json
    python -m closetscan.score ./out groundtruth/IMG_1027.json --dedup

Metrics are counted over *units*: one candidate row each by default, one
deduplicated garment each under --dedup.

  found        garments with a unit of their own             (higher is better)
  absorbed     garments present only as another unit's view  (recoverable, badly)
  missed       garments in no image at all                   (unrecoverable)
  junk         units landing in no garment's span
  over-merged  units spanning more than one labelled garment
  frag         units per found garment; 1.0 is one unit each
"""

from __future__ import annotations

import argparse
import json
import os


def load_gt(path: str) -> list[tuple[str, float, float]]:
    with open(path) as fh:
        raw = json.load(fh)
    items = raw["garments"] if isinstance(raw, dict) else raw
    return [(str(n), float(a), float(b)) for n, a, b in items]


def which(gt: list[tuple[str, float, float]], ts: float) -> str | None:
    for name, a, b in gt:
        if a <= ts <= b:
            return name
    return None


def score(manifest: dict, gt: list[tuple[str, float, float]],
          grouping: dict | None = None) -> dict:
    rows = manifest["garments"]

    if grouping is None:
        units = [{"rows": [i], "label": None} for i in range(len(rows))]
    else:
        junk = set(grouping.get("junk_rows", []))
        units = [g for g in grouping["groups"] if not set(g["rows"]) <= junk]

    found: dict[str, list] = {}
    view_only: set[str] = set()
    junk_units = 0
    over_merged = 0

    for u in units:
        members = [rows[i] for i in u["rows"] if 0 <= i < len(rows)]
        if not members:
            continue
        best_names = {which(gt, m["timestamp_s"]) for m in members} - {None}
        all_names = set(best_names)
        for m in members:
            all_names |= {which(gt, v["timestamp_s"]) for v in m["views"]} - {None}

        if not best_names:
            junk_units += 1
        for n in best_names:
            found.setdefault(n, []).append(u)
        if len(best_names) > 1:
            over_merged += 1
        view_only |= all_names - best_names

    view_only -= set(found)
    missed = [n for n, _, _ in gt if n not in found and n not in view_only]
    n_found = len(found)
    return {
        "units": len(units),
        "n_gt": len(gt),
        "found": n_found,
        "absorbed": len(view_only),
        "absorbed_names": sorted(view_only),
        "missed": len(missed),
        "missed_names": missed,
        "junk_units": junk_units,
        "over_merged": over_merged,
        "frag": round(sum(len(v) for v in found.values()) / n_found, 2) if n_found else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("ground_truth")
    ap.add_argument("--dedup", action="store_true",
                    help="score OUT_DIR/dedup.json instead of the raw rows")
    ap.add_argument("--dedup-file", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    gt = load_gt(args.ground_truth)

    grouping = None
    if args.dedup or args.dedup_file:
        path = args.dedup_file or os.path.join(args.out_dir, "dedup.json")
        with open(path) as fh:
            grouping = json.load(fh)

    r = score(manifest, gt, grouping)
    what = "garments after dedup" if grouping else "rows"
    print(f"{r['units']} {what}  vs  {r['n_gt']} labelled garments\n")
    print(f"  found       {r['found']}/{r['n_gt']}")
    print(f"  absorbed    {r['absorbed']}"
          + (f"  ({', '.join(r['absorbed_names'])})" if r["absorbed"] else ""))
    print(f"  missed      {r['missed']}"
          + (f"  ({', '.join(r['missed_names'])})" if r["missed"] else ""))
    print(f"  junk        {r['junk_units']}")
    print(f"  over-merged {r['over_merged']}"
          + ("   <- a real garment was fused into another" if r["over_merged"] else ""))
    print(f"  frag        {r['frag']}")


if __name__ == "__main__":
    main()
