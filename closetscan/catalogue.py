"""Load a pipeline output directory as one normalized wardrobe catalogue.

The pipeline leaves several files side by side — `manifest.json` (candidate
rows), `dedup.json` (which rows are one garment), `attributes.json` (what each
garment is), `products/` (generated plates) — because each stage is separately
re-runnable. Consumers should not have to know that. This module joins them
into one record per garment and is the only place that knows the layout.

Garments come back in the order they were filmed. Their `index` keeps the
grouping order, because that is what the other files are keyed on.

Everything is optional except `manifest.json` and `dedup.json`: a catalogue
with no plates and no attributes still loads, just with fewer views and an
empty attribute set. Nothing here touches the network or writes to the
catalogue; the wear log is the only mutable file and lives beside it.

A view carries its provenance because the two kinds are not equally
trustworthy. A `frame` view is photographic evidence — this is what the camera
saw, at this second of this clip. A `plate` view is generated from those frames
and may differ from the real garment in any detail the frames did not resolve.
Anything shown to a person, or used to answer a question about what someone
owns, should keep that distinction visible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

WEAR_LOG = "wear_log.jsonl"
"""Append-only, one JSON object per line.

Deliberately JSON Lines rather than a JSON array: appending to an array means
reading, parsing, re-serialising and rewriting the whole file, which is not an
append and loses the log if the process dies mid-write. One line per entry
appends with a single write and survives a partial final line.
"""


@dataclass
class View:
    kind: str                      # "plate" (generated) | "frame" (photographic)
    view: str                      # "front" | "back" | "detail" | "unclear"
    file: str                      # path relative to the catalogue directory
    provenance: dict[str, Any]


@dataclass
class Garment:
    id: str
    index: int
    """Position in the grouping, and the join key into `attributes.json` and
    `products/products.json`, both of which are written against that order.

    Deliberately *not* the position in the list `load_catalogue` returns — that
    is chronological. Renumbering here to match the reading order would attach
    every plate and attribute to the wrong garment."""
    name: str
    description: str
    attributes: dict[str, Any]
    views: list[View] = field(default_factory=list)
    grouping: dict[str, Any] = field(default_factory=dict)
    narration: dict[str, Any] = field(default_factory=dict)
    """What the wearer said about this garment while filming it, if anything.

    Distinct from `attributes`, which is a model's reading of a picture. This is
    testimony: the person who owns the thing saying where it came from and what
    they think of it. Where the two disagree, this one is the authority."""
    observations: int = 0
    first_seen_s: float | None = None
    last_seen_s: float | None = None
    source_clip: str | None = None

    def compact(self) -> dict[str, Any]:
        """The listing record: enough to choose from, not enough to fill a
        context window. Callers that need everything ask for one garment."""
        a = self.attributes
        return {
            "id": self.id,
            "name": self.name,
            "category": a.get("category"),
            "subcategory": a.get("subcategory"),
            "colours": a.get("colours") or [],
            "pattern": a.get("pattern"),
            "season": a.get("season") or [],
            "formality": a.get("formality"),
            "n_views": len(self.views),
            "has_narration": bool(self.narration),
        }

    def full(self) -> dict[str, Any]:
        d = asdict(self)
        d["views"] = [asdict(v) for v in self.views]
        return d


def group_order(group: dict[str, Any]) -> int:
    """Sort key for a dedup group: its earliest candidate row.

    Every consumer orders groups this way, and every one of them used to call
    `min(group["rows"])` directly. An empty `rows` array satisfies the schema —
    a model can and does return one — and `min()` on it raises, so a single
    malformed group took down the grouping summary, the plate renderer, the
    attribute pass and the catalogue loader alike. Empty groups sort last and
    are dropped downstream by the same test that drops all-junk groups.
    """
    rows = group.get("rows") or []
    return min(rows) if rows else 1 << 30


def group_images(group: dict[str, Any],
                 rows: list[dict[str, Any]]) -> list[tuple[str, str, float, bool, int]]:
    """The images belonging to one group: (image_id, file, timestamp, is_best, row).

    Normally read off `rows`. A group with no rows is not necessarily empty,
    though. The grouping pass is shown *images* — row_3 and row_3_alt1 alike —
    but its schema only lets it answer in *rows*, so a garment that appears
    solely in some other row's alternate view has nowhere to be named except an
    empty `rows` plus a `sides` entry pointing at the image. Measured on one
    walkthrough that was two real garments: a blue polo living in row_0_alt2
    and a striped Nike tee in row_1_alt2, each the only record of a thing the
    owner owns. Dropping them loses a garment outright, which is the one
    failure the pipeline is built to avoid, so resolve them here too.
    """
    out: list[tuple[str, str, float, bool, int]] = []
    if group.get("rows"):
        for r in sorted(group["rows"]):
            if not 0 <= r < len(rows):
                continue
            row = rows[r]
            out.append((f"row_{r}", row["best_shot"], row["timestamp_s"], True, r))
            out += [(f"row_{r}_alt{i}", v["file"], v["timestamp_s"], False, r)
                    for i, v in enumerate(row.get("views", []))]
        return out

    index: dict[str, tuple[str, float, bool, int]] = {}
    for r, row in enumerate(rows):
        index[f"row_{r}"] = (row["best_shot"], row["timestamp_s"], True, r)
        for i, v in enumerate(row.get("views", [])):
            index[f"row_{r}_alt{i}"] = (v["file"], v["timestamp_s"], False, r)
    for entry in group.get("sides", []):
        hit = index.get(entry.get("image", ""))
        if hit:
            out.append((entry["image"], *hit))
    return out


def _load(path: str) -> dict[str, Any] | None:
    """Only a missing file is optional; unreadable or invalid data is an error."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid catalogue JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid catalogue JSON in {path}: expected an object")
    return data


def load_catalogue(directory: str) -> list[Garment]:
    """Join the pipeline's outputs into one record per garment."""
    manifest = _load(os.path.join(directory, "manifest.json"))
    if manifest is None:
        raise FileNotFoundError(
            f"{directory} is not a catalogue: no manifest.json "
            f"(run closetscan.run first)")
    if not isinstance(manifest.get("garments"), list):
        raise ValueError(f"Invalid {directory}/manifest.json: garments must be a list")

    dedup = _load(os.path.join(directory, "dedup.json"))
    has_grouping = dedup is not None
    if dedup is None:
        # Pre-dedup state: every candidate row stands alone. Worth rendering
        # rather than refusing — it is exactly the over-split output someone
        # wants to look at before deciding whether to run the grouping pass.
        dedup = {"groups": [{"rows": [i], "label": f"candidate row {i}",
                             "confidence": "low", "reason": "", "sides": []}
                            for i in range(len(manifest["garments"]))],
                 "junk_rows": []}
    elif not isinstance(dedup.get("groups"), list) or not isinstance(dedup.get("junk_rows", []), list):
        raise ValueError(f"Invalid {directory}/dedup.json: groups and junk_rows must be lists")

    # Enrichments use grouping indices. Without that grouping, candidate row
    # indices cannot safely identify which garment these records describe.
    attrs = ((_load(os.path.join(directory, "attributes.json")) or {}).get("garments", {})
             if has_grouping else {})
    narr = ((_load(os.path.join(directory, "narration.json")) or {}).get("garments", {})
            if has_grouping else {})
    prod_doc = ((_load(os.path.join(directory, "products", "products.json")) or {})
                if has_grouping else {})
    products = prod_doc.get("images", [])
    plate_model = prod_doc.get("model")

    rows = manifest["garments"]
    junk = set(dedup.get("junk_rows", []))
    def carries_a_garment(g: dict[str, Any]) -> bool:
        if g.get("rows"):
            return not set(g["rows"]) <= junk
        # No rows, but it may still name images directly — see group_images.
        # These sort last (group_order), so every index already assigned to a
        # rows-based group is unchanged and attributes.json / products.json
        # keep pointing at the garment they were written for.
        return any(e.get("image") for e in g.get("sides", []))

    groups = [g for g in sorted(dedup["groups"], key=group_order)
              if carries_a_garment(g)]

    out: list[Garment] = []
    for index, g in enumerate(groups):
        sides = {e["image"]: e["side"] for e in g.get("sides", [])}
        a = dict(attrs.get(str(index), {}))
        name = a.pop("name", None) or g.get("label") or f"garment {index}"
        description = a.pop("description", "") or ""

        frames: list[View] = []
        stamps: list[float] = []
        clips: set[str] = set()
        for image_id, file, ts, is_best, r in group_images(g, rows):
            clips.add(rows[r].get("source_clip"))
            stamps.append(ts)
            frames.append(View(
                kind="frame",
                view=sides.get(image_id, "unclear"),
                file=file,
                provenance={
                    "source_clip": rows[r].get("source_clip"),
                    "timestamp_s": round(ts, 2),
                    "candidate_row": r,
                    "image_id": image_id,
                    "best_of_row": is_best,
                    "note": "frame captured from the walkthrough video",
                },
            ))

        plates: list[View] = []
        for rec in products:
            if rec.get("group") != index or not rec.get("file"):
                continue
            built_from = [v.provenance["image_id"] for v in frames
                          if v.view == rec["view"]] or [v.provenance["image_id"]
                                                        for v in frames]
            plates.append(View(
                kind="plate",
                view=rec["view"],
                file=os.path.join("products", rec["file"]),
                provenance={
                    "generated": True,
                    "model": rec.get("model", plate_model),
                    "built_from": built_from,
                    "note": "generated flat-lay, not a photograph of the garment",
                },
            ))

        out.append(Garment(
            id=f"g{index:03d}",
            index=index,
            name=name,
            description=description,
            attributes=a,
            narration=dict(narr.get(str(index), {})),
            views=plates + frames,
            grouping={
                "label": g.get("label"),
                "confidence": g.get("confidence"),
                "reason": g.get("reason"),
                "candidate_rows": sorted(g["rows"]),
            },
            observations=sum(rows[r]["observations"] for r in g.get("rows", [])
                             if 0 <= r < len(rows)) or len(frames),
            first_seen_s=round(min(stamps), 2) if stamps else None,
            last_seen_s=round(max(stamps), 2) if stamps else None,
            source_clip=next(iter(c for c in clips if c), None),
        ))

    # Chronological. The grouping order is inherited from `cluster.group`,
    # which sorts candidate rows by observation count, so reading it straight
    # through jumps around the walkthrough at random. The order someone filmed
    # their closet in is the order they expect to read it back in.
    out.sort(key=lambda g: (g.first_seen_s is None, g.first_seen_s or 0.0, g.index))

    # Resolve narration cross-references once every garment exists. The model
    # answers in its own numbering ("garment_18"); consumers want a real id and
    # a name, and a reference to something that is not in the catalogue should
    # disappear rather than dangle.
    by_index = {g.index: g for g in out}
    for g in out:
        raw = str(g.narration.get("cross_reference", "") or "")
        targets = []
        for token in raw.replace(",", " ").split():
            digits = "".join(c for c in token if c.isdigit())
            other = by_index.get(int(digits)) if digits else None
            if other is not None and other.index != g.index:
                targets.append({"id": other.id, "name": other.name})
        if targets:
            g.narration["refers_to"] = targets
        g.narration.pop("cross_reference", None)
    return out


def attribute_values(garments: list[Garment]) -> dict[str, list[str]]:
    """Which attributes this catalogue actually has, and their values.

    Filters should be offered against what is present rather than a fixed
    vocabulary — a catalogue built by a different attribute pass, or enriched
    by hand, will carry different keys."""
    seen: dict[str, set[str]] = {}
    for g in garments:
        for k, v in g.attributes.items():
            values = v if isinstance(v, list) else [v]
            for item in values:
                if isinstance(item, (str, int, float)) and str(item):
                    seen.setdefault(k, set()).add(str(item))
    return {k: sorted(v) for k, v in sorted(seen.items())}


# ---- wear log -------------------------------------------------------------

def wear_log_path(directory: str) -> str:
    return os.path.join(directory, WEAR_LOG)


def append_wear(directory: str, garment_id: str, date: str,
                note: str | None = None) -> dict[str, Any]:
    entry = {
        "garment_id": garment_id,
        "date": date,
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if note:
        entry["note"] = note
    with open(wear_log_path(directory), "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def read_wear(directory: str) -> list[dict[str, Any]]:
    path = wear_log_path(directory)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # tolerate a torn final line
    return entries
