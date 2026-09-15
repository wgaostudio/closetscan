"""Describe each catalogued garment: category, colour, material, season.

Grouping (`closetscan.dedup`) answers "which images are one garment". This
answers "what is that garment", which is a different question and deserves its
own pass — not least because it reads better input. The generated plates are
clean, evenly lit and flat, so they are far easier to describe than the dim,
angled, hand-held frames the grouping pass has to work from.

Everything here is a guess made from pictures. The schema therefore allows
"unknown" everywhere and the prompt asks for it freely: a wardrobe assistant
that says nothing about fabric is more useful than one that says "cashmere"
about acrylic.

    python -m closetscan.attributes ./out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

from .catalogue import group_images, group_order
from .product_shots import API_URL, encode, read_key

DEFAULT_MODEL = "openai/gpt-6-astra"

PROMPT = """Each image below shows one garment from a single person's wardrobe,
labelled garment_0, garment_1 and so on. Describe each one.

The images are either clean generated plates or raw frames from a hand-held
walkthrough — a garment held up at an angle against a cluttered closet, often
warmly lit. Describe the garment, not the photograph, and do not let the room's
light colour your reading of the fabric. Describe only what you can see.

For every field, "unknown" is a correct and useful answer. Prefer it to a
plausible guess: someone will filter their wardrobe on these values and act on
the result, and a confident wrong fabric is worse than an honest gap. In
particular, material is rarely knowable from a photograph — say "unknown"
unless the weave or knit structure genuinely tells you (a chunky rib, a denim
twill, a ripstop shell).

Give each garment a short natural name someone would actually use for it
("mustard knit crewneck", "navy palm-print camp shirt"), and a one-sentence
description covering what it is, its colour and its most distinctive feature."""

FIELDS = {
    "name": {"type": "string", "description": "Short natural name, 2-5 words."},
    "description": {"type": "string", "description": "One sentence."},
    "category": {"type": "string",
                 "enum": ["top", "bottom", "outerwear", "knitwear", "dress",
                          "suiting", "footwear", "accessory", "unknown"]},
    "subcategory": {"type": "string",
                    "description": "Specific garment type, e.g. t-shirt, "
                                   "trousers, blazer. 'unknown' if unsure."},
    "colours": {"type": "array", "items": {"type": "string"},
                "description": "Plain colour words, most dominant first, "
                               "e.g. ['mustard','cream']. Empty if unclear."},
    "pattern": {"type": "string",
                "enum": ["solid", "graphic-print", "all-over-print", "stripe",
                         "check", "marl", "colour-block", "other", "unknown"]},
    "material": {"type": "string",
                 "description": "Only if the structure shows it; else 'unknown'."},
    "season": {"type": "array",
               "items": {"type": "string",
                         "enum": ["spring", "summer", "autumn", "winter",
                                  "all-season"]},
               "description": "When this would be worn."},
    "formality": {"type": "string",
                  "enum": ["casual", "smart-casual", "formal", "athletic",
                           "unknown"]},
}

SCHEMA = {
    "name": "garment_attributes",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["garments"],
        "properties": {
            "garments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id"] + list(FIELDS),
                    "properties": {
                        "id": {"type": "integer",
                               "description": "The garment_N number."},
                        **FIELDS,
                    },
                },
            },
        },
    },
}


def plate_for(out_dir: str, products: list[dict], gid: int) -> str | None:
    for view in ("front", "back"):
        for r in products:
            if r.get("group") == gid and r.get("view") == view and r.get("file"):
                return os.path.join(out_dir, "products", r["file"])
    return None


def frame_for(out_dir: str, manifest: dict, group: dict) -> str | None:
    """Best source frame for a garment, for catalogues with no plates.

    Generating plates costs real money and is not a prerequisite for knowing
    what colour something is. Describing from the raw frame is worse — the
    garment is held at an angle against a cluttered closet, so colour reads
    warm and fabric reads not at all — but it is much better than refusing to
    describe the wardrobe."""
    rows = manifest["garments"]
    if group.get("rows"):
        best = max((r for r in group["rows"] if 0 <= r < len(rows)),
                   key=lambda r: rows[r]["quality"], default=None)
        return os.path.join(out_dir, rows[best]["best_shot"]) if best is not None else None
    # A garment named only by image id has exactly those images and no others.
    imgs = group_images(group, rows)
    return os.path.join(out_dir, imgs[0][1]) if imgs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, "dedup.json")) as fh:
        dedup = json.load(fh)
    with open(os.path.join(args.out_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    ppath = os.path.join(args.out_dir, "products", "products.json")
    products = json.load(open(ppath))["images"] if os.path.exists(ppath) else []
    groups = sorted(dedup["groups"], key=group_order)

    junk = set(dedup.get("junk_rows", []))
    groups = [g for g in groups
              if (not set(g["rows"]) <= junk if g.get("rows")
                  else any(e.get("image") for e in g.get("sides", [])))]

    content = [{"type": "text", "text": PROMPT}]
    described = []
    n_frames = 0
    for gid, g in enumerate(groups):
        plate = plate_for(args.out_dir, products, gid)
        if not plate:
            plate = frame_for(args.out_dir, manifest, g)
            n_frames += 1
        if not plate:
            print(f"  skip garment_{gid} ({g['label']}): no image")
            continue
        described.append((gid, g))
        content.append({"type": "text",
                        "text": f"garment_{gid} — grouped as \"{g['label']}\":"})
        content.append({"type": "image_url",
                        "image_url": {"url": encode(plate, args.max_dim)}})
    if not described:
        sys.exit("nothing to describe — run closetscan.dedup first")
    if n_frames:
        print(f"  note: {n_frames} garment(s) described from raw video frames, "
              f"not plates — colour and fabric are less reliable")

    print(f"describing {len(described)} garments via {args.model}")
    t0 = time.time()
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {read_key()}",
                 "Content-Type": "application/json"},
        json={"model": args.model,
              "messages": [{"role": "user", "content": content}],
              "response_format": {"type": "json_schema", "json_schema": SCHEMA},
              "reasoning": {"effort": "low"},
              "max_tokens": args.max_tokens},
        timeout=args.timeout,
    )
    if r.status_code != 200:
        sys.exit(f"OpenRouter {r.status_code}: {r.text[:500]}")
    payload = r.json()
    usage = payload.get("usage", {})
    print(f"  {time.time() - t0:.1f}s  in={usage.get('prompt_tokens')} "
          f"out={usage.get('completion_tokens')} cost=${usage.get('cost', 0):.4f}")
    result = json.loads(payload["choices"][0]["message"]["content"])

    by_id = {g["id"]: g for g in result["garments"]}
    missing = [gid for gid, _ in described if gid not in by_id]
    if missing:
        print(f"  WARNING: no attributes returned for {missing}")

    out = {"model": args.model, "garments": {}}
    for gid, g in described:
        a = dict(by_id.get(gid, {}))
        a.pop("id", None)
        a.setdefault("name", g["label"])
        out["garments"][str(gid)] = a
        unknowns = sum(1 for k, v in a.items()
                       if v == "unknown" or (isinstance(v, list) and not v))
        print(f"  garment_{gid:<2} {a.get('name','?')[:34]:<36}"
              f"{a.get('category','?'):<12}{','.join(a.get('colours') or []) or '-':<22}"
              f"{unknowns} unknown")

    dest = os.path.join(args.out_dir, "attributes.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
