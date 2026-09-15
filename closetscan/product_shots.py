"""Turn each deduplicated garment into clean retailer-style product photos.

Takes the grouping from `closetscan.dedup`, gathers every frame the pipeline
kept for one garment — best shots and alternate views across all its rows — and
asks an image model for a front and a back flat-lay built from those references.

Model choice is not free here. On OpenRouter the `gpt-image-2.5-*` tiers route
through /api/v1/images, which accepts only `model` and `prompt`: reference
images are silently stripped and the model invents a garment from the text
alone. Use a model whose output modalities include *text* as well as image, so
the request goes through /chat/completions where image input survives.

    python -m closetscan.product_shots ./out
    python -m closetscan.product_shots ./out --only 3 --views front
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests

from .catalogue import group_images, group_order

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = os.path.expanduser("~/.config/openrouter/key")
DEFAULT_MODEL = "openai/gpt-5-image-mini"

BASE_PROMPT = """Generate one clean e-commerce product photo of the exact garment
shown in the input images — the {view} view — as if freshly photographed for a
retail catalog.

FIDELITY (no hallucination):
- Preserve exact colors, patterns, logos, printed text, trims, and fabric
  texture visible in the input.
- Do not invent, add, remove, or alter any design element.
- {unseen}

GARMENT PRESENTATION:
- Smooth, symmetrical flat-lay: sleeves/straps arranged neatly, no twists, no
  wrinkles obscuring any design or text.
- Frame the garment at the same physical scale and crop as you would its
  opposite side, so front and back are directly comparable.

COMPOSITION & CAMERA:
- Top-down flat-lay, perfectly rectified (no perspective distortion), centered.
- No model, mannequin, hands, or props.

LIGHTING & BACKGROUND:
- Pure white seamless studio background.
- Even, soft, shadow-free lighting (or a single soft contact shadow for depth
  only — no harsh directional shadows).
- No reflections, color cast, or background texture.

OUTPUT:
- Square canvas, 1:1 aspect ratio, product centered with even margin.
- High resolution, no watermark, no text overlays, no UI/mockup elements.

The input images are frames from a handheld video: they are dim, angled and
sometimes motion-blurred, and the garment is held up by hand. Read the garment
through that — reproduce the item, not the photography.

USING THE REFERENCES:
Each reference below is captioned with the side it shows.
- Captioned "{view}": authoritative for this drawing. Reproduce every design
  element visible in these, including faint ones — if a graphic appears in even
  one of them, it is on this side and must be drawn.
- Captioned "side uncertain": may show either side. Use these ONLY for colour,
  fabric, texture, collar shape and silhouette. Never take the placement or
  presence of a graphic from them, in either direction — their graphics are
  not evidence for this side, and their absence is not evidence against it.

Invent nothing that is absent from every reference. But do not omit something
that is present in a "{view}" reference merely because it is faint or appears
only once."""

UNSEEN_FRONT = ("If parts of the front are obscured by hands or folds in the "
                "input, reconstruct them as a plain continuation of the "
                "surrounding fabric — no invented graphics, seams or closures.")
UNSEEN_BACK = ("If the back of the garment is not visible in the input, render "
               "it as a plain, consistent continuation of the front's fabric "
               "color and texture — no invented graphics, seams, logos or "
               "closures on the unseen side.")


def read_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        with open(KEY_FILE) as fh:
            return fh.read().strip()
    except OSError:
        sys.exit(f"no API key: set OPENROUTER_API_KEY or write one to {KEY_FILE}")


def encode(path: str, max_dim: int) -> str:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _paths_by_id(manifest: dict, group: dict) -> dict[str, str]:
    """Map the image ids dedup used (row_3, row_3_alt1) back to files."""
    return {image_id: file
            for image_id, file, _ts, _best, _r
            in group_images(group, manifest["garments"])}


def order_views(views: list[str], group: dict, anchor: str) -> list[str]:
    """Decide which view to render first, i.e. which one anchors the chain.

    Chaining only constrains the views rendered *after* the first: the anchor
    itself has nothing to hold it and is the plate free to drift on silhouette
    or invent small print. So render the side the footage actually evidences —
    the one with the most frames confidently labelled as that side — and chain
    the weaker side off it. Convention would put the front first; on a garment
    filmed mostly from the back, that anchors the pair to the guess."""
    if anchor != "evidence" or len(views) < 2:
        return views
    counts = {v: 0 for v in views}
    for entry in group.get("sides", []):
        if entry["side"] in counts:
            counts[entry["side"]] += 1
    if len(set(counts.values())) == 1:          # no evidence either way
        return views
    return sorted(views, key=lambda v: (-counts[v], views.index(v)))


def gather(out_dir: str, manifest: dict, group: dict, max_dim: int,
           max_refs: int, view: str | None = None) -> list[tuple[str, str]]:
    """References for one garment, best shots first.

    When dedup labelled which side each image shows, send only the images for
    the side being drawn. Mixing both sides into one request is what makes a
    model copy a large back graphic onto the front: it cannot tell which
    reference is which, so it reproduces whatever dominates. Filtering removes
    the ambiguity at the source rather than arguing with it in the prompt.

    Fall back to everything when there are no labels, or when the labels leave
    this side with nothing — a bad reference beats no reference."""
    by_id = _paths_by_id(manifest, group)
    ordered = list(by_id)                       # best shots first, then alts
    ordered.sort(key=lambda k: ("_alt" in k, k))
    sides = {e["image"]: e["side"] for e in group.get("sides", [])}

    if view:
        keep = [k for k in ordered if sides.get(k) == view]
        keep += [k for k in ordered if sides.get(k) in ("detail", "unclear")]
        if keep:
            ordered = keep

    out = []
    for k in ordered[:max_refs]:
        side = sides.get(k, "unclear")
        caption = view if (view and side == view) else "side uncertain"
        out.append((encode(os.path.join(out_dir, by_id[k]), max_dim), caption))
    return out


PRIOR_NOTE = """This is the finished plate for the opposite side of this same
garment, already rendered. Match it exactly on colourway, fabric texture and
weave, silhouette, garment scale, crop and lighting, so the pair sits together
on a catalogue page as one item.

It is a colour and material reference ONLY. Do not copy any graphic, print,
logo, number or text from it — those belong to the other side. Take the cloth
from this plate and the design from the photographic references above."""


def generate(refs: list[tuple[str, str]], view: str, label: str, model: str,
             key: str, timeout: int,
             prior: bytes | None = None,
             max_tokens: int = 24000) -> tuple[bytes | None, float, str]:
    prompt = BASE_PROMPT.format(
        view=view, unseen=UNSEEN_BACK if view == "back" else UNSEEN_FRONT)
    content = [{"type": "text", "text": f"{prompt}\n\nThe garment is a {label}."}]
    for url, caption in refs:
        content.append({"type": "text", "text": f"reference — {caption}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    if prior:
        content.append({"type": "text", "text": PRIOR_NOTE})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(prior).decode()}})
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": content}],
              "modalities": ["image", "text"],
              # An image model advertises a very large completion maximum and
              # the provider reserves it against the balance. One plate costs
              # ~7k tokens; leaving this unset can 402 on a healthy account.
              "max_tokens": max_tokens},
        timeout=timeout,
    )
    if r.status_code != 200:
        return None, 0.0, f"http {r.status_code}: {r.text[:200]}"
    payload = r.json()
    cost = float(payload.get("usage", {}).get("cost") or 0.0)
    images = payload["choices"][0]["message"].get("images") or []
    if not images:
        text = payload["choices"][0]["message"].get("content")
        return None, cost, f"no image returned ({str(text)[:160]})"
    url = images[0]["image_url"]["url"] if isinstance(images[0], dict) else images[0]
    return base64.b64decode(url.split(",", 1)[1]), cost, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", help="a directory containing manifest.json and dedup.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--views", default="front,back",
                    help="comma-separated: front, back (default both)")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--max-refs", type=int, default=6,
                    help="reference images per request (default 6)")
    ap.add_argument("--only", type=int, action="append",
                    help="only this group index; repeatable. Use for a cheap trial.")
    ap.add_argument("--workers", type=int, default=4,
                    help="requests in flight at once")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--anchor", default="evidence", choices=["front", "evidence"],
                    help="which view anchors the chain: always the front, or "
                         "the side with the most confident source frames")
    ap.add_argument("--no-chain", dest="chain", action="store_false",
                    help="render every view independently instead of feeding "
                         "the first rendered view in as a colour reference")
    ap.add_argument("--dest", default=None,
                    help="output directory (default OUT_DIR/products)")
    args = ap.parse_args()
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if not views or any(v not in ("front", "back") for v in views):
        ap.error("--views must contain front and/or back")
    views = list(dict.fromkeys(views))

    with open(os.path.join(args.out_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    dedup_path = os.path.join(args.out_dir, "dedup.json")
    if not os.path.exists(dedup_path):
        sys.exit(f"{dedup_path} not found — run closetscan.dedup first")
    with open(dedup_path) as fh:
        dedup = json.load(fh)

    # Junk-only and empty groups are dropped here exactly as catalogue.py and
    # attributes.py drop them, so that the index this stage stamps into
    # products.json means the same garment they will look it up by.
    junk = set(dedup.get("junk_rows", []))
    groups = [g for g in sorted(dedup["groups"], key=group_order)
              if (not set(g["rows"]) <= junk if g.get("rows")
                  else any(e.get("image") for e in g.get("sides", [])))]
    # Number the groups BEFORE --only filters them. The index is stamped into
    # products.json and is how catalogue.py finds a garment's plates, so it has
    # to survive being subset: re-deriving it after the filter made `--only 3`
    # write group 3's plate as group 0, and the catalogue then hung it on a
    # different garment entirely.
    indexed = list(enumerate(groups))
    if args.only:
        indexed = [(i, g) for i, g in indexed if i in set(args.only)]
        if not indexed:
            sys.exit(f"--only {sorted(set(args.only))} matched no group; "
                     f"there are {len(groups)}, numbered 0 to {len(groups) - 1}")
    dest = args.dest or os.path.join(args.out_dir, "products")
    os.makedirs(dest, exist_ok=True)
    index = os.path.join(dest, "products.json")
    previous = []
    if os.path.exists(index):
        with open(index) as fh:
            previous_doc = json.load(fh)
        previous = [dict(r, model=r.get("model", previous_doc.get("model")))
                    for r in previous_doc["images"]]
    key = read_key()

    chain = args.chain and len(views) > 1
    n_jobs = len(indexed) * len(views)
    print(f"{len(indexed)} garments x {len(views)} views = {n_jobs} images "
          f"via {args.model}" + ("  [chained]" if chain else ""))

    def render(i, g):
        """All views of one garment. Chained, the first rendered view is passed
        into the rest as a colour reference — the two plates are shown side by
        side in a catalogue, so drifting a shade apart is more obvious there
        than any single plate being slightly off."""
        out, prior = [], None
        for v in order_views(views, g, args.anchor):
            refs = gather(args.out_dir, manifest, g, args.max_dim,
                          args.max_refs, view=v)
            try:
                data, cost, err = generate(refs, v, g["label"], args.model, key,
                                           args.timeout,
                                           prior=prior if chain else None)
            except Exception as exc:                      # network, decode, ...
                data, cost, err = None, 0.0, str(exc)[:200]
            if data and prior is None:
                prior = data
            out.append((i, g, v, data, cost, err))
        return out

    results, total, t0 = [], 0.0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render, i, g) for i, g in indexed]
        for fut in as_completed(futures):
            for i, g, v, data, cost, err in fut.result():
                total += cost
                slug = "".join(c if c.isalnum() else "_"
                               for c in g["label"])[:40].strip("_")
                name = f"{i:02d}_{slug}_{v}.png"
                if data:
                    with open(os.path.join(dest, name), "wb") as fh:
                        fh.write(data)
                    print(f"  ok    {name}  ${cost:.3f}")
                    results.append({"group": i, "label": g["label"], "view": v,
                                    "rows": sorted(g["rows"]), "file": name,
                                    "cost_usd": round(cost, 4)})
                else:
                    print(f"  FAIL  {name}  {err}")
                    results.append({"group": i, "label": g["label"], "view": v,
                                    "rows": sorted(g["rows"]), "file": None,
                                    "error": err, "cost_usd": round(cost, 4)})

    ok = sum(1 for r in results if r["file"])
    print(f"\n{ok}/{n_jobs} generated in {time.time() - t0:.0f}s, "
          f"total ${total:.2f}")
    # A trial or retry must not hide the plates for other garments/views.
    updated = {(r["group"], r["view"]) for r in results}
    retained = [r for r in previous
                if (r["group"], r["view"]) not in updated
                and 0 <= r["group"] < len(groups)
                and sorted(r.get("rows", [])) == sorted(groups[r["group"]]["rows"])]
    for r in results:
        r["model"] = args.model
    with open(index, "w") as fh:
        json.dump({"model": args.model, "total_cost_usd": round(total, 4),
                   "images": sorted(retained + results, key=lambda r: (r["group"], r["view"]))},
                  fh, indent=2)
    print(f"wrote {index}")
    if ok != n_jobs:
        sys.exit(f"{n_jobs - ok}/{n_jobs} image generations failed; see {index}")


if __name__ == "__main__":
    main()
