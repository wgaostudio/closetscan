"""Collapse the garment rows in a manifest into real garments, using a VLM.

The clustering stage upstream is deliberately biased toward splitting: a false
split costs a duplicate row, a false merge silently deletes a garment. That
leaves two kinds of slack in `manifest.json` for something with eyes to take
up — rows that are the same physical item seen twice, and rows that are not a
garment at all (an empty rail, a laundry pile, the floor between items).

Both are things a vision model does well and the pipeline cannot do at all.
What it cannot do is invent a garment that was never emitted, so this stage
can only ever recover what upstream already found.

    python -m closetscan.dedup ./out
    python -m closetscan.dedup ./out --include-views --model "$VISION_MODEL"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import cv2
import requests

from .catalogue import group_order

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = os.path.expanduser("~/.config/openrouter/key")
DEFAULT_MODEL = "openai/gpt-6-astra"

PROMPT = """You are looking at frames from one continuous video walkthrough of a
single person's closet. They pull out one garment at a time and hold it up.

An automatic pipeline cut the video into candidate garments, one image per
candidate, labelled row_0, row_1, and so on. The pipeline was tuned to
over-split on purpose, so the set you are given contains three things:

1. Several rows that are the SAME physical garment — either the same view
   caught twice, or the front and the back of one item after it was turned
   over. Group these together.
2. Rows showing a garment that appears only once. These are groups of one.
3. Rows that contain no garment being presented at all: an empty rail, a pile
   of laundry, the floor, a person walking, a soft toy. Mark these as junk.

Rules that matter:

- Two garments of the same colour and rough shape are NOT automatically the
  same garment. Look for the distinguishing detail — a print, a logo, a
  collar, a button placket, a texture. If you cannot find positive evidence
  that two rows show the same item, keep them apart.
- A wrong split is cheap: it leaves a duplicate row in a catalogue. A wrong
  merge is expensive: a real garment disappears and nobody notices. When you
  are unsure, DO NOT merge.
- The front and the back of one garment can look very different. Use context:
  they will usually be adjacent in time (the timestamps are given) and the
  fabric, colour and silhouette will match even when the print does not.
- Junk is judged on whether a garment is being *presented*, not on whether any
  clothing is visible. A rail with clothes hanging on it in the background,
  with nothing held up, is junk.

Return every row exactly once, either inside a group or in the junk list.

SIDES

Some rows come with extra images labelled row_N_alt0, row_N_alt1 and so on.
These are other frames of the same row, and they are often the *other side* of
the garment — the walkthrough turns items over.

For every group, also report which side each of its images shows:

- "front"   — the side normally worn facing out. For a printed tee this is
              usually the side with the smaller chest-level logo; for a shirt
              or jacket, the side with the button placket, zip or opening.
- "back"    — the opposite side. On graphic tees this is often the side with
              the single large full-width print.
- "detail"  — a close crop of part of the garment, a label, or a cuff, where
              you cannot tell which side it belongs to.
- "unclear" — the garment is bunched, blurred, or too obscured to say.

Judge this per image, not per row: two images of one row can show different
sides. Use "unclear" freely rather than guessing — a wrong side label sends the
wrong reference downstream, and it is better to say you cannot tell."""

SCHEMA = {
    "name": "garment_grouping",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["groups", "junk_rows"],
        "properties": {
            "groups": {
                "type": "array",
                "description": "One entry per distinct physical garment.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rows", "label", "confidence", "reason", "sides"],
                    "properties": {
                        "rows": {
                            "type": "array",
                            "description": "Row indices showing this garment.",
                            "items": {"type": "integer"},
                        },
                        "label": {
                            "type": "string",
                            "description": "Short human label, e.g. "
                                           "'mustard knit crewneck'.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "For multi-row groups, the specific "
                                           "shared detail that identifies them "
                                           "as one item.",
                        },
                        "sides": {
                            "type": "array",
                            "description": "Which side of the garment each "
                                           "supplied image shows. One entry per "
                                           "image belonging to this group.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["image", "side"],
                                "properties": {
                                    "image": {
                                        "type": "string",
                                        "description": "Image id exactly as "
                                                       "labelled, e.g. row_3 or "
                                                       "row_3_alt1.",
                                    },
                                    "side": {
                                        "type": "string",
                                        "enum": ["front", "back", "detail",
                                                 "unclear"],
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "junk_rows": {
                "type": "array",
                "description": "Rows with no garment being presented.",
                "items": {"type": "integer"},
            },
        },
    },
}


def _read_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        with open(KEY_FILE) as fh:
            return fh.read().strip()
    except OSError:
        sys.exit(f"no API key: set OPENROUTER_API_KEY or write one to {KEY_FILE}")


def _encode(path: str, max_dim: int) -> str:
    """JPEG data URL, longest side capped. Detail is the whole job here — this
    stage exists to tell two white t-shirts apart — so do not shrink further
    than needed to keep the request sane."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_content(out_dir: str, manifest: dict, max_dim: int,
                  include_views: bool) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": PROMPT}]
    for n, g in enumerate(manifest["garments"]):
        shots = [("", g["best_shot"], g["timestamp_s"])]
        if include_views:
            shots += [(f"_alt{i}", v["file"], v["timestamp_s"])
                      for i, v in enumerate(g["views"])]
        for suffix, rel, ts in shots:
            content.append({
                "type": "text",
                "text": f"image row_{n}{suffix}  (t={ts:.1f}s, "
                        f"row_{n} has {g['observations']} frames)",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode(os.path.join(out_dir, rel), max_dim)},
            })
    content.append({
        "type": "text",
        "text": f"There are {len(manifest['garments'])} rows, numbered 0 to "
                f"{len(manifest['garments']) - 1}. Group them now.",
    })
    return content


class AttemptFailed(RuntimeError):
    """One failed grouping attempt.

    `retryable` separates "ask again and it may well work" — a dropped
    connection, a rate limit, a model that miscounted — from "this request is
    wrong and will fail identically forever", such as a provider that caps
    image count or a model id that does not exist. Retrying the second kind
    just spends money slowly.
    """

    def __init__(self, message: str, retryable: bool = True, raw: str = ""):
        super().__init__(message)
        self.retryable = retryable
        self.raw = raw


def token_budget(n_images: int) -> int:
    """Completion budget for a grouping reply, scaled to the request.

    The reply carries one `sides` entry per *image*, not per row, so it grows
    with --include-views. The old flat 16000 was comfortable for 68 images and
    silently truncated at 116: the reply came back cut mid-JSON, unparseable,
    and the whole paid call was lost. Still bounded, because an unbounded
    budget makes the provider reserve the model's maximum against the balance
    and reject the request for credit it was never going to spend.
    """
    return 6000 + 200 * n_images


def call_model(content: list[dict], model: str, key: str, effort: str,
               timeout: int, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "reasoning": {"effort": effort},
        # Bound the completion budget. Left unset, the provider reserves the
        # model's maximum (65k here) against the account balance and rejects
        # the request for want of credit it was never going to spend — a real
        # grouping reply is a couple of thousand tokens.
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    try:
        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # Where a closed laptop lands. Previously this propagated and killed
        # the run, losing a call that had already been paid for.
        raise AttemptFailed(f"network error: {type(exc).__name__}: {exc}") from exc

    if r.status_code != 200:
        # 4xx other than 429 are the request's fault — too many images for the
        # provider, unknown model, prompt over the context limit — and asking
        # again changes nothing.
        retryable = r.status_code == 429 or r.status_code >= 500
        raise AttemptFailed(f"OpenRouter {r.status_code}: {r.text[:600]}", retryable)

    payload = r.json()
    if "choices" not in payload:
        raise AttemptFailed(f"unexpected response: {json.dumps(payload)[:600]}")
    choice = payload["choices"][0]
    text = choice["message"]["content"]
    usage = payload.get("usage", {})
    print(f"  {model}  {time.time() - t0:.1f}s  "
          f"in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')} "
          f"cost=${usage.get('cost', 0):.4f}")
    if choice.get("finish_reason") == "length":
        raise AttemptFailed(
            f"reply hit the {max_tokens}-token completion cap and was cut off; "
            f"raise --max-tokens", raw=text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AttemptFailed(f"model did not return JSON: {exc}", raw=text) from exc


def check_coverage(result: dict, n_rows: int,
                   valid_images: set[str] | None = None) -> list[str]:
    """The schema cannot enforce that every row appears exactly once, so verify
    it. A dropped row is a silently deleted garment, which is the one failure
    this whole pipeline is built to avoid."""
    seen: dict[int, int] = {}
    for g in result["groups"]:
        for r in g["rows"]:
            seen[r] = seen.get(r, 0) + 1
    for r in result["junk_rows"]:
        seen[r] = seen.get(r, 0) + 1
    problems = []
    missing = [r for r in range(n_rows) if r not in seen]
    dupes = sorted(r for r, c in seen.items() if c > 1)
    strays = sorted(r for r in seen if not 0 <= r < n_rows)
    # Image-only groups are supported by catalogue.group_images: a real
    # garment may appear only in another candidate's alternate frame.
    # Accept those only when every referenced image was actually supplied.
    empty = 0
    for g in result["groups"]:
        if g.get("rows"):
            continue
        images = [entry.get("image") for entry in g.get("sides", [])]
        if not images or valid_images is None or any(i not in valid_images for i in images):
            empty += 1
    if empty:
        problems.append(f"groups with no rows or valid image references: {empty}")
    if missing:
        problems.append(f"rows never assigned: {missing}")
    if dupes:
        problems.append(f"rows assigned more than once: {dupes}")
    if strays:
        problems.append(f"rows out of range: {strays}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", help="a directory written by closetscan.run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-dim", type=int, default=1024,
                    help="cap the longest side of each image (default 1024)")
    ap.add_argument("--include-views", action="store_true",
                    help="also send each row's alternate views (~3x the tokens)")
    ap.add_argument("--reasoning-effort", default="high",
                    choices=["none", "low", "medium", "high"],
                    help="'low' costs a few cents less and measurably loses "
                         "garments: on one walkthrough it silently folded two "
                         "alternate-view garments into their neighbours, where "
                         "'high' spotted both and said so")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="completion budget (default: scaled to the number of "
                         "images sent). Providers reserve this against your "
                         "balance, so it is bounded rather than unset")
    ap.add_argument("--retries", type=int, default=2,
                    help="extra attempts when a call drops, is rate-limited, "
                         "is truncated, or comes back unusable (default 2)")
    ap.add_argument("--result", default=None,
                    help="where to write the grouping (default OUT_DIR/dedup.json)")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    n = len(manifest["garments"])
    print(f"{n} rows from {manifest_path}")

    content = build_content(args.out_dir, manifest, args.max_dim, args.include_views)
    n_images = sum(1 for c in content if c["type"] == "image_url")
    print(f"sending {n_images} images at <= {args.max_dim}px")
    valid_images = {f"row_{i}" for i in range(n)}
    if args.include_views:
        valid_images.update(f"row_{i}_alt{j}" for i, row in enumerate(manifest["garments"])
                            for j, _ in enumerate(row.get("views", [])))

    budget = args.max_tokens or token_budget(n_images)
    dest = args.result or os.path.join(args.out_dir, "dedup.json")
    key = _read_key()

    result, problems = None, []
    for attempt in range(1, args.retries + 2):
        try:
            result = call_model(content, args.model, key, args.reasoning_effort,
                                args.timeout, budget)
            problems = check_coverage(result, n, valid_images)
            # Out-of-range rows mean the model answered in a different id space
            # than the one it was asked about — in practice numbering the
            # images rather than the rows. Nothing downstream can recover from
            # that: catalogue.py drops the unknown rows and manufactures
            # garments with a confident label and no photograph behind them.
            # Better to spend one more call than to ship a wardrobe of ghosts.
            fatal = [p for p in problems if p.startswith("rows out of range")]
            if fatal:
                raise AttemptFailed("; ".join(fatal))
            break
        except AttemptFailed as exc:
            print(f"  attempt {attempt}/{args.retries + 1} failed: {exc}")
            if exc.raw:
                raw_path = f"{dest}.attempt{attempt}.raw.txt"
                with open(raw_path, "w") as fh:
                    fh.write(exc.raw)
                print(f"    raw reply saved to {raw_path}")
            if not exc.retryable:
                sys.exit("  not retryable; giving up")
            if attempt > args.retries:
                sys.exit(f"  no usable grouping after {attempt} attempts; "
                         f"nothing written")
            result = None

    for p in problems:
        print(f"  WARNING: {p}")

    groups = result["groups"]
    placed = [g for g in groups if g.get("rows")]
    image_only = [g for g in groups if not g.get("rows") and g.get("sides")
                  and all(e.get("image") in valid_images for e in g["sides"])]
    orphans = [g for g in groups if not g.get("rows") and g not in image_only]
    merged = [g for g in placed if len(g["rows"]) > 1]
    print(f"\n{n} rows -> {len(placed) + len(image_only)} garments "
          f"({len(merged)} merged from multiple rows), "
          f"{len(result['junk_rows'])} junk")
    for g in sorted(groups, key=group_order):
        rows = (",".join(f"row_{r}" for r in sorted(g["rows"]))
                or ",".join(e["image"] for e in g.get("sides", [])))
        flag = "" if g["confidence"] == "high" else f"  [{g['confidence']}]"
        print(f"  {g['label']:<34} {rows}{flag}")
    if result["junk_rows"]:
        print(f"  {'(junk)':<34} "
              + ",".join(f"row_{r}" for r in sorted(result["junk_rows"])))

    if image_only:
        print(f"\n{len(image_only)} garment(s) recovered from alternate images; "
              "included in the catalogue")
    if orphans:
        print(f"\n{len(orphans)} group(s) have no rows or valid image references:")
        for g in orphans:
            imgs = ",".join(s["image"] for s in g.get("sides", [])) or "unspecified"
            print(f"  {g['label']:<34} {imgs}")
            if g.get("reason"):
                print(f"      {g['reason'][:110]}")

    with open(dest, "w") as fh:
        json.dump({"model": args.model,
                   "source_manifest": manifest_path,
                   "n_rows": n,
                   "coverage_problems": problems,
                   **result}, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
