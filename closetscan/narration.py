"""Attach what was said about each garment to the garment.

People narrate a walkthrough with the things a photograph cannot carry: what a
fabric is, where an item came from, whether they still like it. `audio.py`
already uses that speech as a *timing* signal for segmentation and throws the
words away. This keeps the words.

Three steps, deliberately separated by what each is good at:

1. **Transcribe** locally with Whisper. Timing is the load-bearing part — an
   utterance attached to the wrong garment is a confident lie about someone's
   own clothes — and an ASR model built for timestamps is far more trustworthy
   there than a language model asked to guess them. Word-level timings are
   re-chunked into short utterances so one sentence cannot straddle two items.

2. **Chunk against the garments.** Every *word* is tagged with whatever was on
   screen when it was spoken, and the transcript is cut wherever that changes.
   Splitting on pauses first and aligning afterwards is what produced the worst
   errors — Whisper punctuates unreliably, and one nineteen-second run with no
   gap in it swallowed three garments' speech and dumped all of it on the one
   it happened to overlap most. Cutting on the garment boundary makes a
   straddling utterance impossible. No language understanding involved here, so
   nothing to hallucinate.

3. **Assign and distil** with a language model, given the whole transcript, the
   timing as a *hint*, and a picture of every garment. The hint is not the
   answer: speech and camera drift apart, and a remark like "…but overpriced,
   to be fair" routinely lands two seconds after the item has left the frame.
   Only something reading the sentences can put that back where it belongs. Whisper mangles brand names it has no reason to expect —
   Qdrant becomes "quadrant", Uniqlo becomes "unique low", Reigning Champ
   becomes "Ring Champs" — and recovering them needs both halves: the sound of
   the mis-hearing and the sight of the garment. Seeing the whole wardrobe at
   once also lets "I got this in another colour" be pinned to the other colour.
   The same pass separates a remark about fabric from a memory from an opinion.

Verbatim quotes are kept alongside the distilled fields. The summary is a
reading of what someone said; the quote is what they actually said.

    python -m closetscan.narration ./out_phone
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

from .catalogue import load_catalogue
from .product_shots import API_URL, encode, read_key

DEFAULT_MODEL = "openai/gpt-6-astra"
DEFAULT_ASR = "medium.en"

PROMPT = """Below is the full transcript of someone walking through their own
closet, talking about each item as they hold it up, followed by a picture of
every garment in the catalogue.

Your job is to decide **which garment each line is about**, and then to distil
what was said.

## Assigning lines

Each line carries its timestamp and, as a hint, whichever garment was on screen
at that moment. **The hint is a prior, not the answer.** Speech and camera drift
apart constantly:

- A remark trails the item. Someone finishes "…but overpriced, to be fair"
  two seconds after they have already put it down and picked up the next thing.
  That sentence belongs to what they were just holding, not to what is on
  screen as the words come out.
- A sentence split across two lines belongs *whole* to one garment. If half of
  it carries an on-screen hint for the previous item, take the whole sentence
  there.
- People refer backwards: "the green one", "that one I mentioned", "the Quince
  is wool, by the way". Send those to the garment they name, however far back
  it is.
- "This one" almost always means the item that has just come up, so track the
  running order — the hints, read in sequence, tell you what that order is.

Assign a line to no garment only when it is genuinely about nothing — "okay",
"let me see", "oh dear", "here's my closet". Do not park a real remark on a
neighbouring garment just because the clock puts it there; leaving it out is
better than attaching it to the wrong item, and getting it right is better
still.

## Fixing the transcription

The recogniser had no idea what a closet contains, so it rendered unfamiliar
names phonetically and got them wrong. Treat every odd proper noun as a probable
mis-hearing of a real clothing brand and work out which, using the sound of it,
what the garment plainly is, and any logo or wordmark visible in its picture.
"Night Republic" on a plain beige tee is Banana Republic; "Ring Champs" on a
heavyweight grey hoodie is Reigning Champ; "quadrant" is Qdrant; "unique low" is
Uniqlo. Where nothing plausible fits, keep it as heard and say the name is
uncertain. Correct names and recognition slips only — do not tidy the grammar or
add words they did not say.

## Per garment, report

- `utterances` — the line numbers you assign to it, in order.
- `corrected_quotes` — those same lines with recognition errors fixed, one entry
  per line number, in the same order.
- `material` — what it is made of, if stated. Their claim, not your inference.
- `provenance` — where it came from, who gave it, what it is associated with.
- `sentiment` — how they feel about it, in their words. Include dislikes plainly;
  "I don't like the material" is the useful half of a wardrobe audit. Be specific
  about *what* they dislike — the colour, the collar, the fabric, the fit.
- `summary` — one sentence, only where there is something to tie together.
- `cross_reference` — another garment's number when a line points at it
  ("I also bought this in another colour"), else empty.

Leave a field empty when they said nothing of that kind. Most garments will have
one or two filled and some will have none. An invented memory is far worse than
a blank field.

Omit a garment entirely if no line belongs to it."""


SCHEMA = {
    "name": "garment_narration",
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
                    "required": ["id", "utterances", "corrected_quotes",
                                 "material", "provenance", "sentiment",
                                 "summary", "cross_reference"],
                    "properties": {
                        "id": {"type": "integer"},
                        "utterances": {
                            "type": "array", "items": {"type": "integer"},
                            "description": "Line numbers assigned to this "
                                           "garment, in order.",
                        },
                        "corrected_quotes": {
                            "type": "array", "items": {"type": "string"},
                            "description": "The garment's lines, with only "
                                           "clear recognition errors fixed.",
                        },
                        "material": {"type": "string"},
                        "provenance": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "summary": {"type": "string"},
                        "cross_reference": {
                            "type": "string",
                            "description": "Other garment number(s) this one "
                                           "refers to, e.g. 'garment_13'. "
                                           "Empty if none or unclear.",
                        },
                    },
                },
            },
        },
    },
}


# ---- 1. transcribe --------------------------------------------------------

def transcribe_words(clip: str, model_size: str = DEFAULT_ASR) -> list[dict]:
    """Raw word-level timings. Chunking happens later, against the garments."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "narration needs faster-whisper:\n  pip install faster-whisper"
        ) from exc
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        clip, vad_filter=True, word_timestamps=True,
        condition_on_previous_text=False,   # stops one bad guess propagating
    )
    return [{"start": w.start, "end": w.end, "word": w.word}
            for s in segments for w in (s.words or [])]


def which_garment(t: float, windows, tolerance: float = 2.5):
    """The garment on screen at time `t`, or None."""
    best, best_gap = None, 1e9
    for gid, _, wins in windows:
        for lo, hi in wins:
            if lo <= t <= hi:
                return gid
            gap = lo - t if t < lo else t - hi
            if gap < best_gap:
                best, best_gap = gid, gap
    return best if best_gap <= tolerance else None


def chunk_by_garment(words: list[dict], windows, pause: float = 0.45,
                     min_run: int = 2) -> list[dict]:
    """Cut the transcript wherever the garment on screen changes.

    Chunking on pauses alone and aligning afterwards is what produced the worst
    errors: Whisper punctuates unreliably, so a nineteen-second run with no gap
    and no full stop swallowed three garments' worth of speech and dumped all
    of it — including a remark about a shirt that was never mentioned again —
    onto whichever garment happened to overlap it most.

    Assigning each *word* to whatever was on screen when it was spoken, then
    breaking wherever that changes, makes a straddling utterance impossible by
    construction. Pauses and sentence ends still split within a garment, for
    readability."""
    if not words:
        return []
    tagged = [(w, which_garment((w["start"] + w["end"]) / 2, windows))
              for w in words]

    # A single stray word tagged to a neighbour mid-sentence is timing jitter,
    # not a real handoff; absorb runs shorter than min_run into what precedes.
    runs: list[list] = []
    for w, gid in tagged:
        if runs and runs[-1][0][1] == gid:
            runs[-1].append((w, gid))
        else:
            runs.append([(w, gid)])
    merged: list[list] = []
    for run in runs:
        if merged and len(run) < min_run and run[0][1] is not None:
            merged[-1].extend((w, merged[-1][0][1]) for w, _ in run)
        else:
            merged.append(run)

    out: list[dict] = []
    for run in merged:
        gid = run[0][1]
        current: list = []
        for w, _ in run:
            if current and (w["start"] - current[-1]["end"] > pause
                            or current[-1]["word"].strip().endswith((".", "?", "!"))):
                out.append({"start": round(current[0]["start"], 2),
                            "end": round(current[-1]["end"], 2),
                            "text": "".join(x["word"] for x in current).strip(),
                            "garment": gid})
                current = []
            current.append(w)
        if current:
            out.append({"start": round(current[0]["start"], 2),
                        "end": round(current[-1]["end"], 2),
                        "text": "".join(x["word"] for x in current).strip(),
                        "garment": gid})
    return [u for u in out if u["text"]]


# ---- 2. align -------------------------------------------------------------

def garment_windows(directory: str) -> list[tuple[int, str, list[tuple[float, float]]]]:
    """When each garment was on screen, as one interval per candidate row.

    Per row, not per garment: a garment seen at 0:10 and again at 3:20 occupies
    two short windows, and the span between them belongs to other items."""
    manifest = json.load(open(os.path.join(directory, "manifest.json")))
    rows = manifest["garments"]
    out = []
    for g in load_catalogue(directory):
        windows = []
        for r in g.grouping["candidate_rows"]:
            if not 0 <= r < len(rows):
                continue
            stamps = [rows[r]["timestamp_s"]] + [v["timestamp_s"]
                                                 for v in rows[r].get("views", [])]
            windows.append((min(stamps), max(stamps)))
        if not windows:
            # Named by image id rather than by row: its only moments on screen
            # are the frames the catalogue resolved for it.
            stamps = [v.provenance["timestamp_s"] for v in g.views
                      if v.kind == "frame"]
            if stamps:
                windows.append((min(stamps), max(stamps)))
        out.append((g.index, g.name, sorted(windows)))
    return out


# ---- 3. distil ------------------------------------------------------------

def front_plates(directory: str, max_dim: int = 640) -> dict[int, str]:
    """One front plate per garment, encoded, keyed by garment index.

    Sent as separate captioned images rather than a single contact sheet: at
    this size the whole wardrobe still costs a few cents either way, but a
    dedicated image keeps a chest logo legible and lets each one be referred to
    by number without counting tiles."""
    path = os.path.join(directory, "products", "products.json")
    if not os.path.exists(path):
        return {}
    out = {}
    for rec in json.load(open(path)).get("images", []):
        if rec.get("view") == "front" and rec.get("file"):
            full = os.path.join(directory, "products", rec["file"])
            if os.path.exists(full):
                out[rec["group"]] = encode(full, max_dim)
    return out


def distil(aligned: list[dict], windows, model: str, timeout: int,
           max_tokens: int, plates: dict[int, str] | None = None) -> dict:
    """One call: the whole transcript, every garment picture, and the timing as
    a hint the model is free to overrule."""
    names = {gid: name for gid, name, _ in windows}
    plates = plates or {}
    spans = {gid: (min(w[0] for w in ws), max(w[1] for w in ws))
             for gid, _, ws in windows if ws}

    lines = ["## Transcript", ""]
    for i, u in enumerate(aligned):
        hint = (f"on screen: garment_{u['garment']}"
                if u["garment"] is not None else "on screen: nothing clear")
        lines.append(f"[{i:>3}] {u['start']:7.1f}s  ({hint})  {u['text']}")

    content: list[dict] = [{"type": "text", "text": PROMPT},
                           {"type": "text", "text": "\n".join(lines)},
                           {"type": "text", "text": "\n## The garments"}]
    for gid in sorted(names):
        lo, hi = spans.get(gid, (0.0, 0.0))
        content.append({"type": "text",
                        "text": f'garment_{gid} — "{names[gid]}", '
                                f'on screen {lo:.0f}-{hi:.0f}s'})
        if gid in plates:
            content.append({"type": "image_url",
                            "image_url": {"url": plates[gid]}})

    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {read_key()}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": content}],
              "response_format": {"type": "json_schema", "json_schema": SCHEMA},
              "reasoning": {"effort": "medium"},
              "max_tokens": max_tokens},
        timeout=timeout,
    )
    if r.status_code != 200:
        sys.exit(f"OpenRouter {r.status_code}: {r.text[:400]}")
    payload = r.json()
    usage = payload.get("usage", {})
    print(f"  distilled  in={usage.get('prompt_tokens')} "
          f"out={usage.get('completion_tokens')} cost=${usage.get('cost', 0):.4f}")
    return json.loads(payload["choices"][0]["message"]["content"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--clip", default=None,
                    help="video to transcribe (default: the clip in the manifest)")
    ap.add_argument("--clips-dir", default="./clips")
    ap.add_argument("--asr-model", default=DEFAULT_ASR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reuse-transcript", action="store_true",
                    help="reuse the transcript already in narration.json; "
                         "transcription is deterministic and slow, the prompt "
                         "is what you iterate on")
    ap.add_argument("--no-distil", action="store_true",
                    help="transcribe and align only; no API call, no cost")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--pause", type=float, default=0.45,
                    help="split within a garment on a gap this long")
    ap.add_argument("--plate-dim", type=int, default=640,
                    help="size of the garment pictures sent as context")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.out_dir, "manifest.json")))
    clip = args.clip
    if not clip:
        name = manifest["garments"][0]["source_clip"]
        clip = os.path.join(args.clips_dir, name)
    if not os.path.exists(clip):
        sys.exit(f"clip not found: {clip} (pass --clip)")

    dest = os.path.join(args.out_dir, "narration.json")
    words = None
    if args.reuse_transcript and os.path.exists(dest):
        prev = json.load(open(dest))
        if prev.get("asr_model") == args.asr_model and prev.get("words"):
            words = prev["words"]
            print(f"  reusing {len(words)} cached words ({args.asr_model})")
    if words is None:
        print(f"transcribing {os.path.basename(clip)} with {args.asr_model} (local)")
        t0 = time.time()
        words = transcribe_words(clip, args.asr_model)
        print(f"  {len(words)} words  ({time.time() - t0:.0f}s)")

    windows = garment_windows(args.out_dir)
    aligned = chunk_by_garment(words, windows, args.pause)
    placed = sum(1 for u in aligned if u["garment"] is not None)
    covered = len({u["garment"] for u in aligned if u["garment"] is not None})
    longest = max((u["end"] - u["start"] for u in aligned), default=0)
    print(f"  {len(aligned)} utterances, {placed} placed, "
          f"covering {covered}/{len(windows)} garments "
          f"(longest {longest:.1f}s)")

    plates = {} if args.no_distil else front_plates(args.out_dir, args.plate_dim)
    if plates:
        print(f"  attaching {len(plates)} front plates for context")
    distilled = {"garments": []} if args.no_distil else \
        distil(aligned, windows, args.model, args.timeout, args.max_tokens, plates)
    by_id = {g["id"]: g for g in distilled.get("garments", [])}

    out = {"asr_model": args.asr_model,
           "model": None if args.no_distil else args.model,
           "clip": os.path.basename(clip),
           "words": words,
           "garments": {}}

    claimed: set[int] = set()
    moved = 0
    for gid, name, _ in windows:
        d = by_id.get(gid)
        if d is not None:
            # The model assigned the lines; the text and timestamps stay ours,
            # so a re-worded quote can never be passed off as a transcript.
            idx = [i for i in d.get("utterances", []) if 0 <= i < len(aligned)]
        elif args.no_distil:
            idx = [i for i, u in enumerate(aligned) if u["garment"] == gid]
        else:
            idx = []
        if not idx:
            continue
        claimed.update(idx)
        moved += sum(1 for i in idx if aligned[i]["garment"] != gid)
        out["garments"][str(gid)] = {
            "quotes": [{"start": aligned[i]["start"], "end": aligned[i]["end"],
                        "text": aligned[i]["text"]} for i in idx],
            "corrected_quotes": (d or {}).get("corrected_quotes", []),
            "material": (d or {}).get("material", ""),
            "provenance": (d or {}).get("provenance", ""),
            "sentiment": (d or {}).get("sentiment", ""),
            "summary": (d or {}).get("summary", ""),
            "cross_reference": (d or {}).get("cross_reference", ""),
        }
    out["unassigned"] = [{"start": u["start"], "text": u["text"]}
                         for i, u in enumerate(aligned) if i not in claimed]
    if not args.no_distil:
        print(f"  model moved {moved} line(s) off the garment the clock "
              f"suggested")

    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    for gid, rec in sorted(out["garments"].items(), key=lambda kv: int(kv[0]))[:40]:
        name = next(n for i, n, _ in windows if i == int(gid))
        bits = " | ".join(f"{k}: {rec[k]}" for k in
                          ("material", "provenance", "sentiment") if rec[k])
        print(f"  g{gid:<3} {name[:30]:<32} {bits[:96] or '(quotes only)'}")
    print(f"\n{len(out['garments'])} garments have narration, "
          f"{len(out['unassigned'])} utterances unassigned")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
