"""Write the catalogue as one self-contained page you can open by double-click.

    python -m closetscan.html_export ./out
    python -m closetscan.run ./clips --out ./out --html

Constraints that shaped this file, all of them from `file://`:

- No `fetch`. A page opened from disk cannot read a sibling JSON file, so the
  catalogue is inlined into the HTML as a literal.
- No CDN, so no web fonts. The page asks for the grotesque already on the
  machine (Helvetica Neue, then Arial) and the platform mono, so it looks the
  same offline as online.
- Images are *referenced*, not embedded. A wardrobe of plates is tens of
  megabytes; inlining it produces a file no browser enjoys. The page therefore
  travels with its directory, which is the natural unit anyway.

The page lets the clothes supply the colour: every garment's swatches are
sampled from its own plate at export time, and the page itself stays white and
ink so nothing competes with them. It follows the system light/dark setting and
carries a toggle to override it.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter

import cv2
import numpy as np

from .catalogue import Garment, attribute_values, load_catalogue, read_wear


# ---- colour sampling ------------------------------------------------------

def swatches(path: str, k: int = 3) -> list[str]:
    """Dominant garment colours, as hex. Background is dropped first: these
    plates are a garment on white, so keeping the white would return white."""
    img = cv2.imread(path)
    if img is None:
        return []
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = (grey < 242) | (sat > 26)
    px = img[mask]
    if len(px) < 200:
        px = img.reshape(-1, 3)
    px = np.float32(px)
    k = min(k, max(1, len(px) // 50))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    order = [c for c, _ in Counter(labels.flatten().tolist()).most_common()]
    return ["#%02x%02x%02x" % (int(centres[i][2]), int(centres[i][1]),
                               int(centres[i][0])) for i in order]


def build_data(directory: str, garments: list[Garment]) -> dict:
    manifest = json.load(open(os.path.join(directory, "manifest.json")))
    wear = Counter(e["garment_id"] for e in read_wear(directory))
    items = []
    for g in garments:
        plates = [v for v in g.views if v.kind == "plate"]
        frames = [v for v in g.views if v.kind == "frame"]
        hero = next((v for v in plates if v.view == "front"),
                    plates[0] if plates else (frames[0] if frames else None))
        items.append({
            "id": g.id,
            "name": g.name,
            "description": g.description,
            "attributes": g.attributes,
            "hero": hero.file if hero else None,
            "swatches": swatches(os.path.join(directory, hero.file)) if hero else [],
            "plates": [{"view": v.view, "file": v.file,
                        "model": v.provenance.get("model")} for v in plates],
            "frames": [{"view": v.view, "file": v.file,
                        "t": v.provenance.get("timestamp_s"),
                        "row": v.provenance.get("candidate_row")} for v in frames],
            "observations": g.observations,
            "first_seen_s": g.first_seen_s,
            "last_seen_s": g.last_seen_s,
            "clip": g.source_clip,
            "grouping": g.grouping,
            "narration": g.narration,
            "worn": wear.get(g.id, 0),
        })
    first = min((i["first_seen_s"] or 0) for i in items) if items else 0
    last = max((i["last_seen_s"] or 0) for i in items) if items else 0
    return {
        "items": items,
        "facets": attribute_values(garments),
        "meta": {
            "clip": items[0]["clip"] if items else None,
            "n_garments": len(items),
            "n_rows": len(manifest.get("garments", [])),
            "n_frames": manifest.get("n_frames"),
            "embedder": manifest.get("embedder"),
            "span": f"{round(first)}-{round(last)}s",
            "last_seen_s": round(last),
            "has_plates": any(i["plates"] for i in items),
            "n_narrated": sum(1 for i in items if i.get("narration")),
            "any_wear": sum(wear.values()),
        },
    }


# ---- page -----------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --paper:#ffffff; --surface:#f2f2f2; --ink:#1c1c1c; --muted:#767676;
  --faint:#9a9a9a; --rule:#e6e6e6; --plate:#f7f7f7; --plate-edge:#0000000d;
  --plate-edge-hi:#00000021; --accent:#1c1c1c;
  --sans:"Helvetica Neue",Helvetica,Arial,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#101010; --surface:#1a1a1a; --ink:#ededed; --muted:#8f8f8f;
    --faint:#6e6e6e; --rule:#272727; --plate:#efefef; --plate-edge:#ffffff14;
    --plate-edge-hi:#ffffff33; --accent:#ededed;
  }
}
:root[data-theme="dark"]{
  --paper:#101010; --surface:#1a1a1a; --ink:#ededed; --muted:#8f8f8f;
  --faint:#6e6e6e; --rule:#272727; --plate:#efefef; --plate-edge:#ffffff14;
  --plate-edge-hi:#ffffff33; --accent:#ededed;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto;padding-inline:max(24px,4vw)}
:focus-visible{outline:2px solid var(--accent);outline-offset:4px}

/* masthead */
header{padding-block:56px 24px}
.kicker{font-family:var(--mono);font-size:11px;color:var(--faint);margin:0 0 16px}
h1{font-weight:500;font-size:clamp(2rem,4.4vw,3.6rem);line-height:1.05;
  letter-spacing:-.035em;margin:0;text-wrap:balance}
.lede{margin:18px 0 0;max-width:56ch;color:var(--muted);font-size:15px;
  line-height:1.65}
.facts{display:flex;flex-wrap:wrap;gap:6px 26px;margin:28px 0 0;padding:16px 0 0;
  border-top:1px solid var(--rule);font-family:var(--mono);font-size:11px;
  color:var(--muted);font-variant-numeric:tabular-nums}
.facts b{color:var(--ink);font-weight:400}

/* filter rail */
.rail{position:sticky;top:0;z-index:5;background:var(--paper);
  border-bottom:1px solid var(--rule);padding-block:12px;margin-top:28px}
.rail-in{display:flex;gap:7px;align-items:center;flex-wrap:wrap;
  max-width:1280px;margin:0 auto;padding-inline:max(24px,4vw)}
.chip{font-size:12px;padding:8px 14px;border:1px solid var(--rule);
  border-radius:999px;background:none;color:var(--muted);cursor:pointer;
  font-family:inherit;transition:border-color .18s,color .18s,background-color .18s}
.chip:hover{border-color:var(--muted);color:var(--ink)}
.chip[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);
  color:var(--paper)}
.count{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--faint);
  font-variant-numeric:tabular-nums}
#theme{margin-left:10px}

/* grid */
.grid{display:grid;gap:30px 20px;padding-block:34px 64px;
  grid-template-columns:repeat(auto-fill,minmax(224px,1fr))}
.card{border:0;background:none;padding:0;text-align:left;cursor:pointer;
  font:inherit;color:inherit;display:flex;flex-direction:column;gap:12px}
/* The plates carry their own near-white ground, and it is not identical from
   image to image. Letting each one fill its tile hides that variation; a tile
   colour showing through as a border would not. */
.shot{background:var(--plate);border-radius:16px;aspect-ratio:1/1;
  overflow:hidden;display:block;box-shadow:inset 0 0 0 1px var(--plate-edge);
  transition:box-shadow .2s}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.card:hover .shot{box-shadow:inset 0 0 0 1px var(--plate-edge-hi)}
.cap{display:flex;flex-direction:column;gap:6px}
.cname{font-size:14px;font-weight:500;line-height:1.3;letter-spacing:-.01em}
.cmeta{font-family:var(--mono);font-size:11px;color:var(--faint)}
.dots{display:flex;gap:5px;margin-top:2px}
.dot{width:10px;height:10px;border-radius:50%;box-shadow:inset 0 0 0 1px #00000012}
.empty{grid-column:1/-1;color:var(--muted);padding-block:40px}

/* detail */
dialog{border:0;padding:0;max-width:min(1080px,94vw);width:100%;
  border-radius:24px;background:var(--paper);color:var(--ink);max-height:92vh;
  box-shadow:0 30px 80px #00000026}
dialog::backdrop{background:#0c0c0ccc;backdrop-filter:blur(6px)}
.d-in{padding:28px 30px 38px}
.d-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.d-name{font-weight:500;font-size:clamp(1.4rem,3vw,2rem);line-height:1.1;
  margin:0;letter-spacing:-.03em}
.d-desc{color:var(--muted);margin:10px 0 0;max-width:62ch;font-size:13px}
.x{font-family:var(--mono);font-size:14px;background:none;
  border:1px solid var(--rule);border-radius:999px;width:34px;height:34px;
  cursor:pointer;color:var(--muted);flex:none}
.x:hover{background:var(--surface);color:var(--ink)}
.d-plates{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px}
.d-plates figure{margin:0}
figcaption{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:8px}
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:16px 22px;margin-top:24px;padding:20px;border-radius:16px;
  background:var(--surface)}
.spec dt{font-family:var(--mono);font-size:11px;color:var(--faint)}
.spec dd{margin:5px 0 0;font-size:13px}
.spec dd.un{color:var(--faint)}
.said{margin-top:26px;padding-top:24px;border-top:1px solid var(--rule)}
.said h3{font-size:15px;font-weight:500;margin:0 0 14px}
.pull{font-size:clamp(1rem,2vw,1.25rem);line-height:1.45;margin:0 0 18px;
  max-width:52ch;letter-spacing:-.015em;text-wrap:balance}
.said-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:16px 24px;margin:0 0 20px}
.said-grid dt{font-family:var(--mono);font-size:11px;color:var(--faint)}
.said-grid dd{margin:5px 0 0;font-size:13px}
.quotes{list-style:none;margin:0;padding:0}
.quotes li{padding:6px 0;display:flex;gap:11px;align-items:baseline}
.quotes time{font-family:var(--mono);font-size:11px;color:var(--faint);
  font-variant-numeric:tabular-nums;flex:none;background:var(--surface);
  border-radius:999px;padding:4px 9px}
.quotes span{color:var(--muted);font-size:13px}
.badge{font-size:15px;line-height:0;vertical-align:-4px;
  margin-left:3px;color:var(--faint)}
.prov{margin-top:26px;padding-top:24px;border-top:1px solid var(--rule)}
.prov h3{font-size:15px;font-weight:500;margin:0 0 6px}
.prov p{color:var(--muted);margin:0 0 16px;font-size:13px;max-width:64ch}
.strip{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}
.strip figure{margin:0;flex:none;width:116px}
.strip img{width:100%;aspect-ratio:3/4;object-fit:cover;display:block;
  border-radius:10px;box-shadow:inset 0 0 0 1px var(--plate-edge)}
.strip figcaption{margin-top:6px}
.tag{display:inline-block;font-family:var(--mono);font-size:10px;
  border:1px solid var(--rule);padding:2px 7px;border-radius:999px;
  color:var(--muted);margin-right:6px}
footer{border-top:1px solid var(--rule);padding-block:26px 56px;
  color:var(--faint);font-size:13px;max-width:66ch;line-height:1.7}
footer code{font-family:var(--mono);font-size:12px}
@media (max-width:640px){
  .d-plates{grid-template-columns:1fr}
  .grid{grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:24px 12px}
  .d-in{padding:20px 20px 30px}
  header{padding-block:36px 20px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <p class="kicker">__KICKER__</p>
    <h1>__H1__</h1>
    <p class="lede">__LEDE__</p>
    <div class="facts" id="facts"></div>
  </header>
</div>

<nav class="rail"><div class="rail-in" id="rail"></div></nav>

<div class="wrap">
  <div class="grid" id="grid"></div>
  <footer>
    <p>Every plate on this page is <b>generated</b> from video frames, not
    photographed. Open a garment to see the frames it was built from: those are
    the evidence. Where a plate shows a detail no frame shows, the detail is
    invented.</p>
    <p id="narrnote" hidden>Quotes are transcribed from the walkthrough audio
    and matched to whichever garment was on screen as they were spoken. The
    summary lines are a reading of those words; the timestamped quotes are the
    words themselves.</p>
    <p>Built by closetscan from <code id="fclip"></code>. This page reads only
    files in this folder &mdash; move the folder, not the file.</p>
  </footer>
</div>

<dialog id="d"><div class="d-in" id="dbody"></div></dialog>

<script>
const DATA = __DATA__;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const titleCase = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
const UNKNOWN = v => v === undefined || v === null || v === "" ||
  v === "unknown" || (Array.isArray(v) && v.length === 0);

/* facts strip */
const m = DATA.meta;
$("#facts").innerHTML = [
  ["Garments", m.n_garments], ["Candidate rows", m.n_rows],
  ["Frames scored", m.n_frames], ["Seen over", m.span],
  ...(m.n_narrated ? [["Narrated", m.n_narrated]] : []),
  ["Embedder", m.embedder],
].filter(([, v]) => v !== undefined && v !== null)
 .map(([k, v]) => `${k} <b>${esc(v)}</b>`).join("");
$("#fclip").textContent = m.clip || "a closet walkthrough";
if (m.n_narrated) $("#narrnote").hidden = false;

/* filters, built from the attributes this catalogue actually has */
let active = "all";
const cats = [...new Set(DATA.items.map(i => i.attributes.category)
  .filter(c => c && c !== "unknown"))].sort();
function drawRail() {
  const chips = [["all", "Everything"], ...cats.map(c => [c, titleCase(c)])];
  $("#rail").innerHTML = chips.map(([v, label]) =>
    `<button class="chip" data-v="${esc(v)}" aria-pressed="${active === v}">${esc(label)}</button>`
  ).join("") +
  `<span class="count" id="count"></span>` +
  `<button class="chip" id="theme" title="Toggle light and dark">Theme</button>`;
}
function shown() {
  return active === "all" ? DATA.items
    : DATA.items.filter(i => i.attributes.category === active);
}
function drawGrid() {
  const items = shown();
  $("#count").textContent = items.length + (items.length === 1 ? " item" : " items");
  $("#grid").innerHTML = items.length ? items.map(i => `
    <button class="card" data-id="${i.id}">
      <span class="shot">${i.hero ? `<img src="${esc(i.hero)}" alt="${esc(i.name)}" loading="lazy">` : ""}</span>
      <span class="cap">
        <span class="cname">${esc(i.name)}</span>
        <span class="cmeta">${esc(i.attributes.subcategory && i.attributes.subcategory !== "unknown"
            ? i.attributes.subcategory : (i.attributes.category || ""))}${
          i.narration && (i.narration.summary || (i.narration.quotes || []).length)
            ? ` <span class="badge" title="the wearer talked about this one">&#8220;</span>` : ""}</span>
        <span class="dots">${i.swatches.map(s =>
          `<span class="dot" style="background:${esc(s)}"></span>`).join("")}</span>
      </span>
    </button>`).join("")
    : `<p class="empty">Nothing in this category.</p>`;
}
function render() { drawRail(); drawGrid(); }
render();

document.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip[data-v]");
  if (chip) { active = chip.dataset.v; render(); return; }
  if (e.target.closest("#theme")) {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    return;
  }
  const card = e.target.closest(".card");
  if (card) open(card.dataset.id);
});

/* detail */
const SPEC = ["category", "subcategory", "pattern", "material", "formality",
              "season", "colours"];
function open(id) {
  const i = DATA.items.find(x => x.id === id);
  if (!i) return;
  const plates = i.plates.map(p => `
    <figure>
      <span class="shot"><img src="${esc(p.file)}" alt="${esc(i.name)}, ${esc(p.view)}"></span>
      <figcaption>${esc(p.view)} &middot; generated</figcaption>
    </figure>`).join("");
  const spec = SPEC.filter(k => k in i.attributes).map(k => {
    const v = i.attributes[k];
    const text = Array.isArray(v) ? v.join(", ") : v;
    return `<div><dt>${esc(k)}</dt><dd class="${UNKNOWN(v) ? "un" : ""}">${
      UNKNOWN(v) ? "not known" : esc(text)}</dd></div>`;
  }).join("");
  const frames = i.frames.map(f => `
    <figure>
      <img src="${esc(f.file)}" alt="source frame at ${esc(f.t)} seconds" loading="lazy">
      <figcaption>${esc(f.t)}s &middot; ${esc(f.view)}</figcaption>
    </figure>`).join("");
  const n = i.narration || {};
  const refs = (n.refers_to || []).map(r => esc(r.name)).join(", ");
  const said = ["material", "provenance", "sentiment"]
    .filter(k => n[k]).map(k =>
      `<div><dt>${k === "provenance" ? "where it&rsquo;s from" : k}</dt>` +
      `<dd>${esc(n[k])}</dd></div>`).join("");
  const raw = n.quotes || [];
  const fixed = n.corrected_quotes || [];
  // Only pair corrected text with timestamps when the arrays line up one to
  // one. The correction pass drops filler, so a mismatch means every stamp
  // after the first dropped line belongs to a different sentence.
  const qlist = (fixed.length === raw.length && raw.length)
    ? raw.map((q, k) => ({ text: fixed[k], start: q.start }))
    : raw;
  const quotes = qlist.map(q =>
    `<li><time>${q.start != null ? esc(q.start) + "s" : ""}</time>` +
    `<span>${esc(q.text)}</span></li>`).join("");
  const narration = (n.summary || said || quotes || refs) ? `
    <div class="said">
      <h3>In their words</h3>
      ${n.summary ? `<p class="pull">${esc(n.summary)}</p>` : ""}
      ${said || refs ? `<dl class="said-grid">${said}${
        refs ? `<div><dt>also mentioned</dt><dd>${refs}</dd></div>` : ""}</dl>` : ""}
      ${quotes ? `<ul class="quotes">${quotes}</ul>` : ""}
    </div>` : "";
  $("#dbody").innerHTML = `
    <div class="d-top">
      <div>
        <h2 class="d-name">${esc(i.name)}</h2>
        ${i.description ? `<p class="d-desc">${esc(i.description)}</p>` : ""}
      </div>
      <button class="x" id="close" aria-label="Close">&times;</button>
    </div>
    ${plates ? `<div class="d-plates">${plates}</div>` : ""}
    <dl class="spec">${spec}
      <div><dt>seen in clip</dt><dd>${esc(i.first_seen_s)}&ndash;${esc(i.last_seen_s)}s</dd></div>
      <div><dt>frames kept</dt><dd>${esc(i.observations)}</dd></div>
      ${i.worn ? `<div><dt>times worn</dt><dd>${esc(i.worn)}</dd></div>` : ""}
    </dl>
    ${narration}
    <div class="prov">
      <h3>Source frames &mdash; the evidence</h3>
      <p><span class="tag">plate</span>generated above.
         <span class="tag">frame</span>photographed below, from
         ${esc(i.clip || "the walkthrough")}.
         ${i.grouping && i.grouping.reason ? esc(i.grouping.reason) : ""}</p>
      <div class="strip">${frames}</div>
    </div>`;
  $("#d").showModal();
}
$("#d").addEventListener("click", (e) => {
  if (e.target.id === "close" || e.target.id === "d") $("#d").close();
});
document.addEventListener("keydown", (e) => {
  if (!$("#d").open || !["ArrowRight", "ArrowLeft"].includes(e.key)) return;
  const items = shown();
  const cur = items.findIndex(x => x.id === $("#d").dataset.id);
  const next = items[(cur + (e.key === "ArrowRight" ? 1 : -1) + items.length) % items.length];
  if (next) open(next.id);
});
const _open = open;
open = (id) => { $("#d").dataset.id = id; _open(id); };
</script>
</body>
</html>
"""


def render(data: dict, title: str | None = None) -> str:
    m = data["meta"]
    name = title or "Wardrobe"
    lede = (f"{m['n_garments']} garments recovered from a single "
            f"hands-free closet walkthrough, grouped from "
            f"{m['n_rows']} candidates and rendered as AI-generated flat-lays. "
            f"Open any piece to see the frames behind it.")
    if not m["has_plates"]:
        lede = (f"{m['n_garments']} candidates from a closet "
                f"walkthrough, straight from the clustering pass. "
                f"No plates generated yet.")
    return (PAGE
            .replace("__TITLE__", html.escape(name))
            .replace("__KICKER__", f"{html.escape(m['clip'] or 'walkthrough')} &middot; "
                                   f"{m['span']} &middot; closetscan")
            .replace("__H1__", html.escape(name))
            .replace("__LEDE__", lede)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"),
                                            default=str).replace("<", "\\u003c")))


def write_html(directory: str, filename: str = "catalogue.html",
               title: str | None = None, quiet: bool = False) -> str:
    garments = load_catalogue(directory)
    data = build_data(directory, garments)
    dest = os.path.join(directory, filename)
    with open(dest, "w") as fh:
        fh.write(render(data, title))
    if not quiet:
        kb = os.path.getsize(dest) // 1024
        print(f"wrote {dest}  ({len(garments)} garments, {kb}KB, "
              f"open it by double-clicking)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--filename", default="catalogue.html")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    write_html(args.out_dir, args.filename, args.title)


if __name__ == "__main__":
    main()
