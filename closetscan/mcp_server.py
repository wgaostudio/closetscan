"""Local stdio MCP server exposing a digitized wardrobe.

    python -m closetscan.mcp_server [CATALOGUE_DIR]

Catalogue directory resolution, first match wins:
    1. the positional CLI argument
    2. $CLOSETSCAN_CATALOGUE
    3. ./out

Speaks JSON-RPC 2.0 over stdin/stdout. Implemented against the standard
library alone, on purpose: this is a personal tool pointed at a directory of
photographs, and it should start on a laptop with no network, no install step
and no package resolution. It opens no sockets and makes no outbound requests.

The catalogue itself is read-only. `log_wear` is the single write, and it
appends to its own file beside the catalogue rather than touching anything the
pipeline produced — a re-run of the pipeline must never be able to destroy
wear history, and a wear log must never be able to corrupt a catalogue.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import date as _date, datetime
from typing import Any

from .catalogue import (Garment, append_wear, attribute_values, load_catalogue,
                        read_wear, WEAR_LOG)

PROTOCOL_VERSION = "2024-11-05"


def _package_version() -> str:
    """Report the installed package's version rather than a second literal that
    can drift from pyproject. The fallback covers running from the source
    folder without installing, which the README supports."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("closetscan")
    except PackageNotFoundError:
        return "0.1.0"


SERVER_INFO = {"name": "closetscan-wardrobe", "version": _package_version()}


# ---- catalogue access -----------------------------------------------------

class Wardrobe:
    """Loads lazily and reloads when the catalogue changes on disk, so a
    pipeline re-run in another terminal is picked up without a restart."""

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        self._garments: list[Garment] | None = None
        self._stamp: tuple | None = None

    def _current_stamp(self) -> tuple:
        out = []
        for name in ("manifest.json", "dedup.json", "attributes.json",
                     "narration.json", os.path.join("products", "products.json")):
            p = os.path.join(self.directory, name)
            out.append(os.path.getmtime(p) if os.path.exists(p) else 0)
        return tuple(out)

    @property
    def garments(self) -> list[Garment]:
        stamp = self._current_stamp()
        if self._garments is None or stamp != self._stamp:
            self._garments = load_catalogue(self.directory)
            self._stamp = stamp
        return self._garments

    def by_id(self, gid: str | None) -> Garment | None:
        if not isinstance(gid, str):
            return None
        match = re.fullmatch(r"[gG]?([0-9]+)", gid.strip())
        if match is None:
            return None
        # tolerate "3", "g3", "G003"
        norm = match.group(1).lstrip("0") or "0"
        for g in self.garments:
            if str(g.index) == norm:
                return g
        return None


# ---- tool implementations -------------------------------------------------

def _matches(g: Garment, key: str, wanted: str) -> bool:
    value = g.attributes.get(key)
    if value is None:
        return False
    values = value if isinstance(value, list) else [value]
    return any(str(v).lower() == wanted.lower() for v in values)


def tool_list_garments(w: Wardrobe, args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 50)), 200))
    offset = max(0, int(args.get("offset", 0)))

    filters: dict[str, str] = {}
    for key in ("category", "colour", "season"):
        if args.get(key):
            filters["colours" if key == "colour" else key] = str(args[key])
    for key, value in (args.get("attributes") or {}).items():
        filters[str(key)] = str(value)

    hits = [g for g in w.garments
            if all(_matches(g, k, v) for k, v in filters.items())]
    page = hits[offset:offset + limit]
    return {
        "total": len(hits),
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "filters_applied": filters or None,
        "garments": [g.compact() for g in page],
        "hint": ("call get_garment for full detail including views and "
                 "provenance") if page else
               ("no garments matched; call list_attributes to see the "
                "attributes and values this catalogue actually has"),
    }


def tool_get_garment(w: Wardrobe, args: dict) -> dict:
    g = w.by_id(args.get("garment_id"))
    if not g:
        return {"error": f"no garment {args.get('garment_id')!r}",
                "known_ids": [x.id for x in w.garments]}
    record = g.full()
    wear = [e for e in read_wear(w.directory) if e.get("garment_id") == g.id]
    record["wear"] = {"times_worn": len(wear),
                      "last_worn": max((e["date"] for e in wear), default=None)}
    record["files_are_relative_to"] = w.directory
    return record


def _score(g: Garment, terms: list[str]) -> float:
    """Weighted substring match. Name is worth most because it is what a person
    would say; attributes next; free text last."""
    fields = [(g.name.lower(), 3.0),
              (str(g.attributes.get("subcategory", "")).lower(), 2.0),
              (" ".join(str(v) for v in (g.attributes.get("colours") or [])).lower(), 2.0),
              (" ".join(f"{k} {v}" for k, v in g.attributes.items()
                        if not isinstance(v, list)).lower(), 1.5),
              (" ".join(str(v) for v in (g.attributes.get("season") or [])).lower(), 1.0),
              (g.description.lower(), 1.0),
              (str(g.grouping.get("label", "")).lower(), 1.0)]
    total = 0.0
    for term in terms:
        for text, weight in fields:
            if term in text:
                total += weight * (2.0 if text.startswith(term) else 1.0)
                break
    return total


def tool_search_garments(w: Wardrobe, args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    limit = max(1, min(int(args.get("limit", 20)), 100))
    if not query:
        return {"error": "query is required"}
    terms = [t for t in query.lower().split() if t]
    scored = [(s, g) for g in w.garments if (s := _score(g, terms)) > 0]
    scored.sort(key=lambda p: (-p[0], p[1].index))
    return {
        "query": query,
        "matched": len(scored),
        "garments": [{**g.compact(), "score": round(s, 1)}
                     for s, g in scored[:limit]],
        "hint": None if scored else
                "nothing matched; this catalogue is one person's wardrobe and "
                "may simply not contain it — say so rather than substituting "
                "a near match",
    }


def tool_search_narration(w: Wardrobe, args: dict) -> dict:
    """Search what the wearer said, not what a model saw.

    Kept separate from search_garments because the two answer different
    questions. "navy polo" is a question about appearance; "what did I say I
    never wear" is a question about testimony, and only the narration can
    answer it."""
    query = str(args.get("query", "")).strip().lower()
    field = str(args.get("field", "any")).lower()
    limit = max(1, min(int(args.get("limit", 20)), 100))
    if not query:
        return {"error": "query is required"}
    terms = [t for t in query.split() if t]

    hits = []
    for g in w.garments:
        n = g.narration
        if not n:
            continue
        pools = {
            "material": n.get("material", ""),
            "provenance": n.get("provenance", ""),
            "sentiment": n.get("sentiment", ""),
            "summary": n.get("summary", ""),
            "quotes": " ".join(q["text"] for q in n.get("quotes", [])),
        }
        searchable = pools if field == "any" else {field: pools.get(field, "")}
        blob = " ".join(searchable.values()).lower()
        score = sum(1 for t in terms if t in blob)
        if not score:
            continue
        matched = [q for q in n.get("quotes", [])
                   if any(t in q["text"].lower() for t in terms)]
        hits.append((score, {
            "id": g.id, "name": g.name, "matched_terms": score,
            "material": n.get("material", ""),
            "provenance": n.get("provenance", ""),
            "sentiment": n.get("sentiment", ""),
            "summary": n.get("summary", ""),
            "matching_quotes": matched or n.get("quotes", [])[:2],
        }))
    hits.sort(key=lambda p: -p[0])
    return {
        "query": query, "field": field, "matched": len(hits),
        "garments": [h for _, h in hits[:limit]],
        "note": ("quotes are the wearer's own words, transcribed from the "
                 "walkthrough audio; material/provenance/sentiment are a "
                 "summary of those words, so quote the quotes"),
    }


def tool_log_wear(w: Wardrobe, args: dict) -> dict:
    g = w.by_id(args.get("garment_id"))
    if not g:
        return {"error": f"no garment {args.get('garment_id')!r}; nothing logged",
                "known_ids": [x.id for x in w.garments]}
    when = str(args.get("date") or _date.today().isoformat())
    try:
        datetime.strptime(when, "%Y-%m-%d")
    except ValueError:
        return {"error": f"date must be YYYY-MM-DD, got {when!r}; nothing logged"}
    entry = append_wear(w.directory, g.id, when, args.get("note"))
    worn = sum(1 for e in read_wear(w.directory) if e["garment_id"] == g.id)
    return {"logged": entry, "garment_name": g.name, "times_worn": worn,
            "log_file": os.path.join(w.directory, WEAR_LOG)}


def tool_get_wear_history(w: Wardrobe, args: dict) -> dict:
    entries = read_wear(w.directory)
    gid = args.get("garment_id")
    if gid:
        g = w.by_id(gid)
        if not g:
            return {"error": f"no garment {gid!r}"}
        entries = [e for e in entries if e.get("garment_id") == g.id]
    since, until = args.get("since"), args.get("until")
    if since:
        entries = [e for e in entries if e.get("date", "") >= str(since)]
    if until:
        entries = [e for e in entries if e.get("date", "") <= str(until)]

    counts = Counter(e["garment_id"] for e in entries)
    names = {g.id: g.name for g in w.garments}
    per_garment = []
    for cat_id, n in counts.most_common():
        dates = [e["date"] for e in entries if e["garment_id"] == cat_id]
        rec = {"garment_id": cat_id, "name": names.get(cat_id, "(not in catalogue)"),
               "times_worn": n, "first": min(dates), "last": max(dates)}
        g = w.by_id(cat_id)
        cost = (g.attributes.get("cost") if g else None)
        if isinstance(cost, (int, float)) and n:
            rec["cost"] = cost
            rec["cost_per_wear"] = round(cost / n, 2)
        per_garment.append(rec)

    never = [g.compact() for g in w.garments if g.id not in counts] if not gid else []
    return {
        "entries": sorted(entries, key=lambda e: e.get("date", "")),
        "total_entries": len(entries),
        "per_garment": per_garment,
        "never_worn": never,
        "note": ("cost_per_wear is only reported for garments carrying a "
                 "numeric 'cost' attribute; this pipeline does not infer "
                 "prices, so add them yourself to use it"),
    }


def tool_list_attributes(w: Wardrobe, args: dict) -> dict:
    return {
        "catalogue_dir": w.directory,
        "n_garments": len(w.garments),
        "attributes": attribute_values(w.garments),
        "note": "filter list_garments on any of these keys via 'attributes'",
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_garments",
        "description": ("List garments in the wardrobe, optionally filtered. "
                        "Returns compact records; use get_garment for full "
                        "detail. Paginated via limit/offset."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "description": "e.g. top, bottom, outerwear, knitwear"},
                "colour": {"type": "string", "description": "e.g. navy, cream"},
                "season": {"type": "string",
                           "description": "spring, summer, autumn, winter, all-season"},
                "attributes": {"type": "object",
                               "description": "Any other attribute present in "
                                              "this catalogue, as key/value. "
                                              "See list_attributes.",
                               "additionalProperties": {"type": "string"}},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
        "handler": tool_list_garments,
    },
    {
        "name": "get_garment",
        "description": ("Full record for one garment: all attributes, every "
                        "view with its provenance, and file paths."),
        "inputSchema": {
            "type": "object",
            "properties": {"garment_id": {"type": "string",
                                          "description": "e.g. g003"}},
            "required": ["garment_id"],
        },
        "handler": tool_get_garment,
    },
    {
        "name": "search_garments",
        "description": ("Free-text search over names, descriptions and "
                        "attributes. Returns compact records, best match "
                        "first."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
        "handler": tool_search_garments,
    },
    {
        "name": "search_narration",
        "description": ("Search what the wearer said about their clothes while "
                        "filming — materials, where things came from, and how "
                        "they feel about them. Use this for questions about "
                        "history, feelings or origin; use search_garments for "
                        "questions about appearance."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "field": {"type": "string",
                          "enum": ["any", "material", "provenance",
                                   "sentiment", "summary", "quotes"],
                          "description": "restrict the search (default any)"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
        "handler": tool_search_narration,
    },
    {
        "name": "log_wear",
        "description": ("Record that a garment was worn. Appends to the wear "
                        "log; never modifies the catalogue. Date defaults to "
                        "today."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "garment_id": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "note": {"type": "string"},
            },
            "required": ["garment_id"],
        },
        "handler": tool_log_wear,
    },
    {
        "name": "get_wear_history",
        "description": ("Read the wear log: individual entries plus per-garment "
                        "totals, first/last worn, never-worn items, and "
                        "cost-per-wear where a cost attribute exists."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "garment_id": {"type": "string",
                               "description": "omit for the whole wardrobe"},
                "since": {"type": "string", "description": "YYYY-MM-DD"},
                "until": {"type": "string", "description": "YYYY-MM-DD"},
            },
        },
        "handler": tool_get_wear_history,
    },
    {
        "name": "list_attributes",
        "description": ("Which attributes this catalogue actually carries and "
                        "what values they take. Call this before guessing "
                        "filter values."),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_list_attributes,
    },
]

HANDLERS = {t["name"]: t["handler"] for t in TOOLS}
TOOL_SPECS = [{k: v for k, v in t.items() if k != "handler"} for t in TOOLS]


# ---- JSON-RPC plumbing ----------------------------------------------------

def _result(rid: Any, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": payload}


def _error(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle(message: dict, wardrobe: Wardrobe) -> dict | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
    if "id" not in message:
        return None
    if message.get("params") is not None and not isinstance(message["params"], dict):
        return _error(message.get("id"), -32602, "params must be an object")
    method = message.get("method")
    rid = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _result(rid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "A single person's wardrobe, digitized from a video walkthrough. "
                "Garment images are of two kinds: 'frame' views are photographs "
                "from the video, 'plate' views are generated reconstructions. "
                "Treat plates as a likeness, not evidence."),
        })

    if method in ("notifications/initialized", "initialized"):
        return None                              # notification: no reply

    if method == "ping":
        return _result(rid, {})

    if method == "tools/list":
        return _result(rid, {"tools": TOOL_SPECS})

    if method == "tools/call":
        name = params.get("name")
        handler = HANDLERS.get(name)
        if not handler:
            return _error(rid, -32602, f"unknown tool {name!r}")
        try:
            payload = handler(wardrobe, params.get("arguments") or {})
            is_error = isinstance(payload, dict) and "error" in payload
        except FileNotFoundError as exc:
            payload, is_error = {"error": str(exc)}, True
        except Exception as exc:                 # never take the server down
            payload, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
        return _result(rid, {
            "content": [{"type": "text",
                         "text": json.dumps(payload, indent=2, default=str)}],
            "isError": is_error,
        })

    if rid is None:
        return None                              # unknown notification
    return _error(rid, -32601, f"unknown method {method!r}")


def resolve_directory(argv: list[str]) -> str:
    if len(argv) > 1 and not argv[1].startswith("-"):
        return argv[1]
    return os.environ.get("CLOSETSCAN_CATALOGUE") or "./out"


def main() -> None:
    directory = resolve_directory(sys.argv)
    wardrobe = Wardrobe(directory)
    # stderr only: stdout is the protocol channel and anything else corrupts it
    print(f"closetscan wardrobe MCP: {wardrobe.directory}", file=sys.stderr)
    if not os.path.exists(os.path.join(wardrobe.directory, "manifest.json")):
        print(f"  warning: no manifest.json there yet; tools will report the "
              f"error until the catalogue exists", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_error(None, -32700, "Parse error")) + "\n")
            sys.stdout.flush()
            continue
        try:
            response = handle(message, wardrobe)
        except Exception as exc:                 # last-resort guard
            response = _error(message.get("id"), -32603, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
