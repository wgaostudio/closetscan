"""Check saved full-run artifacts independently of the notebook kernel."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import cv2
from closetscan.dedup import check_coverage


def audit(directory):
    out = directory / "catalogue"
    read = lambda path: json.loads(path.read_text())
    manifest = read(out / "manifest.json")
    grouping = read(out / "dedup.json")
    products = read(out / "products/products.json")
    attrs = read(out / "attributes.json")
    narration = read(out / "narration.json")
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(manifest["embedder"] == "dinov2", "DINOv2 was not used")
    n_rows = len(manifest["garments"])
    counts = Counter(grouping["junk_rows"])
    for g in grouping["groups"]:
        counts.update(g["rows"])
    check(counts == Counter(range(n_rows)), "Grouping does not cover each row exactly once")
    valid_images = {f"row_{i}" for i in range(n_rows)}
    valid_images.update(f"row_{i}_alt{j}" for i, row in enumerate(manifest["garments"])
                        for j, _ in enumerate(row.get("views", [])))
    check(not check_coverage(grouping, n_rows, valid_images), "Grouping coverage is invalid")

    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "list_garments", "arguments": {"limit": 200}}}
    reply = subprocess.run([sys.executable, "-m", "closetscan.mcp_server", str(out)],
                           input=json.dumps(request) + "\n", capture_output=True,
                           text=True, check=True)
    payload = json.loads(reply.stdout)["result"]
    check(not payload["isError"], "MCP listing failed")
    listing = json.loads(payload["content"][0]["text"])
    n_garments = listing["total"]
    expected_ids = {str(i) for i in range(n_garments)}
    check(set(attrs["garments"]) == expected_ids, "Attribute garment IDs do not match the catalogue")
    check(set(narration["garments"]) <= expected_ids, "Narration contains unknown garment IDs")
    check(bool(narration.get("words")), "Transcript has no words")
    check(bool(narration["garments"]), "No garments have narration")

    images = products["images"]
    pairs = Counter((p["group"], p["view"]) for p in images if p.get("file"))
    check(pairs == Counter((i, side) for i in range(n_garments)
                          for side in ("front", "back")), "Missing or duplicate product views")
    check(all(p.get("file") and not p.get("error") for p in images), "Product generation failures remain")
    paths = {out / row["best_shot"] for row in manifest["garments"]}
    paths.update(out / view["file"] for row in manifest["garments"] for view in row.get("views", []))
    paths.update(out / "products" / p["file"] for p in images if p.get("file"))
    for path in paths:
        img = cv2.imread(str(path))
        check(img is not None and img.size > 0, f"Undecodable image: {path.relative_to(out)}")

    nb = read(directory / "closetscan.executed.ipynb")
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    check(all(c.get("execution_count") is not None for c in cells), "Some code cells were not executed")
    check(not any(o["output_type"] == "error" for c in cells for o in c.get("outputs", [])), "Notebook contains cell errors")
    stage_costs = {}
    cost_cells = [c for path in sorted(directory.glob("attempt-*.ipynb"))
                  for c in read(path)["cells"] if c["cell_type"] == "code"] + cells
    for cell in cost_cells:
        if cell["id"] not in ("stage-dedup", "stage-attributes", "stage-narration"):
            continue
        text = "".join("".join(o.get("text", [])) for o in cell.get("outputs", [])
                       if o["output_type"] == "stream")
        costs = [float(x) for x in re.findall(r"cost=\$(\d+\.\d+)", text)]
        if costs:
            stage = cell["id"].removeprefix("stage-")
            stage_costs[stage] = round(stage_costs.get(stage, 0) + sum(costs), 4)
    stage_costs["products"] = products["total_cost_usd"]
    with zipfile.ZipFile(directory / "catalogue.zip") as archive:
        check(archive.testzip() is None, "ZIP integrity check failed")
        check("catalogue.html" in archive.namelist(), "ZIP is missing the HTML catalogue")
        for path in paths:
            check(path.relative_to(out).as_posix() in archive.namelist(), "ZIP is missing an image")

    report = {
        "passed": not errors, "errors": errors,
        "original_grouping_warnings": grouping.get("coverage_problems", []),
        "coverage_revalidated_with_image_references": True,
        "frames": manifest["n_frames"], "candidate_rows": n_rows,
        "garments": n_garments, "product_images": len(images),
        "decoded_images": len(paths), "transcript_words": len(narration["words"]),
        "garments_with_narration": len(narration["garments"]),
        "unassigned_utterances": len(narration.get("unassigned", [])),
        "reported_stage_costs_usd": stage_costs,
        "reported_total_cost_usd": round(sum(stage_costs.values()), 4),
        "cost_note": "Sum of recorded successful-stage usage and any logged dedup attempts; excludes charges not returned by the provider.",
        "models": {"vision": grouping["model"], "images": products["model"],
                   "asr": narration["asr_model"]},
    }
    (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    sys.exit(0 if audit(args.directory.resolve()) else 1)
