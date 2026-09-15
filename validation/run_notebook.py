"""Execute the release notebook with the invoking Python's Jupyter kernel.

Install .[pipeline], nbformat, nbclient, ipykernel and nbconvert first.
Run from any directory: python validation/run_notebook.py
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "validation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run DINOv2 and live paid AI stages, including narration")
    parser.add_argument("--resume", type=Path, help="Resume a failed run without repeating successful paid stages")
    parser.add_argument("--vision-model", default="openai/gpt-6-astra")
    parser.add_argument("--image-model", default="openai/gpt-5-image-mini")
    args = parser.parse_args()
    if args.full and args.resume:
        parser.error("choose --full or --resume")
    nb = nbformat.read(ROOT / "closetscan_colab.ipynb", as_version=4)
    nbformat.validate(nb)
    dest = DEST
    completed = []
    if args.full or args.resume:
        values = {"RUN_AI": True, "INCLUDE_NARRATION": True, "EMBEDDER": "dinov2",
                  "VISION_MODEL": args.vision_model, "IMAGE_MODEL": args.image_model}
        if args.resume:
            dest = args.resume.resolve()
            previous = nbformat.read(dest / "closetscan.executed.ipynb", as_version=4)
            for node in ast.parse(previous.cells[1].source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in values:
                        values[name] = ast.literal_eval(node.value)
            by_id = {c.id: c for c in previous.cells}
            stages = [("run", "cell-04", "manifest.json"),
                      ("dedup", "stage-dedup", "dedup.json"),
                      ("product_shots", "stage-products", "products/products.json"),
                      ("attributes", "stage-attributes", "attributes.json"),
                      ("narration", "stage-narration", "narration.json")]
            for module, cell_id, artifact in stages:
                cell = by_id.get(cell_id, {})
                if cell.get("execution_count") is None or any(
                    o.output_type == "error" for o in cell.get("outputs", [])
                ) or not (dest / "catalogue" / artifact).is_file():
                    break
                completed.append(module)
            attempt = len(list(dest.glob("attempt-*.ipynb"))) + 1
            shutil.copy2(dest / "closetscan.executed.ipynb", dest / f"attempt-{attempt:02d}.ipynb")
        else:
            dest = DEST / ("full-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            dest.mkdir()
        values["WORKSPACE_DIR"] = str(dest)
        defaults = {"RUN_AI": "False", "INCLUDE_NARRATION": "False", "EMBEDDER": "'hist'",
                    "VISION_MODEL": "''", "IMAGE_MODEL": "''", "WORKSPACE_DIR": "''"}
        replacements = {f"{name} = {defaults[name]}": f"{name} = {value!r}"
                        for name, value in values.items()}
        for old, new in replacements.items():
            if nb.cells[1].source.count(old) != 1:
                raise ValueError(f"Notebook option not found exactly once: {old}")
            nb.cells[1].source = nb.cells[1].source.replace(old, new)
        print(f"Full paid run: {dest}", flush=True)
        if args.resume:
            nb.cells[1].source += (
                f"\n# Resume: these stages already succeeded in the saved prior attempt.\n"
                f"completed_stages = {completed!r}\n"
                "original_run = run\n"
                "def run(module, *args):\n"
                "    if module in completed_stages:\n"
                "        print('Reusing completed stage:', module)\n"
                "    else:\n"
                "        original_run(module, *args)\n"
            )
            nb.cells[3].source = nb.cells[3].source.replace("out.mkdir()", "out.mkdir(exist_ok=True)")
            print(f"Reusing successful stages: {completed}", flush=True)
    def save_checkpoint(**kwargs):
        nbformat.write(nb, dest / "closetscan.executed.ipynb")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="closetscan-kernel-") as runtime:
        spec = Path(runtime) / "python3"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ClosetScan validation", "language": "python",
        }))
        os.environ["IPYTHONDIR"] = str(Path(runtime) / "ipython")
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[runtime]),
        )
        client = NotebookClient(nb, km=manager, timeout=7200,
                                resources={"metadata": {"path": str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index: print(
            f"Starting cell {cell_index + 1}/{len(nb.cells)}", flush=True)
        client.on_cell_executed = save_checkpoint
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            save_checkpoint()
            for cell in nb.cells:
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        print(output.text, end="", flush=True)
            html, _ = HTMLExporter().from_notebook_node(nb)
            (dest / "closetscan.executed.html").write_text(html)
    print(f"Notebook completed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
