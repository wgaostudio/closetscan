# Notebook validation

This folder holds the scripts used to execute and audit the release notebook.
The recorded run artifacts (executed notebooks, catalogue, ZIP, screenshots)
are not distributed with the release because of their size; the results below
summarize them. Rerun the commands in this file to regenerate them locally.

## Full live run

The full demo video was processed in local Jupyter on Python 3.12.13,
macOS arm64. DINOv2, all paid AI stages, and fresh Whisper transcription
completed successfully.

| Stage | Result | Reported API cost |
| --- | --- | ---: |
| DINOv2 extraction | 1,120 retained frames, 68 candidate rows; embedding took 55.6 seconds | Local |
| Grouping, `openai/gpt-6-astra` | 32 row-based garments plus 2 recovered from alternate frames; 9 junk rows | $1.1837 |
| Images, `openai/gpt-5-image-mini` | 68/68 front/back images generated at 1024×1024 | $3.3836 |
| Attributes, `openai/gpt-6-astra` | All 34 garments described | $0.4405 |
| Whisper `medium.en`, CPU int8 | 345 words, 68 utterances; fresh transcription took 49 seconds | Local |
| Narration, `openai/gpt-6-astra` | All 34 garments received narration; 7 utterances remained unassigned | $0.4266 |
| Total reported API usage | No API failures reported | **$5.4344** |

Costs are the usage amounts returned by the provider, not an account invoice.
Image generation took 1,079 seconds. Package installation, model downloads,
and approval/wait time are not included in those stage timings.

The catalogue produced by this run is the one published as the
[demo](https://github.com/YOUR-GITHUB-USERNAME/closetscan/releases/latest); its
`catalogue.html` was restyled afterwards, so the presentation differs from the
recorded run, and its generated images were recompressed from PNG to quality-95
JPEG for distribution.

### Fix found during execution

The grouping model identified two additional garments using `row_0_alt2` and
`row_1_alt2`. The catalogue loader already supported these image-only groups,
but `check_coverage` incorrectly called every group without a primary row
invalid, so the first notebook stopped at its final assertion.

The checker now accepts image-only groups only when all their references exist
among the supplied images. Empty groups and invalid references still fail
validation. The CLI summary includes recovered garments, and the notebook
revalidates the actual grouping rather than trusting a stale warning. A
regression test covers valid and invalid image-only groups. Validation was
resumed after the fix; successful paid stages were not repeated.

### Verification

An independent audit decoded all 184 referenced source/generated images and
checked garment IDs, product pairs, attributes, narration, notebook completion,
MCP listing, and ZIP contents. Headless Chrome opened all 34 garment details,
loaded both generated views and source frames, exercised all four category
filters, and reported no JavaScript errors or mobile horizontal overflow.
Desktop, mobile, and detail screenshots were visually inspected. Browser checks
passed for all 34 garments on September 15, 2026.

## Default histogram smoke test

The release notebook was also executed top-to-bottom in a real local Jupyter
kernel on Python 3.12.13, macOS arm64, using the full included demo video, with
the default configuration `EMBEDDER='hist'`, `RUN_AI=False`, and
`INCLUDE_NARRATION=False`.

The run retained 1,120 frames and produced 64 candidate rows, three contact
sheets, source images, HTML, and a ZIP. All six code cells completed without
errors. The notebook's image-file, MCP listing, and ZIP integrity checks
passed. No paid API calls were made.

The first contact sheet was visually inspected: it contains garment frames,
duplicates, and background-only candidates. The candidate count is not a
measurement of garment recall or accuracy.

## Repeating the run

From the release folder, in an isolated Python environment:

```bash
python -m pip install '.[pipeline]' nbformat nbclient ipykernel nbconvert
python validation/run_notebook.py
python -m unittest discover -s tests -v
```

The runner uses the invoking Python for the Jupyter kernel and saves outputs
even if execution fails. It needs permission to open local kernel ports;
the notebook installation cell also needs package-index access.

To run DINOv2 and all live paid stages, including local Whisper transcription
and paid narration assignment:

```bash
python validation/run_notebook.py --full \
  --vision-model openai/gpt-6-astra \
  --image-model openai/gpt-5-image-mini
```

This uses `OPENROUTER_API_KEY` or `~/.config/openrouter/key` and creates a new
`validation/full-<UTC timestamp>` directory. It downloads the required models
and incurs API charges. The regular notebook defaults remain unpaid.

After fixing a failed stage, resume its saved run with:

```bash
python validation/run_notebook.py --resume validation/full-<UTC timestamp>
python validation/audit_full_run.py validation/full-<UTC timestamp>
node validation/check_browser.mjs validation/full-<UTC timestamp>
```

Resume preserves the previous attempt as a separate notebook and skips the
successful stages that still have their artifacts. Failed stages run again
and can incur charges. Browser checks require Node with built-in WebSocket
support and Chrome; set `CHROME_PATH` if Chrome is not in its default macOS
location. They use an isolated temporary profile and save screenshots.

## Limits

Successful execution does not establish garment-recognition accuracy or image
fidelity. A spot comparison of the purple Allen School tee preserved the chest
lettering but changed the sleeve mark. Generated images remain reconstructions.
No complete hand-labelled inventory or quote-by-quote narration accuracy
evaluation was performed.

All 18 tests pass, including regression coverage for image-only groups,
missing/malformed garment IDs, and corrupt catalogue files. Remaining platform
validation: actual Colab upload/download behavior and Linux/Colab dependency
installation. The live pipeline has been tested locally.
