# Motional Scenario Tagging

Integrated autonomous-driving motional scenario tagging pipeline.

The current pipeline converts OD ALT annotations plus ego trajectory into canonical frame JSON, builds overlapping 5-second motional windows, creates model-facing `refined.json` plus BEV keyframes, runs local/server OpenAI-compatible vLLM inference, validates the model output schema, and compares results with separate GT labels when available.

## Pipeline

```text
OD annotations + ego trajectory
  -> canonical frame JSON
  -> overlapping 5-second motional windows
  -> refined.json + BEV keyframes
  -> local/server model inference
  -> schema and deterministic validation
  -> separate GT evaluation
```

## Layout

- `src/motional_tagging/`: reusable pipeline code.
- `scripts/`: thin CLI wrappers.
- `schemas/`: model output schema.
- `prompts/`: system and user prompt templates.
- `configs/`: local/server example configuration only.
- `tests/fixtures/`: small GT and synthetic test fixtures.
- `data/`: local data mount point; private/raw data is ignored.
- `outputs/`: generated artifacts; ignored except `outputs/README.md`.

## Installation

```bash
cd /path/to/motional-scenario-tagging
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the server that hosts vLLM, install the server extras only in the intended server environment:

```bash
python -m pip install -e ".[server]"
```

## Configuration

Copy `.env.example` values into your shell or private `.env` manager. Do not commit real `.env` files.

Important variables:

- `MOTIONAL_DATA_ROOT`: directory containing recording folders with `annotations.json` and `traj_lcs.txt`.
- `MOTIONAL_OUTPUT_ROOT`: generated output root.
- `MOTIONAL_MODEL_INPUT_ROOT`: model input root for inference, defaults to `<output_root>/model_inputs`.
- `VLLM_BASE_URL`: OpenAI-compatible chat completions endpoint.
- `VLLM_MODEL`: model id used by inference.
- `GT_LABEL_ROOT`: private GT label location, if not using `tests/fixtures/gt`.

`configs/local.example.yaml` and `configs/server.example.yaml` show the intended path structure. The current CLIs use command-line arguments and environment variables directly; YAML loading is not implemented yet.

## Server Setup

Keep vLLM bound to localhost. A verified server endpoint shape is:

```bash
http://127.0.0.1:8001/v1/chat/completions
```

Use SSH tunneling from Windows instead of binding the server publicly:

```bash
ssh -L 8001:localhost:8001 stradvision@<server-host>
```

## Windows Setup

Clone the repository, create a Python environment, install the package, and point `MOTIONAL_DATA_ROOT` at a local or mounted copy of the private data.

```powershell
git clone <PRIVATE_REPO_URL>
cd motional-scenario-tagging
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Commands

Canonical JSON:

```bash
python scripts/build_canonical_od_json.py \
  --source-root "$MOTIONAL_DATA_ROOT" \
  --output-root outputs/canonical_frames \
  Rec_Drv_GER_MACHET18_20260227_153128
```

Motional windows:

```bash
python scripts/build_motional_windows.py \
  --canonical-dir outputs/canonical_frames \
  --output-dir outputs/motional_windows
```

Refined input and BEV generation:

```bash
python scripts/build_bev_model_inputs.py \
  --input-dir outputs/motional_windows \
  --output-dir outputs/model_inputs
```

Refined input schema validation:

```bash
python scripts/validate_refined_model_input_schema.py \
  --model-input-dir outputs/model_inputs/Rec_Drv_GER_MACHET18_20260227_153128 \
  --motional-window-file outputs/motional_windows/Rec_Drv_GER_MACHET18_20260227_153128_motional_windows.json
```

Local/server model inference:

```bash
python scripts/run_local_vllm_eval.py \
  --recording Rec_Drv_GER_MACHET18_20260227_153128 \
  --window Rec_Drv_GER_MACHET18_20260227_153128_000-049 \
  --mode json_only \
  --endpoint "${VLLM_BASE_URL:-http://127.0.0.1:8001/v1/chat/completions}" \
  --model-input-root outputs/model_inputs \
  --output-root outputs/local_vllm_eval
```

GT evaluation is part of the inference validation step when `--gt-labels` points at a GT JSON file or when the default `gt_labels/<recording>_gt.json` is available in a compatible deployment. Example:

```bash
python scripts/run_local_vllm_eval.py \
  --recording Rec_Drv_GER_MACHET18_20260227_153128 \
  --window Rec_Drv_GER_MACHET18_20260227_153128_000-049 \
  --gt-labels tests/fixtures/gt/Rec_Drv_GER_MACHET18_20260227_153128_gt.json
```

## Tracked And Ignored Data

Tracked:

- source code and CLI wrappers
- prompt templates and output schema
- example configs
- small GT fixtures
- tests and docs

Ignored:

- raw ALT data
- LiDAR/PCAP/video/model-weight files
- generated canonical/window/refined/BEV/model-output/report artifacts
- virtual environments and caches
- real `.env` files and machine-specific configs

## Sync Through Git

Use the server repository as the source of truth for code. Commit only portable source, tests, prompts, schemas, docs, and small fixtures. Keep raw data and generated outputs on each machine outside Git or under ignored paths.

After a private remote exists:

```bash
git remote add origin <PRIVATE_REPO_URL>
git push -u origin main
```

Windows machines should clone from that remote and configure local data paths through environment variables or private config files.

## Known Limitations

- YAML example files are documentation/config examples only; CLIs do not load YAML yet.
- Full production sample reproduction requires private raw data under `MOTIONAL_DATA_ROOT`.
- vLLM server lifecycle scripts were not imported into this repository.
- The rejected smoothed-kinematics experiment and planned Prompt 4-12 improvements are intentionally excluded.
- Builder defaults were made repo-relative; pass explicit roots when reproducing older server output locations.
