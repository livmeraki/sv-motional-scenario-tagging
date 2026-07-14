from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from motional_tagging.canonical.build_canonical_od_json import build_recording as build_canonical
from motional_tagging.inference.run_local_vllm_eval import (
    load_gt_labels,
    output_window_ids,
    retry_prompt,
    validate_against_gt,
    validate_output,
)
from motional_tagging.model_inputs.build_bev_model_inputs import build_refined_json
from motional_tagging.validation.validate_refined_model_input_schema import validate_refined
from motional_tagging.windows.build_motional_windows import build_recording as build_windows


RECORDING = "Rec_Drv_GER_MACHET18_20260227_153128"


def write_synthetic_recording(root: Path) -> None:
    rec_dir = root / RECORDING
    rec_dir.mkdir(parents=True)
    annotations = {"scene": {"frameCount": 60}, "objects": []}
    (rec_dir / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    rows = []
    for idx in range(60):
        timestamp = idx * 0.1
        rows.append(f"{timestamp:.1f} {idx * 1.0:.3f} 0.0 0.0 0.0 0.0 0.0 1.0")
    (rec_dir / "traj_lcs.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_imports_work() -> None:
    import motional_tagging

    assert motional_tagging.__version__


def test_cli_help_works() -> None:
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    for script in (
        "build_canonical_od_json.py",
        "build_motional_windows.py",
        "build_bev_model_inputs.py",
        "run_local_vllm_eval.py",
    ):
        result = subprocess.run(
            [sys.executable, str(repo / "scripts" / script), "--help"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_sample_pipeline_and_schema_validation(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    canonical_dir = tmp_path / "canonical"
    windows_dir = tmp_path / "windows"
    model_inputs_dir = tmp_path / "model_inputs"
    write_synthetic_recording(source_root)
    windows_dir.mkdir(parents=True)

    canonical_path, canonical = build_canonical(source_root, canonical_dir, RECORDING)
    assert canonical_path.is_file()
    assert canonical["recording"]["frame_count"] == 60

    windows_path, windows = build_windows(canonical_path, windows_dir)
    assert windows_path.is_file()
    assert windows["windowing"]["window_count"] == 1
    assert windows["windows"][0]["window_id"] == f"{RECORDING}:000-049"

    recording = dict(windows)
    recording["_source_file"] = windows_path
    window = windows["windows"][0]
    window_dir = model_inputs_dir / RECORDING / f"{RECORDING}_000-049"
    window_dir.mkdir(parents=True)
    keyframe_files = {
        "start": {"frame_index": 0, "path": "bev_start.png"},
        "middle": {"frame_index": 25, "path": "bev_middle.png"},
        "end": {"frame_index": 49, "path": "bev_end.png"},
    }
    refined = build_refined_json(recording, window, keyframe_files, max_objects=24, include_preliminary_candidates=False)
    refined_path = window_dir / "refined.json"
    refined_path.write_text(json.dumps(refined), encoding="utf-8")

    errors = validate_refined(refined_path, {window["window_id"]: window})
    assert errors == []


def test_output_schema_loads() -> None:
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "schemas" / "output_schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["labels"]["required"]
    assert "stationary" in schema["properties"]["labels"]["required"]
    decision_schema = schema["$defs"]["label_decision"]["properties"]
    assert decision_schema["evidence_frames"]["maxItems"] == 3
    assert decision_schema["object_ids"]["maxItems"] == 2


def test_model_output_rejects_long_evidence_arrays() -> None:
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "schemas" / "output_schema.json").read_text(encoding="utf-8"))
    label_names = schema["properties"]["labels"]["required"]
    labels = {
        label: {
            "value": False,
            "confidence": 0.1,
            "evidence_summary": "none",
            "evidence_frames": [0, 5, 25, 49],
            "object_ids": ["1", "2", "3"],
        }
        for label in label_names
    }
    output = {
        "schema_version": "motional-scenario-model-output-v1",
        "recording_id": RECORDING,
        "window_id": f"{RECORDING}:000-049",
        "model_mode": "json_bev",
        "labels": labels,
        "overall_quality": {"confidence": 0.5, "data_issues": []},
        "review_priority": "low",
    }
    refined = {"ego_summary": {"median_speed_mps": 0.0, "minimum_speed_mps": 0.0}}

    errors = validate_output(output, RECORDING, f"{RECORDING}_000-049", "json_bev", refined)

    assert "labels.stationary.evidence_frames must contain at most 3 items" in errors
    assert "labels.stationary.object_ids must contain at most 2 items" in errors


def test_retry_prompt_mentions_shortening_arrays() -> None:
    prompt = retry_prompt(["labels.stationary.object_ids must contain at most 2 items"])
    assert "Shorten arrays" in prompt
    assert "evidence_frames array must have at most 3 items" in prompt
    assert "object_ids array must have at most 2 items" in prompt


def test_gt_window_id_matching() -> None:
    gt_path = Path(__file__).resolve().parent / "fixtures" / "gt" / f"{RECORDING}_gt.json"
    labels = load_gt_labels(gt_path, RECORDING, f"{RECORDING}_000-049")
    assert labels is not None
    assert output_window_ids(RECORDING, f"{RECORDING}_000-049") == {
        f"{RECORDING}_000-049",
        f"{RECORDING}:000-049",
    }


def test_gt_mismatch_does_not_create_validation_error() -> None:
    gt_labels = {"stationary": True}
    output = {"labels": {"stationary": {"value": False}}}
    result = validate_against_gt(output, gt_labels)
    assert result["status"] == "failed"
    assert result["mismatches"] == [{"label": "stationary", "expected": True, "actual": False}]


def test_no_tracked_secret_or_server_absolute_path_patterns() -> None:
    repo = Path(__file__).resolve().parents[1]
    banned = (
        "sk" + "-",
        "BEGIN " + "PRIVATE KEY",
        "/home/" + "stradvision",
        "/media/" + "stradvision",
        "C:" + "\\Users",
    )
    for path in repo.rglob("*"):
        if path.is_dir() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".png", ".pyc"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned:
            assert pattern not in text, f"{pattern} found in {path.relative_to(repo)}"
