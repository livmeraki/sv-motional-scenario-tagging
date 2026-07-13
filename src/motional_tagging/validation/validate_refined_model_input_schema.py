#!/usr/bin/env python3
"""Validate refined model-input JSON against the local v2 schema contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DATA_QUALITY = {
    "missing_object_frames",
    "frames_with_static_snapshot_spike",
    "nearby_count_source_available",
    "warnings",
}

REQUIRED_COUNT_FIELDS = {
    "dynamic_object_count",
    "static_visible_object_count",
    "total_object_count",
    "classes_dynamic",
    "classes_static",
    "classes_total",
    "object_count",
    "classes",
    "nearby_pedestrian_count_30m",
    "nearby_motorcycle_count_30m",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_refined(path, motional_by_window):
    errors = []
    refined = load_json(path)
    if refined.get("schema_version") != "od-motional-model-input-v2":
        errors.append(f"{path}: expected schema_version od-motional-model-input-v2")
    if "preliminary_candidates" in refined:
        errors.append(f"{path}: model-facing refined.json must not include preliminary_candidates")
    if "\\" in str(refined.get("source_window_file", "")):
        errors.append(f"{path}: source_window_file contains backslashes")
    metadata = refined.get("schema_metadata", {})
    if "deprecated_fields" not in metadata:
        errors.append(f"{path}: schema_metadata.deprecated_fields missing")
    data_quality = refined.get("data_quality", {})
    missing_quality = REQUIRED_DATA_QUALITY - set(data_quality)
    if missing_quality:
        errors.append(f"{path}: data_quality missing {sorted(missing_quality)}")

    window = motional_by_window.get(refined.get("source_window_id"))
    if not window:
        errors.append(f"{path}: source window not found in motional windows")
        return errors
    frame_by_index = {frame["frame_index"]: frame for frame in window["frames"]}

    for counts in refined.get("per_frame_counts", []):
        missing_count_fields = REQUIRED_COUNT_FIELDS - set(counts)
        if missing_count_fields:
            errors.append(
                f"{path}: frame {counts.get('frame_index')} count fields missing {sorted(missing_count_fields)}"
            )
            continue
        if counts["object_count"] != counts["total_object_count"]:
            errors.append(f"{path}: deprecated object_count does not mirror total_object_count")
        if counts["classes"] != counts["classes_total"]:
            errors.append(f"{path}: deprecated classes does not mirror classes_total")
        if counts["dynamic_object_count"] + counts["static_visible_object_count"] != counts["total_object_count"]:
            errors.append(f"{path}: dynamic + static count mismatch at frame {counts['frame_index']}")
        frame = frame_by_index[counts["frame_index"]]
        source_counts = frame.get("scenario_signals", {}).get("nearby_30m_counts")
        if source_counts is None:
            if counts["nearby_pedestrian_count_30m"] is not None:
                errors.append(f"{path}: pedestrian nearby count should be null when source unavailable")
            if counts["nearby_motorcycle_count_30m"] is not None:
                errors.append(f"{path}: motorcycle nearby count should be null when source unavailable")
        else:
            for class_name, refined_key in (
                ("pedestrian", "nearby_pedestrian_count_30m"),
                ("motorcycle", "nearby_motorcycle_count_30m"),
            ):
                expected = source_counts.get(class_name)
                actual = counts[refined_key]
                if actual != expected:
                    errors.append(
                        f"{path}: {refined_key} frame {counts['frame_index']} expected {expected}, got {actual}"
                    )
                if expected == 0 and actual is None:
                    errors.append(
                        f"{path}: {refined_key} frame {counts['frame_index']} converted valid zero to null"
                    )

    return errors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-dir", type=Path, required=True)
    parser.add_argument("--motional-window-file", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    motional = load_json(args.motional_window_file)
    motional_by_window = {window["window_id"]: window for window in motional["windows"]}
    errors = []
    refined_files = sorted(args.model_input_dir.glob("*/refined.json"))
    if not refined_files:
        raise SystemExit(f"No refined.json files found under {args.model_input_dir}")
    for path in refined_files:
        errors.extend(validate_refined(path, motional_by_window))
    if errors:
        print("Schema validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Schema validation passed for {len(refined_files)} refined.json files.")


if __name__ == "__main__":
    main()
