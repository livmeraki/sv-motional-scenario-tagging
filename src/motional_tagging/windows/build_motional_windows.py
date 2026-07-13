#!/usr/bin/env python3
"""Build overlapping motional-scenario windows from canonical OD JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_BASE = Path(os.environ.get("MOTIONAL_OUTPUT_ROOT", REPO_ROOT / "outputs"))

CONFIG = {
    "window_duration_s": 5.0,
    "window_stride_s": 2.5,
    "stationary": {"speed_lt_mps": 0.5, "minimum_duration_s": 1.0},
    "speed_bands": {
        "low_magnitude_speed": [0.5, 5.0],
        "medium_magnitude_speed": [5.0, 15.0],
        "high_magnitude_speed": [15.0, None],
    },
    "following": {
        "interval_s": 3.0,
        "ego_speed_gte_mps": 2.0,
        "with_lead_min_fraction": 0.60,
        "without_lead_min_fraction": 0.80,
    },
    "turn_onset": {
        "yaw_rate_abs_gte_radps": 0.08,
        "onset_confirmation_s": 0.5,
        "previous_straight_duration_s": 1.0,
        "previous_yaw_rate_abs_lt_radps": 0.04,
        "minimum_speed_mps": 2.0,
        "positive_yaw_is_left": True,
    },
    "stopping": {
        "approach_speed_gte_mps": 2.0,
        "stopped_speed_lte_mps": 0.5,
        "stopped_sustain_s": 1.0,
        "approach_lookback_s": 2.0,
        "lead_lookback_s": 2.0,
        "with_lead_min_fraction": 0.60,
        "without_lead_min_fraction": 0.80,
    },
    "near_multiple_pedestrians": {
        "minimum_count": 2,
        "distance_m": 15.0,
        "minimum_duration_s": 0.3,
    },
    "near_multiple_motorcycle": {
        "minimum_count": 2,
        "distance_m": 20.0,
        "minimum_duration_s": 0.3,
    },
}

TAXONOMY = [
    "stationary",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "medium_magnitude_speed",
    "following_lane_with_lead",
    "following_lane_without_lead",
    "starting_left_turn",
    "starting_right_turn",
    "stopping_with_lead",
    "stopping_without_lead",
    "near_multiple_pedestrians",
    "near_multiple_motorcycle",
]


def contiguous_true_runs(values):
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index - 1
            start = None


def required_samples(duration_s, frame_interval_s):
    return max(1, round(duration_s / frame_interval_s))


def interval_evidence(frames, start, end):
    return {
        "start_frame": frames[start]["frame_index"],
        "end_frame": frames[end]["frame_index"],
        "start_time_s": frames[start]["time_since_start_s"],
        "end_time_s": frames[end]["time_since_start_s"],
    }


def detect_turn_events(frames, frame_interval):
    config = CONFIG["turn_onset"]
    previous_count = required_samples(config["previous_straight_duration_s"], frame_interval)
    confirmation_count = required_samples(config["onset_confirmation_s"], frame_interval)
    events = []
    for index in range(previous_count, len(frames) - confirmation_count + 1):
        previous_rates = [
            frame["ego"]["yaw_rate_radps"] for frame in frames[index - previous_count : index]
        ]
        confirmation = [
            frame["ego"]["yaw_rate_radps"]
            for frame in frames[index : index + confirmation_count]
        ]
        if any(value is None for value in previous_rates + confirmation):
            continue
        if max(abs(value) for value in previous_rates) >= config["previous_yaw_rate_abs_lt_radps"]:
            continue
        if frames[index]["ego"]["speed_mps"] < config["minimum_speed_mps"]:
            continue
        direction = None
        threshold = config["yaw_rate_abs_gte_radps"]
        if min(confirmation) >= threshold:
            direction = "left"
        elif max(confirmation) <= -threshold:
            direction = "right"
        if direction:
            events.append(
                {
                    "frame_index": frames[index]["frame_index"],
                    "time_s": frames[index]["time_since_start_s"],
                    "direction": direction,
                    "onset_yaw_rate_radps": frames[index]["ego"]["yaw_rate_radps"],
                    "confirmation_median_yaw_rate_radps": round(median(confirmation), 4),
                    "previous_max_abs_yaw_rate_radps": round(
                        max(abs(value) for value in previous_rates), 4
                    ),
                }
            )
    return events


def detect_stopping_events(frames, frame_interval):
    config = CONFIG["stopping"]
    sustain_count = required_samples(config["stopped_sustain_s"], frame_interval)
    approach_count = required_samples(config["approach_lookback_s"], frame_interval)
    lead_count = required_samples(config["lead_lookback_s"], frame_interval)
    stopped = [
        frame["ego"]["speed_mps"] is not None
        and frame["ego"]["speed_mps"] <= config["stopped_speed_lte_mps"]
        for frame in frames
    ]
    events = []
    for start, end in contiguous_true_runs(stopped):
        if end - start + 1 < sustain_count:
            continue
        prior_start = max(0, start - approach_count)
        prior_speeds = [
            frame["ego"]["speed_mps"] for frame in frames[prior_start:start]
            if frame["ego"]["speed_mps"] is not None
        ]
        if not prior_speeds or max(prior_speeds) < config["approach_speed_gte_mps"]:
            continue
        lead_start = max(0, start - lead_count)
        lead_flags = [
            frame["scenario_signals"]["lead_candidate"] is not None
            for frame in frames[lead_start:start]
        ]
        lead_fraction = sum(lead_flags) / len(lead_flags) if lead_flags else 0.0
        events.append(
            {
                "frame_index": frames[start]["frame_index"],
                "time_s": frames[start]["time_since_start_s"],
                "prior_max_speed_mps": round(max(prior_speeds), 4),
                "stopped_duration_s": round(
                    (end - start + 1) * frame_interval, 3
                ),
                "lead_fraction_final_2s": round(lead_fraction, 4),
                "with_lead": lead_fraction >= config["with_lead_min_fraction"],
                "without_lead": (1.0 - lead_fraction)
                >= config["without_lead_min_fraction"],
            }
        )
    return events


def following_candidate(frames, frame_interval, with_lead):
    config = CONFIG["following"]
    interval_count = required_samples(config["interval_s"], frame_interval)
    threshold = (
        config["with_lead_min_fraction"]
        if with_lead
        else config["without_lead_min_fraction"]
    )
    best = None
    for start in range(0, len(frames) - interval_count + 1):
        segment = frames[start : start + interval_count]
        speeds = [frame["ego"]["speed_mps"] for frame in segment]
        if median(speeds) < config["ego_speed_gte_mps"]:
            continue
        lead_fraction = sum(
            frame["scenario_signals"]["lead_candidate"] is not None for frame in segment
        ) / len(segment)
        qualifying_fraction = lead_fraction if with_lead else 1.0 - lead_fraction
        if best is None or qualifying_fraction > best["qualifying_fraction"]:
            best = {
                **interval_evidence(frames, start, start + interval_count - 1),
                "median_ego_speed_mps": round(median(speeds), 4),
                "lead_fraction": round(lead_fraction, 4),
                "qualifying_fraction": round(qualifying_fraction, 4),
            }
    return best if best and best["qualifying_fraction"] >= threshold else None


def near_multiple_candidate(frames, frame_interval, class_name, config_key):
    config = CONFIG[config_key]
    flags = []
    counts = []
    for frame in frames:
        count = sum(
            state["class"] == class_name
            and state["position_ego_m"]["distance"] <= config["distance_m"]
            for state in frame["objects"]
        )
        counts.append(count)
        flags.append(count >= config["minimum_count"])
    minimum_samples = required_samples(config["minimum_duration_s"], frame_interval)
    qualifying = [
        (start, end)
        for start, end in contiguous_true_runs(flags)
        if end - start + 1 >= minimum_samples
    ]
    if not qualifying:
        return None
    start, end = max(qualifying, key=lambda pair: pair[1] - pair[0])
    return {
        **interval_evidence(frames, start, end),
        "maximum_count": max(counts[start : end + 1]),
        "distance_threshold_m": config["distance_m"],
        "sustained_duration_s": round((end - start + 1) * frame_interval, 3),
    }


def build_candidates(window_frames, all_turn_events, all_stop_events, frame_interval):
    speeds = [frame["ego"]["speed_mps"] for frame in window_frames]
    median_speed = median(speeds)
    evidence = {}
    stationary_flags = [
        speed < CONFIG["stationary"]["speed_lt_mps"] for speed in speeds
    ]
    minimum_stationary = required_samples(
        CONFIG["stationary"]["minimum_duration_s"], frame_interval
    )
    stationary_runs = [
        (start, end)
        for start, end in contiguous_true_runs(stationary_flags)
        if end - start + 1 >= minimum_stationary
    ]
    if stationary_runs:
        start, end = max(stationary_runs, key=lambda pair: pair[1] - pair[0])
        evidence["stationary"] = {
            **interval_evidence(window_frames, start, end),
            "duration_s": round((end - start + 1) * frame_interval, 3),
            "maximum_speed_mps": round(max(speeds[start : end + 1]), 4),
        }

    if 0.5 <= median_speed < 5.0:
        evidence["low_magnitude_speed"] = {"median_ego_speed_mps": round(median_speed, 4)}
    elif 5.0 <= median_speed < 15.0:
        evidence["medium_magnitude_speed"] = {"median_ego_speed_mps": round(median_speed, 4)}
    elif median_speed >= 15.0:
        evidence["high_magnitude_speed"] = {"median_ego_speed_mps": round(median_speed, 4)}

    with_lead = following_candidate(window_frames, frame_interval, True)
    without_lead = following_candidate(window_frames, frame_interval, False)
    if with_lead:
        evidence["following_lane_with_lead"] = with_lead
    if without_lead:
        evidence["following_lane_without_lead"] = without_lead

    first_frame = window_frames[0]["frame_index"]
    last_frame = window_frames[-1]["frame_index"]
    for event in all_turn_events:
        if first_frame <= event["frame_index"] <= last_frame:
            evidence.setdefault(f"starting_{event['direction']}_turn", []).append(event)

    for event in all_stop_events:
        if not first_frame <= event["frame_index"] <= last_frame:
            continue
        if event["with_lead"]:
            evidence.setdefault("stopping_with_lead", []).append(event)
        if event["without_lead"]:
            evidence.setdefault("stopping_without_lead", []).append(event)

    pedestrian = near_multiple_candidate(
        window_frames, frame_interval, "pedestrian", "near_multiple_pedestrians"
    )
    motorcycle = near_multiple_candidate(
        window_frames, frame_interval, "motorcycle", "near_multiple_motorcycle"
    )
    if pedestrian:
        evidence["near_multiple_pedestrians"] = pedestrian
    if motorcycle:
        evidence["near_multiple_motorcycle"] = motorcycle

    return {
        "candidate_tags": [tag for tag in TAXONOMY if tag in evidence],
        "candidate_flags": {tag: tag in evidence for tag in TAXONOMY},
        "evidence": evidence,
        "disclaimer": "Heuristic candidates are retrieval hints, not ground-truth labels.",
    }


def summarize_objects(frames):
    by_id = {}
    for frame in frames:
        for state in frame["objects"]:
            entry = by_id.setdefault(
                state["object_id"],
                {
                    "object_id": state["object_id"],
                    "class": state["class"],
                    "minimum_distance_m": None,
                    "observed_frame_count": 0,
                },
            )
            entry["observed_frame_count"] += 1
            distance = state["position_ego_m"]["distance"]
            if entry["minimum_distance_m"] is None or distance < entry["minimum_distance_m"]:
                entry["minimum_distance_m"] = distance
    return sorted(by_id.values(), key=lambda item: item["minimum_distance_m"])


def build_recording(source_path, output_dir):
    with source_path.open(encoding="utf-8") as handle:
        canonical = json.load(handle)
    frames = canonical["frames"]
    frame_interval = canonical["recording"]["median_frame_interval_s"]
    window_count = required_samples(CONFIG["window_duration_s"], frame_interval)
    stride_count = required_samples(CONFIG["window_stride_s"], frame_interval)
    turn_events = detect_turn_events(frames, frame_interval)
    stop_events = detect_stopping_events(frames, frame_interval)

    windows = []
    for start in range(0, len(frames) - window_count + 1, stride_count):
        selected = frames[start : start + window_count]
        windows.append(
            {
                "window_id": f"{canonical['recording_id']}:{start:03d}-{start + window_count - 1:03d}",
                "start_frame": selected[0]["frame_index"],
                "end_frame": selected[-1]["frame_index"],
                "start_time_s": selected[0]["time_since_start_s"],
                "end_time_s": selected[-1]["time_since_start_s"],
                "duration_s": round(
                    selected[-1]["timestamp_unix_s"]
                    - selected[0]["timestamp_unix_s"],
                    4,
                ),
                "ego_summary": {
                    "minimum_speed_mps": round(min(f["ego"]["speed_mps"] for f in selected), 4),
                    "median_speed_mps": round(median(f["ego"]["speed_mps"] for f in selected), 4),
                    "maximum_speed_mps": round(max(f["ego"]["speed_mps"] for f in selected), 4),
                    "minimum_acceleration_mps2": round(
                        min(f["ego"]["acceleration_mps2"] for f in selected), 4
                    ),
                    "maximum_acceleration_mps2": round(
                        max(f["ego"]["acceleration_mps2"] for f in selected), 4
                    ),
                    "minimum_yaw_rate_radps": round(
                        min(f["ego"]["yaw_rate_radps"] for f in selected), 4
                    ),
                    "maximum_yaw_rate_radps": round(
                        max(f["ego"]["yaw_rate_radps"] for f in selected), 4
                    ),
                },
                "object_summary": summarize_objects(selected),
                "preliminary_candidates": build_candidates(
                    selected, turn_events, stop_events, frame_interval
                ),
                "frames": selected,
            }
        )

    result = {
        "schema_version": "od-motional-window-v1",
        "recording_id": canonical["recording_id"],
        "source_canonical_json": source_path.name,
        "scenario_taxonomy": TAXONOMY,
        "candidate_configuration": CONFIG,
        "recording_events": {
            "turn_onsets": turn_events,
            "stopping_events": stop_events,
        },
        "windowing": {
            "window_duration_target_s": CONFIG["window_duration_s"],
            "window_stride_target_s": CONFIG["window_stride_s"],
            "window_frame_count": window_count,
            "stride_frame_count": stride_count,
            "full_windows_only": True,
            "window_count": len(windows),
        },
        "windows": windows,
    }
    output_path = output_dir / source_path.name.replace(
        "_canonical_frames.json", "_motional_windows.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, separators=(",", ":"))
    return output_path, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / "canonical_frames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / "motional_windows",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "od-motional-window-manifest-v1",
        "scenario_taxonomy": TAXONOMY,
        "candidate_configuration": CONFIG,
        "recordings": [],
    }
    for source_path in sorted(args.canonical_dir.glob("*_canonical_frames.json")):
        output_path, result = build_recording(source_path, args.output_dir)
        tag_counts = {
            tag: sum(
                tag in window["preliminary_candidates"]["candidate_tags"]
                for window in result["windows"]
            )
            for tag in TAXONOMY
        }
        manifest["recordings"].append(
            {
                "recording_id": result["recording_id"],
                "path": output_path.name,
                "window_count": len(result["windows"]),
                "candidate_window_counts": tag_counts,
            }
        )
        print(f"Wrote {output_path}")

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
