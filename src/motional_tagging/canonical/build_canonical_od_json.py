#!/usr/bin/env python3
"""Convert OD ALT annotations and ego trajectories into model-ready frame JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "2600_MV2_OD_traj_annotations"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MOTIONAL_OUTPUT_ROOT", REPO_ROOT / "outputs")) / "canonical_frames"

DEFAULT_RECORDINGS = [
    "Rec_Drv_GER_MACHET18_20260227_153128",
    "Rec_Drv_GER_MACHET18_20260227_155628",
    "Rec_Drv_GER_MACHET18_20260227_151728",
]

SCENARIO_TAXONOMY = [
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

MOTIONAL_CLASSES = {
    "car",
    "truck",
    "truck_head",
    "bus",
    "trailer",
    "special_vehicle",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider_other",
    "animal",
}

LEAD_CLASSES = {
    "car",
    "truck",
    "truck_head",
    "bus",
    "trailer",
    "special_vehicle",
    "motorcycle",
    "bicycle",
}

MAX_KINEMATIC_SAMPLE_GAP_S = 0.25
INTERACTION_HORIZON_S = 5.0
EGO_LENGTH_M = 4.8
EGO_WIDTH_M = 2.0
PATH_MARGIN_M = 0.5


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def round_or_none(value, digits=4):
    return round(value, digits) if finite_number(value) else None


def quaternion_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def central_derivative(values, timestamps, index):
    if len(values) < 2:
        return None
    left = max(0, index - 1)
    right = min(len(values) - 1, index + 1)
    dt = timestamps[right] - timestamps[left]
    return None if dt <= 0 else (values[right] - values[left]) / dt


def parse_trajectory(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 columns")
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            rows.append(
                {
                    "timestamp": timestamp,
                    "position": (tx, ty, tz),
                    "quaternion": (qx, qy, qz, qw),
                    "yaw": quaternion_yaw(qx, qy, qz, qw),
                }
            )

    timestamps = [row["timestamp"] for row in rows]
    for index, row in enumerate(rows):
        vx = central_derivative([r["position"][0] for r in rows], timestamps, index)
        vy = central_derivative([r["position"][1] for r in rows], timestamps, index)
        vz = central_derivative([r["position"][2] for r in rows], timestamps, index)
        row["velocity"] = (vx, vy, vz)
        row["speed"] = math.hypot(vx, vy) if vx is not None and vy is not None else None

    speeds = [row["speed"] for row in rows]
    unwrapped_yaw = [rows[0]["yaw"]]
    for row in rows[1:]:
        unwrapped_yaw.append(unwrapped_yaw[-1] + wrap_angle(row["yaw"] - unwrapped_yaw[-1]))
    for index, row in enumerate(rows):
        row["acceleration"] = central_derivative(speeds, timestamps, index)
        row["yaw_rate"] = central_derivative(unwrapped_yaw, timestamps, index)
    return rows


def valid_bbox(bbox):
    required = ("x", "y", "z", "length", "width", "height", "qx", "qy", "qz", "qw")
    return isinstance(bbox, dict) and all(finite_number(bbox.get(key)) for key in required)


def bbox_yaw(bbox):
    return quaternion_yaw(bbox["qx"], bbox["qy"], bbox["qz"], bbox["qw"])


def ego_relative(position, ego_position, ego_yaw):
    dx = position[0] - ego_position[0]
    dy = position[1] - ego_position[1]
    cosine = math.cos(ego_yaw)
    sine = math.sin(ego_yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def build_object_samples(objects, timestamps):
    samples_by_object = {}
    for obj in objects:
        samples = []
        for frame_key, frame_data in obj.get("frames", {}).items():
            frame_index = int(frame_data.get("frameIndex", frame_key))
            bbox = frame_data.get("bbox3d")
            if 0 <= frame_index < len(timestamps) and valid_bbox(bbox):
                samples.append((frame_index, bbox))
        samples.sort(key=lambda item: item[0])

        velocities = {}
        for index, (frame_index, bbox) in enumerate(samples):
            left = max(0, index - 1)
            right = min(len(samples) - 1, index + 1)
            left_frame, left_bbox = samples[left]
            right_frame, right_bbox = samples[right]
            dt = timestamps[right_frame] - timestamps[left_frame]
            left_gap = timestamps[frame_index] - timestamps[left_frame]
            right_gap = timestamps[right_frame] - timestamps[frame_index]
            if (
                dt > 0
                and left != right
                and left_gap <= MAX_KINEMATIC_SAMPLE_GAP_S
                and right_gap <= MAX_KINEMATIC_SAMPLE_GAP_S
            ):
                velocities[frame_index] = (
                    (right_bbox["x"] - left_bbox["x"]) / dt,
                    (right_bbox["y"] - left_bbox["y"]) / dt,
                    (right_bbox["z"] - left_bbox["z"]) / dt,
                )

        accelerations = {}
        velocity_samples = sorted(velocities.items())
        for index, (frame_index, velocity) in enumerate(velocity_samples):
            left = max(0, index - 1)
            right = min(len(velocity_samples) - 1, index + 1)
            left_frame, left_velocity = velocity_samples[left]
            right_frame, right_velocity = velocity_samples[right]
            dt = timestamps[right_frame] - timestamps[left_frame]
            left_gap = timestamps[frame_index] - timestamps[left_frame]
            right_gap = timestamps[right_frame] - timestamps[frame_index]
            if (
                dt > 0
                and left != right
                and left_gap <= MAX_KINEMATIC_SAMPLE_GAP_S
                and right_gap <= MAX_KINEMATIC_SAMPLE_GAP_S
            ):
                accelerations[frame_index] = (
                    (right_velocity[0] - left_velocity[0]) / dt,
                    (right_velocity[1] - left_velocity[1]) / dt,
                    (right_velocity[2] - left_velocity[2]) / dt,
                )
        samples_by_object[str(obj["objectId"])] = {
            "bbox_by_frame": dict(samples),
            "velocity_by_frame": velocities,
            "acceleration_by_frame": accelerations,
        }
    return samples_by_object


def attributes_as_object(attributes):
    result = {}
    for attribute in attributes or []:
        if isinstance(attribute, dict) and "key" in attribute:
            result[str(attribute["key"])] = attribute.get("value")
    return result


def make_object_state(obj, frame_index, ego, samples):
    object_id = str(obj["objectId"])
    sample = samples[object_id]
    bbox = sample["bbox_by_frame"].get(frame_index)
    geometry_source = "per_frame_bbox3d"

    if bbox is None and obj.get("type") == "static" and frame_index in obj.get("visible_frames", []):
        candidate = obj.get("bbox3d")
        if valid_bbox(candidate):
            bbox = candidate
            geometry_source = "object_bbox3d"
    if bbox is None:
        return None

    position = (bbox["x"], bbox["y"], bbox["z"])
    rel_x, rel_y = ego_relative(position, ego["position"], ego["yaw"])
    velocity = sample["velocity_by_frame"].get(frame_index)
    acceleration = sample["acceleration_by_frame"].get(frame_index)
    relative_velocity = None
    if velocity and all(component is not None for component in ego["velocity"]):
        world_relative = (
            velocity[0] - ego["velocity"][0],
            velocity[1] - ego["velocity"][1],
        )
        cosine = math.cos(ego["yaw"])
        sine = math.sin(ego["yaw"])
        relative_velocity = (
            cosine * world_relative[0] + sine * world_relative[1],
            -sine * world_relative[0] + cosine * world_relative[1],
        )

    return {
        "object_id": object_id,
        "class": obj.get("className"),
        "subclass": obj.get("subclassName"),
        "annotation_type": obj.get("type"),
        "geometry_source": geometry_source,
        "position_lcs_m": [round_or_none(value) for value in position],
        "position_ego_m": {
            "longitudinal": round_or_none(rel_x),
            "lateral": round_or_none(rel_y),
            "distance": round_or_none(math.hypot(rel_x, rel_y)),
            "bearing_deg": round_or_none(math.degrees(math.atan2(rel_y, rel_x)), 2),
        },
        "dimensions_m": {
            "length": round_or_none(bbox["length"]),
            "width": round_or_none(bbox["width"]),
            "height": round_or_none(bbox["height"]),
        },
        "heading_relative_rad": round_or_none(wrap_angle(bbox_yaw(bbox) - ego["yaw"])),
        "velocity_lcs_mps": (
            [round_or_none(value) for value in velocity] if velocity else None
        ),
        "acceleration_lcs_mps2": (
            [round_or_none(value) for value in acceleration] if acceleration else None
        ),
        "relative_velocity_ego_mps": (
            {
                "longitudinal": round_or_none(relative_velocity[0]),
                "lateral": round_or_none(relative_velocity[1]),
            }
            if relative_velocity
            else None
        ),
        "static_attributes": attributes_as_object(obj.get("staticAttributes")),
        "dynamic_attributes": attributes_as_object(obj.get("dynamicAttributes")),
    }


def constant_velocity_interaction(state):
    relative_velocity = state["relative_velocity_ego_mps"]
    if relative_velocity is None:
        return {
            "closing_speed_mps": None,
            "ttc_constant_velocity_s": None,
            "time_to_closest_approach_s": None,
            "predicted_min_distance_m": None,
        }

    x = state["position_ego_m"]["longitudinal"]
    y = state["position_ego_m"]["lateral"]
    vx = relative_velocity["longitudinal"]
    vy = relative_velocity["lateral"]
    distance = math.hypot(x, y)
    radial_rate = (x * vx + y * vy) / distance if distance > 1e-6 else 0.0
    closing_speed = max(0.0, -radial_rate)
    velocity_squared = vx * vx + vy * vy
    time_to_closest = (
        max(0.0, min(INTERACTION_HORIZON_S, -(x * vx + y * vy) / velocity_squared))
        if velocity_squared > 1e-6
        else 0.0
    )
    closest_x = x + vx * time_to_closest
    closest_y = y + vy * time_to_closest
    min_distance = math.hypot(closest_x, closest_y)
    ttc = distance / closing_speed if closing_speed > 1e-3 else None
    if ttc is not None and ttc > INTERACTION_HORIZON_S:
        ttc = None
    return {
        "closing_speed_mps": round_or_none(closing_speed),
        "ttc_constant_velocity_s": round_or_none(ttc),
        "time_to_closest_approach_s": round_or_none(time_to_closest),
        "predicted_min_distance_m": round_or_none(min_distance),
    }


def observed_future_path_overlap(state, frame_index, trajectory, sample, timestamps):
    """Compare future observed object boxes with future ego footprints.

    This is an offline annotation feature and intentionally uses future frames.
    It is not an online collision prediction.
    """
    object_id = state["object_id"]
    max_timestamp = timestamps[frame_index] + INTERACTION_HORIZON_S
    first_overlap = None
    minimum_center_distance = None
    minimum_frame = None

    for future_frame, bbox in sample["bbox_by_frame"].items():
        if future_frame < frame_index or timestamps[future_frame] > max_timestamp:
            continue
        future_ego = trajectory[future_frame]
        rel_x, rel_y = ego_relative(
            (bbox["x"], bbox["y"], bbox["z"]),
            future_ego["position"],
            future_ego["yaw"],
        )
        center_distance = math.hypot(rel_x, rel_y)
        if minimum_center_distance is None or center_distance < minimum_center_distance:
            minimum_center_distance = center_distance
            minimum_frame = future_frame

        longitudinal_limit = EGO_LENGTH_M / 2.0 + bbox["length"] / 2.0 + PATH_MARGIN_M
        lateral_limit = EGO_WIDTH_M / 2.0 + bbox["width"] / 2.0 + PATH_MARGIN_M
        if (
            first_overlap is None
            and abs(rel_x) <= longitudinal_limit
            and abs(rel_y) <= lateral_limit
        ):
            first_overlap = future_frame

    return {
        "object_id": object_id,
        "horizon_s": INTERACTION_HORIZON_S,
        "observed_future_path_overlap": first_overlap is not None,
        "first_overlap_frame": first_overlap,
        "first_overlap_time_s": (
            round_or_none(timestamps[first_overlap] - timestamps[frame_index])
            if first_overlap is not None
            else None
        ),
        "minimum_observed_center_distance_m": round_or_none(minimum_center_distance),
        "minimum_distance_frame": minimum_frame,
        "method": "future observed object bbox vs future ego footprint",
    }


def scenario_signals(ego, object_states):
    nearby = [
        state
        for state in object_states
        if state["position_ego_m"]["distance"] is not None
        and state["position_ego_m"]["distance"] <= 30.0
    ]
    lead_candidates = [
        state
        for state in object_states
        if state["class"] in LEAD_CLASSES
        and 0.0 < state["position_ego_m"]["longitudinal"] <= 80.0
        and abs(state["position_ego_m"]["lateral"]) <= 3.5
    ]
    lead_candidates.sort(key=lambda state: state["position_ego_m"]["longitudinal"])
    lead = lead_candidates[0] if lead_candidates else None

    return {
        "ego_speed_mps": round_or_none(ego["speed"]),
        "ego_acceleration_mps2": round_or_none(ego["acceleration"]),
        "ego_yaw_rate_radps": round_or_none(ego["yaw_rate"]),
        "lead_candidate": (
            {
                "object_id": lead["object_id"],
                "class": lead["class"],
                "longitudinal_distance_m": lead["position_ego_m"]["longitudinal"],
                "lateral_offset_m": lead["position_ego_m"]["lateral"],
            }
            if lead
            else None
        ),
        "nearby_30m_counts": {
            "pedestrian": sum(state["class"] == "pedestrian" for state in nearby),
            "motorcycle": sum(state["class"] == "motorcycle" for state in nearby),
            "all_motional": sum(state["class"] in MOTIONAL_CLASSES for state in nearby),
        },
        "heuristic_parameters": {
            "nearby_radius_m": 30.0,
            "lead_longitudinal_range_m": [0.0, 80.0],
            "lead_lateral_abs_max_m": 3.5,
        },
    }


def build_recording(source_root, output_root, recording):
    recording_dir = source_root / recording
    annotation_path = recording_dir / "annotations.json"
    trajectory_path = recording_dir / "traj_lcs.txt"
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    trajectory = parse_trajectory(trajectory_path)

    frame_count = annotations["scene"]["frameCount"]
    if frame_count != len(trajectory):
        raise ValueError(
            f"{recording}: annotation frameCount={frame_count}, trajectory rows={len(trajectory)}"
        )

    timestamps = [row["timestamp"] for row in trajectory]
    samples = build_object_samples(annotations["objects"], timestamps)
    visible_at_frame = defaultdict(list)
    for obj in annotations["objects"]:
        frame_indices = set(obj.get("visible_frames", []))
        frame_indices.update(samples[str(obj["objectId"])]["bbox_by_frame"])
        for frame_index in frame_indices:
            if 0 <= frame_index < frame_count:
                visible_at_frame[frame_index].append(obj)

    frames = []
    missing_geometry_count = 0
    for frame_index, ego in enumerate(trajectory):
        object_states = []
        for obj in visible_at_frame[frame_index]:
            state = make_object_state(obj, frame_index, ego, samples)
            if state is None:
                missing_geometry_count += 1
            else:
                object_states.append(state)
        object_states.sort(
            key=lambda state: (
                state["position_ego_m"]["distance"],
                state["class"],
                state["object_id"],
            )
        )
        interaction_candidates = []
        for state in object_states:
            if state["class"] not in MOTIONAL_CLASSES:
                continue
            current_metrics = constant_velocity_interaction(state)
            future_metrics = observed_future_path_overlap(
                state,
                frame_index,
                trajectory,
                samples[state["object_id"]],
                timestamps,
            )
            if (
                state["position_ego_m"]["distance"] <= 100.0
                or future_metrics["observed_future_path_overlap"]
            ):
                interaction_candidates.append(
                    {
                        "object_id": state["object_id"],
                        "class": state["class"],
                        **current_metrics,
                        **future_metrics,
                    }
                )

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_unix_s": round(ego["timestamp"], 6),
                "time_since_start_s": round(ego["timestamp"] - timestamps[0], 4),
                "ego": {
                    "position_lcs_m": [round_or_none(value) for value in ego["position"]],
                    "orientation_lcs_quaternion_xyzw": [
                        round_or_none(value) for value in ego["quaternion"]
                    ],
                    "heading_lcs_rad": round_or_none(ego["yaw"]),
                    "velocity_lcs_mps": [
                        round_or_none(value) for value in ego["velocity"]
                    ],
                    "speed_mps": round_or_none(ego["speed"]),
                    "acceleration_mps2": round_or_none(ego["acceleration"]),
                    "yaw_rate_radps": round_or_none(ego["yaw_rate"]),
                },
                "objects": object_states,
                "scenario_signals": scenario_signals(ego, object_states),
                "interaction_candidates": interaction_candidates,
            }
        )

    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    output = {
        "schema_version": "od-trajectory-canonical-frame-v1",
        "recording_id": recording,
        "source": {
            "annotations": str(annotation_path.resolve()),
            "trajectory": str(trajectory_path.resolve()),
            "coordinate_system": "LCS",
            "frame_alignment": "annotation frameIndex maps directly to trajectory row index",
        },
        "recording": {
            "frame_count": frame_count,
            "start_timestamp_unix_s": round(timestamps[0], 6),
            "end_timestamp_unix_s": round(timestamps[-1], 6),
            "duration_s": round(timestamps[-1] - timestamps[0], 4),
            "median_frame_interval_s": round(median(deltas), 6),
            "nominal_frame_rate_hz": round(1.0 / median(deltas), 4),
        },
        "scenario_taxonomy": SCENARIO_TAXONOMY,
        "data_quality": {
            "trajectory_rows_match_annotation_frames": True,
            "object_states_without_usable_geometry": missing_geometry_count,
            "notes": [
                "No object state is forward-filled.",
                "Dynamic geometry uses exact per-frame bbox3d only.",
                "Static object-level bbox3d is used only on listed visible frames.",
                "Lead detection is a geometric candidate, not a lane-topology assertion.",
                "Object derivatives are omitted across observation gaps over 0.25 s.",
                "TTC uses current relative position and constant relative velocity.",
                "Observed future path overlap uses future annotation frames and is offline evidence.",
            ],
        },
        "frames": frames,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{recording}_canonical_frames.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, separators=(",", ":"))
    return output_path, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("MOTIONAL_DATA_ROOT", DEFAULT_DATA_ROOT)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("recordings", nargs="*", default=DEFAULT_RECORDINGS)
    args = parser.parse_args()

    manifest = {
        "schema_version": "od-trajectory-canonical-manifest-v1",
        "scenario_taxonomy": SCENARIO_TAXONOMY,
        "recordings": [],
    }
    for recording in args.recordings:
        path, output = build_recording(args.source_root, args.output_root, recording)
        manifest["recordings"].append(
            {
                "recording_id": recording,
                "path": path.name,
                **output["recording"],
                **output["data_quality"],
            }
        )
        print(f"Wrote {path}")

    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
