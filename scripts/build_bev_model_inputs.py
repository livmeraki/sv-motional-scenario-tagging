#!/usr/bin/env python3
"""Build model-ready JSON and BEV keyframe images from motional windows.

Default output is model-facing only: one `refined.json` plus three PNG BEV
keyframes per window. Debug SVG generation is intentionally not part of the
normal pipeline because it adds clutter and can leak formula/candidate text into
model inputs.
"""

from __future__ import annotations

import argparse
import struct
import zlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_DIR = Path("quick_exploration_outputs/motional_windows")
DEFAULT_OUTPUT_DIR = Path("quick_exploration_outputs/scenario_tagging_pipeline/03_model_inputs")

SCHEMA_VERSION = "od-motional-model-input-v2"
BEV_SCHEMA_VERSION = "od-bev-keyframe-v2"
DEPRECATED_REFINED_FIELDS = {
    "per_frame_counts[].object_count": "Use total_object_count.",
    "per_frame_counts[].classes": "Use classes_total.",
}

CLASS_COLORS = {
    "car": "#2f6fed",
    "truck": "#7c3aed",
    "truck_head": "#6d28d9",
    "bus": "#8b5cf6",
    "trailer": "#a855f7",
    "special_vehicle": "#9333ea",
    "motorcycle": "#4f9a38",
    "bicycle": "#0d9488",
    "rider_other": "#14b8a6",
    "pedestrian": "#e11d48",
    "traffic_sign": "#10b981",
    "traffic_sign_chevron": "#f97316",
    "traffic_light_car": "#0891b2",
    "traffic_light_ped": "#2563eb",
    "barrier_solid": "#a16207",
    "barrier_fence": "#92400e",
    "barrier_rail": "#b45309",
    "barrier_temporary": "#d97706",
    "obstacle_pole": "#ef4444",
    "obstacle_cylinder": "#dc2626",
    "obstacle_bollard": "#f43f5e",
    "obstacle_beacon": "#fb923c",
    "obstacle_cone": "#f97316",
    "obstacle_etc": "#475569",
}

VEHICLE_CLASSES = {
    "car",
    "truck",
    "truck_head",
    "bus",
    "trailer",
    "special_vehicle",
    "motorcycle",
}


def r4(value):
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 4)
    if isinstance(value, list):
        return [r4(v) for v in value]
    if isinstance(value, dict):
        return {k: r4(v) for k, v in value.items()}
    return value


def safe_name(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def portable_path(path):
    if path is None:
        return None
    return str(path).replace("\\", "/")


def ego_to_screen(point, center_x, center_y, scale):
    """Map ego coordinates to SVG pixels.

    Ego coordinates use longitudinal positive forward and lateral positive
    left. The SVG shows forward upward and left on the left side of the image.
    """
    longitudinal, lateral = point
    return center_x - lateral * scale, center_y - longitudinal * scale


def lcs_to_ego(point_lcs, ego_lcs, ego_heading):
    dx = point_lcs[0] - ego_lcs[0]
    dy = point_lcs[1] - ego_lcs[1]
    c = math.cos(ego_heading)
    s = math.sin(ego_heading)
    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy
    return longitudinal, lateral


def object_corners(obj):
    pos = obj.get("position_ego_m") or {}
    dims = obj.get("dimensions_m") or {}
    longitudinal = pos.get("longitudinal")
    lateral = pos.get("lateral")
    length = dims.get("length")
    width = dims.get("width")
    heading = obj.get("heading_relative_rad") or 0.0
    if None in (longitudinal, lateral, length, width):
        return None
    half_l = max(float(length), 0.5) / 2.0
    half_w = max(float(width), 0.5) / 2.0
    c = math.cos(heading)
    s = math.sin(heading)
    corners = []
    for forward, left in (
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ):
        rot_forward = c * forward - s * left
        rot_left = s * forward + c * left
        corners.append((longitudinal + rot_forward, lateral + rot_left))
    return corners


def polygon_points(points, center_x, center_y, scale):
    return " ".join(
        f"{ego_to_screen(point, center_x, center_y, scale)[0]:.1f},"
        f"{ego_to_screen(point, center_x, center_y, scale)[1]:.1f}"
        for point in points
    )


def svg_polyline(points, center_x, center_y, scale):
    return " ".join(
        f"{ego_to_screen(point, center_x, center_y, scale)[0]:.1f},"
        f"{ego_to_screen(point, center_x, center_y, scale)[1]:.1f}"
        for point in points
    )


def class_color(class_name):
    return CLASS_COLORS.get(class_name, "#64748b")


def ego_heading(ego):
    return (
        ego.get("heading_lcs_rad")
        if ego.get("heading_lcs_rad") is not None
        else ego.get("heading_rad", 0.0)
    )


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def blend_rgb(dst, src, alpha):
    return tuple(round(dst[i] * (1.0 - alpha) + src[i] * alpha) for i in range(3))


def frame_by_index(window):
    return {frame["frame_index"]: frame for frame in window["frames"]}


def choose_keyframes(window):
    frames = window["frames"]
    frames_with_objects = [frame for frame in frames if frame.get("objects")]
    if frames_with_objects:
        middle_target = frames[len(frames) // 2]["frame_index"]
        middle_frame = min(
            frames_with_objects,
            key=lambda frame: abs(frame["frame_index"] - middle_target),
        )
        return {
            "start": frames_with_objects[0]["frame_index"],
            "middle": middle_frame["frame_index"],
            "end": frames_with_objects[-1]["frame_index"],
        }
    return {
        "start": frames[0]["frame_index"],
        "middle": frames[len(frames) // 2]["frame_index"],
        "end": frames[-1]["frame_index"],
    }


def lead_ids_in_window(window):
    ids = set()
    for frame in window["frames"]:
        lead = frame.get("scenario_signals", {}).get("lead_candidate")
        if lead and lead.get("object_id") is not None:
            ids.add(str(lead["object_id"]))
    return ids


def interaction_index(window):
    by_frame_object = {}
    important_ids = set()
    for frame in window["frames"]:
        frame_idx = frame["frame_index"]
        for candidate in frame.get("interaction_candidates", []):
            object_id = str(candidate.get("object_id"))
            by_frame_object[(frame_idx, object_id)] = candidate
            predicted_min_distance = candidate.get("predicted_min_distance_m")
            if predicted_min_distance is None:
                predicted_min_distance = 9999.0
            if (
                candidate.get("ttc_constant_velocity_s") is not None
                or candidate.get("observed_future_path_overlap")
                or predicted_min_distance <= 8.0
            ):
                important_ids.add(object_id)
    return by_frame_object, important_ids


def build_object_tracks(window):
    tracks = defaultdict(list)
    for frame in window["frames"]:
        for obj in frame.get("objects", []):
            tracks[str(obj["object_id"])].append((frame, obj))
    return tracks


def instantaneous_track_vector_ego(tracks, object_id, frame_idx, key_heading):
    """Return object instantaneous motion in keyframe ego axes.

    The vector is derived from adjacent object positions in LCS, then rotated
    into the keyframe ego orientation. It represents observed object motion,
    not ego-relative apparent motion.
    """
    states = tracks.get(str(object_id), [])
    current_index = None
    for idx, (frame, obj) in enumerate(states):
        if frame["frame_index"] == frame_idx and obj.get("position_lcs_m"):
            current_index = idx
            break
    if current_index is None:
        return None

    prev_state = None
    next_state = None
    for idx in range(current_index - 1, -1, -1):
        if states[idx][1].get("position_lcs_m"):
            prev_state = states[idx]
            break
    for idx in range(current_index + 1, len(states)):
        if states[idx][1].get("position_lcs_m"):
            next_state = states[idx]
            break

    if prev_state and next_state:
        start_frame, start_obj = prev_state
        end_frame, end_obj = next_state
    elif prev_state:
        start_frame, start_obj = prev_state
        end_frame, end_obj = states[current_index]
    elif next_state:
        start_frame, start_obj = states[current_index]
        end_frame, end_obj = next_state
    else:
        return None

    dt = end_frame["time_since_start_s"] - start_frame["time_since_start_s"]
    if dt <= 0 or dt > 1.0:
        return None
    start_pos = start_obj.get("position_lcs_m")
    end_pos = end_obj.get("position_lcs_m")
    if not start_pos or not end_pos:
        return None
    dx = (end_pos[0] - start_pos[0]) / dt
    dy = (end_pos[1] - start_pos[1]) / dt
    c = math.cos(key_heading)
    s = math.sin(key_heading)
    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy
    speed = math.hypot(longitudinal, lateral)
    if speed < 0.25:
        return None
    return {"longitudinal": longitudinal, "lateral": lateral, "speed": speed}


def arrow_endpoint_from_motion(point, motion, min_length=4.0, max_length=14.0):
    speed = motion["speed"]
    length = min(max(speed * 0.75, min_length), max_length)
    return (
        point[0] + motion["longitudinal"] / speed * length,
        point[1] + motion["lateral"] / speed * length,
    )


def select_relevant_objects(window, max_objects):
    tracks = build_object_tracks(window)
    lead_ids = lead_ids_in_window(window)
    interaction_by_frame_object, interaction_ids = interaction_index(window)
    selected = set(lead_ids) | set(interaction_ids)
    ranked = []
    for object_id, states in tracks.items():
        min_distance = min(
            (
                state.get("position_ego_m", {}).get("distance", 9999)
                for _, state in states
            ),
            default=9999,
        )
        class_name = states[0][1].get("class")
        near_pedestrian = class_name == "pedestrian" and min_distance <= 25.0
        near_motorcycle = class_name == "motorcycle" and min_distance <= 30.0
        if near_pedestrian or near_motorcycle:
            selected.add(object_id)
        ranked.append((min_distance, object_id))
    for _, object_id in sorted(ranked):
        if len(selected) >= max_objects:
            break
        selected.add(object_id)
    return selected, tracks, interaction_by_frame_object


def compact_track(window, object_id, states, interaction_by_frame_object):
    sample_every = 5
    keyframes = set(choose_keyframes(window).values())
    samples = []
    for idx, (frame, obj) in enumerate(states):
        if idx % sample_every != 0 and frame["frame_index"] not in keyframes:
            continue
        pos = obj.get("position_ego_m") or {}
        interaction = interaction_by_frame_object.get((frame["frame_index"], object_id))
        samples.append(
            {
                "frame_index": frame["frame_index"],
                "time_s": r4(frame["time_since_start_s"]),
                "longitudinal_m": r4(pos.get("longitudinal")),
                "lateral_m": r4(pos.get("lateral")),
                "distance_m": r4(pos.get("distance")),
                "bearing_deg": r4(pos.get("bearing_deg")),
                "relative_velocity_ego_mps": r4(obj.get("relative_velocity_ego_mps")),
                "ttc_s": r4(interaction.get("ttc_constant_velocity_s")) if interaction else None,
                "path_overlap": bool(interaction.get("observed_future_path_overlap")) if interaction else False,
            }
        )
    first_obj = states[0][1]
    distances = [
        state.get("position_ego_m", {}).get("distance")
        for _, state in states
        if state.get("position_ego_m", {}).get("distance") is not None
    ]
    return {
        "object_id": object_id,
        "class": first_obj.get("class"),
        "subclass": first_obj.get("subclass"),
        "annotation_type": first_obj.get("annotation_type"),
        "geometry_source": first_obj.get("geometry_source"),
        "observed_frame_count": len(states),
        "first_frame": states[0][0]["frame_index"],
        "last_frame": states[-1][0]["frame_index"],
        "minimum_distance_m": r4(min(distances) if distances else None),
        "dimensions_m": r4(first_obj.get("dimensions_m")),
        "samples": samples,
    }


def compact_ego_series(window):
    sample_every = 5
    keyframes = set(choose_keyframes(window).values())
    series = []
    for idx, frame in enumerate(window["frames"]):
        if idx % sample_every != 0 and frame["frame_index"] not in keyframes:
            continue
        ego = frame["ego"]
        series.append(
            {
                "frame_index": frame["frame_index"],
                "time_s": r4(frame["time_since_start_s"]),
                "speed_mps": r4(ego.get("speed_mps")),
                "acceleration_mps2": r4(ego.get("acceleration_mps2")),
                "yaw_rate_radps": r4(ego.get("yaw_rate_radps")),
                "heading_lcs_rad": r4(ego_heading(ego)),
            }
        )
    return series


def count_objects_by_type(objects):
    dynamic_classes = defaultdict(int)
    static_classes = defaultdict(int)
    total_classes = defaultdict(int)
    dynamic_count = 0
    static_visible_count = 0
    for obj in objects:
        class_name = obj.get("class", "unknown")
        total_classes[class_name] += 1
        if obj.get("annotation_type") == "static":
            static_visible_count += 1
            static_classes[class_name] += 1
        else:
            dynamic_count += 1
            dynamic_classes[class_name] += 1
    return {
        "dynamic_object_count": dynamic_count,
        "static_visible_object_count": static_visible_count,
        "total_object_count": len(objects),
        "classes_dynamic": dict(sorted(dynamic_classes.items())),
        "classes_static": dict(sorted(static_classes.items())),
        "classes_total": dict(sorted(total_classes.items())),
    }


def nearby_count(frame, class_name):
    counts = frame.get("scenario_signals", {}).get("nearby_30m_counts")
    if counts is None:
        return None
    return counts.get(class_name)


def refined_data_quality(window, per_frame_counts):
    missing_object_frames = [
        frame["frame_index"]
        for frame, counts in zip(window["frames"], per_frame_counts)
        if counts["total_object_count"] == 0
    ]
    static_counts = [counts["static_visible_object_count"] for counts in per_frame_counts]
    sorted_static = sorted(static_counts)
    median_static = sorted_static[len(sorted_static) // 2] if sorted_static else 0
    spike_threshold = max(20, median_static * 3)
    frames_with_static_snapshot_spike = [
        counts["frame_index"]
        for counts in per_frame_counts
        if counts["static_visible_object_count"] >= spike_threshold
        and counts["static_visible_object_count"] > counts["dynamic_object_count"]
    ]
    nearby_count_source_available = all(
        "nearby_30m_counts" in frame.get("scenario_signals", {})
        for frame in window["frames"]
    )
    warnings = []
    if missing_object_frames:
        warnings.append("Some frames contain no object states; this can be normal before OD annotations begin.")
    if frames_with_static_snapshot_spike:
        warnings.append("Some frames contain high static-visible object counts; review for static snapshot clutter.")
    if not nearby_count_source_available:
        warnings.append("At least one frame lacks scenario_signals.nearby_30m_counts.")
    return {
        "missing_object_frames": missing_object_frames,
        "frames_with_static_snapshot_spike": frames_with_static_snapshot_spike,
        "nearby_count_source_available": nearby_count_source_available,
        "warnings": warnings,
    }


def build_refined_json(
    recording,
    window,
    keyframe_files,
    max_objects,
    include_preliminary_candidates=True,
):
    selected, tracks, interaction_by_frame_object = select_relevant_objects(
        window, max_objects=max_objects
    )
    objects = [
        compact_track(window, object_id, tracks[object_id], interaction_by_frame_object)
        for object_id in sorted(
            selected,
            key=lambda oid: min(
                state.get("position_ego_m", {}).get("distance", 9999)
                for _, state in tracks[oid]
            ),
        )
    ]
    per_frame_counts = []
    for frame in window["frames"]:
        objects_in_frame = frame.get("objects", [])
        counts = count_objects_by_type(objects_in_frame)
        lead_candidate = frame.get("scenario_signals", {}).get("lead_candidate")
        per_frame_counts.append({
            "frame_index": frame["frame_index"],
            **counts,
            "object_count": counts["total_object_count"],
            "classes": counts["classes_total"],
            "lead_object_id": lead_candidate.get("object_id") if lead_candidate else None,
            "nearby_pedestrian_count_30m": nearby_count(frame, "pedestrian"),
            "nearby_motorcycle_count_30m": nearby_count(frame, "motorcycle"),
        })
    refined = {
        "schema_version": SCHEMA_VERSION,
        "schema_metadata": {
            "migration_from": "od-motional-model-input-v1",
            "migration_notes": [
                "scenario_signals.nearby_30m_counts is now the single source for nearby pedestrian/motorcycle counts.",
                "Valid zero nearby counts are preserved as 0; null means unavailable/not computed.",
                "per_frame_counts now separates dynamic_object_count, static_visible_object_count, total_object_count, classes_dynamic, classes_static, and classes_total.",
                "object_count and classes are retained for backward compatibility and deprecated.",
                "data_quality was added for missing object frames, static snapshot spikes, nearby count source availability, and warnings.",
                "source_window_file is normalized to a portable POSIX-style path.",
            ],
            "deprecated_fields": DEPRECATED_REFINED_FIELDS,
        },
        "recording_id": recording["recording_id"],
        "source_window_id": window["window_id"],
        "source_window_file": portable_path(recording.get("_source_file")),
        "time_window": {
            "start_frame": window["start_frame"],
            "end_frame": window["end_frame"],
            "start_time_s": r4(window["start_time_s"]),
            "end_time_s": r4(window["end_time_s"]),
            "duration_s": r4(window["duration_s"]),
        },
        "bev_keyframes": keyframe_files,
        "taxonomy": recording.get("scenario_taxonomy"),
        "ego_summary": r4(window.get("ego_summary")),
        "ego_series_sampled": compact_ego_series(window),
        "per_frame_counts": per_frame_counts,
        "relevant_objects": objects,
        "data_quality": refined_data_quality(window, per_frame_counts),
        "data_notes": [
            "Object geometry is not forward-filled.",
            "Dynamic objects use exact per-frame bbox3d only.",
            "Static object-level bbox3d is used only on frames marked visible.",
            "BEV keyframes are ego-centric at the selected frame; trails use LCS positions transformed into that keyframe ego pose.",
            "Preliminary candidates are deterministic evidence hints, not final labels.",
            "Deprecated per-frame fields object_count/classes mirror total_object_count/classes_total for backward compatibility.",
        ],
    }
    if include_preliminary_candidates:
        refined["preliminary_candidates"] = window.get("preliminary_candidates")
    return refined


class PngCanvas:
    def __init__(self, width, height, bg=(248, 250, 252)):
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * (width * height))

    def _inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def set_pixel(self, x, y, color, alpha=1.0):
        x = int(round(x))
        y = int(round(y))
        if not self._inside(x, y):
            return
        idx = (y * self.width + x) * 3
        if alpha >= 1:
            self.pixels[idx : idx + 3] = bytes(color)
        else:
            dst = tuple(self.pixels[idx + i] for i in range(3))
            self.pixels[idx : idx + 3] = bytes(blend_rgb(dst, color, alpha))

    def line(self, x1, y1, x2, y2, color, width=1, alpha=1.0):
        dx = x2 - x1
        dy = y2 - y1
        steps = max(1, int(max(abs(dx), abs(dy))))
        radius = max(0, int(width // 2))
        for step in range(steps + 1):
            t = step / steps
            x = x1 + dx * t
            y = y1 + dy * t
            for yy in range(int(y) - radius, int(y) + radius + 1):
                for xx in range(int(x) - radius, int(x) + radius + 1):
                    if (xx - x) ** 2 + (yy - y) ** 2 <= (radius + 0.6) ** 2:
                        self.set_pixel(xx, yy, color, alpha)

    def polyline(self, points, color, width=1, alpha=1.0):
        for a, b in zip(points, points[1:]):
            self.line(a[0], a[1], b[0], b[1], color, width=width, alpha=alpha)

    def circle(self, cx, cy, radius, color, alpha=1.0):
        r = int(math.ceil(radius))
        for y in range(int(cy) - r, int(cy) + r + 1):
            for x in range(int(cx) - r, int(cx) + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                    self.set_pixel(x, y, color, alpha)

    def polygon(self, points, fill, outline=None, alpha=1.0, outline_width=1):
        if not points:
            return
        ys = [p[1] for p in points]
        min_y = max(0, int(math.floor(min(ys))))
        max_y = min(self.height - 1, int(math.ceil(max(ys))))
        for y in range(min_y, max_y + 1):
            intersections = []
            for i, p1 in enumerate(points):
                p2 = points[(i + 1) % len(points)]
                y1, y2 = p1[1], p2[1]
                if (y1 <= y < y2) or (y2 <= y < y1):
                    x = p1[0] + (y - y1) * (p2[0] - p1[0]) / (y2 - y1)
                    intersections.append(x)
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 >= len(intersections):
                    break
                start = max(0, int(math.ceil(intersections[i])))
                end = min(self.width - 1, int(math.floor(intersections[i + 1])))
                for x in range(start, end + 1):
                    self.set_pixel(x, y, fill, alpha)
        if outline:
            for p1, p2 in zip(points, points[1:] + points[:1]):
                self.line(p1[0], p1[1], p2[0], p2[1], outline, width=outline_width)

    def save_png(self, path):
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])

        def chunk(tag, data):
            body = tag + data
            return (
                struct.pack(">I", len(data))
                + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
            )

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
            )
        )
        png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(png)


def render_bev_svg(recording, window, frame_idx, label, output_path, extent, size):
    frames = frame_by_index(window)
    frame = frames[frame_idx]
    key_ego = frame["ego"]
    key_pos = key_ego["position_lcs_m"]
    key_heading = ego_heading(key_ego)
    width, height = size
    left_m, right_m, back_m, forward_m = extent
    scale = min(width / (left_m + right_m), height / (back_m + forward_m))
    center_x = left_m * scale
    center_y = forward_m * scale
    center_x += (width - (left_m + right_m) * scale) / 2.0
    center_y += (height - (back_m + forward_m) * scale) / 2.0
    selected, tracks, interaction_by_frame_object = select_relevant_objects(
        window, max_objects=80
    )
    lead_ids = lead_ids_in_window(window)
    current_objects = {str(obj["object_id"]): obj for obj in frame.get("objects", [])}

    def visible(point):
        longitudinal, lateral = point
        return -back_m <= longitudinal <= forward_m and -right_m <= lateral <= left_m

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    title = f"{recording['recording_id']} {window['window_id']} {label} frame {frame_idx}"
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append('<rect width="100%" height="100%" fill="#f8fafc"/>')

    grid_color = "#dbe4ee"
    for lateral in range(-int(right_m), int(left_m) + 1, 10):
        x, _ = ego_to_screen((0, lateral), center_x, center_y, scale)
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
            f'stroke="{grid_color}" stroke-width="1"/>'
        )
    for longitudinal in range(-int(back_m), int(forward_m) + 1, 10):
        _, y = ego_to_screen((longitudinal, 0), center_x, center_y, scale)
        parts.append(
            f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" '
            f'stroke="{grid_color}" stroke-width="1"/>'
        )
    x0, y0 = ego_to_screen((0, 0), center_x, center_y, scale)
    parts.append(
        f'<line x1="0" y1="{y0:.1f}" x2="{width}" y2="{y0:.1f}" '
        'stroke="#94a3b8" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{x0:.1f}" y1="0" x2="{x0:.1f}" y2="{height}" '
        'stroke="#94a3b8" stroke-width="1.5"/>'
    )

    ego_path_points = []
    elapsed_path_points = []
    for path_frame in window["frames"]:
        ego_pos = path_frame["ego"]["position_lcs_m"]
        point = lcs_to_ego(ego_pos, key_pos, key_heading)
        if visible(point):
            ego_path_points.append(point)
            if path_frame["frame_index"] <= frame_idx:
                elapsed_path_points.append(point)
    if len(ego_path_points) >= 2:
        parts.append(
            f'<polyline points="{svg_polyline(ego_path_points, center_x, center_y, scale)}" '
            'fill="none" stroke="#94a3b8" stroke-width="4" opacity="0.65"/>'
        )
    if len(elapsed_path_points) >= 2:
        parts.append(
            f'<polyline points="{svg_polyline(elapsed_path_points, center_x, center_y, scale)}" '
            'fill="none" stroke="#2563eb" stroke-width="5" opacity="0.95"/>'
        )

    for object_id in selected:
        states = tracks.get(object_id, [])
        trail = []
        for trail_frame, obj in states:
            if trail_frame["frame_index"] > frame_idx:
                continue
            pos_lcs = obj.get("position_lcs_m")
            if not pos_lcs:
                continue
            point = lcs_to_ego(pos_lcs, key_pos, key_heading)
            if visible(point):
                trail.append(point)
        if len(trail) >= 2:
            class_name = states[0][1].get("class", "unknown") if states else "unknown"
            parts.append(
                f'<polyline points="{svg_polyline(trail, center_x, center_y, scale)}" '
                f'fill="none" stroke="{class_color(class_name)}" '
                'stroke-width="2" opacity="0.35"/>'
            )

    for object_id, obj in current_objects.items():
        pos = obj.get("position_ego_m") or {}
        point = (pos.get("longitudinal"), pos.get("lateral"))
        if None in point or not visible(point):
            continue
        class_name = obj.get("class", "unknown")
        color = class_color(class_name)
        interaction = interaction_by_frame_object.get((frame_idx, object_id))
        important = object_id in lead_ids or (
            interaction
            and (
                interaction.get("observed_future_path_overlap")
                or interaction.get("ttc_constant_velocity_s") is not None
            )
        )
        corners = object_corners(obj)
        if corners:
            stroke_width = 3 if important else 1.5
            fill_opacity = 0.22 if important else 0.12
            parts.append(
                f'<polygon points="{polygon_points(corners, center_x, center_y, scale)}" '
                f'fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" '
                f'stroke-width="{stroke_width}"/>'
            )
        sx, sy = ego_to_screen(point, center_x, center_y, scale)
        radius = 5 if important else 3.5
        parts.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius}" fill="{color}" '
            'stroke="#0f172a" stroke-width="0.8"/>'
        )
        if important or class_name in {"pedestrian", "motorcycle"}:
            label_text = object_id
            if object_id in lead_ids:
                label_text = f"L {object_id}"
            parts.append(
                f'<text x="{sx + 6:.1f}" y="{sy - 6:.1f}" font-family="Arial, sans-serif" '
                'font-size="11" fill="#0f172a" paint-order="stroke" '
                f'stroke="#fff" stroke-width="3">{html.escape(label_text)}</text>'
            )
        motion = instantaneous_track_vector_ego(tracks, object_id, frame_idx, key_heading)
        if motion:
            end = arrow_endpoint_from_motion(point, motion)
            ex, ey = ego_to_screen(end, center_x, center_y, scale)
            parts.append(
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                f'stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    ego_corners = [(2.4, 1.0), (2.4, -1.0), (-2.4, -1.0), (-2.4, 1.0)]
    parts.append(
        f'<polygon points="{polygon_points(ego_corners, center_x, center_y, scale)}" '
        'fill="#22c55e" fill-opacity="0.28" stroke="#166534" stroke-width="3"/>'
    )
    nose = [(3.0, 0.0), (1.6, 0.7), (1.6, -0.7)]
    parts.append(
        f'<polygon points="{polygon_points(nose, center_x, center_y, scale)}" '
        'fill="#166534" opacity="0.85"/>'
    )

    parts.insert(
        2,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
        'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="context-stroke"/>'
        "</marker></defs>",
    )
    candidate_tags = ", ".join(window.get("preliminary_candidates", {}).get("candidate_tags", []))
    info = (
        f"{recording['recording_id']} | {window['window_id']} | {label} | "
        f"frame {frame_idx} | t={frame['time_since_start_s']:.2f}s | "
        f"speed={key_ego.get('speed_mps', 0):.2f} m/s"
    )
    parts.append(
        '<rect x="12" y="12" width="620" height="58" rx="6" fill="#ffffff" '
        'fill-opacity="0.88" stroke="#cbd5e1"/>'
    )
    parts.append(
        f'<text x="24" y="36" font-family="Arial, sans-serif" font-size="16" '
        f'font-weight="700" fill="#0f172a">{html.escape(info)}</text>'
    )
    parts.append(
        f'<text x="24" y="58" font-family="Arial, sans-serif" font-size="12" '
        f'fill="#334155">candidates: {html.escape(candidate_tags or "none")}</text>'
    )
    legend_y = 92
    parts.append(
        f'<text x="24" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" '
        'fill="#475569">BEV: forward up, ego lateral left is screen-left; keyframe-relative view</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_bev_model_png(recording, window, frame_idx, label, output_path, extent, size):
    """Render a model-facing BEV PNG with no formula/candidate text leakage."""
    frames = frame_by_index(window)
    frame = frames[frame_idx]
    key_ego = frame["ego"]
    key_pos = key_ego["position_lcs_m"]
    key_heading = ego_heading(key_ego)
    width, height = size
    left_m, right_m, back_m, forward_m = extent
    scale = min(width / (left_m + right_m), height / (back_m + forward_m))
    center_x = left_m * scale + (width - (left_m + right_m) * scale) / 2.0
    center_y = forward_m * scale + (height - (back_m + forward_m) * scale) / 2.0
    selected, tracks, interaction_by_frame_object = select_relevant_objects(
        window, max_objects=80
    )
    current_objects = {str(obj["object_id"]): obj for obj in frame.get("objects", [])}

    def visible(point):
        longitudinal, lateral = point
        return -back_m <= longitudinal <= forward_m and -right_m <= lateral <= left_m

    def screen(point):
        return ego_to_screen(point, center_x, center_y, scale)

    canvas = PngCanvas(width, height)
    grid = hex_to_rgb("#dbe4ee")
    axis = hex_to_rgb("#94a3b8")
    for lateral in range(-int(right_m), int(left_m) + 1, 10):
        x, _ = screen((0, lateral))
        canvas.line(x, 0, x, height - 1, grid, width=1, alpha=0.75)
    for longitudinal in range(-int(back_m), int(forward_m) + 1, 10):
        _, y = screen((longitudinal, 0))
        canvas.line(0, y, width - 1, y, grid, width=1, alpha=0.75)
    x0, y0 = screen((0, 0))
    canvas.line(0, y0, width - 1, y0, axis, width=2, alpha=0.8)
    canvas.line(x0, 0, x0, height - 1, axis, width=2, alpha=0.8)

    ego_path_points = []
    elapsed_path_points = []
    for path_frame in window["frames"]:
        ego_pos = path_frame["ego"]["position_lcs_m"]
        point = lcs_to_ego(ego_pos, key_pos, key_heading)
        if visible(point):
            ego_path_points.append(screen(point))
            if path_frame["frame_index"] <= frame_idx:
                elapsed_path_points.append(screen(point))
    if len(ego_path_points) >= 2:
        canvas.polyline(ego_path_points, hex_to_rgb("#94a3b8"), width=5, alpha=0.55)
    if len(elapsed_path_points) >= 2:
        canvas.polyline(elapsed_path_points, hex_to_rgb("#2563eb"), width=6, alpha=0.95)

    for object_id in selected:
        states = tracks.get(object_id, [])
        trail = []
        for trail_frame, obj in states:
            if trail_frame["frame_index"] > frame_idx:
                continue
            pos_lcs = obj.get("position_lcs_m")
            if not pos_lcs:
                continue
            point = lcs_to_ego(pos_lcs, key_pos, key_heading)
            if visible(point):
                trail.append(screen(point))
        if len(trail) >= 2:
            class_name = states[0][1].get("class", "unknown") if states else "unknown"
            canvas.polyline(trail, hex_to_rgb(class_color(class_name)), width=3, alpha=0.38)

    for object_id, obj in current_objects.items():
        pos = obj.get("position_ego_m") or {}
        point = (pos.get("longitudinal"), pos.get("lateral"))
        if None in point or not visible(point):
            continue
        class_name = obj.get("class", "unknown")
        color = hex_to_rgb(class_color(class_name))
        interaction = interaction_by_frame_object.get((frame_idx, object_id))
        important = bool(
            interaction
            and (
                interaction.get("observed_future_path_overlap")
                or interaction.get("ttc_constant_velocity_s") is not None
            )
        )
        corners = object_corners(obj)
        if corners:
            screen_corners = [screen(corner) for corner in corners]
            canvas.polygon(
                screen_corners,
                fill=color,
                outline=color,
                alpha=0.24 if important else 0.16,
                outline_width=3 if important else 2,
            )
        sx, sy = screen(point)
        if class_name in {"pedestrian", "motorcycle"}:
            canvas.circle(sx, sy, 9 if important else 7, color, alpha=0.22)
        canvas.circle(sx, sy, 5 if important else 4, color, alpha=1.0)

        motion = instantaneous_track_vector_ego(tracks, object_id, frame_idx, key_heading)
        if motion:
            end = arrow_endpoint_from_motion(point, motion)
            ex, ey = screen(end)
            canvas.line(sx, sy, ex, ey, color, width=3, alpha=0.9)
            # Minimal arrowhead without text labels.
            angle = math.atan2(ey - sy, ex - sx)
            for delta in (2.55, -2.55):
                hx = ex + math.cos(angle + delta) * 8
                hy = ey + math.sin(angle + delta) * 8
                canvas.line(ex, ey, hx, hy, color, width=3, alpha=0.9)

    ego_corners = [(2.4, 1.0), (2.4, -1.0), (-2.4, -1.0), (-2.4, 1.0)]
    canvas.polygon(
        [screen(point) for point in ego_corners],
        fill=hex_to_rgb("#22c55e"),
        outline=hex_to_rgb("#166534"),
        alpha=0.34,
        outline_width=4,
    )
    nose = [(3.0, 0.0), (1.6, 0.7), (1.6, -0.7)]
    canvas.polygon(
        [screen(point) for point in nose],
        fill=hex_to_rgb("#166534"),
        outline=hex_to_rgb("#166534"),
        alpha=0.9,
        outline_width=1,
    )
    canvas.save_png(output_path)


def write_readme(output_dir):
    readme = output_dir / "README.md"
    readme.write_text(
        """# Motional Model Inputs

Generated by `quick_exploration_outputs/build_bev_model_inputs.py`.

Each recording folder contains one folder per five-second motional window:

- `refined.json`: compact model input with preliminary/formula candidate fields
  removed.
- `bev_start.png`, `bev_middle.png`, `bev_end.png`: model-facing BEV keyframes
  with no candidate tags, formula labels, object text labels, or answer hints.

BEV convention: forward is up, ego lateral-left is screen-left, and each image
is centered on the ego pose at that keyframe. Object trails are transformed from
LCS into the keyframe ego pose, so the current frame and trail are spatially
consistent.

Formula-only baseline labels remain in the upstream motional-window JSON and
should be compared outside this model input package.

## Schema v2 migration notes

- Nearby object counts use `scenario_signals.nearby_30m_counts` as the canonical
  source field.
- Valid zero nearby counts are preserved as `0`; `null` means unavailable or not
  computed.
- `per_frame_counts` separates `dynamic_object_count`,
  `static_visible_object_count`, `total_object_count`, `classes_dynamic`,
  `classes_static`, and `classes_total`.
- Deprecated compatibility fields `object_count` and `classes` mirror
  `total_object_count` and `classes_total`.
- `data_quality` reports missing object frames, static snapshot spikes, nearby
  count source availability, and warnings.
- `source_window_file` uses a normalized POSIX-style path.
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create model-facing refined JSON and BEV PNG keyframes from motional windows."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--recording",
        action="append",
        help="Recording id to convert. Can be repeated. Defaults to all files.",
    )
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--left-m", type=float, default=45.0)
    parser.add_argument("--right-m", type=float, default=45.0)
    parser.add_argument("--back-m", type=float, default=25.0)
    parser.add_argument("--forward-m", type=float, default=95.0)
    parser.add_argument(
        "--window-limit",
        type=int,
        default=None,
        help="Optional limit per recording for timing/smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    write_readme(args.output_dir)
    recordings = set(args.recording or [])
    files = sorted(args.input_dir.glob("*_motional_windows.json"))
    converted = []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bev_schema_version": BEV_SCHEMA_VERSION,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "bev_extent_m": {
            "left": args.left_m,
            "right": args.right_m,
            "back": args.back_m,
            "forward": args.forward_m,
        },
        "bev_size_px": {"width": args.width, "height": args.height},
        "recordings": [],
    }
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_source_file"] = str(path)
        recording_id = data["recording_id"]
        if recordings and recording_id not in recordings:
            continue
        recording_dir = args.output_dir / safe_name(recording_id)
        ensure_dir(recording_dir)
        windows = data["windows"]
        if args.window_limit is not None:
            windows = windows[: args.window_limit]
        recording_summary = {
            "recording_id": recording_id,
            "source_file": str(path),
            "window_count": len(windows),
            "windows": [],
        }
        for window in windows:
            window_dir = recording_dir / safe_name(window["window_id"])
            ensure_dir(window_dir)
            keyframes = choose_keyframes(window)
            keyframe_files = {}
            for label, frame_idx in keyframes.items():
                filename = f"bev_{label}.png"
                output_path = window_dir / filename
                render_bev_model_png(
                    data,
                    window,
                    frame_idx,
                    label,
                    output_path,
                    (args.left_m, args.right_m, args.back_m, args.forward_m),
                    (args.width, args.height),
                )
                keyframe_files[label] = {
                    "frame_index": frame_idx,
                    "path": filename,
                    "format": "png",
                    "audience": "model",
                }
            refined = build_refined_json(
                data,
                window,
                keyframe_files,
                max_objects=args.max_objects,
                include_preliminary_candidates=False,
            )
            refined_path = window_dir / "refined.json"
            refined_path.write_text(
                json.dumps(refined, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            recording_summary["windows"].append(
                {
                    "window_id": window["window_id"],
                    "directory": str(window_dir),
                    "refined_json": str(refined_path),
                    "bev_keyframes": {
                        label: str(window_dir / info["path"])
                        for label, info in keyframe_files.items()
                    },
                }
            )
        manifest["recordings"].append(recording_summary)
        converted.append((recording_id, len(windows)))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_windows = sum(count for _, count in converted)
    print(f"Converted {len(converted)} recording(s), {total_windows} window(s).")
    print(f"Wrote {manifest_path}")
    for recording_id, count in converted:
        print(f"- {recording_id}: {count} windows")


if __name__ == "__main__":
    main()
