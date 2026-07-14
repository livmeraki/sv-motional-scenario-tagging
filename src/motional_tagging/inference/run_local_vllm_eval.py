#!/usr/bin/env python3
"""Run one local vLLM scenario-tagging evaluation request."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ROOT / "prompts"
SCHEMAS_DIR = ROOT / "schemas"
INPUTS_DIR = Path(os.environ.get("MOTIONAL_MODEL_INPUT_ROOT", ROOT / "outputs" / "model_inputs"))
OUTPUTS_DIR = Path(os.environ.get("MOTIONAL_OUTPUT_ROOT", ROOT / "outputs")) / "local_vllm_eval"

DEFAULT_RECORDING = "Rec_Drv_GER_MACHET18_20260227_153128"
DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1/chat/completions"
DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
BEV_CACHE_DIR = OUTPUTS_DIR / "_bev_png_cache"
BEV_RENDER_MAX_SIDE = 256
RUNS_DIR = OUTPUTS_DIR / "runs"
REPORT_TSV = OUTPUTS_DIR / "run_report.tsv"
REPORT_MD = OUTPUTS_DIR / "run_report.md"
REPORT_COLUMNS = [
    "Run",
    "Scene",
    "Frames",
    "Model",
    "Mode",
    "Max_Model_Len",
    "BEV Max Side",
    "Max Tokens",
    "Prompt Tokens",
    "Completion Tokens",
    "Total Tokens",
    "Windows",
    "Valid JSON",
    "Correct Speed Band",
    "Lead Correct",
    "GT Correct",
    "GT Mismatches",
    "Major Issue",
]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def strip_preliminary_candidates(refined: dict[str, Any]) -> dict[str, Any]:
    clean = dict(refined)
    clean.pop("preliminary_candidates", None)
    return clean


def compact_object(obj: dict[str, Any]) -> dict[str, Any]:
    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    keep_frames = {5, 25, 49}
    compact_samples = []
    for sample in samples:
        if sample.get("frame_index") in keep_frames:
            compact_samples.append(
                {
                    key: sample.get(key)
                    for key in (
                        "frame_index",
                        "time_s",
                        "longitudinal_m",
                        "lateral_m",
                        "distance_m",
                        "ttc_s",
                        "path_overlap",
                    )
                    if key in sample
                }
            )
    if not compact_samples and samples:
        for sample in (samples[0], samples[len(samples) // 2], samples[-1]):
            if sample not in compact_samples:
                compact_samples.append(
                    {
                        key: sample.get(key)
                        for key in (
                            "frame_index",
                            "time_s",
                            "longitudinal_m",
                            "lateral_m",
                            "distance_m",
                            "ttc_s",
                            "path_overlap",
                        )
                        if key in sample
                    }
                )
    return {
        key: obj.get(key)
        for key in (
            "object_id",
            "class",
            "subclass",
            "observed_frame_count",
            "first_frame",
            "last_frame",
            "minimum_distance_m",
        )
        if key in obj
    } | {"samples": compact_samples[:3]}


def compact_for_bev(refined: dict[str, Any]) -> dict[str, Any]:
    frame_keep = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49}
    per_frame_counts = [
        item
        for item in refined.get("per_frame_counts", [])
        if isinstance(item, dict) and item.get("frame_index") in frame_keep
    ]
    objects = refined.get("relevant_objects", [])
    if isinstance(objects, list):
        sorted_objects = sorted(
            [obj for obj in objects if isinstance(obj, dict)],
            key=lambda obj: (
                obj.get("minimum_distance_m") is None,
                obj.get("minimum_distance_m", 9999),
            ),
        )
    else:
        sorted_objects = []
    compact_objects = [
        compact_object(obj)
        for obj in sorted_objects
        if obj.get("minimum_distance_m", 9999) <= 40
        or obj.get("class") in {"pedestrian", "motorcycle", "bicycle"}
    ][:16]

    keep = {
        "schema_version",
        "recording_id",
        "source_window_id",
        "time_window",
        "bev_keyframes",
        "taxonomy",
        "ego_summary",
        "ego_series_sampled",
        "data_notes",
    }
    compact = {key: refined[key] for key in keep if key in refined}
    compact["per_frame_counts_sampled"] = per_frame_counts
    compact["relevant_objects_compact"] = compact_objects
    compact["input_compaction_note"] = (
        "BEV mode uses compact numeric JSON plus start/middle/end BEV images. "
        "Preliminary candidate fields are omitted unless explicitly requested."
    )
    return compact


def first_window_dir(recording_id: str) -> Path:
    recording_dir = INPUTS_DIR / recording_id
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"Recording folder not found: {recording_dir}")
    windows = sorted(path for path in recording_dir.iterdir() if path.is_dir())
    if not windows:
        raise FileNotFoundError(f"No window folders found under: {recording_dir}")
    return windows[0]


def parse_number(value: str | None, default: float = 0.0, reference: float | None = None) -> float:
    if value is None:
        return default
    value = value.strip()
    if value.endswith("%") and reference is not None:
        return reference * float(value[:-1]) / 100.0
    match = re.match(r"[-+]?\d*\.?\d+", value)
    return float(match.group(0)) if match else default


def parse_color(value: str | None, opacity: float = 1.0) -> tuple[int, int, int, int] | None:
    if not value or value == "none" or value.startswith("url(") or value == "context-stroke":
        return None
    try:
        red, green, blue = ImageColor.getrgb(value)[:3]
    except ValueError:
        return None
    alpha = max(0, min(255, int(round(255 * opacity))))
    return red, green, blue, alpha


def parse_points(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    nums = [float(num) for num in re.findall(r"[-+]?\d*\.?\d+", value)]
    return list(zip(nums[0::2], nums[1::2]))


def draw_with_alpha(image: Image.Image, draw_fn: Any) -> None:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw_fn(draw)
    image.alpha_composite(layer)


def rasterize_svg_to_png(svg_path: Path, png_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    width = int(parse_number(root.attrib.get("width"), 1000))
    height = int(parse_number(root.attrib.get("height"), 900))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    font_cache: dict[int, ImageFont.ImageFont] = {}

    def font(size: int) -> ImageFont.ImageFont:
        if size not in font_cache:
            try:
                font_cache[size] = ImageFont.truetype("DejaVuSans.ttf", size)
            except OSError:
                font_cache[size] = ImageFont.load_default()
        return font_cache[size]

    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        attrs = element.attrib
        opacity = parse_number(attrs.get("opacity"), 1.0)
        fill_opacity = parse_number(attrs.get("fill-opacity"), opacity)
        stroke_opacity = parse_number(attrs.get("stroke-opacity"), opacity)
        fill = parse_color(attrs.get("fill"), fill_opacity)
        stroke = parse_color(attrs.get("stroke"), stroke_opacity)
        stroke_width = max(1, int(round(parse_number(attrs.get("stroke-width"), 1.0))))

        if tag == "rect":
            x = parse_number(attrs.get("x"), 0)
            y = parse_number(attrs.get("y"), 0)
            w = parse_number(attrs.get("width"), width, width)
            h = parse_number(attrs.get("height"), height, height)

            def draw_rect(draw: ImageDraw.ImageDraw) -> None:
                draw.rectangle([x, y, x + w, y + h], fill=fill, outline=stroke, width=stroke_width)

            draw_with_alpha(image, draw_rect)
        elif tag == "line" and stroke:
            x1 = parse_number(attrs.get("x1"))
            y1 = parse_number(attrs.get("y1"))
            x2 = parse_number(attrs.get("x2"))
            y2 = parse_number(attrs.get("y2"))
            draw_with_alpha(image, lambda draw: draw.line([x1, y1, x2, y2], fill=stroke, width=stroke_width))
        elif tag == "polyline":
            points = parse_points(attrs.get("points"))
            if len(points) >= 2 and stroke:
                draw_with_alpha(image, lambda draw, pts=points: draw.line(pts, fill=stroke, width=stroke_width, joint="curve"))
        elif tag == "polygon":
            points = parse_points(attrs.get("points"))
            if len(points) >= 2:
                def draw_polygon(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]] = points) -> None:
                    if fill:
                        draw.polygon(pts, fill=fill)
                    if stroke:
                        draw.line(pts + [pts[0]], fill=stroke, width=stroke_width, joint="curve")

                draw_with_alpha(image, draw_polygon)
        elif tag == "circle":
            cx = parse_number(attrs.get("cx"))
            cy = parse_number(attrs.get("cy"))
            r = parse_number(attrs.get("r"), 1.0)

            def draw_circle(draw: ImageDraw.ImageDraw) -> None:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=stroke, width=stroke_width)

            draw_with_alpha(image, draw_circle)
        elif tag == "text" and element.text:
            x = parse_number(attrs.get("x"))
            y = parse_number(attrs.get("y"))
            size = int(parse_number(attrs.get("font-size"), 12))
            text_fill = fill or (15, 23, 42, 255)
            if attrs.get("stroke"):
                outline = parse_color(attrs.get("stroke"), stroke_opacity)
                if outline:
                    draw_with_alpha(
                        image,
                        lambda draw, text=element.text: draw.text(
                            (x, y - size),
                            text,
                            fill=outline,
                            font=font(size),
                            stroke_width=2,
                            stroke_fill=outline,
                        ),
                    )
            draw_with_alpha(
                image,
                lambda draw, text=element.text: draw.text((x, y - size), text, fill=text_fill, font=font(size)),
            )

    rgb = image.convert("RGB")
    scale = min(1.0, BEV_RENDER_MAX_SIDE / max(rgb.size))
    if scale < 1.0:
        new_size = (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale))))
        rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(png_path, format="PNG")


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_png_for_prompt(source_png: Path, prompt_png: Path) -> Path:
    with Image.open(source_png) as image:
        rgb = image.convert("RGB")
        scale = min(1.0, BEV_RENDER_MAX_SIDE / max(rgb.size))
        if scale >= 1.0:
            return source_png
        if prompt_png.exists() and prompt_png.stat().st_mtime >= source_png.stat().st_mtime:
            return prompt_png
        new_size = (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale))))
        rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        prompt_png.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(prompt_png, format="PNG")
    return prompt_png


def bev_png_paths(window_dir: Path) -> list[Path]:
    labels = ["start", "middle", "end"]
    cache_dir = BEV_CACHE_DIR / f"max{BEV_RENDER_MAX_SIDE}" / window_dir.parent.name / window_dir.name
    png_paths: list[Path] = []
    for label in labels:
        direct_png = window_dir / f"bev_{label}.png"
        if direct_png.is_file():
            png_paths.append(prepare_png_for_prompt(direct_png, cache_dir / f"bev_{label}.png"))
            continue
        svg_path = window_dir / f"bev_{label}.svg"
        if not svg_path.is_file():
            raise FileNotFoundError(f"Missing BEV keyframe PNG or SVG: {direct_png} / {svg_path}")
        png_path = cache_dir / f"bev_{label}.png"
        if not png_path.exists() or png_path.stat().st_mtime < svg_path.stat().st_mtime:
            rasterize_svg_to_png(svg_path, png_path)
        png_paths.append(png_path)
    return png_paths


def build_messages(refined: dict[str, Any], mode: str, window_dir: Path) -> list[dict[str, Any]]:
    system_prompt = load_text(PROMPTS_DIR / "system_prompt.md")
    user_template = load_text(PROMPTS_DIR / f"{mode}_user_prompt.md")
    if mode == "json_bev":
        refined_json = json.dumps(refined, ensure_ascii=False, separators=(",", ":"))
    else:
        refined_json = json.dumps(refined, ensure_ascii=False, indent=2)
    if mode == "json_bev":
        schema_block = (
            "Output schema summary: return one JSON object with exactly these top-level keys: "
            "schema_version='motional-scenario-model-output-v1', recording_id, window_id, "
            "model_mode='json_bev', labels, overall_quality, review_priority. "
            "labels must contain all 12 taxonomy labels. Each label value must be an object with "
            "boolean value, numeric confidence in [0,1], string evidence_summary, integer-array "
            "evidence_frames, and string-array object_ids. overall_quality has confidence and "
            "data_issues. review_priority is low, medium, or high.\n\n"
            "Compact-output rules for local vLLM: every label object MUST include the boolean key "
            "`value`. Use at most 3 evidence_frames per label, chosen only from [0,5,25,49]. "
            "Use at most 2 object_ids per label. Keep every evidence_summary under 90 characters. "
            "For low_magnitude_speed, medium_magnitude_speed, and high_magnitude_speed, use median_speed_mps only; "
            "set evidence_frames=[] and object_ids=[] because speed bands do not depend on frame or object evidence. "
            "Do not list all frames. Do not explain false labels in detail. "
            "Use terse numeric evidence, for example `median speed 10.05 m/s => medium`.\n\n"
        )
    else:
        output_schema = json.dumps(load_json(SCHEMAS_DIR / "output_schema.json"), ensure_ascii=False, indent=2)
        schema_block = (
            "Output schema:\n\n"
            "```json\n"
            f"{output_schema}\n"
            "```\n\n"
        )
    user_text = schema_block + user_template.replace("{{REFINED_JSON}}", refined_json)
    if mode == "json_bev":
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for png_path in bev_png_paths(window_dir):
            content.append({"type": "image_url", "image_url": {"url": image_data_url(png_path)}})
        user_content: str | list[dict[str, Any]] = content
    else:
        user_content = user_text
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def post_chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def models_endpoint(chat_endpoint: str) -> str:
    return chat_endpoint.rsplit("/v1/", 1)[0] + "/v1/models"


def fetch_server_model_info(endpoint: str, model: str, timeout_s: float) -> dict[str, Any]:
    try:
        request = urllib.request.Request(models_endpoint(endpoint), method="GET")
        with urllib.request.urlopen(request, timeout=min(timeout_s, 10)) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
    except Exception:
        return {}
    for item in payload.get("data", []):
        if item.get("id") == model:
            return item
    data = payload.get("data", [])
    return data[0] if data else {}


def extract_content(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Response did not contain choices[0].message.content") from exc


def parse_model_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model content was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON content was not an object")
    return parsed


def expected_speed_labels(refined: dict[str, Any]) -> dict[str, bool] | None:
    median_speed = refined.get("ego_summary", {}).get("median_speed_mps")
    if not isinstance(median_speed, (int, float)):
        return None
    return {
        "low_magnitude_speed": 0.5 <= median_speed < 5.0,
        "medium_magnitude_speed": 5.0 <= median_speed < 15.0,
        "high_magnitude_speed": median_speed >= 15.0,
    }


def default_gt_path(recording_id: str) -> Path:
    return Path(os.environ.get("GT_LABEL_ROOT", ROOT / "gt_labels")) / f"{recording_id}_gt.json"


def output_window_ids(recording_id: str, window_id: str) -> set[str]:
    suffix = window_id.split("_")[-1]
    return {window_id, f"{recording_id}:{suffix}"}


def load_gt_labels(gt_path: Path, recording_id: str, window_id: str) -> dict[str, bool] | None:
    if not gt_path.is_file():
        return None
    gt = load_json(gt_path)
    if gt.get("recording_id") != recording_id:
        raise ValueError(f"GT recording_id in {gt_path} must be {recording_id}")

    valid_window_ids = output_window_ids(recording_id, window_id)
    for window in gt.get("windows", []):
        if not isinstance(window, dict) or window.get("window_id") not in valid_window_ids:
            continue
        labels = window.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"GT window {window.get('window_id')} has no labels object")
        return {key: value for key, value in labels.items() if isinstance(value, bool)}
    raise ValueError(f"GT labels not found for window {window_id} in {gt_path}")


def validate_against_gt(output: dict[str, Any] | None, gt_labels: dict[str, bool] | None) -> dict[str, Any]:
    if gt_labels is None:
        return {"status": "not_available", "mismatches": []}

    output_labels = output.get("labels") if isinstance(output, dict) else None
    if not isinstance(output_labels, dict):
        return {
            "status": "failed",
            "checked_count": len(gt_labels),
            "correct_count": 0,
            "accuracy": 0.0,
            "mismatches": [
                {"label": label, "expected": expected, "actual": None}
                for label, expected in sorted(gt_labels.items())
            ],
        }

    mismatches = []
    correct_count = 0
    for label, expected in sorted(gt_labels.items()):
        actual = output_labels.get(label, {}).get("value")
        if actual == expected:
            correct_count += 1
        else:
            mismatches.append({"label": label, "expected": expected, "actual": actual})
    checked_count = len(gt_labels)
    accuracy = correct_count / checked_count if checked_count else None
    return {
        "status": "passed" if not mismatches else "failed",
        "checked_count": checked_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "mismatches": mismatches,
    }


def validate_output(
    output: dict[str, Any],
    recording_id: str,
    window_id: str,
    mode: str,
    refined: dict[str, Any],
) -> list[str]:
    schema = load_json(SCHEMAS_DIR / "output_schema.json")
    required_top = schema["required"]
    label_names = schema["properties"]["labels"]["required"]
    errors: list[str] = []

    for key in required_top:
        if key not in output:
            errors.append(f"missing top-level key: {key}")

    if output.get("schema_version") != "motional-scenario-model-output-v1":
        errors.append("schema_version must be motional-scenario-model-output-v1")
    if output.get("recording_id") != recording_id:
        errors.append(f"recording_id must be {recording_id}")
    if output.get("window_id") not in output_window_ids(recording_id, window_id):
        errors.append(f"window_id should identify {window_id}")
    if output.get("model_mode") != mode:
        errors.append(f"model_mode must be {mode}")

    labels = output.get("labels")
    if not isinstance(labels, dict):
        errors.append("labels must be an object")
        return errors

    for label in label_names:
        decision = labels.get(label)
        if not isinstance(decision, dict):
            errors.append(f"labels.{label} must be an object")
            continue
        if not isinstance(decision.get("value"), bool):
            errors.append(f"labels.{label}.value must be boolean")
        confidence = decision.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"labels.{label}.confidence must be in [0, 1]")
        if not isinstance(decision.get("evidence_summary"), str):
            errors.append(f"labels.{label}.evidence_summary must be string")
        if not isinstance(decision.get("evidence_frames"), list):
            errors.append(f"labels.{label}.evidence_frames must be list")
        elif len(decision["evidence_frames"]) > 3:
            errors.append(f"labels.{label}.evidence_frames must contain at most 3 items")
        if not isinstance(decision.get("object_ids"), list):
            errors.append(f"labels.{label}.object_ids must be list")
        elif len(decision["object_ids"]) > 2:
            errors.append(f"labels.{label}.object_ids must contain at most 2 items")
        if label in {"low_magnitude_speed", "medium_magnitude_speed", "high_magnitude_speed"}:
            if decision.get("evidence_frames") != []:
                errors.append(f"labels.{label}.evidence_frames must be empty for speed-band labels")
            if decision.get("object_ids") != []:
                errors.append(f"labels.{label}.object_ids must be empty for speed-band labels")

    expected_speed = expected_speed_labels(refined)
    if expected_speed:
        for label, expected_value in expected_speed.items():
            actual_value = labels.get(label, {}).get("value")
            if isinstance(actual_value, bool) and actual_value != expected_value:
                median_speed = refined["ego_summary"]["median_speed_mps"]
                errors.append(
                    f"labels.{label}.value must be {expected_value} for median_speed_mps={median_speed}"
                )

    minimum_speed = refined.get("ego_summary", {}).get("minimum_speed_mps")
    if isinstance(minimum_speed, (int, float)) and minimum_speed > 0.5:
        actual_stationary = labels.get("stationary", {}).get("value")
        if isinstance(actual_stationary, bool) and actual_stationary:
            errors.append(f"labels.stationary.value must be false because minimum_speed_mps={minimum_speed}")

    quality = output.get("overall_quality")
    if not isinstance(quality, dict):
        errors.append("overall_quality must be an object")
    else:
        confidence = quality.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append("overall_quality.confidence must be in [0, 1]")
        if not isinstance(quality.get("data_issues"), list):
            errors.append("overall_quality.data_issues must be list")

    if output.get("review_priority") not in {"low", "medium", "high"}:
        errors.append("review_priority must be low, medium, or high")

    return errors


def retry_prompt(validation_errors: list[str]) -> str:
    error_lines = "\n".join(f"- {error}" for error in validation_errors)
    return (
        "Your previous response failed validation:\n"
        f"{error_lines}\n\n"
        "Return only a corrected JSON object matching the requested schema. "
        "Fix numeric threshold mistakes. Shorten arrays: each evidence_frames array must have at most "
        "3 items and each object_ids array must have at most 2 items. "
        "For speed-band labels, set evidence_frames=[] and object_ids=[] because speed bands use only median_speed_mps. "
        "Do not include markdown or explanation outside the JSON object."
    )


def write_outputs(
    output_dir: Path,
    request_payload: dict[str, Any],
    raw_response: dict[str, Any],
    parsed_output: dict[str, Any] | None,
    validation_errors: list[str],
    gt_validation: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_files(output_dir)
    (output_dir / "request.json").write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw_response.json").write_text(
        json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if parsed_output is not None:
        (output_dir / "model_output.json").write_text(
            json.dumps(parsed_output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    validation_payload = {"errors": validation_errors, "gt_validation": gt_validation}
    (output_dir / "validation.json").write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def clear_generated_files(output_dir: Path) -> None:
    for name in (
        "request.json",
        "raw_response.json",
        "model_output.json",
        "validation.json",
        "raw_response_parse_error.json",
        "model_output_parse_error.txt",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_parse_error_outputs(
    output_dir: Path,
    request_payload: dict[str, Any],
    raw_response: dict[str, Any] | None,
    content: str | None,
    error: str,
    gt_validation: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_files(output_dir)
    (output_dir / "request.json").write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if raw_response is not None:
        (output_dir / "raw_response_parse_error.json").write_text(
            json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if content is not None:
        (output_dir / "model_output_parse_error.txt").write_text(content, encoding="utf-8")
    validation_payload = {"errors": [error], "gt_validation": gt_validation}
    (output_dir / "validation.json").write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def next_run_id() -> str:
    max_seen = 0
    if REPORT_TSV.exists():
        with REPORT_TSV.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                run = row.get("Run", "")
                if re.fullmatch(r"R\d{3,}", run):
                    max_seen = max(max_seen, int(run[1:]))
    if RUNS_DIR.exists():
        for path in RUNS_DIR.iterdir():
            if path.is_dir() and re.fullmatch(r"R\d{3,}", path.name):
                max_seen = max(max_seen, int(path.name[1:]))
    return f"R{max_seen + 1:03d}"


def yes_no(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def speed_band_correct(output: dict[str, Any] | None, refined: dict[str, Any]) -> bool | None:
    if output is None:
        return False
    labels = output.get("labels")
    if not isinstance(labels, dict):
        return False
    expected = expected_speed_labels(refined)
    if not expected:
        return None
    for label, expected_value in expected.items():
        if labels.get(label, {}).get("value") != expected_value:
            return False
    return True


def lead_correct(output: dict[str, Any] | None, refined: dict[str, Any]) -> bool | None:
    expected = refined.get("preliminary_candidates", {}).get("candidate_flags", {}).get("following_lane_with_lead")
    if not isinstance(expected, bool):
        return None
    if output is None:
        return False
    actual = output.get("labels", {}).get("following_lane_with_lead", {}).get("value")
    return actual == expected if isinstance(actual, bool) else False


def summarize_issue(
    parsed_output: dict[str, Any] | None,
    validation_errors: list[str],
    gt_validation: dict[str, Any],
    raw_response: dict[str, Any] | None,
    parse_error: str | None,
    refined: dict[str, Any],
) -> str:
    finish_reason = None
    if raw_response:
        finish_reason = raw_response.get("choices", [{}])[0].get("finish_reason")
    if parse_error:
        if finish_reason == "length":
            return "Truncated output / invalid JSON"
        return f"Invalid JSON: {parse_error.splitlines()[0]}"
    if validation_errors:
        speed_errors = [err for err in validation_errors if "magnitude_speed" in err or "median_speed_mps" in err]
        if speed_errors:
            median = refined.get("ego_summary", {}).get("median_speed_mps")
            return f"Failed speed threshold; median was {median} m/s"
        return "; ".join(validation_errors[:2])
    gt_mismatches = gt_validation.get("mismatches", [])
    if gt_mismatches:
        labels = ", ".join(str(item.get("label")) for item in gt_mismatches[:3])
        suffix = "" if len(gt_mismatches) <= 3 else f" +{len(gt_mismatches) - 3} more"
        return f"GT mismatch: {labels}{suffix}"
    if lead_correct(parsed_output, refined) is False:
        return "Marked following_lane_with_lead false"
    return "none"


def report_row(
    run_id: str,
    scene: str,
    frames: str,
    model: str,
    mode: str,
    max_model_len: Any,
    max_tokens: int,
    raw_response: dict[str, Any] | None,
    parsed_output: dict[str, Any] | None,
    validation_errors: list[str],
    gt_validation: dict[str, Any],
    parse_error: str | None,
    refined: dict[str, Any],
) -> dict[str, str]:
    usage = raw_response.get("usage", {}) if raw_response else {}
    valid_json = parsed_output is not None
    display_mode = "JSON+BEV" if mode == "json_bev" else "JSON"
    gt_status = gt_validation.get("status")
    gt_correct = None if gt_status == "not_available" else gt_status == "passed"
    return {
        "Run": run_id,
        "Scene": scene,
        "Frames": frames,
        "Model": model,
        "Mode": display_mode,
        "Max_Model_Len": str(max_model_len or ""),
        "BEV Max Side": str(BEV_RENDER_MAX_SIDE) if mode == "json_bev" else "n/a",
        "Max Tokens": str(max_tokens),
        "Prompt Tokens": str(usage.get("prompt_tokens", "")),
        "Completion Tokens": str(usage.get("completion_tokens", "")),
        "Total Tokens": str(usage.get("total_tokens", "")),
        "Windows": "1",
        "Valid JSON": yes_no(valid_json),
        "Correct Speed Band": yes_no(speed_band_correct(parsed_output, refined)),
        "Lead Correct": yes_no(lead_correct(parsed_output, refined)),
        "GT Correct": yes_no(gt_correct),
        "GT Mismatches": str(len(gt_validation.get("mismatches", []))),
        "Major Issue": summarize_issue(parsed_output, validation_errors, gt_validation, raw_response, parse_error, refined),
    }


def write_report_files(new_row: dict[str, str]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if REPORT_TSV.exists():
        with REPORT_TSV.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    rows.append(new_row)
    with REPORT_TSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in REPORT_COLUMNS) + " |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global INPUTS_DIR, OUTPUTS_DIR, BEV_CACHE_DIR, RUNS_DIR, REPORT_TSV, REPORT_MD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--window", help="Window folder name. Defaults to the first sorted window.")
    parser.add_argument("--mode", choices=["json_only", "json_bev"], default="json_only")
    parser.add_argument("--endpoint", default=os.environ.get("VLLM_BASE_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--model")
    parser.add_argument("--model-input-root", type=Path, default=INPUTS_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument("--retry-on-invalid", action="store_true")
    parser.add_argument("--keep-preliminary-candidates", action="store_true")
    parser.add_argument("--full-input", action="store_true", help="Do not compact refined.json for BEV mode.")
    parser.add_argument(
        "--gt-labels",
        type=Path,
        help="GT labels JSON. Defaults to gt_labels/<recording>_gt.json when present.",
    )
    parser.add_argument("--no-gt-validation", action="store_true", help="Skip GT label comparison.")
    args = parser.parse_args()
    INPUTS_DIR = args.model_input_root.resolve()
    OUTPUTS_DIR = args.output_root.resolve()
    BEV_CACHE_DIR = OUTPUTS_DIR / "_bev_png_cache"
    RUNS_DIR = OUTPUTS_DIR / "runs"
    REPORT_TSV = OUTPUTS_DIR / "run_report.tsv"
    REPORT_MD = OUTPUTS_DIR / "run_report.md"
    model = args.model or os.environ.get("VLLM_MODEL") or (DEFAULT_VISION_MODEL if args.mode == "json_bev" else DEFAULT_TEXT_MODEL)

    window_dir = INPUTS_DIR / args.recording / args.window if args.window else first_window_dir(args.recording)
    refined_path = window_dir / "refined.json"
    refined = load_json(refined_path)
    gt_path = args.gt_labels or default_gt_path(args.recording)
    if args.gt_labels and not gt_path.is_file():
        print(f"GT labels file not found: {gt_path}", file=sys.stderr)
        return 2
    try:
        gt_labels = None if args.no_gt_validation else load_gt_labels(gt_path, args.recording, window_dir.name)
    except ValueError as exc:
        print(f"GT validation setup failed: {exc}", file=sys.stderr)
        return 2
    model_input = refined if args.keep_preliminary_candidates else strip_preliminary_candidates(refined)
    if args.mode == "json_bev" and not args.full_input:
        model_input = compact_for_bev(model_input)
    messages = build_messages(model_input, args.mode, window_dir)
    run_id = next_run_id()
    server_model_info = fetch_server_model_info(args.endpoint, model, args.timeout_s)
    request_payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "model": model,
        "server_model_info": server_model_info,
        "max_tokens": args.max_tokens,
        "recording": args.recording,
        "window": window_dir.name,
        "refined_json": display_path(refined_path),
        "mode": args.mode,
        "gt_labels": None if args.no_gt_validation or gt_labels is None else display_path(gt_path),
        "bev_max_side": BEV_RENDER_MAX_SIDE if args.mode == "json_bev" else None,
        "compacted_input": args.mode == "json_bev" and not args.full_input,
        "stripped_preliminary_candidates": not args.keep_preliminary_candidates,
        "messages": messages,
    }
    run_output_dir = RUNS_DIR / run_id
    latest_output_dir = OUTPUTS_DIR / args.recording / window_dir.name / args.mode

    started_at = time.time()
    raw_response: dict[str, Any] | None = None
    content: str | None = None
    parsed_output: dict[str, Any] | None = None
    validation_errors: list[str] = []
    gt_validation: dict[str, Any] = validate_against_gt(None, gt_labels)
    try:
        raw_response = post_chat_completion(
            args.endpoint,
            model,
            messages,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        content = extract_content(raw_response)
        parsed_output = parse_model_json(content)
        validation_errors = validate_output(parsed_output, args.recording, window_dir.name, args.mode, model_input)
        gt_validation = validate_against_gt(parsed_output, gt_labels)

        if validation_errors and args.retry_on_invalid:
            retry_messages = messages + [{"role": "user", "content": retry_prompt(validation_errors)}]
            request_payload["messages"] = retry_messages
            raw_response = post_chat_completion(
                args.endpoint,
                model,
                retry_messages,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
            )
            content = extract_content(raw_response)
            parsed_output = parse_model_json(content)
            validation_errors = validate_output(parsed_output, args.recording, window_dir.name, args.mode, model_input)
            gt_validation = validate_against_gt(parsed_output, gt_labels)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Request failed for {args.endpoint}: HTTP {exc.code} {exc.reason}: {detail}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Request failed for {args.endpoint}: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        error = str(exc)
        write_parse_error_outputs(run_output_dir, request_payload, raw_response, content, error, gt_validation)
        write_parse_error_outputs(latest_output_dir, request_payload, raw_response, content, error, gt_validation)
        row = report_row(
            run_id,
            args.recording,
            window_dir.name.split("_")[-1],
            model,
            args.mode,
            server_model_info.get("max_model_len"),
            args.max_tokens,
            raw_response,
            None,
            [error],
            gt_validation,
            error,
            refined,
        )
        write_report_files(row)
        print(f"Response parsing failed: {exc}", file=sys.stderr)
        print(f"run_id: {run_id}")
        print(f"report: {display_path(REPORT_TSV)}")
        return 3

    write_outputs(run_output_dir, request_payload, raw_response, parsed_output, validation_errors, gt_validation)
    write_outputs(latest_output_dir, request_payload, raw_response, parsed_output, validation_errors, gt_validation)
    row = report_row(
        run_id,
        args.recording,
        window_dir.name.split("_")[-1],
        model,
        args.mode,
        server_model_info.get("max_model_len"),
        args.max_tokens,
        raw_response,
        parsed_output,
        validation_errors,
        gt_validation,
        None,
        refined,
    )
    write_report_files(row)

    elapsed_s = time.time() - started_at
    print(f"run_id: {run_id}")
    print(f"recording: {args.recording}")
    print(f"window: {window_dir.name}")
    print(f"mode: {args.mode}")
    print(f"model: {model}")
    print(f"endpoint: {args.endpoint}")
    print(f"output_dir: {display_path(run_output_dir)}")
    print(f"latest_dir: {display_path(latest_output_dir)}")
    print(f"report: {display_path(REPORT_TSV)}")
    print(f"elapsed_s: {elapsed_s:.1f}")
    if validation_errors:
        print("validation: failed")
        for error in validation_errors:
            print(f"- {error}")
        return 1
    if gt_validation.get("status") == "failed":
        print("gt_validation: failed")
        for item in gt_validation.get("mismatches", []):
            print(f"- {item['label']}: expected {item['expected']}, actual {item['actual']}")
        return 1
    if gt_validation.get("status") == "passed":
        print(
            "gt_validation: ok "
            f"({gt_validation.get('correct_count')}/{gt_validation.get('checked_count')})"
        )
    else:
        print("gt_validation: not_available")
    print("validation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
