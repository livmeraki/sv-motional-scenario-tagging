# Prompt Usage Notes

## Recommended first experiment

Recording:

```text
Rec_Drv_GER_MACHET18_20260227_153128
```

Runs:

1. Formula-only baseline from `preliminary_candidates.candidate_flags`.
2. JSON-only Qwen3.5 9B Vision using `json_only_user_prompt.md`.
3. JSON + BEV Qwen3.5 9B Vision using `json_bev_user_prompt.md`.

## Bias control

For the cleanest model test, strip these fields before sending to the model:

```text
preliminary_candidates.candidate_tags
preliminary_candidates.candidate_flags
preliminary_candidates.evidence
```

Keep them locally for later comparison. This prevents the model from simply
copying formula outputs.

## Suggested compact model input

Keep:

- `schema_version`
- `recording_id`
- `source_window_id`
- `time_window`
- `bev_keyframes`
- `taxonomy`
- `ego_summary`
- `ego_series_sampled`
- `per_frame_counts`
- `relevant_objects`
- `data_notes`

Remove:

- formula/preliminary candidate outputs
- long object tracks beyond selected samples
- any fields not used by the schema

## Output handling

Reject and retry if:

- output is not valid JSON
- a required label is missing
- label values are strings instead of booleans
- confidence is outside `[0, 1]`
- `recording_id`, `window_id`, or `model_mode` is missing

Retry prompt suffix:

```text
Your previous response did not match the required JSON schema. Return only valid JSON. Do not include markdown or explanation outside the JSON object.
```

