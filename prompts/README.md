# Motional Scenario Prompt Pack

Reusable prompts for evaluating OD + ego-trajectory windows with LLM/VLM APIs.

The prompts are designed for the current model input package:

```text
quick_exploration_outputs/model_inputs/<recording>/<window>/
  refined.json
  bev_start.svg or .png
  bev_middle.svg or .png
  bev_end.svg or .png
```

For Together Qwen3.5 9B Vision:

- Use `system_prompt.md` as the system message.
- Use `json_only_user_prompt.md` for the JSON-only run.
- Use `json_bev_user_prompt.md` for the JSON + BEV run.
- Require output matching `output_schema.json`.

Efficiency notes:

- Do not include `preliminary_candidates` in the model input for the first
  unbiased test. Keep it only for later comparison against the formula-only
  baseline.
- Send only one window per request.
- Keep output concise and schema-valid.
- Ask for evidence summaries, not hidden chain-of-thought.
- Use PNG/JPEG BEV images for true vision input. SVG text should only be used
  as a fallback text artifact.

