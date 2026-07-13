Evaluate this single driving window using the supplied `refined.json` only.

Return only JSON matching the provided output schema.

Important:
- Do not use BEV imagery for this run.
- Do not overfit to formula/preliminary candidates if they are present.
- Use `ego_summary`, `ego_series_sampled`, `per_frame_counts`, `relevant_objects`, and interaction fields as the primary evidence.
- For each true label, cite compact evidence using frame indices, time range, object IDs/classes, distance, speed, yaw rate, lead presence, or TTC/path-overlap when relevant.
- For each false label, give a very short reason only when useful.
- Keep each `evidence_summary` under 280 characters.

Input JSON:

```json
{{REFINED_JSON}}
```

