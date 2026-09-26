<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Report templates

These templates are the contract between the pipeline and the user-facing markdown. **Do not paraphrase**. The pipeline's [scripts/generate_final_reports.py](scripts/generate_final_reports.py) already renders these files at `<OutputDir>/model_compatibility_report.md` and `<OutputDir>/model_compatibility_details.md`. When you read those generated files back to the user, preserve the section order and naming verbatim.

## Non-negotiable rules

1. Never invent operators, counts, mappings, or reasons. Every number must match `compatibility/report_input.json`.
2. Unsupported reason text policy — the text must say which of these it is, never a blanket "not implemented":
   - No converter matches the operator: `No Hip Dialect implementation available.`
   - A converter exists but refused every instance: the generated text names the converter and quotes what the conversion reported. Keep it; do not shorten it to the sentence above.
   - Compile-time ops: keep the specific compile-time reason text.
   An `It reported:` clause is the compiler's own message, so quote it in your prose rather than restating it. Text without one means the converter refused without a message; [diagnose.md](diagnose.md) is how you find the guard.
3. Status display rule:
   - input `full` -> displayed as `supported`
   - input `partial` -> kept as `partial`
   - input `unsupported` -> kept as `unsupported`
4. If a field is missing, render as `—` and add one short note in the "Data quality notes" section of the details file.
5. For `unsupported` recommendations, use [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json) as the primary capability matrix, not just current repo wrappers.
6. In summary, append percentage for `Supported instances` when total is available (e.g. `1294 (62.3%)`).

## Output files (rendered by the pipeline)

| File | Purpose |
|---|---|
| `model_compatibility_report.md` | Executive summary + key tables (primary deliverable) |
| `model_compatibility_details.md` | Per-operator diagnostics (supports, partials with reason codes, data quality notes) |
| `compatibility/report_input.json` | Normalized data behind both files, validated against `report_input.schema.json` |

## model_compatibility_report.md — section order

Exact order, do not reorder:

1. `# Model compatibility report`
2. Metadata bullets (one bullet each):
   - `Analyzed graph` — the EP input, or the original ONNX in `-SkipDump` mode
   - `Original model`
   - `Generated UTC`
3. *(conditional)* `> **Source:** original ONNX, conversion probe skipped` — when the pipeline ran in `-SkipDump` mode (the orchestrator injects this badge automatically)
4. `## Summary`
   - Total node instances
   - Supported instances `<n> (<pct>%)`
   - Unsupported instances
   - Total Operator Types
   - Fully Compatible
   - Partially Compatible
   - Unsupported
5. *(conditional)* `## Original vs EP input` — only when the dump ran; rendered from `op_distribution_comparison.json`, with the totals table, the operators unique to either side, and the per-operator delta table
6. `## Operator Distribution with Compatibility Status`
   - Columns (exact, in order): `Op Type | Domain | Count | Data Types | Recommended Rocm Implementation | Status | Op Description`
7. `### Compatibility Summary`
   - `#### Fully Compatible Operator (<count>)`
   - `#### Partially Compatible Operators (<count>):`
   - `#### Unsupported Operators (<count>):`
8. `Unsupported operator recommendation buckets` (one bucket per recommended path)
9. `## ONNX to Hip to runtime mapping` — rendered from `mapping_chain`; columns `ONNX Op | Domain | Hip Op | Runtime Func | Backend | Instances | Status`
10. Final pointer line: `Detailed compatibility diagnostics are in model_compatibility_details.md`

## model_compatibility_details.md — section order

This file carries only what the main report does not: the evidence behind a non-supported row.

1. Title + metadata, with a pointer back to `model_compatibility_report.md`
2. *(conditional)* `Source: original ONNX, conversion probe skipped` badge when applicable
3. `## Partially compatible details` — columns: `Op Type | Domain | Reason Codes | Reason Texts | Evidence`
4. `## Data quality notes`

## Agent rendering rules

- When you echo the report back to the user, do not re-render or reformat; **read the generated markdown** and quote it.
- Chat Summary **must** equal `model_compatibility_report.md`. If a diagnose finding changes a status, fix the pipeline and re-run; never keep a private "true" percentage that disagrees with the file.
- The percentage in Summary is the headline number. Follow it with the constraint behind each unsupported operator (from [diagnose.md](diagnose.md)), because that is what tells the user whether a fix is small or structural.
- For every `unsupported` recommendation bucket include the closest existing wrapper / entry point (if any) and a short family-based rationale per [reference.md](reference.md).
- When the report header carries the `Source: original ONNX, conversion probe skipped` badge, lead your summary with that caveat: nothing was verified against the compiler.
- When it says support comes from converting each operator on its own, lead with the failing step and keep the numbers qualified: they say which operators are not what stopped the model, not that it runs. A row whose only reason codes are `SLICE_*` rests on that weaker evidence.
