<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Report templates

These templates are the contract between the pipeline and the user-facing markdown. **Do not paraphrase**. The pipeline's [scripts/generate_final_reports.py](scripts/generate_final_reports.py) already renders these files at `<OutputDir>/model_compatibility_report.md` and `<OutputDir>/model_compatibility_details.md`. When you read those generated files back to the user, preserve the section order and naming verbatim.

## Non-negotiable rules

1. Never invent operators, counts, mappings, or reasons. Every number must match `compatibility/report_input.json`.
2. Unsupported reason text policy:
   - Compile-time ops: keep the specific compile-time reason text.
   - All others: exactly `No Hip Dialect implementation available.`
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
| `compatibility/unsupported_reco_runtime.json` | Machine-readable unsupported recommendations |

## model_compatibility_report.md — section order

Exact order, do not reorder:

1. `# Model compatibility report`
2. Metadata bullets (one bullet each):
   - `EP input (compatibility target)` — only present when dump ran
   - `Original model`
   - `Generated UTC`
3. *(conditional)* `> **Source:** original ONNX (no EP rewrites)` — when pipeline ran in `-SkipDump` mode (orchestrator injects this badge automatically)
4. `## Summary`
   - Total node instances
   - Supported instances `<n> (<pct>%)`
   - Unsupported instances
   - Total Operator Types
   - Fully Compatible
   - Partially Compatible
   - Unsupported
5. *(conditional)* `## Original vs EP input (operator distribution)` — only when dump ran; embedded from `op_distribution_comparison.json`
6. `## Operator Distribution with Compatibility Status`
   - Columns (exact, in order): `Op Type | Domain | Count | Data Types | Recommended Rocm Implementation | Status | Op Description`
7. `### Compatibility Summary`
   - `#### Fully Compatible Operator (<count>)`
   - `#### Partially Compatible Operators (<count>):`
   - `#### Unsupported Operators (<count>):`
8. `Unsupported operator recommendation buckets` (one bucket per recommended path)
9. `## Hip Ops Summary` — copy the `## Operator Summary` table from `step2_hip_ops.md` when available
10. `## ONNX-HIP-RUNTIME Mapping` — render from `mapping_chain`
11. Final pointer line: `Detailed compatibility diagnostics are in model_compatibility_details.md`

## model_compatibility_details.md — section order

1. Title + metadata
2. *(conditional)* `Source: original ONNX (no EP rewrites)` badge when applicable
3. *(conditional)* `## Original vs EP input (operator distribution)` block
4. `## Supported operators table (full)`
5. `## Partially compatible details` — columns: `Op Type | Domain | Reason Codes | Reason Texts | Evidence`
6. `## Unsupported operators` — columns: `Op Type | Domain | Count | Reason`
7. `## Data quality notes`

## Agent rendering rules

- When you echo the report back to the user, do not re-render or reformat; **read the generated markdown** and paste / quote it.
- When you summarize verbally, the percentage in Summary is the headline number; mention any tool-FP rescued by diagnose and the resulting "true" supported percentage.
- For every `unsupported` recommendation bucket you should include closest existing wrapper / entry point (if any) and short family-based rationale per [reference.md](reference.md).
- When the report header carries the `Source: original ONNX` badge, lead your user-facing summary with that caveat — the numbers do not reflect EP fusions/folds.
