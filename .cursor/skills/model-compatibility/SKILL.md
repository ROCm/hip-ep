<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: model-compatibility
description: Analyze ONNX models for AMD HIP / ROCm EP compatibility against the onnx-hipdnn-ep dialect. Dumps the EP-input graph via VOE when available, classifies every operator as supported / partial / unsupported, generates markdown reports, and verifies every non-supported entry against the actual source in lib/Conversion and include/hip/Dialect/IR/HipOps.td to catch parser false positives. Use when the user asks to analyze an ONNX model, check op compatibility, diagnose EP fallbacks, decide if a model can run on hipdnn EP, or batch-compare multiple ONNX models.
---

# model-compatibility

Run the ONNX -> HIP / ROCm EP compatibility pipeline against a given ONNX model. The pipeline lives entirely under [scripts/](scripts/); this document tells you how to drive it, interpret outputs, and avoid the well-known parser false-positive class.

## Gather inputs

| Input | Required | Default if omitted |
|---|---|---|
| `<model.onnx>` | Yes | — |
| `<VoePackageRoot>` | No (optional) | `$env:VOE_PACKAGE_ROOT` if set; otherwise must use `-SkipDump` |
| `<OutputDir>` | No | `D:\temp\<meaningful-path-name>_ep_compat` (auto-derived from path; see below) |

Before running, ask the user only for `<model.onnx>` if missing. **Never invent paths**. Do not reuse a path from an earlier chat unless the user confirms it again.

## Workflow

### 1. Configure (VOE is optional)

VOE detection order inside [scripts/run_ep_compatibility_check.ps1](scripts/run_ep_compatibility_check.ps1):

1. `-VoePackageRoot <path>` parameter
2. `$env:VOE_PACKAGE_ROOT`
3. Neither: the orchestrator emits `[VOE_NOT_CONFIGURED] ...` to stdout and exits with code **10** (does NOT silently continue).

**When you see `[VOE_NOT_CONFIGURED]`** you MUST call `AskQuestion` with exactly two options:

- **Provide VOE path** — re-run with `-VoePackageRoot <path>`. Also suggest the user run `setx VOE_PACKAGE_ROOT <path>` once for persistence.
- **Skip dump** — re-run with `-SkipDump`. The pipeline then analyzes the **original** ONNX (not the EP-rewritten graph). The generated `model_compatibility_report.md` will carry a `> **Source:** original ONNX (no EP rewrites)` badge at the top so the limitation is impossible to miss.

### 2. Single-model run

```powershell
.\scripts\run_ep_compatibility_check.ps1 -ModelPath <model.onnx>
# add -SkipDump when the user chose to skip dump
# add -VoePackageRoot <path> when not on env
# add -OutputDir <dir> only if the user wants a fixed location
```

The orchestrator auto-derives a human-readable `OutputDir` from the model's path so that closing the chat and returning later still lets you see which model a directory belongs to. The rule: take the last 3 parent path segments, skip generic ones (`onnx`, `models`), append the basename when it is non-generic, lowercase and sanitize, suffix with `_ep_compat`. Examples:

| Input model path | Auto OutputDir |
|---|---|
| `...\blip\onnx\decoder\fp16\model.onnx` | `D:\temp\blip_decoder_fp16_ep_compat` |
| `...\blip\onnx\encoder\fp16\model.onnx` | `D:\temp\blip_encoder_fp16_ep_compat` |
| `D:\bar\custom_v2.onnx` | `D:\temp\bar_custom_v2_ep_compat` |

Re-runs against the same model reuse the same dir. If two genuinely-distinct models would collide on the auto name, pass `-OutputDir <dir>` explicitly.

Key produced artifacts (under `<OutputDir>`):

| File | Purpose |
|---|---|
| `model_compatibility_report.md` | Primary deliverable; the markdown you read back to the user |
| `model_compatibility_details.md` | Full per-operator diagnostics |
| `op_distribution_comparison.md` | Original vs EP-input operator counts (only when dump ran) |
| `compatibility/report_input.json` | Normalized intermediate data (input to report generators) |
| `compatibility/unsupported_reco_runtime.json` | Machine-readable unsupported recommendations |
| `ep_input/onnx.onnx` | EP-rewritten graph (only when dump ran) |
| `pipeline_status.md` | Pipeline run metadata + warnings |

### 3. Triage every non-supported entry (mandatory diagnose pass)

Open `model_compatibility_report.md`. For each operator with status `unsupported` or `partial`, you MUST run the playbook in [diagnose.md](diagnose.md). The parser is **not** a source of truth; it has known false-positive classes. Recent example: the parser mis-tagged `onnx.LayerNormalization` as `unsupported` because of a regex substring collision; the actual source has full support. The diagnose playbook catches this in ~30 seconds via three grep calls against `lib/Conversion`, `include/hip/Dialect/IR/HipOps.td`, and `lib/Runtime/real/`.

Each non-supported entry resolves to one of three labels:

- **truly supported (tool-FP)** — source proves the mapping exists; report as supported and record the parser bug location for future fix in [scripts/step2_1_onnx_to_hip_parser.py](scripts/step2_1_onnx_to_hip_parser.py).
- **partial (real attribute / shape mismatch)** — keep `partial` status; cite the reason code from [reference.md](reference.md).
- **truly unsupported** — apply ROCm family routing from [reference.md](reference.md) (and the machine-readable [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json)) to recommend a wrapper extension target.

### 4. Batch (inline, no extra script)

When the user asks for multiple models, drive the loop yourself. The ONNX file name is **not** standardized (it may be `model.onnx`, `<model_name>.onnx`, `decoder.onnx`, etc.); ask the user for the filter pattern or default to `*.onnx`:

```powershell
$skill = "<repo>\.cursor\skills\model-compatibility"
# Pick ONE of:
#   - Explicit list:           $models = @("D:\a\foo.onnx", "D:\b\bar.onnx")
#   - All .onnx under a root:  $models = Get-ChildItem -Recurse -Filter *.onnx <models-root> | % FullName
#   - Custom pattern:          $models = Get-ChildItem -Recurse -Filter <user-glob> <models-root> | % FullName
foreach ($m in $models) {
    & "$skill\scripts\run_ep_compatibility_check.ps1" -ModelPath $m -ContinueOnDumpFailure
}
```

If the user just points at a directory without specifying a pattern, ask: "filter as `*.onnx` (all ONNX files) or a more specific glob like `decoder*.onnx`?". Aggregate the per-model reports manually. Only consider promoting batch / diff / aggregation to a dedicated script after you have done this loop more than three times.

### 5. Output to the user

Render the markdown templates from [report_template.md](report_template.md) verbatim. Status display rule (mandatory):

- input `full` -> displayed `supported`
- input `partial` -> displayed `partial`
- input `unsupported` -> displayed `unsupported`

In the summary line append the percentage to `Supported instances`, e.g. `682 (94.7%)`.

### 6. Validation before responding

- [ ] Counts in summary equal computed counts from `operator_distribution`
- [ ] Diagnose pass executed for every non-supported entry (no shortcuts)
- [ ] No unsupported non-compile-time line uses any reason text other than `No Hip Dialect implementation available.`
- [ ] Every operator in the compatibility summary appears in operator distribution
- [ ] `mapping_chain` table rows match input JSON exactly
- [ ] If `-SkipDump` mode was used, the report header shows the `Source: original ONNX (no EP rewrites)` badge

## Agent checklist

```
- [ ] User provided <model.onnx> (no guessed path)
- [ ] VOE configured OR user chose -SkipDump via AskQuestion
- [ ] Ran scripts/run_ep_compatibility_check.ps1 with auto OutputDir
- [ ] Read model_compatibility_report.md from <OutputDir>
- [ ] Ran diagnose.md playbook on every unsupported / partial entry
- [ ] Promoted tool-FP entries to supported in the user-facing summary
- [ ] Reported using report_template.md format with correct status display
- [ ] Validation checklist all green
```

## Additional resources

- [reference.md](reference.md) — status semantics, reason code catalog, ROCm family routing matrix
- [report_template.md](report_template.md) — exact markdown templates for the final report (do not paraphrase)
- [diagnose.md](diagnose.md) — false-positive detection playbook (mandatory for every non-supported entry); contains the BLIP fp16 decoder walked example
- [scripts/](scripts/) — full pipeline source; orchestrator entry is [scripts/run_ep_compatibility_check.ps1](scripts/run_ep_compatibility_check.ps1)
