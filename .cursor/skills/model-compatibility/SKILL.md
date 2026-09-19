<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: model-compatibility
description: Analyze ONNX models for AMD HIP / hip-ep compatibility by running the compiler. Dumps the EP-input graph hip-ep receives from ONNX Runtime, runs the real ONNX-to-HIP conversion over it, classifies every operator as supported / partial / unsupported from what the conversion produced, and generates markdown reports. Use when the user asks to analyze an ONNX model, check op compatibility, diagnose EP fallbacks, decide if a model can run on hipdnn EP, or batch-compare multiple ONNX models.
---

# model-compatibility

Report whether hip-ep can run a model, using the compiler itself as the oracle. The pipeline lives under [scripts/](scripts/); this document tells you how to drive it, what each step produces, and how to check it.

The classification never comes from reading conversion source code. It comes from running `convert-onnx-to-hip` over the graph the compiler actually receives:

- an operator still present as `onnx.*` afterwards is **unsupported**;
- an operator that converted but lost an ONNX attribute carrying a non-default value is **partial**;
- everything else is **supported**.

Source is read for two things, neither of which decides support:

- **Explaining a leftover.** "Unsupported" covers two very different findings, and the report must say which one applies — no converter exists, or a converter exists and refused this model's instances (a dtype guard, an operand count). The second is the common case on real models and points at a much smaller fix.
- **Naming the runtime path.** Which runtime function executes a `hip.*` op is written down once in its HIP-to-LLVM lowering, so it is a lookup keyed on an op the conversion actually produced, not an inference about support.

## Gather inputs

| Input | Required | Default if omitted |
|---|---|---|
| `<model.onnx>` | Yes | — |
| `<HipEpPackageRoot>` | Yes, unless `-SkipDump` | `$env:HIP_EP_PACKAGE_ROOT` |
| `<OutputDir>` | No | `$env:TEMP\<meaningful-path-name>_ep_compat` (auto-derived) |

Ask the user only for `<model.onnx>` if it is missing. **Never invent paths**, and do not reuse a path from an earlier chat unless the user confirms it.

The package must contain `bin\hip-onnx-runner.exe` (with `--no-run`, `--provider-options` and `--allow-cpu-fallback`), `bin\hipgpu.dll` and `bin\hip-mlir-opt.exe`. A local build tree works as a package.

## Workflow

### 1. Configure

Package detection order in [scripts/run_ep_compatibility_check.ps1](scripts/run_ep_compatibility_check.ps1):

1. `-HipEpPackageRoot <path>`
2. `$env:HIP_EP_PACKAGE_ROOT`
3. Neither: the orchestrator prints `[HIP_EP_NOT_CONFIGURED] ...` and exits **10**.

**On `[HIP_EP_NOT_CONFIGURED]`** call `AskQuestion` with exactly two options:

- **Provide the package path** — re-run with `-HipEpPackageRoot <path>`; suggest `setx HIP_EP_PACKAGE_ROOT <path>` for persistence.
- **Skip the dump** — re-run with `-SkipDump`. The pipeline then counts operators in the original ONNX but **verifies nothing**; every row is reported `partial / CONVERSION_NOT_PROBED` and the report carries a badge saying so.

`[HIP_EP_RUNNER_TOO_OLD]` (exit 11) means the package predates the runner flags; ask for a newer package.

### 2. Run

```powershell
.\scripts\run_ep_compatibility_check.ps1 -ModelPath <model.onnx>
# add -HipEpPackageRoot <path> when not on the environment
# add -SkipDump when the user chose to skip
# add -OutputDir <dir> only when the user wants a fixed location
```

`OutputDir` is derived from the model path so a directory is still identifiable later: take the last 3 parent segments, drop generic ones (`onnx`, `models`), append a non-generic basename, lowercase, sanitize, suffix `_ep_compat`.

| Input model path | Auto OutputDir |
|---|---|
| `...\blip\onnx\decoder\fp16\model.onnx` | `$env:TEMP\blip_decoder_fp16_ep_compat` |
| `<any-drive>\bar\custom_v2.onnx` | `$env:TEMP\bar_custom_v2_ep_compat` |

Re-runs reuse the directory, and an existing `compatibility\ep_input.mlir` is reused instead of re-dumping. Pass `-OutputDir` explicitly when two distinct models would collide, or when an earlier result must be kept for comparison.

### 3. Steps, outputs and checkpoints

The three markdown files sit at the top of `<OutputDir>`; everything the pipeline produced on the way to them is in `<OutputDir>\compatibility`, named below without that prefix.

| Step | Script | Produces | Check before continuing |
|---|---|---|---|
| 1 dump | `dump_ep_input.ps1` | `ep_input.mlir`, `dump_meta.json` | file is text MLIR starting with `module` and contains `onnx.` ops |
| 2 count | `op_distribution.py` | `step1_original_onnx_ops.json`, `step1_ep_input_ops.json`, `op_distribution_comparison.json` | `_analysis_meta.excluded_carrier_ops` lists the `onnx.Constant` carriers, not compute ops; deltas explain themselves (ORT fusions, `Swish` to `Sigmoid`+`Mul`) |
| 3 convert | `run_convert_probe.ps1` | `converted.mlir`, `convert_log.txt` | probe exited 0; the log's unconverted list matches step 4 |
| — failure | orchestrator, or step 3 | `pipeline_failure.json` | written only when step 1 or 3 failed; step 4 is then skipped |
| 4 analyze | `analyze_conversion.py` | `leftover_onnx.json`, `leftover_reasons.json`, `attr_transfer.json`, `hip_runtime_map.json` | every leftover also appears in the EP-input counts and says whether a converter exists; `unpaired_instances` is small and explainable; every observed `hip.*` op resolves to a runtime function |
| 5 report | `build_report_input.py`, `generate_final_reports.py` | `report_input.json`, and the three markdown files above it | `report_input.json` is validated against its schema as it is written; summary numbers match it |

Each fact is written once. The comparison and the operator distribution are rendered only in the report, and the evidence behind non-supported rows only in the details file; `op_distribution_comparison.json` is the report's input, not a second copy for the reader. `ep_input_loc.mlir` is a location-carrying copy the probe recreates on demand, so it is removed after the analysis reads it.

The whole run is seconds, not minutes; nothing here compiles a kernel or touches the GPU.

### 4. Explain every non-supported entry

For each `unsupported` or `partial` operator, follow [diagnose.md](diagnose.md). The conversion result is trustworthy, but it only says *that* an operator did not convert, not *why*. The playbook finds the bail-out in `lib/Conversion/OnnxToHip/` so the report can state the actual constraint (a dtype the converter rejects, a missing pattern, an attribute it cannot honour).

Report the finding; do not silently promote a leftover to supported. If the diagnose pass shows the pipeline itself is wrong (a pairing miss, a bad default), fix the script and re-run rather than editing the generated markdown.

When the dump or the conversion probe fails, the run still produces a report, and its `## Where it failed` section is the result: the step, the reason quoted from the log, and the source line for a crash. Lead your summary with it. Nothing was verified, so the 0% headline is the absence of a measurement rather than a measurement — say that, and treat the failing step as the model's blocking issue.

### 5. Batch

Drive the loop yourself; ONNX file names are not standardized, so ask for the pattern or default to `*.onnx`:

```powershell
$skill = ".cursor\skills\model-compatibility"
$models = Get-ChildItem -Recurse -Filter *.onnx <models-root> | % FullName
foreach ($m in $models) {
    & "$skill\scripts\run_ep_compatibility_check.ps1" -ModelPath $m -ContinueOnDumpFailure
}
```

Aggregate the per-model reports manually. Consider a dedicated script only after doing this more than three times.

### 6. Answer the user

The deliverable is `<OutputDir>\model_compatibility_report.md`. Quote that file; do not keep a second summary in chat with different numbers. Status display: `full` reads as `supported`, `partial` and `unsupported` keep their names. Append the percentage to `Supported instances`.

Lead with what the model would do on hip-ep: the unsupported instance count is the part that falls back to CPU, and one unsupported operator in a hot path (a quantized matmul, say) matters far more than its type count suggests.

### 7. Validate before responding

- [ ] Checkpoints in the step table are green
- [ ] Diagnose ran for every `unsupported` / `partial` row
- [ ] Chat numbers equal `model_compatibility_report.md`
- [ ] Every operator in the compatibility summary appears in the distribution
- [ ] `-SkipDump` runs carry the unverified badge and the caveat leads your summary

## Agent checklist

```
- [ ] User provided <model.onnx> (no guessed path)
- [ ] hip-ep package configured OR user chose -SkipDump via AskQuestion
- [ ] Ran scripts/run_ep_compatibility_check.ps1
- [ ] Read model_compatibility_report.md from <OutputDir>
- [ ] Ran diagnose.md for every unsupported / partial entry
- [ ] Reported the cause of each unsupported operator, not just its name
- [ ] Chat Summary matches the generated markdown
- [ ] Validation checklist all green
```

## Additional resources

- [reference.md](reference.md) — status semantics, reason codes, ROCm family routing
- [report_template.md](report_template.md) — exact report structure
- [diagnose.md](diagnose.md) — finding why an operator did not convert
- [scripts/](scripts/) — pipeline source; entry point is [scripts/run_ep_compatibility_check.ps1](scripts/run_ep_compatibility_check.ps1)
