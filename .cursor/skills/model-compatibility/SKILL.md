<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: model-compatibility
description: Find out which operators in an ONNX model hip-ep can run, and what work the rest needs. Dumps the graph the EP actually compiles, then asks the compiler itself -- one whole-graph pass for whether a converter exists, one per-operator lowering for whether the chain reaches a runtime symbol, and attribute perturbation for whether converters read what the model sets. Sorts every operator into supported, partial, lowering-broken, blocked or unsupported, each naming a different piece of work, and renders a report that leads with that worklist. Use when asked to analyze an ONNX model, check operator compatibility, decide whether a model can run on hip-ep, work out why the EP falls back to CPU, or plan which operators to add or extend.
---

# model-compatibility

Answers two questions about a model: which operators run today, and for the
rest, what specifically needs doing. Support is decided by compiling, not by
pattern-matching source, so a converter that exists but rejects this model's
variant is reported as such rather than as working.

## Inputs

| Input | Required | Default |
|---|---|---|
| `<model.onnx>` | yes | — |
| `<GpuTestPackageRoot>` | yes | `$env:GPU_TEST_PACKAGE_ROOT` |
| `<OutputDir>` | no | `$env:TEMP\<derived-from-path>_ep_compat` |

Ask for the model path if it is missing. **Never invent one**, and do not
reuse a path from an earlier conversation without confirming it.

The package supplies `hip-onnx-runner.exe` and `hip-mlir-opt.exe`; no build
tree or GPU is needed. Exit code **10** with
`[GPU_TEST_PACKAGE_NOT_CONFIGURED]` means the package was not found -- ask
the user for the path, and suggest `setx GPU_TEST_PACKAGE_ROOT <path>` so it
persists.

## Run

```powershell
.\scripts\run_ep_compatibility_check.ps1 -ModelPath <model.onnx>
# -GpuTestPackageRoot <path>   when not in the environment
# -OutputDir <dir>             to pin the location
# -SkipDump -EpMlirPath <mlir> to reuse an existing dump
```

One command, about two minutes on a 1.8B model. Artifacts land under
`<OutputDir>`; `model_compatibility_report.md` is the one to read.

## The five statuses

Each names a different piece of work. This distinction is the reason the
skill exists -- "unsupported" alone does not say whether the job is an
afternoon or a fortnight.

| Status | Meaning | Work |
|---|---|---|
| `supported` | converts and lowers | none |
| `partial` | converts, but a converter ignores an attribute this model sets | teach the converter that attribute |
| `lowering-broken` | front end converts, back end does not follow | add the HipToLLVM lowering or runtime function |
| `blocked` | no conversion, but an implementation exists | relax a dtype, shape or attribute restriction |
| `unsupported` | no implementation anywhere | write the operator |

`blocked` and `unsupported` differ by roughly an order of magnitude in
effort. Reporting them as one number misleads planning.

An operator can be `supported` and still be marked "lowering unverified".
That means the front end converted it but the probe could not build a
standalone module to check the rest of the chain, usually because the
operator carries a region, as `Loop` and `If` do. The summary counts it as
supported, because that is what was observed, and the worklist carries a
row saying what went wrong so the reader can judge whether it matters.

## After the run

Read `model_compatibility_report.md`. It builds up in four steps: the
counts, the full distribution, the operators grouped by status, then a
worklist of what is not simply supported and why.
`model_compatibility_details.md` carries the evidence behind each worklist
row -- signature, dtypes, shapes, attributes, and for a blocked operator
which operand the converter objects to, established by changing one
operand's type at a time.

Two things the report cannot produce, because they mean reading the
converter. Add them for every `blocked`, `lowering-broken` and `partial`
entry:

- **Where the converter is** — search `lib/Conversion/OnnxToHip/` for the
  operator name, usually `<Op>Conversion.cpp`.
- **Which check rejects it** — read the matcher and quote the line, so the
  fix is obvious.

For an `unsupported` operator, also name the closest existing
implementation in `lib/Runtime/real/`. There is no rule table for this; a
keyword-matched one existed and was removed after it went stale.

There is no verification pass to run. Earlier versions inferred support with
regexes over C++ and required every finding to be re-checked by hand; the
compiler now answers directly.

## Evidence levels

The report states one in its header. Anything below A means a fallback was
taken.

| Level | Condition | Effect |
|---|---|---|
| A | whole-graph probe succeeded | full results |
| B | probe failed; per-operator slices used | under-reports support, since fusions needing context cannot fire |
| C | no whole-graph MLIR; single-operator models used | per-operator results, independent failures |
| D | no compilation at all | **cannot detect `blocked`; treat numbers as an upper bound** |

At level D, say so first when reporting. An operator whose implementation
rejects this model reads as supported, which is the failure mode the
compiler-based approach exists to prevent.

## Reporting back

Read the generated markdown; do not re-derive the numbers. Lead with the
supported percentage and the worklist, in that order. Mention the evidence
level only when it is not A.

## Batch

Drive the loop; there is no batch script. Operator names vary between
exports, so ask for the filter or default to `*.onnx`.

```powershell
$skill = ".cursor\skills\model-compatibility"
$models = Get-ChildItem -Recurse -Filter *.onnx <models-root> | % FullName
foreach ($m in $models) {
    & "$skill\scripts\run_ep_compatibility_check.ps1" -ModelPath $m -ContinueOnDumpFailure
}
```

The probe dominates the runtime, so batches are slow. Results are per
operator signature, not per model, and could be cached across models -- do
that only after the loop has proved itself more than a few times.

## Checklist

```
- [ ] Model path came from the user
- [ ] gpu-test-package resolved (or the user supplied one after exit 10)
- [ ] Ran run_ep_compatibility_check.ps1
- [ ] Read model_compatibility_report.md
- [ ] Filled in converter source and root cause for every blocked,
      lowering-broken and partial entry
- [ ] Reported the evidence level if it was not A
- [ ] At level D, led with the upper-bound caveat
```

## Also here

- [reference.md](reference.md) — how each status is decided, and what the probe does
- [report_template.md](report_template.md) — report structure
- [scripts/](scripts/) — pipeline; entry point is
  [scripts/run_ep_compatibility_check.ps1](scripts/run_ep_compatibility_check.ps1)
