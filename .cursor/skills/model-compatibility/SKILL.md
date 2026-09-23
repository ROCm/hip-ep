<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: model-compatibility
description: Find out which operators in an ONNX model hip-ep can run, and what work the rest needs. Dumps the graph the EP actually compiles, then asks the compiler itself -- one whole-graph pass for whether a converter exists, one per-operator lowering for whether the chain reaches a runtime symbol, and attribute perturbation for whether converters read what the model sets. Falls back to one-node models built from the original ONNX when the EP cannot import the model at all, so the verdicts still come from compiling. Sorts every operator into supported, partial, lowering-broken, blocked or unsupported, each naming a different piece of work, and renders a report ending in a worklist of what to do. Use when asked to analyze an ONNX model, check operator compatibility, decide whether a model can run on hip-ep, work out why the EP falls back to CPU, or plan which operators to add or extend.
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

One command, seconds to a minute depending on the model. Artifacts land
under `<OutputDir>`; `model_compatibility_report.md` is the one to read.

## The five statuses

Each names a different piece of work. This distinction is the reason the
skill exists -- "unsupported" alone does not say whether the job is an
afternoon or a fortnight.

| Status | Meaning | Work |
|---|---|---|
| `supported` | converts and lowers | none |
| `partial` | converts, but a converter ignores an attribute this model sets | teach the converter that attribute |
| `lowering-broken` | converts, but the rest of the pipeline does not finish | read the error: it can be any pass, not only a missing lowering |
| `blocked` | no conversion, but an implementation exists | relax a dtype, shape or attribute restriction |
| `unsupported` | no implementation anywhere | write the operator |

`blocked` and `unsupported` differ by roughly an order of magnitude in
effort. Reporting them as one number misleads planning.

`supported` is not one piece of evidence but two, of unequal strength. The
conversion half is checked on the real graph; the lowering half is checked
on the operator alone, because a whole graph cannot be lowered outside the
EP. So a supported operator is one that converts in context and lowers in
isolation -- a failure that needs the size or the neighbours of a real
model, such as memory planning, is outside what this can see.

An operator can be `supported` and still be marked "lowering unverified".
That means the front end converted it but the probe could not build a
standalone module to check the rest of the chain, usually because the
operator carries a region, as `Loop` and `If` do. The summary counts it as
supported, because that is what was observed, and the worklist carries a
row saying what went wrong so the reader can judge whether it matters.

The worklist has one more entry of its own: **import-blocked**, for an
operator the EP refused before any conversion ran. The status stays
`blocked` or `unsupported` according to the documentation, but the work is
on the MorphiZen ONNX importer, and adding a HIP operator would achieve
nothing. `SequenceAt` and friends land here on a model using sequence
types.

When the EP cannot import the model at all, the skill rebuilds each
operator as a one-node model and probes those instead of falling back to
reading documentation. That is evidence level C, and it comes with a
warning worth repeating to the user: the support percentage is then per
operator type and says nothing about whether the model runs. A model can be
90% supported and still fail to load.

## After the run

Read `model_compatibility_report.md`. It narrows as it goes: how far the
pipeline got, the counts, the full distribution, the same operators grouped
by status, then only those needing work.
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
implementation in `lib/Runtime/real/`. There is no rule table for this.

There is no verification pass to run: the verdicts come from the compiler,
so there is nothing to second-guess.

## Evidence levels

The report states one in its header. Anything below A means a fallback was
taken.

| Level | Condition | Effect |
|---|---|---|
| A | whole-graph probe succeeded | full results |
| B | probe failed; per-operator slices used | under-reports support, since fusions needing context cannot fire |
| C | no whole-graph import; one-node models used | per operator type; says nothing about the model as a whole |
| D | no compilation at all | **cannot detect `blocked`; treat numbers as an upper bound** |

At level D, say so first when reporting: an operator whose implementation
rejects this model reads as supported, which is the failure this skill
exists to prevent. It is reached only when the one-node probe cannot run
either.

At level C, report the two conclusions separately: the operators that are
supported, and the fact that the model does not load. Quoting only the
percentage would be the same failure in a different form.

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
