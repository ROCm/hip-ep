<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Reference — how each verdict is reached

Definitions behind the report. Source of truth for status semantics; do not
restate them elsewhere.

## The pipeline

| Step | Script | Produces |
|---|---|---|
| Dump the graph the EP compiles | `dump_ep_input.ps1` | `ep_input.mlir` |
| Count operators in it | `mlir_op_parser.py` | `ep_input_ops.json` |
| Count operators in the original | `step1_onnx_parser.py` | `step1_onnx_ops.json` |
| Compare the two | `compare_op_distribution.py` | `op_distribution_comparison.json` |
| Ask the compiler | `probe.py` | `probe_result.json` |
| Assemble | `build_report_input.py` | `report_input.json` |
| Render | `generate_final_reports.py` | the two markdown files |

## The probe

### stage1 — whole graph, does a converter exist

Runs the first six passes of `--onnx-to-hip-pipeline`, ending at
`convert-onnx-to-hip`, plus `--mlir-print-debuginfo`.

An operator with no converter **does not fail the pass** --
`applyPatternsGreedily` leaves unmatched operations alone -- so one run
covers the whole model. Operations still named `onnx.*` afterwards are the
ones with no converter, or whose matcher declined.

The verdict comes from diffing the IR, not from parsing the
`[convert-onnx-to-hip] N unconverted ...` line. That line is still read, as
a cross-check.

Results pair back to inputs through `loc("<file>":<line>:<col>)`, which is
why the operator counts carry line numbers.

### stage2 — one operator, does it lower

Runs `--onnx-to-hip-pipeline --hip-to-llvm-pipeline` on a single-operator
module, and collects the `wrap_*` symbols it reaches.

Two constraints, both learned the hard way:

- The **full** `--onnx-to-hip-pipeline` is required. stage1's six-pass
  prefix omits bufferization, and lowering straight from tensor-level HIP
  reports `failed to legalize operation` for every operator.
- It must be **per-operator**. Whole-graph lowering fails in constant
  externalization, which needs a FileSystem `hip-mlir-opt` cannot inject.

No operator-specific symbol means the lowering is entirely compile-time,
which is a result, not a failure -- `Reshape` via `tensor.expand_shape` is
the usual case.

### Slicing

| Operand's definition | Treatment | Why |
|---|---|---|
| inline `onnx.Constant` | copied in | converters read the value; `Pow` only decomposes for a constant scalar exponent |
| weight `onnx.Constant` | becomes an argument | the value cannot change the decision, and a memory-address source breaks externalization |
| `onnx.NoValue` | rebuilt in place | `none` cannot be a function argument |
| anything else | becomes an argument | |

`none` results are dropped from the function signature: module metadata
rejects a non-tensor in `@main_graph`.

Which instance gets sliced matters. Shape inference leaves some uses of an
operator unranked and others not, and an unranked type cannot go in
`@main_graph`'s signature, so every instance is tried before the operator
is given up on. On one vision model the first three of 51
`SkipLayerNormalization` uses are unranked and the remaining 48 are usable.

When no instance is usable, a shape is borrowed from the widest ranked
operand, and also tried with one and two trailing dimensions dropped, since
reductions need a narrower result than their operand. That is a guess, so
it only counts in one direction: a slice that lowers proves the operator is
handled, while one that fails proves nothing and is reported as unverified.
A guessed shape can produce a semantically impossible operation -- a
`ReduceMax` with `keepdims = 0` and a result as wide as its operand
segfaults `hip-mlir-opt` about four times in five -- which is exactly the
kind of failure that must not be read as a verdict.

What is left after all that is operators carrying a region, `Loop` and
`If`, whose bodies the line-based parser cannot lift out.

### Attributes

Two steps. The diff finds attributes absent from the conversion output, for
free, from the whole-graph result. Perturbation then says why they are
absent, since a converter that read one and omitted it as a default looks
identical to one that never read it.

| Perturbed result | Verdict |
|---|---|
| output changed | `handled` |
| converter rejected the value | `handled` |
| conversion stopped terminating | `handled` |
| output byte-for-byte identical | **`ignored`** |
| no valid different value exists | `inconclusive` |

Any behaviour difference means the converter read the attribute; an
indifferent converter behaves identically. `Gather.axis` shows the third
case: `0` converts instantly, `1` never finishes.

Perturbation preserves the MLIR type annotation. Rewriting `axis = 0 : si64`
to `axis = 1` drops the signedness and sends the converter down paths it
never takes on real input.

### Attribution

For an operator that did not convert, one operand's element type is changed
at a time. If the conversion then succeeds, that operand is the reason --
a single-variable result naming the fix.

## Status

| stage1 | stage2 | In `docs/supported-operations.md` | Status |
|---|---|---|---|
| converted | ok | — | `supported` |
| converted | ok, no runtime symbol | — | `supported` (compile-time) |
| converted | ok, an ignored attribute the model sets | — | `partial` |
| converted | failed | — | `lowering-broken` |
| not converted | — | yes | `blocked` |
| not converted | — | no | `unsupported` |
| converted | ok | no | `supported`, listed under documentation drift |
| converted | could not be sliced | — | `supported`, marked "lowering unverified" |

The probe outranks the doc. The doc only separates `blocked` from
`unsupported`.

An unverified lowering is not a sixth status. stage1 converted the
operator, and that is a real observation, so it counts as supported; what
is missing is the second half of the check. It gets a worklist row anyway,
carrying the reason and any error, because deciding whether that gap
matters is a person's call and nothing else in the report would raise it.

### Why "the model sets it" matters

ORT fills in ONNX schema defaults during `Graph::Resolve`, so an attribute
present in the EP input does not mean the model author wrote it: `Reshape`
ships with no attributes and arrives carrying `allowzero`. Only the file on
disk distinguishes them.

An ignored attribute the model does not set is still a real gap -- another
model will set it -- so it appears under capability gaps regardless. The
status says whether it bites here.

Without the original model, every ignored attribute counts as a defect and
the evidence note says so.

## Evidence levels

| Level | Condition | Effect |
|---|---|---|
| A | whole-graph probe succeeded | full results |
| B | probe failed; per-operator slices | under-reports; context-dependent fusions cannot fire |
| C | no whole-graph MLIR; single-operator models from the original ONNX | independent per-operator results |
| D | no compilation | **cannot detect `blocked`** |

D must be stated prominently. `MatMulNBits` is listed as supported in
`docs/supported-operations.md`, so at level D its 224 blocked instances read
as working.

## Counting

Support percentage counts `supported` only. `partial`, `lowering-broken`,
`blocked` and `unsupported` are reported on their own lines.

The denominator excludes weight constants: ONNX initializers become
`onnx.Constant` operations in MLIR, and on a 1.8B model that is 805 of 809
of them -- more than the compute operators put together.

`onnx.Return`, `onnx.Yield` and `onnx.NoValue` are not operators and are
excluded. `onnx.Custom` is normalized back to the operator in its
`function_name`, without which the comparison against the original ONNX
misaligns every `com.microsoft` row.

## Recommending a path for `unsupported`

There is no rule table for this; the recommendation is yours to make while
filling in the root cause.

A keyword-matched one used to exist and was removed. It aged badly -- four
of its ten families still pointed at MIOpen after the dependency was taken
out of the tree, so it recommended paths that no longer existed. Matching on
the operator name is also weak: `conv` catches `ConvTranspose` and
`CausalConvWithState` alike.

You are already reading the converter to explain a failure. Naming the
closest existing implementation from what is actually in
`lib/Runtime/real/` is both more accurate and cannot go stale.
