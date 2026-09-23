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
| Dump the graph the EP compiles | `ep_dump.py` | `ep_input.mlir` |
| Count operators in it | `mlir_op_parser.py` | `ep_input_ops.json` |
| Count operators in the original | `step1_onnx_parser.py` | `step1_onnx_ops.json` |
| Compare the two | `compare_op_distribution.py` | `op_distribution_comparison.json` |
| Ask the compiler | `probe.py` | `probe_result.json` |
| Assemble | `build_report_input.py` | `report_input.json` |
| Render | `generate_final_reports.py` | the two markdown files |

When the dump fails there is no `ep_input.mlir` to probe, and
`single_op_probe.py` stands in for the first five rows: it builds a one-node
ONNX model per operator, imports each one, and writes the same
`probe_result.json`. See [Evidence levels](#evidence-levels).

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

Being per-operator sets the limit of what stage2 can see. A two-line module
does not exercise memory planning, buffer reuse, or anything else that only
appears at the size of a real graph, so it can only find failures an
operator has on its own. Conversion is checked on the whole graph and
lowering is not: the two halves of `supported` are not equally strong.

A failure anywhere in those 57 passes counts as `lowering-broken`, not just
a missing `HipToLLVM` pattern -- bufferization, shape reification, DPS init
construction and memory planning are all inside stage2. The error text says
which; the status does not.

### Slicing

Both probes build their modules through `mlir_slice.py`, from a whole-graph
dump or a one-node import respectively.

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
is given up on.

When none is usable, a shape is borrowed from the widest ranked operand and
also tried with one and two trailing dimensions dropped, since reductions
need a narrower result than their operand. Being a guess, it counts in one
direction only: a slice that lowers proves the operator is handled, one
that fails proves nothing and is reported as unverified. A guessed shape
can even be semantically impossible, and the compiler does not always
diagnose that gracefully -- another reason its failure is not a verdict.

What remains is operators carrying a region, `Loop` and `If`, whose bodies
the line-based parser cannot lift out.

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
| the importer refused it | — | yes / no | `blocked` / `unsupported`, in the `import-blocked` worklist |

The probe outranks the doc. The doc only separates `blocked` from
`unsupported`.

An unverified lowering is not a sixth status: stage1 converted the operator,
which is a real observation, and only the second half of the check is
missing. It still earns a worklist row carrying the reason, since nothing
else in the report would raise it.

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

### Level C — one-node models

Reached when the EP cannot import the model at all. `single_op_probe.py`
rebuilds each operator as a model of its own and imports that, so the
verdicts still come from compiling. Construction mirrors the MLIR slicer:
weights and values captured from an enclosing scope become graph inputs,
small inline Constants are carried in. A node with a subgraph travels with
it, and whatever that subgraph reads from outside becomes an input too.

`onnx.checker` advises here rather than decides. It rejects a signature of
unknown rank, which the importer accepts and turns into an unranked tensor
the slicer already handles; its complaint is kept only to explain an import
failure if one follows.

Two things level C says that no other level can. An operator the importer
turns away is recorded as `import-blocked`, which is work on the MorphiZen
importer and not on any HIP operator -- calling it `unsupported` would send
someone to the wrong repository. And because a model can fail to import
while its operators are individually fine, the report states the two
conclusions separately: the support percentage is per operator type, not a
verdict on the model.

Level C shares B's blind spot. A one-node model has no surrounding graph,
so fusions never fire and support is a lower bound.

## Counting

Support percentage counts `supported` only. `partial`, `lowering-broken`,
`blocked` and `unsupported` are reported on their own lines.

The denominator excludes weight constants: ONNX initializers become
`onnx.Constant` operations in MLIR, and can outnumber the compute operators
several times over.

`onnx.Return`, `onnx.Yield` and `onnx.NoValue` are not operators and are
excluded. `onnx.Custom` is normalized back to the operator in its
`function_name`, without which the comparison against the original ONNX
misaligns every `com.microsoft` row.

## Recommending a path for `unsupported`

There is no rule table for this, and one keyed on operator names is not
worth building: it goes stale as the runtime changes, and `conv` catches
`ConvTranspose` and `CausalConvWithState` alike.

You are already reading the converter to explain a failure. Name the
closest existing implementation from what is in `lib/Runtime/real/`.
