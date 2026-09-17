<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Reference — compatibility rules, reason codes, recommendation routing

Canonical definitions consumed by SKILL.md, diagnose.md, and report_template.md. **Source of truth** for status semantics and reason text wording — do not paraphrase elsewhere.

## Status definitions (data layer)

`report_input.json` stores one of three canonical statuses per `(onnx_op, domain)`, all derived from running `convert-onnx-to-hip` over the compiler-input MLIR:

- `full` — every instance converted, and every ONNX attribute with a non-default value reached the resulting op.
- `partial` — some instances did not convert, or an attribute with a non-default value was dropped.
- `unsupported` — no instance converted; the operator is still `onnx.*` in the converted module.

## Status display rule (report layer)

| Data status | Displayed in report as |
|---|---|
| `full` | `supported` |
| `partial` | `partial` |
| `unsupported` | `unsupported` |

## Compile-time operators

When every op an ONNX operator converted into is outside the `hip.` dialect (`tensor.expand_shape`, `arith.*`, and similar):

- status = `full`
- reason code = `COMPILE_TIME_TENSOR_OP`
- reason text = `Handled at compile time.`

## Attribute transfer check

Instances are paired pre- and post-conversion by their MLIR location, then each ONNX attribute is looked for on the ops that replaced the node.

- An attribute that landed is fine.
- An attribute that did not land, whose value equals the operator's default, is recorded under `dropped_default_attrs` and does **not** affect status: dropping a default changes no behaviour. Defaults come from the ONNX schema, or from [scripts/contrib_attr_defaults.json](scripts/contrib_attr_defaults.json) for contrib operators the `onnx` package has no schema for.
- An attribute that did not land with a non-default value gives reason code `EXTRA_ONNX_ATTR_NOT_IN_HIP` and status `partial`.

Bookkeeping attributes (`onnx_node_name`, `node.outputs`, and the `function_name` / `domain_name` selectors on `onnx.Custom`) are never treated as operator attributes.

Instances with no post-conversion location match are counted in `unpaired_instances` and listed under the details report's data-quality notes rather than being silently treated as clean.

## Unsupported rule

Every instance of `(onnx_op, domain)` is still `onnx.*` after conversion, so status = `unsupported`. Which reason code applies depends on whether the operator has a converter at all, because the two cases need different work:

| Situation | Reason code | Reason text |
|---|---|---|
| No converter matches the operator name | `NO_HIP_DIALECT_IMPL` | **exactly** `No Hip Dialect implementation available.` |
| A converter matches but refused every instance | `CONVERSION_REJECTED_INSTANCES` | names the converter file, the operand element types in this model, and the candidate constraints with file and line |

[scripts/explain_leftovers.py](scripts/explain_leftovers.py) makes that distinction. It finds converters by the quoted operator name they match on (`"MatMulNBits"`, `"onnx.Cast"`), then lists the messages that converter can pass to `notifyMatchFailure`, ordered so constraints the observed element types contradict come first.

Those candidates are a hint, not a verdict: `notifyMatchFailure` messages are compiled out of a release build, so no reason reaches the log at runtime and the ranking is inferred from types. Confirm the real constraint with [diagnose.md](diagnose.md) before telling the user.

When only some instances are left over, status is `partial` with `PARTIAL_INSTANCE_CONVERSION`.

## Reason code catalog

```
NO_HIP_DIALECT_IMPL
CONVERSION_REJECTED_INSTANCES
COMPILE_TIME_TENSOR_OP
EXTRA_ONNX_ATTR_NOT_IN_HIP
PARTIAL_INSTANCE_CONVERSION
CONVERSION_NOT_PROBED
```

`CONVERSION_NOT_PROBED` only appears in `-SkipDump` runs, where nothing was verified.

## Recommended ROCm implementation (report-layer inference)

`Recommended Rocm Implementation` is **agent-inferred at render time** from normalized data; it is NOT stored in `report_input.json`.

### For `full` / `supported` / `partial`

The conversion already chose the implementation, so the column reports what it produced rather than what a rule table would suggest. In order, first match wins:

1. `hip_op` is outside the `hip.` dialect -> `Compile Time Optimization`
2. `hip_op` resolves to a runtime function -> `` <backend> (`<runtime_func>`) ``
3. `hip_op` with no runtime entry -> ``Hip Dialect (`<hip_op>`)``
4. no `hip_op` (the instance had no location match) -> `Unknown`

[scripts/hip_runtime_map.py](scripts/hip_runtime_map.py) builds that mapping from the HIP-to-LLVM lowering, which names one symbol constant per op, and reads the backend off the wrapper's own implementation: a file calling `hipblasLt*` is hipBLASLt, one calling `hipdnn*` is hipDNN, one launching custom kernels is a custom kernel. A wrapper can be several at once (`hipBLASLt + Custom Hip Kernel` for GQA and MatMulNBits, which drive hipBLASLt for their matmuls and custom kernels around them), and one that touches no library and no kernel is a `Runtime helper`.

This is source reading, but it answers a different question from support: the key is a `hip.*` op the conversion actually produced, and a wrong lookup shows up as an unresolved op rather than as a false "supported".

### For `unsupported`

1. Reason code `CONVERSION_REJECTED_INSTANCES` -> `Extend the existing conversion`
2. If reason text indicates compile-time handling -> `Compile Time Optimization`
3. Otherwise run capability-driven recommendation:
   1. Infer op family from ONNX semantics (`op_type` + schema description)
   2. Build the capability inventory from the runtime wrappers in `lib/Runtime/real/` and [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json)
   3. Map family to nearest available ROCm path and name the extension target wrapper
4. If no feasible ROCm/library match -> `Custom Hip Kernel`

For `Extend the existing conversion`, name the converter file and the constraint that has to be widened, so the reader can see the work is a guard and its runtime path rather than a new kernel.

For every unsupported op recommendation include: recommended path, closest existing wrapper / entry point (if any), short rationale.

## ROCm family routing matrix

Machine-readable form: [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json). Human-readable summary:

| # | Family | Preferred path | Fallback |
|---|---|---|---|
| 1 | Matrix multiplication (`matmul`, `gemm`, batched dense linear) | `hipBLASLt` — extend `wrap_hipblasLtMatmul` | `Custom Hip Kernel` |
| 2 | Convolution (`conv`, depthwise / pointwise variants) | extend `wrap_conv` / `wrap_conv_transpose` (in-tree `hip_conv` kernels) | new `Custom Hip Kernel` |
| 3 | Activation (`relu`, `sigmoid`, `tanh`, `softplus`, similar unary) | extend the in-tree activation kernels (`wrap_gelu` / `wrap_softplus` / `wrap_leaky_relu` family) | new `Custom Hip Kernel` |
| 4 | Elementwise arith / cmp / logical | `wrap_elementwise` when the op is representable as add / mul / min / max | new `Custom Hip Kernel` |
| 5 | Reduction (`reduce_*`, cumulative reductions) | extend existing reduction runtime path (`wrap_reduce_sum` family) | `Custom Hip Kernel` |
| 6 | Indexing / data movement (`gather / scatter / slice / split / tile / pad / concat / expand`) | reuse / extend existing custom data-movement kernels | `Custom Hip Kernel` |
| 7 | Control flow / stateful (`loop / if / scan`) | graph-level lowering / runtime orchestration | `Custom Hip Kernel` (unless compile-time eliminable) |
| 8 | Normalization / composite blocks | decompose into supported primitives if possible, else extend the `wrap_layer_normalization` / `wrap_rms_norm` family | new `Custom Hip Kernel` |

### Known runtime capabilities (capability inventory examples)

- `wrap_elementwise` — elementwise add / mul / min / max, with per-axis broadcast
- `wrap_gelu` / `wrap_softplus` / `wrap_leaky_relu` — activation-family elementwise ops
- `wrap_power` — `Reciprocal` and `Sqrt` only; any other exponent is rejected
- `wrap_conv` — forward convolution family (in-tree `hip_conv` kernel)
- `wrap_conv_transpose` — ConvTranspose
- `wrap_hipblasLtMatmul` — matmul family
- `wrap_rms_norm` — RMS / simplified layer norm (custom HIP kernel)
- `wrap_layer_normalization` — standard ONNX-17 LayerNormalization (mean + var)
- `wrap_skip_simplified_layer_norm` — Microsoft SkipSimplifiedLayerNormalization fusion
- `wrap_reduce_sum` — current custom reduction path

## Validation checklist (final pass before responding to user)

- Counts in summary equal computed counts from `operator_distribution`
- No unsupported non-compile-time line uses text other than `No Hip Dialect implementation available.`
- Every operator in compatibility summary appears in operator distribution
- `mapping_chain` table rows exactly match input rows
- No extra sections beyond template order
- **Diagnose pass executed for every non-supported entry** (see [diagnose.md](diagnose.md)), and its finding is in the answer
- If `-SkipDump` mode was used, the report header carries the `Source: original ONNX, conversion probe skipped` badge
