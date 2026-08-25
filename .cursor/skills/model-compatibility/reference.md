<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Reference — compatibility rules, reason codes, recommendation routing

Canonical definitions consumed by SKILL.md, diagnose.md, and report_template.md. **Source of truth** for status semantics and reason text wording — do not paraphrase elsewhere.

## Status definitions (data layer)

`report_input.json` stores one of three canonical statuses per `(onnx_op, domain)`:

- `full` — mapped to a HIP op (or a compile-time `tensor.*` op) and no per-instance schema mismatch found.
- `partial` — mapped to a HIP op, but per-instance schema checks found mismatch.
- `unsupported` — no ONNX -> HIP mapping found in the parsed conversion patterns.

## Status display rule (report layer)

| Data status | Displayed in report as |
|---|---|
| `full` | `supported` |
| `partial` | `partial` |
| `unsupported` | `unsupported` |

## Compile-time operators

When a mapping's `hip_op` starts with `tensor.` (e.g. `tensor.expand_shape`, `tensor.collapse_shape`):

- status = `full`
- reason code = `COMPILE_TIME_TENSOR_OP`
- reason text = `Handled at compile time.`

## Non-compile-time per-instance schema checks

For each `(onnx_op, domain)` mapped to a non-`tensor.*` HIP op, the pipeline runs three checks:

1. **Input edge compatibility**
   - HIP input bounds from TableGen:
     - lower bound = required non-`ctx` operands
     - upper bound = total non-`ctx` operands unless variadic
   - reason codes: `ONNX_INPUT_BELOW_HIP_MIN`, `ONNX_INPUT_ABOVE_HIP_MAX`
2. **Output edge compatibility**
   - HIP output bounds same way.
   - reason codes: `ONNX_OUTPUT_BELOW_HIP_MIN`, `ONNX_OUTPUT_ABOVE_HIP_MAX`
3. **Attribute compatibility**
   - Required attrs = TD non-optional attrs + strict overrides from `compatibility_attr_rules.json` (key `strict_required_attrs`).
   - No strict attrs are hardcoded.
   - reason codes: `MISSING_HIP_REQUIRED_ATTR`, `EXTRA_ONNX_ATTR_NOT_IN_HIP`

Any reason code present => `partial`. No reason code => `full`.

## Unsupported rule

No ONNX -> HIP mapping found for `(onnx_op, domain)`:

- status = `unsupported`
- reason code = `NO_HIP_DIALECT_IMPL`
- reason text:
  - if op is compile-time classifiable elsewhere: keep that specific text
  - otherwise: **exactly** `No Hip Dialect implementation available.`

## Reason code catalog

```
NO_HIP_DIALECT_IMPL
COMPILE_TIME_TENSOR_OP
MISSING_HIP_REQUIRED_ATTR
EXTRA_ONNX_ATTR_NOT_IN_HIP
ONNX_INPUT_BELOW_HIP_MIN
ONNX_INPUT_ABOVE_HIP_MAX
ONNX_OUTPUT_BELOW_HIP_MIN
ONNX_OUTPUT_ABOVE_HIP_MAX
```

## Recommended ROCm implementation (report-layer inference)

`Recommended Rocm Implementation` is **agent-inferred at render time** from normalized data; it is NOT stored in `report_input.json`.

### For `full` / `supported` / `partial`

In order, first match wins:

1. `hip_op` starts with `tensor.` -> `Compile Time Optimization`
2. `backend` AND `runtime_func` exist -> `` `<backend>` (`<runtime_func>`) ``
3. only `backend` exists -> `<backend>`
4. only `runtime_func` exists -> `` `<runtime_func>` ``
5. else -> `Unknown`

### For `unsupported`

1. If reason text indicates compile-time handling -> `Compile Time Optimization`
2. Otherwise run capability-driven recommendation:
   1. Infer op family from ONNX semantics (`op_type` + schema description)
   2. Build capability inventory from `step2_3_backend_analysis.json`, runtime wrappers in `lib/Runtime/real/`, and [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json)
   3. Map family to nearest available ROCm path and name extension target wrapper
3. If no feasible ROCm/library match -> `Custom Hip Kernel`

For every unsupported op recommendation include: recommended path, closest existing wrapper / entry point (if any), short rationale.

## ROCm family routing matrix

Machine-readable form: [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json). Human-readable summary:

| # | Family | Preferred path | Fallback |
|---|---|---|---|
| 1 | Matrix multiplication (`matmul`, `gemm`, batched dense linear) | `hipBLASLt` — extend `wrap_hipblasLtMatmul` | `Custom Hip Kernel` |
| 2 | Convolution (`conv`, depthwise / pointwise variants) | `MIOpen` — extend convolution wrapper | `Custom Hip Kernel` |
| 3 | Activation (`relu`, `sigmoid`, `tanh`, `softplus`, similar unary) | `MIOpen` activation wrapper extension | `Custom Hip Kernel` |
| 4 | Elementwise arith / cmp / logical | `MIOpen` op-tensor path when representable by `wrap_elementwise` | `Custom Hip Kernel` |
| 5 | Reduction (`reduce_*`, cumulative reductions) | extend existing reduction runtime path (`wrap_reduce_sum` family) | `Custom Hip Kernel` |
| 6 | Indexing / data movement (`gather / scatter / slice / split / tile / pad / concat / expand`) | reuse / extend existing custom data-movement kernels | `Custom Hip Kernel` |
| 7 | Control flow / stateful (`loop / if / scan`) | graph-level lowering / runtime orchestration | `Custom Hip Kernel` (unless compile-time eliminable) |
| 8 | Normalization / composite blocks | decompose into supported primitives if possible, else extend `wrap_miopen*LayerNorm*` family | `Custom Hip Kernel` |

### Known runtime capabilities (capability inventory examples)

- `wrap_elementwise` — elementwise add / mul / min / max
- `wrap_miopenActivationForward` — activation-family elementwise ops
- `miopenActivationPOWER` — can express `Reciprocal` / `Sqrt` by parameterization
- `wrap_miopenConvolutionForward` — convolution family
- `wrap_hipblasLtMatmul` — matmul family
- `wrap_miopenT5LayerNormForward` — RMS / T5 layer norm
- `wrap_layer_normalization` — standard ONNX-17 LayerNormalization (mean + var)
- `wrap_skip_simplified_layer_norm` — Microsoft SkipSimplifiedLayerNormalization fusion
- `wrap_reduce_sum` — current custom reduction path

## Validation checklist (final pass before responding to user)

- Counts in summary equal computed counts from `operator_distribution`
- No unsupported non-compile-time line uses text other than `No Hip Dialect implementation available.`
- Every operator in compatibility summary appears in operator distribution
- `mapping_chain` table rows exactly match input rows
- No extra sections beyond template order
- **Diagnose pass executed for every non-supported entry** (see [diagnose.md](diagnose.md))
- If `-SkipDump` mode was used, report header carries `Source: original ONNX (no EP rewrites)` badge
