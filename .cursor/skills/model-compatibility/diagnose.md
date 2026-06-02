<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Diagnose — false-positive triage for `unsupported` / `partial`

The pipeline's ONNX -> HIP parser ([scripts/step2_1_onnx_to_hip_parser.py](scripts/step2_1_onnx_to_hip_parser.py)) is **not** the source of truth. It uses regex over `lib/Conversion/OnnxToHip/*.cpp` and has known classes of false positives. **Every `unsupported` or `partial` entry in the generated report MUST be run through this playbook before reporting to the user.**

Skipping this step has produced wrong answers in the field — most recently, all 38 instances of `onnx.LayerNormalization` in BLIP fp16 decoder were tagged `unsupported` when the source actually has full E2E support (`LayerNormToHip` -> `hip.layer_norm` -> `wrap_layer_normalization`). The playbook below catches this in ~30 seconds.

## Playbook (run for each unsupported / partial op)

Let `<Op>` be the ONNX op type from the report (e.g. `LayerNormalization`).

### Step 1 — Is there an ONNX -> HIP conversion pattern?

```
rg --pcre2 'RewritePattern\("onnx\.<Op>"' lib/Conversion/OnnxToHip/
rg --pcre2 'function_name["'"'"']?\s*==\s*"<Op>"' lib/Conversion/OnnxToHip/
```

- **Hit** in `RewritePattern("onnx.<Op>", ...)` constructor -> standard-domain ONNX op IS mapped.
- **Hit** in `function_name == "<Op>"` check -> `onnx.Custom`-domain op IS mapped (typically `com.microsoft`).
- **No hit anywhere** -> truly unsupported in conversion layer; go to Step 4.

### Step 2 — Is the target HIP op defined?

Open the matched conversion struct and find what it constructs: either `mlir::hip::<XxxOp>::create(...)` or `mlir::OperationState(loc, "hip.<xxx>")`. Then:

```
rg 'def Hip_<XxxOp>\b' include/hip/Dialect/IR/HipOps.td
rg 'def Hip_\w+Op[^{]*<"<xxx>"' include/hip/Dialect/IR/HipOps.td
```

- **Hit** -> the HIP dialect Op exists. Go to Step 3.
- **No hit** -> conversion exists but target dialect Op missing; this is a real bug in `lib/Conversion`, escalate to dialect owner. Mark report entry as `unsupported (missing HIP dialect Op)`.

### Step 3 — Is the runtime wrapper present?

Find the lowering in `lib/Conversion/HipToLLVM/` for `hip.<xxx>`, then locate its `wrap_*` / `hip_*` runtime symbol:

```
rg 'hip\.<xxx>' lib/Conversion/HipToLLVM/
rg '\bwrap_<xxx>\b|\bhip_<xxx>\b' lib/Runtime/real/
```

- **Hit** in `lib/Runtime/real/` -> end-to-end path exists. **The report entry is a tool-FP. Promote it to `supported` in the user-facing summary.**
- **No hit** -> dialect Op exists but runtime impl missing; mark as `unsupported (missing runtime wrapper)`. Recommended path = the wrapper name itself (extension target).

### Step 4 — Truly unsupported (Step 1 had no hit)

Apply ROCm family routing from [reference.md](reference.md):

1. Identify family (matmul / conv / activation / elementwise / reduction / data-movement / control-flow / norm-composite) based on ONNX schema semantics.
2. Map to recommended path: `MIOpen` / `hipBLASLt` / extension of existing wrapper / `Custom Hip Kernel`.
3. Cite the closest existing wrapper if any (machine matrix in [scripts/unsupported_reco_rules.json](scripts/unsupported_reco_rules.json)).
4. Output: `<Op>` -> recommended path -> closest wrapper -> one-line rationale.

## Partial-status triage

For `partial`, the report already names a reason code from the catalog in [reference.md](reference.md). Confirm by:

- For `EXTRA_ONNX_ATTR_NOT_IN_HIP` / `MISSING_HIP_REQUIRED_ATTR`: open the HIP TableGen def and verify the attribute list; if the attr is genuinely required by the lowering, partial is real. If the attr is purely informational and the lowering ignores it, this is a tool-FP — promote to `supported`.
- For `ONNX_INPUT_BELOW_HIP_MIN` / `ABOVE_HIP_MAX` (and `OUTPUT_*` variants): inspect a real model instance in `compatibility/report_input.json` and compare its operand count to the HIP op's TableGen signature.

### Real-partial examples (do NOT promote)

Some partials are genuinely real even after diagnose. Two recurring cases from BLIP fp16 decoder:

- `Cast` with `EXTRA_ONNX_ATTR_NOT_IN_HIP: saturate` — `hip.cast` does not model `saturate`. Most exports use the default value, so it does not block execution in practice, but the partial status is technically correct.
- `Softmax` with `EXTRA_ONNX_ATTR_NOT_IN_HIP: axis` — `hip.miopen.softmax` hard-codes last-dim row-wise softmax; `axis != -1` would require a different runtime path. Keep `partial`.

When reporting partials to the user, distinguish "attribute is non-default in this model" (real risk) vs "attribute is at default and runtime ignores it" (no practical impact).

## Known parser false-positive classes (regression catalog)

| FP class | Symptom | Cause | Fix landed |
|---|---|---|---|
| Substring-shadowed struct scope | Op marked unsupported despite explicit `RewritePattern("onnx.<X>", ...)` in source | `<X>ToHip` is a suffix of a sibling struct name (e.g. `LayerNormToHip` ⊂ `SimplifiedLayerNormToHip`); regex `LayerNormToHip::matchAndRewrite` matched the longer sibling's body | YES — `(?<!\w)` word boundary in `_scope_to_rewrite_pattern_body`; same-file struct disambiguation |
| Whole-file domain heuristic | Standard-domain op mis-tagged `com.microsoft` | Old `_determine_domain_from_code` returned `com.microsoft` if the substring `com.microsoft` appeared **anywhere** in the file; mixed-domain files (Norm, etc.) tripped this | YES — domain now requires a real `domain_name == "com.microsoft"` check inside the struct scope |
| `OperationState` declaration form | Op shows up with empty / wrong `class_name` in mappings | Regex `OperationState\s*\(` only matched `OperationState(loc, ...)` expression form, not `OperationState state(loc, ...)` declaration form | YES — `OperationState(?:\s+\w+)?\s*\(` matches both |
| First `*Op::create` wins | `hip_op` mis-tagged as a sibling struct's target | `_extract_hip_op_from_code` returned the file's first `*Op::create`, not the struct-scoped one | YES — Mode 2 now passes a scoped body to the extractors |

When you discover a new FP class, add a row here and file a fix in [scripts/step2_1_onnx_to_hip_parser.py](scripts/step2_1_onnx_to_hip_parser.py).

## Walked example — LayerNormalization (2026-06-01)

Report said:
```
LayerNormalization | onnx | 38 | float16 | — | unsupported | ...
```

Playbook:

1. `rg 'RewritePattern\("onnx\.LayerNormalization"' lib/Conversion/OnnxToHip/`
   -> hit in [lib/Conversion/OnnxToHip/NormConversion.cpp](lib/Conversion/OnnxToHip/NormConversion.cpp): `struct LayerNormToHip : RewritePattern("onnx.LayerNormalization", ...)`
2. Find construction site in that struct: `mlir::OperationState state(loc, "hip.layer_norm")`.
   `rg 'def Hip_LayerNormOp\b' include/hip/Dialect/IR/HipOps.td` -> hit at line 575.
3. `rg 'wrap_layer_normalization' lib/Runtime/real/` -> hit in [lib/Runtime/real/layer_normalization.cpp](lib/Runtime/real/layer_normalization.cpp) at `int wrap_layer_normalization(...)`.
4. Conclusion: **tool-FP, promote to `supported`**. Recommended ROCm impl = `MIOpen (wrap_layer_normalization)`.
5. Recorded FP class "Substring-shadowed struct scope" in the table above; fixed in `_scope_to_rewrite_pattern_body`.

After the fix, re-running the pipeline on BLIP fp16 decoder returned `720 / 720 (100.0%)` supported, matching ground truth.

## Output for the user

After running the playbook on all non-supported entries, restate the report's headline numbers reflecting any FP promotions, e.g.:

> The pipeline reported 682 / 720 (94.7%) supported and 38 `LayerNormalization` unsupported. After diagnose, all 38 are a known tool-FP (substring-shadowed struct scope); ground-truth supported = **720 / 720 (100.0%)**. Parser fix target: [scripts/step2_1_onnx_to_hip_parser.py](scripts/step2_1_onnx_to_hip_parser.py).
