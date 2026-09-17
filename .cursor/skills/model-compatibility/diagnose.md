<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Diagnose: why an operator did not convert

The pipeline runs the real `convert-onnx-to-hip`, so its verdict is what the compiler would do with this model. What it cannot tell you is the reason. A leftover `onnx.*` op means one of:

1. **No converter exists** for that operator at all.
2. **A converter exists but rejected these instances** — a dtype, rank, operand count or attribute value outside what it supports.
3. **An earlier pass changed the graph** so the pattern no longer matched (an unranked value, an unsupported subgraph form).

Cases 2 and 3 are common on real models and read very differently in a report: "hip-ep does not support MatMulNBits" is wrong when the truth is "hip-ep supports MatMulNBits but not with fp32 zero-points".

The pipeline already separates case 1 from cases 2 and 3 in `compatibility/leftover_reasons.json` and puts the candidate constraints in the report's reason text. That ranking is inferred from the observed element types, so this playbook is what turns a candidate into a confirmed cause.

Run it for every `unsupported` and `partial` row before answering.

## 1. Read the instance the compiler saw

`compatibility/leftover_onnx.json` carries samples straight from the converted MLIR:

```json
{
  "key": "com.microsoft.MatMulNBits",
  "count": 224,
  "samples": [
    {
      "node_name": "model/layers.0/self_attn/v_proj/MatMulNBits_node_13",
      "mlir_op": "onnx.Custom",
      "type_signature": "(tensor<?x?x2048xf32>, tensor<512x16x32xui8>, tensor<512x16xf32>, tensor<512x16xf32>) -> tensor<?x?x512xf32>"
    }
  ]
}
```

The type signature is the evidence. Note the operand dtypes and ranks before opening any source.

## 2. Find the converter

`leftover_reasons.json` already names the candidate files. Confirm, and check the dialect side too:

```bash
rg -l '"<OpType>"' lib/Conversion/OnnxToHip/
rg "def Hip_.*Op|hip\.<expected_mnemonic>" include/hip/Dialect/IR/HipOps.td
```

No file and no dialect op means case 1: genuinely unsupported, and the report's `No Hip Dialect implementation available.` is accurate. Anything else means the report carries `CONVERSION_REJECTED_INSTANCES` and you owe the reader the specific constraint.

## 3. Find the bail-out

When a converter exists, look for the guard that rejected the instance. Converters report these through `notifyMatchFailure`, so grep for it in the matching file:

```bash
rg -n "notifyMatchFailure" lib/Conversion/OnnxToHip/<Op>Conversion.cpp
```

Compare each guard against the type signature from step 1. Worked example, MatMulNBits on a 2-bit quantized model:

```cpp
// lib/Conversion/OnnxToHip/MatMulNBitsConversion.cpp
if (elemTy.isInteger(8)) {
  zpElemSize = 1;
} else if (elemTy.isF16()) {
  zpElemSize = 2;
} else {
  // "unsupported zero_points element type: ... Expected i8 (packed uint4) or f16"
  return rewriter.notifyMatchFailure(op, msg);
}
```

The model's zero-points are `tensor<512x16xf32>`, so every instance failed this guard. The report line becomes "MatMulNBits is supported, but not with fp32 zero-points; this model would fall back to CPU for all 224 of them", which points at a concrete fix.

To watch the matcher live on a small reproduction:

```bash
hip-mlir-opt <input>.mlir --onnx-dialect=stub --simplify-onnx --hip-add-context-arg \
  --onnx-loop-outline --onnx-if-outline --hip-infer-loop-body-shapes \
  --convert-onnx-to-hip --debug-only=dialect-conversion
```

## 4. Check a partial

`partial` comes from one of two places, both in `compatibility/attr_transfer.json` and `report_input.json`:

- `PARTIAL_INSTANCE_CONVERSION` — some instances converted, others did not. Same playbook as above; the difference between converted and leftover instances is the interesting part, so compare their type signatures.
- `EXTRA_ONNX_ATTR_NOT_IN_HIP` — the HIP op does not carry an ONNX attribute whose value is not the default. Confirm against the HIP op in `include/hip/Dialect/IR/HipOps.td`: if the attribute really is absent, the op silently ignores that setting, which is a genuine behaviour difference worth reporting.

Attributes whose value equals their schema default are listed under `dropped_default_attrs` and are not partial. Defaults come from the ONNX schema, or from ONNX Runtime's contrib operator registry for `com.microsoft`.

An attribute can be dropped and still be harmless without the report knowing: ORT records no default for some optional contrib attributes, so the value cannot be compared to anything. The reason text prints the value for exactly this case (`rotary_interleaved=0`), and your job is to say whether that value is the operator's no-op setting. Do not add a defaults table to silence it; the value in the report is the evidence a reader needs.

## 5. When the pipeline itself is wrong

The oracle can still be misread by the scripts. Symptoms and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| An operator is `supported` but its `Hip Op` column is empty | its location had no match after conversion; it appears in `unpaired_instances` | usually a fold into a neighbour, which is fine; investigate if the count is large |
| An attribute is reported dropped but the HIP op clearly has it | the attribute was renamed by the converter | record the rename in the diagnose notes; extend the pairing only if it recurs |
| A whole operator type is missing from the report | the MLIR line form is not recognized | check [scripts/mlir_text.py](scripts/mlir_text.py) against the actual line in `ep_input.mlir` |

Fix the script and re-run the pipeline. Never edit the generated markdown to match a conclusion.

## 6. Report it

State the operator, the instance count, and the constraint that blocked it. For example:

> 224 of 617 nodes are `MatMulNBits` and none of them convert: the conversion accepts i8 or f16 zero-points and this model stores them as fp32, so the whole quantized matmul path would fall back to CPU.

Then apply the ROCm routing in [reference.md](reference.md) to say where a fix would live.
