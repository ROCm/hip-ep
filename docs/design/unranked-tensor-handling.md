<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Unranked tensor handling: cross-repo contract

How "I don't know the shape" is encoded across the importer / EP
boundary, where each side refines it, and why the EP has no
`--onnx-infer-shapes` pass of its own.

This is the cross-repo contract referenced from PR
[ROCm/MorphiZen #228](https://github.com/ROCm/MorphiZen/pull/228) and
from comments in `lib/Conversion/OnnxToHip/LoopOutline.cpp`,
`include/hip/InitAllPasses.h`, and
`lib/Dialect/Transforms/Pipelines.cpp`.

> **TODO (in-flight)**: the importer-side half of this contract ships
> in [MorphiZen PR #228](https://github.com/ROCm/MorphiZen/pull/228).
> Until that PR is merged and the `morphizen` submodule is
> bumped past the merge, the importer in this repo still emits
> `tensor<>` (rank-0) for values whose shape it could not derive,
> not `tensor<*xT>`. The EP-side cleanup (deletion of
> `--onnx-infer-shapes` and `RefineOnnxResultType`) has been done in
> anticipation of #228, but Loop-heavy models (any `onnx.Loop` body
> containing rank-aware ops like `onnx.Concat` / `onnx.Add`) will
> fail conversion until the bump lands. When the submodule is
> bumped: re-run the in-tree LIT suite plus the Python perf tests
> on at least one Loop-heavy model end-to-end, then delete this
> note.

## The problem we are solving

ONNX's protobuf `TensorShapeProto` distinguishes three states:

| ONNX state | Meaning |
|---|---|
| `shape: dim {} dim {} ...` (N entries) | Rank-N, with each dim known or marked dynamic. |
| `shape: {}` (empty repeated field) | Rank-0 scalar. |
| No `shape` field at all | Unknown rank. |

In practice, real HuggingFace exports of vision encoders / text
encoders that contain a counted attention loop (an `onnx.Loop` whose
body is the per-token / per-step attention block) ship hundreds of
body-internal values in the third bucket — "unknown rank" — because
the importer-side shape-inference step has no per-iteration rank to
assign to the body block args, and the inference walk inside the body
fans out from those unranked roots.

MLIR's tensor types distinguish two states:

| MLIR type | Meaning |
|---|---|
| `tensor<d0 x d1 x ... xT>` (`RankedTensorType`) | Rank-N, dims known or `?` (`ShapedType::kDynamic`). |
| `tensor<*xT>` (`UnrankedTensorType`) | Unknown rank. |

The contract: ONNX's "no `shape` field" maps to MLIR's
`UnrankedTensorType`, and ONNX's "empty shape repeated field" maps to
MLIR's rank-0 `RankedTensorType`. Conflating them — historically the
MorphiZen importer did, by emitting `RankedTensorType::get({}, T)` for
both — produces invalid IR like `onnx.Concat(rank-0, rank-3) ->
rank-0` for any op whose source ONNX value lacked a shape field, and
breaks every downstream pass that does rank-aware reasoning (the
`onnx-to-hip` converters, the bufferizer, the loop verifier).

## What each side owns

### Importer side: morphizen (`morphizen`)

- **`mlir-imp/src/mlir-constants.cpp::onnxElementTypeToMlirType`** is
  the boundary that picks the MLIR type from an ONNX element-type +
  shape. A `nullptr` shape pointer means "unknown rank" and emits
  `mlir::UnrankedTensorType`. A non-null pointer to an empty vector
  means "rank-0 scalar" and emits `RankedTensorType::get({}, T)`.
- **`ort-bridge/src/ir-converter-imp.cpp::convert_value_info_proto`**
  is the ORT boundary that decodes the ONNX protobuf. When
  `tensor_type.has_shape() == false`, it passes a `nullptr` shape
  pointer through to `node_arg_new`. When it has a shape (even
  rank-0), it passes a pointer to the (possibly empty) vector.
- **`mlir-imp/src/mlir-graph.cpp::node_arg_new`** dispatches the
  null / non-null shape distinction into the corresponding
  `MLIRNodeArg` constructor. No `CHECK(shape != nullptr)`.

This contract is implemented by [MorphiZen
PR #228](https://github.com/ROCm/MorphiZen/pull/228) (`fix(mlir-imp,
ort-bridge): preserve unranked tensors at the ORT boundary`).

### EP side: this repo (`hip-ep`)

- The EP **does not run an ONNX-level shape inference pass.** ONNX op
  semantics are upstream's responsibility — duplicating per-op shape
  rules in the EP would be reinventing onnx-mlir's
  `--shape-inference` against an unregistered ONNX dialect.
- `--convert-onnx-to-hip` may receive `tensor<*xT>` operands. Most
  converters propagate unranked types through naturally — they read
  operand types via the value's MLIR type, not from a separate
  declared field. Converters that genuinely require a ranked operand
  (e.g. those that index into a specific dim to size a `tensor.empty`)
  bail on unranked input; the rank gap is closed by the next pass.
- **`--hip-infer-shapes`** (lib/Dialect/Transforms/InferShapesPass.cpp)
  refines unranked / under-refined HIP DPS op result types from the
  shapes reified by `ReifyRankedShapedTypeOpInterface` on each op,
  and runs `refineLoopSignatures` (Phase 2) to sync `hip.loop` body
  signatures with v_init types after HIP-dialect conversion. See
  [`hip-shape-inference.md`](hip-shape-inference.md) for the full
  pass design.
- **`--onnx-loop-outline`** (lib/Conversion/OnnxToHip/LoopOutline.cpp)
  drives the body func's v_carry entry args from the source
  `onnx.Loop`'s v_init operand types — not from the under-refined
  body block arg types — so `hip.loop`'s `result_type[i] ==
  v_init[i].type` invariant holds at outline-pass exit even when the
  body block arg is unranked. Cloned body ops keep their source-IR
  result types (typically unranked); `--hip-infer-shapes` refines
  them post-conversion.

## Pipeline ordering

```
              [importer side]                       [EP side]
ONNX file → ort-bridge → mlir-imp → MLIR module → onnx-to-hip-pipeline
                                       (with        ├─ simplify-onnx
                                        tensor<*x>  ├─ hip-add-context-arg
                                        for unknown ├─ onnx-loop-outline
                                        ranks)      ├─ convert-onnx-to-hip
                                                    │  (tail:)
                                                    ├─ hip-infer-shapes  ← single refinement
                                                    ├─ one-shot-bufferize
                                                    ├─ buffer-deallocation
                                                    ├─ hip-pool-allocs
                                                    └─ ...
                                                  → hip-to-llvm-pipeline
                                                    ├─ expand-strided-metadata
                                                    ├─ lower-affine
                                                    ├─ convert-hip-to-llvm
                                                    └─ generate-interface
```

The single refinement step on the EP side is `--hip-infer-shapes`,
which runs once at the head of the ONNX-to-HIP pipeline tail
(immediately after `--convert-onnx-to-hip`, before bufferize). There
is no per-dialect ONNX-level inference pass.

## Why we don't run ONNX shape inference in the EP

It is a fair question — `morphizen-ort-api.cpp` exposes a
`graph_infer_shapes` hook that wraps `morphizen_onnx::shape_inference::InferShapes`,
and the EP could call it before or after import. Three reasons we
deliberately do not:

1. **Loop body subgraphs are the canonical failure mode.** ONNX's
   `infer_shapes` does not recurse into `Loop` / `If` / `Scan` body
   regions in the configurations we ship. Calling it would refine
   main-graph shapes the importer happens to know already, but leave
   the body subgraphs (the actual problem) untouched.
2. **Duplication risk.** Once we have an unranked-aware import path,
   the only shapes left to derive at MLIR level are those that depend
   on operand-type propagation through ops the importer cannot see
   (the body subgraph case above). That work is op-by-op, requires
   knowing each ONNX op's shape contract, and would re-implement in
   C++ what `onnx.shape_inference` does in Python — against an
   unregistered ONNX dialect that lacks the OpInterface scaffolding
   (`InferTypeOpInterface`, `ShapeInferenceOpInterface`) that
   upstream projects (onnx-mlir, torch-mlir, TOSA) rely on.
3. **The HIP-side refinement already covers it.** Once
   `--convert-onnx-to-hip` produces HIP ops, those ops carry
   `ReifyRankedShapedTypeOpInterface` and `--hip-infer-shapes` walks
   them to refine unranked → ranked result types. This works because
   HIP DPS ops have ranked operands (the `outs` operand types come
   from preceding ops or from `tensor.empty` with explicit shape),
   and the result shape is a function of operand shapes via the
   helpers in `lib/Dialect/IR/HipShapeUtils.cpp`. A single pass closes
   the gap for every HIP op the converter emits.

The historical workaround `--onnx-infer-shapes`, with its op-name
keyed rules library (`RefineOnnxResultType.{cpp,h}`), was deleted once
[PR #228](https://github.com/ROCm/MorphiZen/pull/228) landed on the
importer side. That pass:

- Encoded a hand-rolled subset of ONNX shape inference (pointwise
  broadcast, Concat axis-sum, Slice rank-preservation, LayerNorm
  shape-passthrough) — duplicating what ONNX itself already knew.
- Required a per-op-name whitelist as the only way to disambiguate
  "rank-0 placeholder for unknown shape" from "genuine rank-0 scalar"
  — an ambiguity that disappears once the importer encodes them
  differently.
- Was the only EP-side feature whose existence depended on the
  importer being broken.

If a future failure mode genuinely requires ONNX-level shape
inference (e.g. an exporter that ships unranked tensors *and* uses
ONNX ops we have no HIP equivalent for), the right fix is upstream:
either run `onnx.shape_inference.infer_shapes(model, data_prop=True)`
recursively at import (and route the result into MorphiZen), or
extend the importer to emit ranked types where it knows them. Both
are larger changes than this doc covers, but both are smaller than
maintaining a parallel ONNX shape-inference layer in the EP forever.

## Verifying the contract

Two end-to-end signals tell you the contract is healthy:

1. **No `tensor<f16>` (rank-0) in IR dumps for body-internal values.**
   On a Loop-heavy model (canonical: a HuggingFace vision encoder
   with a counted attention loop), dump the post-import IR with
   `HIPDNN_EP_IR_DUMP_PATH=...`. Search for `onnx.Concat`, `onnx.Add`,
   etc. inside outlined body funcs; their operand and result types
   should be `tensor<*xT>` (unranked) or genuinely ranked, never
   rank-0 unless the ONNX source genuinely declared a scalar.

2. **`--hip-infer-shapes` refines all unranked HIP ops.** After the
   `--onnx-to-hip-pipeline` runs, every HIP-dialect op result type
   should be `RankedTensorType` (with `?` dynamic dims allowed). If a
   HIP op result type is `UnrankedTensorType` after pipeline exit,
   either the converter that produced it doesn't propagate operand
   shapes, or `ReifyRankedShapedTypeOpInterface` is missing on that
   op kind. Both are `--hip-infer-shapes`-side bugs to fix; they are
   never resolved by reintroducing ONNX-level inference.

The LIT case `test_loop_outline.mlir`'s `vision_encoder_loop` test
pins the outline-time half of the contract (v_init drives the body
func entry args and `hip.loop` result types even when the body block
arg is unranked).
