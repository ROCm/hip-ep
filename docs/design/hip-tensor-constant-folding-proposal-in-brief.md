<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP Tensor Constant Folding: Proposal in Brief

**Decision requested:** Approve an opt-in compiler pass for exact, inline-only tensor folding. Default enablement is a separate decision after validation. The pass is not yet implemented.

## What exists today vs. what changes

HIP operations with constant-only inputs can still launch GPU kernels on every inference.

| Where it runs | What exists today | Why it does not cover this |
| ------------- | ----------------- | -------------------------- |
| Before hip-ep, on the source graph | Exporters or upstream optimizers may fold constants. | Coverage varies with the frontend and settings; constant-only work can still reach HIP. |
| During ONNX-to-HIP conversion | Targeted ONNX rewrites handle cases such as `Gather(Shape(x), k)` and `ConstantOfShape`. | These recognize specific source ops, not arbitrary HIP tensor branches. Extending this stage would not cover the Torch-to-HIP path. |
| Shared tensor HIP, before externalization | `FoldConstantExtracts` exposes scalar values and `InferShapes` refines result dimensions. | A known shape or scalar does not imply that a tensor's elements are known. Keep these passes early so shape inference can use the scalar facts. |
| ONNX pipeline head and shared tensor HIP tail | The ONNX head runs `FoldBatchNorm` to absorb normalization into Conv weights and bias. The shared tail runs HIP fusion. | Conv still runs on runtime activations. Reuse the constant reader, not the Conv-specific rewrite, to evaluate constant-only branches. |
| MLIR cleanup, including shared HIP after externalization | Canonicalization applies local folds and patterns; CSE merges eligible equivalent ops. | CSE does not compute tensor values. HIP compute ops lack payload folders and HIP has no constant materializer; adding local folders alone would not decide what to store across a branch or enforce storage budgets. |
| LLVM backend, after HIP-to-LLVM lowering | LLVM folds visible scalar arithmetic and eliminates redundant work it can prove equivalent. | The relevant HIP ops are now runtime calls that write destination buffers. LLVM does not evaluate the GPU computation behind those calls. |

**Evidence:** In one saved optimized bitcode artifact, 16 runtime call sites still prepare a fixed attention mask from constant inputs: one each for `Range`, `Less`, `Not`, and `Expand`, plus 12 for `Where`. These are static call sites from an earlier build, not a measured speedup or a guarantee for the current pipeline.

The [full compiler-generated HIP IR capture](../../test/lit/Inputs/hip-tensor-constant-folding-capture.hip.mlir) shows the preparation chain feeding attention. This is the saved import-stage IR, not a dump at the proposed pass insertion point. It references external weight data that is not included, and the installed `hip-mlir-opt` rejects its `hip.gqa` syntax. It lives under LIT `Inputs` as a reference, not a runnable test.

## Proposal

Evaluate supported tensor ops once at compile time when all inputs that determine their result are known. Emit or reuse a `hip.constant` only for results that surviving consumers need. Intermediate tensors stay in compiler memory and are not serialized unless another use still needs them. This is not a fusion rule: the runtime consumer's computation stays the same.

**Why a dedicated pass?** A local fold can compute one op's result. The pass can look across a chain or branches before deciding what to store, what to keep live, and whether removing the runtime work is worth the added constant storage. The graph algorithm is shared across frontends; new coverage comes from adding validated per-op evaluators, not matching model-specific patterns.

## Compiler workflow

Add `hip-fold-tensor-constants` to the shared tensor HIP tail used by the ONNX and Torch entry points, after HIP fusion and before constant externalization:

```text
ONNX or Torch -> shared tensor HIP tail
  -> shape inference -> HIP fusion
  -> NEW: evaluate constants -> plan replacements -> commit safe groups
  -> externalize constants -> existing cleanup -> bufferize -> LLVM
```

This slot keeps existing fusion opportunities intact and lets the pass read inline payloads and retire dead constants before storage is assigned.

**Related work and ordering:**

- [#1553](https://github.com/ROCm/hip-ep/pull/1553) proposes BatchNorm folding in the shared tail. If it lands, keep that fold before tensor constant folding. Phase 1 does not depend on it.
- [#1324](https://github.com/ROCm/hip-ep/pull/1324) proposes device placement for existing constants before externalization. If it lands, test the combined ordering and verify that surviving consumers get device-readable folded constants.
- [#1588](https://github.com/ROCm/hip-ep/pull/1588) evaluates constant dequantized filters inside ConvTranspose decomposition. It is the closest overlap, but serves an earlier, specific rewrite; the proposed shared pass does not replace it.

The pass owns graph traversal, rewrite planning, and budgets. Per-op evaluators own value semantics; reuse `HipConstantData` and HIP shape helpers rather than duplicating shape rules. The existing externalizer owns storage and serialization. Keep canonicalization, CSE, and destination repair in their current order. No runtime, kernel, or ABI changes.

## Phase 1 scope

- Start with validated static views and exact cases of `Range`, `Less`, `Not`, `Expand`, and `Where` from inline constants. For example, `Where` can copy the selected floating-point bits without performing floating-point arithmetic. Check types, layout, shapes, bounds, and any destination values the op reads. Preserve existing validation and error behavior.
- Work within single-block functions. Support shared producers and joins. Skip unknown inputs or unproven semantics. No propagation across control flow.
- Defer file-backed constants, graph-output replacement, and floating-point arithmetic (`Mul`, `Div`, `Sqrt`, reductions) until storage or numerical behavior has been validated separately.

## Algorithm

1. **Evaluate forward without changing IR.** Visit ops in block order. Evaluate each supported op once when its required inputs are known, and memoize results by SSA value. A blocked op stops propagation through that op, not through sibling branches.
2. **Plan rewrites backward.** Group evaluated ops and views connected by SSA dependencies; sharing a literal alone does not join groups. Keep *known value*, *replaceable operand use*, and *removable op* separate. A surviving use must accept the replacement's immutable storage, layout, and placement, including through aliases. A blocked use keeps its producer live. Erase an op only if removal preserves behavior and every use of every result is replaced or belongs to an op already planned dead.
3. **Choose carriers, check budgets, then commit.** Emit only live boundary values. Reuse exact-type, bit-identical dominating carriers when legal. Charge compile-time work and memory before evaluation, and price serialized and resident storage separately before committing. Count only proven retirements; do not credit shared sources across groups. Require removal of at least one runtime compute op. Leave rejected groups unchanged, and do not erase a carrier if doing so could hide a storage validation error.

### Before and after: one constant branch, one runtime input

```text
Before
constants -> A --+
                 +-> Join --+
constants -> B --+          +-> Consumer
runtime input --------------+

After
folded hip.constant --+
                      +-> Consumer
runtime input --------+
```

`A`, `B`, and `Join` disappear only if they have no remaining uses. A blocked side use keeps its producer and required dependencies alive; it doesn't prevent folding the join for a compatible consumer.

## Implementation and acceptance

1. **Capture a pass-off baseline** at the proposed insertion point. Confirm operand types, constant storage, and consumer roles before setting call-removal expectations.
2. **Land the standalone pass and focused LIT tests.** Cover exact values, wide fan-out, a diamond join with a blocked side use, return/view exclusions, carrier reuse, budget refusal, and preserved validation errors.
3. **Wire the default-off option through both frontend entry points.** With the same build and fresh artifacts, compare optimized bitcode call sites and constant payloads. If the fixed-mask chain is present and eligible in the new baseline, verify that its preparation calls disappear, attention still runs, and dead payloads are not serialized.
4. **Gate default enablement on GPU parity and cost.** Prove execution without CPU fallback, then compare warmed-up inference latency, compile time, initialization, and persistent/transient memory with tracing off. No speedup is claimed before measurement.
