<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Technical Note: one-shot-bufferize

**Relevant issues:** #003 (register HipDstBufferizableModel), #011 (per-op aliasing)

---

## What it does

`one-shot-bufferize` lowers tensor-typed IR to memref-typed IR. It is the
standard MLIR pass for the tensor → memref transition. "One-shot" means it
analyzes the entire function in a single pass before making any IR changes —
no iterative fixpoint.

Input: ops with `tensor` operands and results
Output: same ops with `memref` operands, `memref.alloc` for every value that
needs a fresh buffer, in-place reuse where safe

---

## Two-phase execution

### Phase 1 — analysis (no IR changes)

For every `OpOperand`, the pass asks: can this operand reuse an existing
buffer, or does it need a fresh allocation?

The decision is driven by two questions:

1. **Aliasing declaration** — does the op's bufferization model declare that
   this operand's buffer may overlap with an `OpResult`?
2. **Safety check** — does the use-def chain allow aliasing without
   corrupting other consumers of the same value?

Both must be true for in-place bufferization. If either fails, a fresh
`memref.alloc` is emitted.

### Phase 2 — rewriting

Using the analysis result, the pass rewrites each op:
- Replace tensor operands with the resolved memref buffers
- Insert `memref.alloc` where a fresh buffer is required
- Replace tensor results with the tied init buffer (for DPS ops)

---

## BufferizableOpInterface

Every op that holds tensor types must have a registered implementation of
`BufferizableOpInterface`. Without one, the pass aborts:

```
error: op does not implement BufferizableOpInterface
```

For DPS (destination-passing style) ops, MLIR provides
`DstBufferizableOpInterfaceExternalModel` as a CRTP base that handles the
DPS contract automatically. The only method requiring a custom implementation
is `bufferize()`.

---

## DPS contract and the init/result alias

For a DPS op such as `hip.relu(%ctx, %input, %output)`:

- `$output` is the `outs`/init operand — the pre-allocated destination
- The tensor result is **tied** to this init operand

`DstBufferizableOpInterfaceExternalModel` declares:

```
getAliasingValues(outs/init operand) → {result}
```

This means: the buffer for `result` IS the buffer for `outs`. No extra
allocation is needed for the result — it reuses the init buffer.

---

## In-place optimization: input aliasing

The default model is conservative for `ins` operands:

```
getAliasingValues(ins[i]) → {}   ← input cannot alias any result
```

This forces a fresh allocation for the init/result buffer in all cases.

To enable in-place bufferization for element-wise ops (where reading and
writing the same buffer is safe), override `getAliasingValues` for the
relevant `ins` operand:

```cpp
AliasingValueList getAliasingValues(OpOperand &opOperand, ...) const override {
  // ins[0] (input) may alias the result — allows in-place write
  if (opOperand.getOperandNumber() == 1)  // ins[0], after ctx at index 0
    return {{op->getResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  return {};
}
```

`isDefinite=false` means "permitted, not required" — the framework decides
whether to actually alias based on the safety check below.

---

## Safety check: single-use analysis

After the aliasing declaration unlocks in-place, the framework verifies
safety by inspecting the use-def chain of the `ins` operand's value.

**Safe (in-place applied):**
```
%x → ReLU → %y → Conv → %z
          ↑ %x has one use; overwriting %x is safe
```
No `memref.alloc` for `%y`. ReLU writes into `%x`'s buffer.

**Unsafe (falls back to fresh alloc):**
```
%x → ReLU  → %y → Conv1 → %z
%x → Conv2 → %w
     ↑ %x has two uses; in-place would corrupt Conv2's input
```
A fresh `memref.alloc` is inserted for `%y`. ReLU writes to it. `%x` is
unchanged. Behavior is identical to the conservative default.

The aliasing declaration has no downside: it is a no-op for multi-use values
and an optimization for single-use values.

---

## Which HIP ops allow input aliasing

| Op | In-place safe? | Reason |
|----|---------------|--------|
| `hip.relu` | Yes | element-wise: reads index i, writes index i |
| `hip.cast` | Yes | element-wise |
| `hip.layer_norm` | Partial | output does not alias input (different semantics) |
| `hip.conv` | No | each output element depends on a window of input elements |
| `hip.gemm` | No | each output element is a dot product over a row/column |
| `hip.skip_simplified_layer_norm` | No | fused op with multiple inputs |

MIOpen's `miopenActivationForward` (used by `hip.relu`) explicitly supports
aliased src/dst pointers, so the runtime constraint is satisfied.

---

## Effect on memory pooling

In-place bufferization reduces the number of `memref.alloc` ops that reach
the `memory-pooling` pass (#010). Fewer allocs → smaller pool.

For a ReLU followed by Conv in a single-use chain:

```
Without in-place:  pool holds %x, %y (relu out), %z (conv out) — 3 slots
With in-place:     pool holds %x/%y (same slot), %z           — 2 slots
```

The pool size reduction is proportional to the number of element-wise ops
whose inputs are single-use in the graph.
