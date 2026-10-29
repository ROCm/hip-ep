<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ElementMoveOpInterface and quantize sinking

`ElementMoveOpInterface` (`include/hip/Dialect/IR/HipElementMoveInterface.td`)
marks ops whose result elements are copies of elements of some operands, or
are selected among them by comparison, with no arithmetic. `hip-fuse-rocmlir`
uses it to run those ops on 8-bit values in quantized graphs.

## Why a quantize commutes with these ops

A per-tensor quantize `Q` with scale `s`, zero point `z` and integer type `T`
is elementwise and monotonic. For an op `f` that only moves or selects
elements:

```text
Q(f(x1, x2, ...)) = f(Q(x1), Q(x2), ...)
Q(DQ(a))          = a     when DQ has the same (s, z, T)
```

The second rule is exact on 8-bit values: the round trip error is far below
half a unit.

## The rewrite

`sinkQuantizeThroughElementMoves` (`lib/Dialect/Transforms/SinkQuantize.cpp`)
runs first in `hip-fuse-rocmlir`, so only the rocMLIR path uses it. For each
per-tensor `hip.quantize_linear` with inline scalar parameters, a static shape,
block size 0 and saturation on:

1. It walks back through the ops that implement the interface. Each interior
   op must return `movesElements() == true`, have one result and one reader,
   and have a static shape.
2. Each branch must end at a leaf:
   - a per-tensor dequantize with inline parameters. With the same scale bits,
     zero point value and integer type, the leaf becomes its 8-bit input.
     Otherwise it becomes `Q(DQ(a))` with the quantize's parameters, and the
     producing integer kernel takes that requantize into its epilogue;
   - a `tensor.empty`, which becomes an empty tensor of type `T`.
3. If any node fails, the IR does not change. Otherwise each interior op is
   rebuilt through `cloneWithElementType`, the quantize is replaced, and the
   dead float ops are erased.

The rewrite removes only `DQ -> Q` identities; every `Q -> DQ` pair stays.

A Concat needs no special code: `ConcatDecompose` emits a chain of
`tensor.insert_slice` ops into a `tensor.empty`, and `insert_slice` moves the
elements of both its source and its destination.

## Contract for an op that implements the interface

- Every result element is a copy of one element of the moved operands, or one
  of them selected by comparison (a max). An op whose modes compute new values
  returns false from `movesElements()` for those modes.
- `getMovedOperands()` lists only the operands whose elements reach the
  result. Shapes, indices and offsets stay as they are.
- `cloneWithElementType()` builds the same op on the given moved values, with
  the given result element type. A HIP DPS op writes into a new empty init.
- The 8-bit form must lower both inside a rocMLIR kernel (HipToTosa) and on
  the HIP runtime, because the rebuilt op can stay outside a kernel.

## Ops that implement it

| Op | Moved operands | Note |
|---|---|---|
| `tensor.insert_slice` | source, destination | External model |
| `tensor.extract_slice` | source | External model |
| `tensor.expand_shape`, `tensor.collapse_shape` | source | External models |
| `hip.transpose` | input | |
| `hip.resize` | input | Nearest mode only; the runtime kernel takes i8 and ui8 for nearest |

The tensor external models are registered by
`registerElementMoveExternalModels`, which `hip-fuse-rocmlir` adds to its
dependent dialect registry, so every tool that runs the pass has them.

Ops such as `hip.slice`, `hip.gather`, `hip.pad` and a max `hip.pool` can
implement the interface when their runtime kernels take 8-bit types. A pad
also needs its constant mapped to the zero point.
