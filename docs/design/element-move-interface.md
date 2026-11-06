<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ElementMoveOpInterface, quantize sinking and the cast round-trip fold

`ElementMoveOpInterface` (`include/hip/Dialect/IR/HipElementMoveInterface.td`)
marks ops whose result elements are copies of elements of some operands, or
are selected among them by comparison, with no arithmetic. `hip-fuse-rocmlir`
uses it to run those ops on 8-bit values in quantized graphs. The `hip.cast`
canonicalizer uses it to run those ops on the narrow float type of a cast
round trip.

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

## Rebias of ui8 trees to i8

rocMLIR's integer conv and gemm take only signed bytes, so a ui8 input costs
the kernel a widen, a subtraction of 128 and a narrow on each load.
`rebiasUnsignedQuantizedTrees` (`lib/Dialect/Transforms/RebiasUnsignedQuantize.cpp`)
runs right after the sinking and removes that cost where it can:

```text
Q_i8(x, s, z - 128)     = Q_ui8(x, s, z) - 128
DQ_i8(q - 128, s, z - 128) = DQ_ui8(q, s, z)
```

It collects each connected set of static ui8 values:

- producers: per-tensor `hip.quantize_linear` ops with an inline scalar zero
  point (or none, which is 0), and the results of `hip.qlinear_conv` and
  `hip.qlinear_matmul` ops that will become integer anchors;
- interior: the ops that implement this interface, in both directions, with
  `tensor.empty` destinations replaced by i8 ones;
- readers: per-tensor `hip.dequantize_linear` ops with an inline scalar zero
  point, and the data operand of the QLinear anchors.

If any value has another producer or reader (a function argument, a return,
`hip.qadd`, a pool), or if no reader feeds an integer anchor, the set keeps
ui8. Otherwise every value becomes i8 and every zero point in the set drops by
128. The QLinear and dequantized-conv planners compute the same correction
`x_zp - offset` for both types, so the integer math does not change.

## Cast round trip through element moves

An exporter often wraps an op that has no low-precision form in a pair of
casts. The op then moves wide elements that hold only narrow values:

```text
%w = hip.cast %x  : f16 -> f32
%r = hip.resize %w {mode = 0}      // nearest
%y = hip.cast %r  : f32 -> f16
```

An exact widening (f16 or bf16 to f32 or f64, f32 to f64) commutes with an op
that only moves elements, and the narrow cast of a widened value gives the
value back:

```text
narrow(f(widen(x))) = f(x)
```

`FoldCastRoundTripThroughElementMoves` (`lib/Dialect/IR/HipCastCanonicalize.cpp`)
is a `hip.cast` canonicalization pattern. From the outer cast, it walks back
through a chain of ops that implement the interface. Each op in the chain
must return `movesElements() == true`, have one result, one moved operand,
one reader and a static shape. The chain must end at an inner `hip.cast`
with one reader. The pattern applies only if:

- the chain has at least one op;
- the inner cast is an exact float widening;
- the outer cast gives back the type of the inner cast input.

The rewrite rebuilds each op of the chain on the narrow type through
`cloneWithElementType`, and erases the two casts. A narrowing inner cast, an
integer source, a linear resize or a value with a second reader stops the
fold, because the result can change in those cases.

The pattern runs in every canonicalizer run, on the default path and on the
rocMLIR path. On the rocMLIR path, the canonicalizer after
`hip-decompose-conv-transpose` folds the casts before `hip-fuse-rocmlir`
plans the kernels.

## Contract for an op that implements the interface

- Every result element is a copy of one element of the moved operands, or one
  of them selected by comparison (a max). An op whose modes compute new values
  returns false from `movesElements()` for those modes.
- `getMovedOperands()` lists only the operands whose elements reach the
  result. Shapes, indices and offsets stay as they are.
- `cloneWithElementType()` builds the same op on the given moved values, with
  the given result element type. A HIP DPS op writes into a new empty init.
- Every element type that a caller rebuilds the op on must lower both inside
  a rocMLIR kernel (HipToTosa) and on the HIP runtime, because the rebuilt op
  can stay outside a kernel. The quantize users rebuild ops on i8 and ui8.
  The cast fold rebuilds ops on f16 and bf16.

## Ops that implement it

| Op | Moved operands | Note |
|---|---|---|
| `tensor.insert_slice` | source, destination | External model |
| `tensor.extract_slice` | source | External model |
| `tensor.expand_shape`, `tensor.collapse_shape` | source | External models |
| `hip.transpose` | input | |
| `hip.resize` | input | Nearest mode only; the runtime kernel takes i8, ui8, f16 and bf16 for nearest |

`registerElementMoveExternalModels` registers the tensor external models.
`registerAllDialects` and `hip-mlir-opt` call it, because the `hip.cast`
canonicalizer runs in every pipeline. `hip-fuse-rocmlir` also adds it to its
dependent dialect registry.

Ops such as `hip.slice`, `hip.gather`, `hip.pad` and a max `hip.pool` can
implement the interface when their runtime kernels take 8-bit types. A pad
also needs its constant mapped to the zero point.
