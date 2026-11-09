<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# rocMLIR kernel fusion

The rocMLIR path outlines each convolution, matmul and gemm of `main_graph`,
together with the pointwise ops and views around it, into one `rock.kernel`
function. `main_graph` calls the kernel through `hip.rocmlir`, and
`convert-hip-to-tosa` lowers the kernel body for the rocMLIR compiler. This
document describes how the fusion chooses the ops of each kernel.

## Pipeline

`addRocMlirFusionPasses` (`lib/Dialect/Transforms/Pipelines.cpp`) builds the
fusion for the EP, `hip-rocmlir-compiler`, `rocmlir-pipeline` and the
`hip-rocmlir-fusion` pipeline of `hip-mlir-opt`:

```
func.func(hip-rocmlir-prepare-quant)
func.func(hip-fuse-rocmlir-elementwise)
func.func(hip-fuse-rocmlir-anchor)
duplicate-function-elimination
hip-rocmlir-fusion-report        (only with a report file)
```

The passes change only `main_graph`. `hip-decompose-conv-transpose` runs
before the fusion, because rocMLIR has no transposed-convolution anchor.

The fusion has two steps. The first step groups the pointwise ops into
`hip.rocmlir_elementwise` modules without any anchor. The second step plans
the kernel of every anchor on these modules, and then outlines the kernels.
A module is one unit: an anchor takes a whole module or none of it.

## hip-rocmlir-prepare-quant

The pass prepares the quantize trees for integer kernels. It runs three
rewrites in this order, because the first fixes the final shape of each tree
and the last plans the integer anchors on that shape.

1. Each per-tensor `hip.quantize_linear` with inline scalar parameters moves
   back through the single-use ops that only move elements
   (`ElementMoveOpInterface`: `tensor.insert_slice`, `tensor.extract_slice`,
   static reshapes, `hip.transpose` and nearest `hip.resize`), down to
   per-tensor dequantizes and empty tensors. Thus a decomposed Concat, a
   transpose or an upsample runs on the integer type. A dequantize with the
   same scale bits, zero point and integer type folds to its input. Any other
   dequantize keeps a requantize of its own, which the producing integer
   kernel then takes into its epilogue. Any other leaf leaves the tree
   unchanged, and no quantize-dequantize pair is removed (see
   [element-move-interface.md](element-move-interface.md)).
2. Each `hip.qlinear_conv` and `hip.qlinear_matmul` with ui8 weights or a
   non-zero weight zero point, static shapes and inline constant parameters,
   weights and bias becomes integer ops. rocMLIR takes only i8 weights and has
   no zero-point operand, so the weights become `w' = w - offset` and the
   zero point `zw' = zw - offset` (offset 128 for ui8, 0 for i8); both fit
   i8, and `w' - zw' = w - zw`. The sum is then

   ```text
   sum (x' - z')(w' - zw') = quant(x, w') - quant(x, zw') - z' * sum(w')
                             + z' * zw' * taps
   ```

   over the taps that read the input. `quant(x, zw')` is a second
   `hip.quant_conv` (or `hip.quant_matmul`) over a filter of `zw'` in the
   weights' shape; it is left out when every `zw'` is 0. A `hip.sub` takes the difference, and the constant terms
   join the i32 bias of the requantize epilogue. The rewrite is exact and
   is declined when an i32 sum could overflow. Each integer op becomes a
   rocMLIR kernel, and the subtraction and the requantize join the epilogue
   of one of them.
3. Each connected ui8 tree that reaches an integer anchor becomes i8, and
   every zero point in it drops by 128. The producers of the tree are
   per-tensor quantizes with an inline scalar zero point and the results of
   the QLinear convs and matmuls that become integer anchors. Its interior is
   the same element-move ops. Its readers are per-tensor dequantizes, the
   data operands of those QLinear ops, and the data operands of
   `hip.quant_conv` and `hip.quant_matmul`, which read a ui8 value as
   `value - 128`. A tree with a function argument, a
   return or another reader keeps ui8, and so does a tree with no integer
   anchor among its readers. The rewrite is exact, and the anchor reads
   signed bytes with no shift.

A second run of the pass changes nothing.

## hip-fuse-rocmlir-elementwise

The pass walks `main_graph` in block order. From each op with a role that is
in no module yet, it grows a module through the readers of the module ops and
wraps the module in one `hip.rocmlir_elementwise` op. The module op has no
anchor semantics. `hip-inline-rocmlir-elementwise` puts its body back, and a
round trip computes the same values.

An op gets a role when it has one ranked result of rank 1 or more, and when
every init is a `tensor.empty`:

- A compute op is a pointwise op that `convert-hip-to-tosa` lowers
  (`isTosaExpressiblePointwiseOp`), or an op that it lowers only inside an
  anchor kernel (`isFusablePointwise`, for example a `hip.where` on a ui8
  mask).
- A transpose or a static reshape is a view, and a TOSA-lowerable
  per-tensor dequantize or quantize is a dequantize or a quantize. A
  dequantize that feeds a convolution of dequantizes that
  becomes an integer anchor gets no role, because the anchor pass reads its
  integer operand directly.

A broadcast, a slice, a dynamic reshape and a rank-0 value get no role and
stay outside as module inputs. A rank-0 value, such as the dequantize of a
scalar constant, is recomputed in each kernel that reads it. Inline scalar
constants also stay outside as inputs.

The growth follows the rules of an anchor epilogue, so that an anchor can
take a module whole:

- An op that broadcasts a module value starts a new module, because rocMLIR
  cannot invert a broadcast of an epilogue value.
- A quantize ends a module, and a dequantize only starts one. A dequantize
  that only the module reads joins it from above, also when it comes after
  the first module op, so that the kernel loads the 8-bit input (the other
  side of a residual add).
- An op whose other input depends on the module through an outside op starts
  a new module, so that no module reads its own result.
- A value with a reader outside the module is a module result. It needs a
  DPS init and a type that the kernel stores, so the pass drops each op that
  breaks this, with its readers in the module. A view read outside has no
  init and stays outside. Every input and result has a static shape, and no
  input or result is an integer wider than 32 bits, because the rocMLIR
  loads and stores take at most 32 bits.
- A module of views alone is not made.

## hip-fuse-rocmlir-anchor

The anchors are `hip.conv`, `hip.matmul`, `hip.gemm`, a `hip.qlinear_conv`
and a `hip.qlinear_matmul` that can become integer anchors, and a `hip.conv`
of dequantizes. An anchor has static shapes and no i32 input. The pass makes
a plan for every anchor first, and then outlines the plans.

### Epilogue

The pass plans the epilogues of all anchors in block order. Thus a module
that one anchor takes is not available to a later anchor, and an epilogue
takes precedence over a prologue. An anchor takes a module that reads its
kernel when these conditions are true:

- Every value that the module computes from a kernel value keeps the shape of
  that value. A transpose and a static reshape are allowed, because rocMLIR
  inverts them to map each output element back to its tile.
- No other module input depends on the kernel.
- A dequantize of a kernel value occurs only after a quantized anchor, because
  after a float anchor it would widen the kernel output again.
- After a quantized anchor, no kernel result is a float value that a
  dequantize widened. Every result is an integer value or a value from before
  the first dequantize.

A transpose or a static reshape outside any module that reads the kernel is
also an epilogue op. So is a chain of single-use reshapes without an init
that leads from a kernel value to such a view or to a module. The kernel
cannot store such a reshape, so it goes in only with the op at the end of the
chain.

A kernel value with a reader outside the kernel is an
extra `hip.rocmlir` result, and the outside readers above the kernel move
below the dispatch. A quantized anchor continues through the modules that
read the modules it took, past each quantize and dequantize, so it takes the
graph of readers and not only a chain.

### Prologue

Then each anchor takes, as the prologue of an activation operand, a module
that only the anchor reads. The module can have several inputs, such as the
two sides of a residual add; the kernel loads each input and recomputes the
module on every load. The anchor loads each activation element about once for
each filter tap that covers it (the filter size divided by the stride) and
once for each tile of 128 output columns. A module with more than one
non-scalar input joins only while this reload count, times the number of
full-size inputs that the prologue loads, is 9 or less. Thus a 1x1 conv or a
narrow matmul takes a residual add, but a 5x5 conv or a wide gemm leaves the
module as its own kernel, which computes it once. A module with one input
always joins. The activation operands are the conv input and the matmul and gemm A
and B operands, not the conv filter or the gemm C operand. For a conv of
dequantizes they are the integer input and weights. The search continues
through the input views and the producer modules of each module input.

A module that no plan takes and that is a valid elementwise kernel (see
[Elementwise kernels](#elementwise-kernels)) stays whole. Every other module
that no plan takes is inlined back before the outline. A kernel can then still
absorb the views and the unary pointwise ops of such a module one by one. This
matters for a module with several readers, such as the views and the scale of a
shared attention operand.

### Outline

The plans outline in reverse block order, so the kernel names and their order
stay the same as in the earlier greedy fusion. Before each outline, a
read-only trial checks the plan on the current IR. If the trial fails, the
pass retries without the prologue, then without the epilogue, and then keeps
the anchor as a HIP op. A QLinear op or a conv of dequantizes is decomposed
only after its trial passes, so a declined op keeps its HIP kernel.

The outline also absorbs the producers of the kernel inputs, when the kernel
is their only reader: the views that the policy allows (`hip.transpose`,
static reshapes, unit-stride `tensor.extract_slice` and `hip.expand` with a
constant target shape), which rocMLIR reads through its input transform maps,
and a unary pointwise op on an activation operand. A producer with another
reader would keep running, so it saves nothing. An absorbed producer's inputs
take its argument slot, so the kernel signature keeps the operand order of the
anchor. View absorption uses the legality of HIP-to-TOSA
(`HipToTosaLegality.h`), so an absorbed view always lowers.

A rank-0 kernel operand is reshaped to rank 1 at the dispatch, and all such
reshapes in `main_graph` share one shape constant.

### Elementwise kernels

After the plans outline, each module that is still in `main_graph` becomes a
`rock.kernel` function with the `rock.elementwise` attribute, behind
`hip.rocmlir`, when these conditions are true:

- Every module op, other than a `tensor.empty` and a static reshape, is a
  pointwise op that `convert-hip-to-tosa` lowers, a TOSA-lowerable per-tensor
  dequantize or quantize, or a `hip.transpose`.
- The module has at least two ops that launch, so the kernel saves a launch.
  A static reshape does not launch.
- A module with more than one result has no result that is a transpose of a
  module input, through static reshapes only. rocMLIR cannot store such a
  result next to the other results of the kernel.

The inline scalar constants that the module reads are cloned into the kernel.
Any other module is inlined back. The rocMLIR compile does not autotune an
elementwise kernel, and outside strict mode a kernel that fails to compile
falls back to its HIP ops.

### Integer anchors

These ops become integer anchors:

- A `hip.qlinear_conv` with static shapes, inline constant parameters,
  weights and bias, a zero weight zero point and per-tensor activation
  parameters becomes `hip.quant_conv` (i8 x i8 to i32 on `x - 128` for a u8
  input, `x` for i8). It is followed by an i32 bias that folds in the
  zero-point correction `-(x_zp - offset) * sum(w)` over the taps that each
  output reads (`[1,K,1,1]` when every output reads the same taps,
  `[1,K,OH,OW]` at a padded border), an f32 multiply by
  `x_scale * w_scale / y_scale`, and a unit-scale `hip.quantize_linear` with
  the output zero point. A channel whose multiplier is below 2^-33 writes the
  output zero point for any i32 accumulator, so it gets multiplier 0 and bias
  0; its correction then cannot overflow (a quantizer can saturate the bias of
  a dead channel to INT32_MIN).
- A `hip.qlinear_matmul` with inline constant parameters and a zero B zero
  point takes the same form around `hip.quant_matmul`, with a `[1,N]`
  correction `-(a_zp - offset) * sum_k(b[k,n])`. B must be an inline constant
  only when that correction is not 0.
- A `hip.conv` of a per-tensor dequantized 8-bit input and dequantized i8
  weights (per tensor or per output channel, with a zero weight zero point),
  with no bias or a constant one, becomes `hip.quant_conv` on the integer
  operands, the same correction, an f32 multiply by `x_scale * w_scale`, and
  an f32 add of the dequantized bias. Its activation and quantize then become
  the epilogue. The weights must be inline only when the correction is not 0.

In the outlined kernel, a per-tensor quantize of an add or a mul of two
dequantizes becomes the QLinearAdd or QLinearMul formula of ONNX Runtime CPU,
with a unit-scale quantize: `a * rA + (b * rB + fixed)` with
`fixed = zC - fma(rA, zA, rB * zB)` (a constant `fixed + b * rB` for a
one-element constant B), and `(a - zA) * (b - zB) * (sA * sB / sC) + zC`. The
result is at most one unit from ONNX Runtime CPU for each quantize: a sum that
rounds at a half can differ by one unit where the compiler does not fuse a
multiply-add.

The QLinear ops with ui8 weights or a non-zero weight zero point are already
integer ops at this point (see `hip-rocmlir-prepare-quant` above). Any other
`hip.qlinear_conv` or `hip.qlinear_matmul` keeps its HIP kernel. A
`hip.quant_conv`, `hip.quant_matmul` or `hip.rocmlir_elementwise` that is
left after the outline fails the pass. `convert-hip-to-tosa` then makes the
unsigned integers in the kernel signless, because rocMLIR takes only signless
types, and its `unsigned_cast` custom op keeps the unsigned meaning.

### Gemm-elementwise-gemm

After the outline, two matmul or gemm kernels merge into one
gemm-elementwise-gemm kernel when the only result of the first kernel feeds
the A operand of the second gemm through pointwise ops and row-only reshapes
(no transpose), and the gates pass:

- Both gemms read an element type in `gemm-gemm-accel-types`. The EP fills
  the list with the types that rocMLIR accelerates on the detected
  architecture, and an empty list merges no pair.
- The shapes are a column-vector intermediate (n = 1, k <= 128, m <= 1024,
  g <= 64) or a skinny-k MLP bottom (m >= 98304, n >= 512, k <= 13,
  g <= 256).

## Options

The fusion rules have no switches. The two options describe the target.

| Option | EP value | Effect |
|---|---|---|
| `arch=<gfx>` | detected | Stamps `rock.arch` on each kernel. |
| `gemm-gemm-accel-types=<types>` | detected | The element types for the gemm-gemm gate. |

## Tests and tools

- `test/lit/Dialect/hip-fuse-rocmlir*.mlir` check the kernels of the whole
  fusion. `hip-fuse-rocmlir-elementwise.mlir` checks one grouping rule in each
  test, and `hip-fuse-rocmlir-anchor.mlir` checks the planning rules.
- `HIPDNN_EP_ROCMLIR_PRE_FUSION_DUMP=<dir>` writes the IR that the fusion
  reads, with the fusion options on its first line, so that
  `hip-mlir-opt --hip-rocmlir-fusion` replays the fusion without a GPU.
- `HIPDNN_EP_ROCMLIR_FUSION_REPORT=<file>` (pipeline option `report`)
  appends one JSON line for each kernel and each host function. Two runs make
  the same kernels when their records are equal as multisets.
