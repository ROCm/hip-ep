<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: add-hip-to-tosa-op
description: Add a 1-1 HIP-to-TOSA conversion pattern to the convert-hip-to-tosa pass so a hip.* op can be fused into a rocMLIR kernel. Use when implementing or extending HipToTosa / hip2tosa, when wiring an op into the rocmlir-pipeline, or when a hip.* op survives the pass unconverted and fails downstream in rocMLIR.
---

# Add a 1-1 HIP to TOSA Conversion

`convert-hip-to-tosa` (`lib/Conversion/HipToTosa/HipToTosa.cpp`) lowers `hip.*`
ops inside outlined `rock.kernel` functions to TOSA, so rocMLIR/rocmlirTriton
can absorb them into a fused kernel. Coverage is opt-in: every op needs its own
pattern added, and `AddConverter` (`hip.add`) is the model to copy.

Pointwise ops need **no Rock counterpart**. rocMLIR's `TosaToRock` marks only
six ops illegal (`Conv2DOp`, `Conv3DOp`, `MatMulOp`, `MatmulTBlockScaledOp`,
`ReduceSumOp`, `ReduceMaxOp`); everything else stays TOSA and is absorbed by
`RockTosaToElementwise`. Emitting valid, *lowerable* TOSA is the whole job.

Every edit below lands in `HipToTosa.cpp` alone. Do not touch `Passes.td`,
`InitAllPasses.h`, or any CMake file unless step 1's parsing note applies.

## 1. Dialect loading is already wired up

Linking `MLIRTosaDialect` only provides the C++ symbols; the dialect also has to
be **loaded into the context** or creating a `tosa.*` op fails with ``Dialect
`tosa' not found``. `ConvertHipToTosaPass` in `Passes.td` already declares

```
let dependentDialects = ["mlir::tosa::TosaDialect"];
```

so nothing is needed to emit TOSA. Do not override `getDependentDialects` in the
pass class — TableGen generates it from that list.

Only if `hip-mlir-opt` must also **parse** IR that already contains `tosa.*`
(round-tripping its own output, or a lit test whose input is TOSA) add
`registry.insert<mlir::tosa::TosaDialect>();` to `registerAllDialects` in
`include/hip/InitAllPasses.h`. Parsing happens before the pass manager runs, so
`dependentDialects` does not cover it.

## 2. Pick the mapping

These pairs are 1-1 **and** lowerable by `RockTosaToElementwise`. Prefer them:

| HIP op | TOSA op | HIP op | TOSA op |
|---|---|---|---|
| `hip.add` | `tosa.add` | `hip.ceil` | `tosa.ceil` |
| `hip.sub` | `tosa.sub` | `hip.floor` | `tosa.floor` |
| `hip.mul` | `tosa.mul` | `hip.tanh` | `tosa.tanh` |
| `hip.abs` | `tosa.abs` | `hip.erf` | `tosa.erf` |
| `hip.neg` | `tosa.negate` | `hip.sigmoid` | `tosa.sigmoid` |
| `hip.exp` | `tosa.exp` | `hip.reciprocal` | `tosa.reciprocal` |
| `hip.log` | `tosa.log` | `hip.min` | `tosa.minimum` |
| `hip.sin` | `tosa.sin` | `hip.max` | `tosa.maximum` |
| `hip.cos` | `tosa.cos` | `hip.cast` | `tosa.cast` |
| `hip.equal` | `tosa.equal` | `hip.where` | `tosa.select` |
| `hip.less` | `tosa.greater`, operands swapped | | |

The set splits by operand signature, and that decides which template an op
plugs into rather than how much new code it needs:

- **Binary** `(ctx, lhs, rhs, output)` — `add`, `sub`, `mul`, `min`, `max`,
  `div`, `equal`, `less`, `and`, `or`, `mod`. Read `adaptor.getLhs()` /
  `getRhs()`. `BinaryConverter` covers the ones that keep the element type, and
  handles the broadcasting mismatch for them: hip rank-extends the way
  ONNX/NumPy do, while TOSA needs both operands to already carry the result's
  rank and broadcasts size-1 dimensions only. The template calls
  `tosa::EqualizeRanks` (from `Tosa/Utils/ConversionUtils.h`) to reshape the
  shorter operand with leading 1s, then re-checks compatibility so a dimension
  that still cannot broadcast is rejected instead of emitting invalid TOSA.
  This is what lets a ReLU arriving as `hip.max(tensor<1x64x112x112xf16>,
  tensor<f16>)` convert at all.
- **Unary** `(ctx, x, y)` — `abs`, `neg`, `ceil`, `floor`, `exp`, `log`, `sin`,
  `cos`, `tanh`, `erf`, `sigmoid`, `reciprocal`. Read `adaptor.getX()`; the outs
  accessor is `Y`. All twelve are already registered via `UnaryConverter`.
- **Ternary** — `where` is `(ctx, condition, x, y, output)`.

Unary ops are the safe ones. Their TOSA counterparts are
`Tosa_ElementwiseUnaryOp`, which carries `SameOperandsAndResultShape` and
`SameOperandsAndResultElementType`, so there is no broadcasting to reason about
at all. Require `operand type == result type` exactly, and do **not** reuse
`isTosaCompatibleOperand` — its size-1 tolerance would admit an operand that
needs broadcasting and emit invalid TOSA.

Special cases within this set:
- `tosa.mul` takes a third **shift** operand (`Tosa_ScalarInt8Tensor`, i.e.
  `tensor<1xi8>`) that right-shifts the product of `i32` inputs, and it has
  **no** convenience builder. `BinaryConverter` handles it with an
  `if constexpr` that materializes a zero shift via `createZeroMulShift`; zero
  is required, not merely conventional, since `tosa::MulOp::verify` rejects a
  nonzero shift for float types and a rescale is not what `hip.mul` means. The
  shift is excluded from that verifier's same-rank check, so equalizing the two
  data operands is still sufficient.
- `tosa.equal` / `tosa.greater` produce `i1`. `hip.equal` / `hip.less` already
  produce `tensor<...xi1>`, so the mapping is 1-1, but they still cannot use
  `BinaryConverter`: `isTosaCompatibleOperand` compares the operand element type
  against the *result* element type and so rejects every comparison. They need a
  check that compares the operands to each other instead.
- **`tosa.negate` is not a plain unary op.** It is a `Tosa_InferShapedTypeOp`
  taking `input1`, `input1_zp` and `output_zp`. It still shares `UnaryConverter`
  unchanged, because `Tosa_NegateOpQuantInfoBuilder` exposes a
  `(Type outputType, Value input)` builder — the same signature as the plain
  unary builder — and `buildNegateOpWithQuantInfo` materializes both zero
  points. It expands to three ops, so a test must expect `tosa.const` ahead of
  `tosa.negate` rather than a single op.
- `tosa.sin` and `tosa.cos` take `Tosa_FloatTensor`, so an integer operand fails
  the TOSA verifier. The other float ops (`ceil`, `floor`, `exp`, `log`, `tanh`,
  `erf`, `sigmoid`, `reciprocal`) take `Tosa_Tensor` but are only available in
  TOSA's FP profile, so an integer would verify and then have no lowering. Gate
  all ten on a float element type; `abs` and `negate` are the only two that
  legitimately accept integers.
- `tosa.maximum` / `tosa.minimum` carry a `nan_mode` attribute, but ODS emits a
  builder defaulting it to `PROPAGATE`, so the two-operand `replaceOpWithNewOp`
  below still works unchanged. `MIGraphXToTosa.cpp` instead passes `IGNORE`
  unless `-disable-fast-math`, which makes `RockTosaToElementwise` emit
  `nnan`-flagged arith ops so a `maximum`/`minimum` clamp pair folds into a
  single `tt.clampf`. Worth matching if hip-ep ever gets a fast-math flag.
- TOSA's `maximum` / `minimum` compare as **signed**. `MaxConverter` in
  `MIGraphXToTosa.cpp` routes unsigned integers to `tosa.custom`
  (`ROCK_CUSTOMOP_UNSIGNED_MAX`) instead; do the same or reject them.

Decompositions — every piece is supported, so these are safe to emit:
- `hip.sqrt` → `tosa.reciprocal(tosa.rsqrt(x))`. TOSA has no sqrt; rocMLIR
  explicitly folds this pair back into a single `math.sqrt`. Use this idiom.
- `hip.div` → `tosa.mul(a, tosa.reciprocal(b))`. TOSA's only division is
  `tosa.intdiv`, restricted to `Tosa_Int32Or64Tensor`, so `BinaryConverter`
  can cover *integer* div only if gated to i32/i64 — anything else would emit
  an invalid `tosa.intdiv`. Take the reciprocal on the rank-equalized `rhs`,
  since `tosa.reciprocal` demands operand type == result type, and let the
  `tosa.mul` broadcast. Flag in review that this adds a rounding step and
  changes divide-by-zero behaviour versus a true divide.
- `hip.silu` → `tosa.mul(x, tosa.sigmoid(x))`
- `hip.softplus` → `tosa.log(tosa.add(tosa.exp(x), 1))`
- `hip.leaky_relu` → `tosa.maximum(x, tosa.mul(x, alpha))`
- `hip.gelu` / `hip.bias_gelu` / `hip.fast_gelu` → compose from `erf` / `tanh`,
  `mul`, `add`

Do **not** map these — no TOSA op exists: `hip.atan`, `hip.mod`, `hip.round`,
`hip.sign`. Leave them unconverted (they end the fusion chain) or use
`tosa.custom`, which rocMLIR handles. `hip.sign` is expressible via
`tosa.select` + `tosa.greater` if needed.

**Trap:** `hip.and` / `hip.or` / `hip.not` look fine as
`tosa.logical_and`/`logical_or`/`logical_not`, but `RockTosaToElementwise` only
has patterns for the **bitwise** variants. The logical forms convert cleanly and
then fail downstream. Emit `tosa.bitwise_and` / `bitwise_or` / `bitwise_not` on
`i1` instead.

## 3. Write the pattern

HIP ops are destination-passing style: they carry a `!hip.context` and an `outs`
buffer. Drop both — the result type already encodes the destination. After
`hip-fuse-rocmlir` the context is a `ub.poison` value.

The pass uses the dialect conversion driver, so patterns are
`OpConversionPattern`s that read operands off the `adaptor`. `BinaryConverter`
and `UnaryConverter` are already templated over the hip/TOSA pair, mirroring
`TrivialConverter` in `MIGraphXToTosa.cpp`, so an op that fits either signature
is **one line** in `HipToTosaPass::runOnOperation`:

```cpp
patterns.add<BinaryConverter<AddOp, tosa::AddOp>,
             BinaryConverter<SubOp, tosa::SubOp>,
             UnaryConverter<AbsOp, tosa::AbsOp>,
             UnaryConverter<ExpOp, tosa::ExpOp, /*FloatOnly=*/true>>(ctx);
```

Write a standalone pattern only when the op needs something the templates do
not express — a third TOSA operand, a rank-equalizing reshape, an attribute
translation, or a comparison's `i1` result. When you do, follow the template
bodies: bail on `getNumResults() != 1` (memref mode has no SSA result to
replace), require a static `RankedTensorType` result, then check operands.

Note the templates take the hip op class unqualified (`AddOp`, not `hip::AddOp`)
because the file is inside `namespace mlir::hip`.

Use `notifyMatchFailure` for anything unsupported (dynamic shapes, memref mode,
unhandled attributes) rather than asserting.

**A rejected op is a hard error, not a passthrough.** The pass runs
`applyFullConversion` with `addIllegalDialect<HipDialect>()`, so any `hip.*` op
left in a `rock.kernel` function fails legalization and the pass reports
`failed to legalize operation 'hip.<op>'`. So a partially supported op takes the
whole kernel down rather than degrading.

For the same reason every op the kernel contains must be marked legal:
`applyFullConversion` treats an op with *no registered legality* as illegal, so
an unlisted dialect fails the pass even when no pattern touches it and even when
the op has no uses. The target lists the `tosa` and `func` dialects plus
`ub::PoisonOp`, which is needed because `hip-fuse-rocmlir` maps the
`!hip.context` to a `ub.poison` inside the outlined kernel. Extend that legality
set if a new op's lowering introduces another dialect (e.g. `arith` for a
materialized constant).

Because of this, a test that passes the context in as a `!hip.context` block
argument does **not** exercise the shape the pipeline actually produces. Cover
the `ub.poison` form too when a pattern is meant for real fused kernels.

## 4. Build and verify

```bash
cmake --build ../build/hip-ep -j 32 --target hip-mlir-opt
```

On Windows this needs the MSVC environment or it dies with `Cannot open include
file: 'stddef.h'`. Either use an "x64 Native Tools Command Prompt for VS 2022",
or import vcvars once per PowerShell session:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vc`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
```

Two gates control whether the pass does anything:
- `convert-hip-to-tosa` early-returns unless the function has `rock.kernel`.
- `hip-fuse-rocmlir` only runs on a function named `main_graph`, and only
  outlines `hip.conv` / `hip.gemm` anchors plus their pointwise consumers.

So test the pattern directly on a pre-stamped function:

```bash
../build/hip-ep/bin/hip-mlir-opt --convert-hip-to-tosa test.mlir
```

Do not try to validate via the full flow. `--hip-fuse-rocmlir` segfaults on a
hand-written `main_graph` with a `hip.gemm` anchor even with
`convert-hip-to-tosa` out of the pipeline, so `--rocmlir-pipeline` cannot
currently confirm a pattern. Verify with `--convert-hip-to-tosa` alone.

Tests live under `test/lit/Conversion/hip-to-tosa/`, which holds exactly two
files, split by **operand group rather than by op** since a whole group shares
one converter template: `test_binary.mlir` and `test_unary.mlir`. Each one
holds both what converts and what is rejected. Add a new op to its group file
instead of creating a per-op file or a separate rejection file.

Cover the op mapping itself plus anything specific to that op's operands. Do
**not** repeat the group's shape and broadcast cases per op — the template
makes them redundant. Those are exercised once (through `hip.add` for binary)
as a size-1 broadcast, a rank-extending operand and a rank-0 scalar, both of
the latter expecting a `tosa.reshape` ahead of the op. The `rock.kernel` guard
is a property of the pass, so one case covers it.

Rejections live in the **same** file, after the converting cases, using
`// expected-error @+1 {{failed to legalize operation 'hip.<op>'}}`. Note this
departs from the rest of `test/lit/Conversion/`, where rejections sit in a
separate `*-invalid.mlir`; the hip-to-tosa tests deliberately keep one file per
operand group. One RUN line serves both:

```
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s
```

Two rules make that work. Keep every converting case in the **first** chunk so
they stay a single module, which is what covers several ops converting in one
pass run. Then give every rejection its **own** chunk: full conversion turns a
rejection into a pass failure that aborts the run for the whole module, so
sharing a chunk would let one rejection mask the cases after it. A failing
chunk contributes no output while the converting chunks still print for
`FileCheck`, so the two kinds of check coexist.

Both kinds of coverage stay live under this layout — verified by mutation, as
deleting an `expected-error` or corrupting a `CHECK` each turn the file red.
The one cost is that converting cases must stay diagnostic-clean: if the pass
ever emits a warning or remark on an op that converts, `--verify-diagnostics`
fails the whole file on the unexpected diagnostic.

**Check how the op prints its result before writing the test.** It varies by op
in `HipOps.td`: ops with `hasCustomAssemblyFormat = 1` print `-> tensor<...>`,
while ops with a declarative `assemblyFormat` ending in
`attr-dict (`:` type($result_tensors)^)?` print `: tensor<...>`. Using the wrong
one fails to parse with `error: cannot name an operation with no results`.

Only five of the 103 ops in `HipOps.td` are custom-format and take `->`:
`hip.add`, `hip.mul`, `hip.silu`, `hip.miopen_add` and `hip.miopen_softmax`.
Three of those are elementwise, so `hip.silu` will hit this the next time the
decomposition list below is picked up. Everything else takes `:`, including
`hip.sub`, `hip.min`, `hip.max` and all twelve unary ops.

Copying one of the `hip.add` cases in `test_binary.mlir` as a starting point
therefore means fixing the result separator, which is easy to miss because the
error points at the *following* line. Do not infer the format from whether an
op is unary or binary — `hip.silu` is unary and still takes `->`. Get the full
list with:

```powershell
$lines = Get-Content include\hip\Dialect\IR\HipOps.td
$cur = $null
for ($i = 0; $i -lt $lines.Count; $i++) {
  if ($lines[$i] -match '^def (Hip_\w+)\s*:') { $cur = $matches[1] }
  if ($lines[$i] -match 'hasCustomAssemblyFormat\s*=\s*1') { $cur }
}
```

```mlir
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK: tosa.add %arg1, %arg2
// CHECK-NOT: hip.add
func.func @add(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
```

Run it with lit:

```bash
cd ../build/hip-ep
python _deps/llvm-project-build/bin/llvm-lit.py -v test/lit/Conversion/hip-to-tosa
```

`Conversion/onnx-to-hip/test_qadd.mlir` fails on this branch already; it never
runs `convert-hip-to-tosa`, so ignore it.

Confirm the output contains no residual `hip.*` compute ops. Anything left
behind will fail later in rocMLIR, not here.

## Reference

- Follow `MIGraphXToTosa.cpp` in rocmlirTriton for prior art; its
  `TrivialConverter<MIGraphXOp, TosaOp>` template is the same 1-1 idea, and its
  `SqrtConverter` is the source of the `reciprocal(rsqrt(x))` idiom.
- The authoritative list of lowerable TOSA ops is the pattern registration in
  rocmlirTriton's `mlir/lib/Dialect/Rock/Transforms/RockTosaToElementwise.cpp`.
  Check it before adding a mapping not listed above.
