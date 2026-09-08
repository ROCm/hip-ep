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

Special cases within this set:
- `tosa.mul` takes a third **shift** operand (`i8` zero const for float).
- `tosa.equal` / `tosa.greater` produce `i1`; cast back to the original element
  type if consumers expect it.
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
  `IntDivOp` (integer-only).
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

The pass uses the dialect conversion driver, so write an `OpConversionPattern`
and read operands off the `adaptor`:

```cpp
struct AddConverter final : public OpConversionPattern<hip::AddOp> {
  using OpConversionPattern<hip::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (!isTosaCompatibleOperand(adaptor.getLhs(), resultType) ||
        !isTosaCompatibleOperand(adaptor.getRhs(), resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};
```

Register it in `HipToTosaPass::runOnOperation`:

```cpp
patterns.add<AddConverter, /* ... */>(ctx);
```

Once a second `lhs`/`rhs`/`output` op needs the same treatment, lift this into a
`template <typename HipOpTy, typename TosaOpTy>` and reduce each op to one
`using` alias, mirroring `TrivialConverter` in `MIGraphXToTosa.cpp`.

Use `notifyMatchFailure` for anything unsupported (dynamic shapes, memref mode,
unhandled attributes) rather than asserting.

**A rejected op is a hard error, not a passthrough.** The pass runs
`applyFullConversion` with `addIllegalDialect<HipDialect>()`, so any `hip.*` op
left in a `rock.kernel` function fails legalization and the pass reports
`failed to legalize operation 'hip.<op>'`. So a partially supported op takes the
whole kernel down rather than degrading.

For the same reason every dialect the kernel contains must be marked legal:
`applyFullConversion` treats an op with *no registered legality* as illegal, so
an unlisted dialect fails the pass even when no pattern touches it. The target
currently lists `tosa` and `func`; add to that `addLegalDialect` call if a new
op's lowering introduces another dialect (e.g. `arith` for a materialized
constant).

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

Add two lit tests per op under `test/lit/Conversion/hip-to-tosa/`:

- `test_<op>.mlir` for what converts — the plain shape, a size-1 broadcast, and
  a function without `rock.kernel` (the pass early-returns, so the op survives).
- `<op>-invalid.mlir` for what the conversion rejects, using
  `--split-input-file --verify-diagnostics` with
  `// expected-error @+1 {{failed to legalize operation 'hip.<op>'}}`. Rejected
  forms cannot go in the positive test, because full conversion makes them a
  pass failure that produces no output for `FileCheck` to read.

**Check how the op prints its result before writing the test.** It varies by op
in `HipOps.td`: ops with `hasCustomAssemblyFormat = 1` (`hip.add`, `hip.mul`)
print `-> tensor<...>`, while ops with a declarative `assemblyFormat` ending in
`attr-dict (`:` type($result_tensors)^)?` (`hip.max`, `hip.min`, `hip.gemm`)
print `: tensor<...>`. Using the wrong one fails to parse with
`error: cannot name an operation with no results`.

```mlir
// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

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
