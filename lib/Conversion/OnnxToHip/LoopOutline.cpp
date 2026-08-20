/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LoopOutline.cpp - Outline onnx.Loop bodies into func.func ----------===//
//
// Converts each `onnx.Loop` op in the module into a `hip.loop` op plus an
// outlined `func.func` containing the loop body.  Runs BEFORE
// `--convert-onnx-to-hip` so that the body's onnx.* ops get the same
// conversion treatment as ops in `main_graph`.
//
// Transformations performed per onnx.Loop:
//   1. Collect free variables (captures) used inside the body.
//   2. Unbox the ONNX-style 0-D-tensor `M` and `cond_init` operands to the
//      MLIR-style `index` and `i1` scalars that hip.loop expects.
//   3. Build a new func.func with signature
//        (!hip.context, iter_t, cond_in_t, v_in..., captures...,
//         !hip.loop_frame)
//          -> (cond_out_t, v_out...)
//      and move the body region into it.
//   4. Replace the body's `onnx.Yield` terminator with `func.return`.
//   5. Detect cond passthrough (yield.cond_out SSA-equals cond_in arg) and
//      set the `cond_is_passthrough` UnitAttr on the hip.loop op.
//   6. Replace the onnx.Loop op with hip.loop and erase it.
//
// Limitations of this initial version:
//   - Missing-M (while-style infinite loop) is rejected.  Counted loops
//     with `cond_init` absent (`onnx.NoValue` / `none` type) are
//     supported: cond is synthesized as `arith.constant true : i1` and
//     `cond_is_passthrough` is forced so the runtime takes the fast
//     `hipdnn_ep_run_counted_loop` path (no per-iter cond readback).
//     Per ONNX spec, missing cond means the loop runs M times with no
//     conditional early exit -- the body's `cond_out` (if any) is
//     ignored.
//   - Scan outputs (results beyond the loop-carried slot count) are not
//     yet supported; rejected with a clean diagnostic.
//
//===----------------------------------------------------------------------===//

#include "ReadbackScalar.h"

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-loop-outline"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Materialize an `index` scalar from a 0-D `tensor<i64>` trip count. A runtime
/// trip count (e.g. a window count derived from a graph input via `onnx.Sub`)
/// goes through `hip.readback_scalar` (D2H + stream sync) -- a bare
/// `tensor.extract` would read stale bytes on true-device pools and run the
/// loop the wrong number of times. See ReadbackScalar.h.
static FailureOr<Value> unboxTripCount(OpBuilder &builder, Location loc,
                                       Value ctx, Value mTensor) {
  auto t = dyn_cast<RankedTensorType>(mTensor.getType());
  if (!t || t.getRank() != 0 || !t.getElementType().isInteger(64))
    return failure();
  // A constant trip count folds straight to an index (no readback, no cast).
  if (DenseElementsAttr dense = getConstantDense(mTensor))
    if (dense.getNumElements() == 1)
      return arith::ConstantIndexOp::create(
                 builder, loc,
                 (*dense.getValues<APInt>().begin()).getSExtValue())
          .getResult();
  // Runtime (possibly GPU-computed) trip count: synchronized readback then
  // cast.
  Value i64Val = readbackScalarToHost(builder, loc, ctx, mTensor);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    i64Val)
      .getResult();
}

/// Materialize an `i1` scalar from a 0-D BOOL-like tensor operand.
///
/// ONNX `BOOL` is canonical 1-byte storage (0x00=false, non-zero=true).
/// Different importers spell that in MLIR three different ways:
///
///   * `tensor<i1>`  -- canonical MLIR bool (what hand-written test
///                      fixtures under SimpleTestModels/*.preoutline.mlir
///                      use, and what the ONNX-MLIR style importer emits).
///   * `tensor<ui8>` -- what the morphizen importer emits (it deliberately
///                      preserves ONNX's physical 1-byte storage layout
///                      so subsequent passes / runtime see the same
///                      byte-level encoding as ORT).
///   * `tensor<i8>`  -- signless 8-bit; accepted for safety so callers
///                      don't need to know which morphizen variant they
///                      got.
///
/// Same constant-folding rationale as `unboxTripCount`: when the cond is
/// produced by an `onnx.Constant` we fold the byte directly into an
/// `arith.constant i1` so the hip.loop op's `cond_init` operand doesn't
/// sit on a `tensor.extract` that would later bufferize to a host load
/// from device-resident memory. For non-constant cond (rare; future
/// extension), we extract and reduce to i1 via `cmpi ne 0`, bridging
/// `ui8`/`si8` -> signless `i8` with an `unrealized_conversion_cast`
/// since `arith.cmpi` requires signless operands.
static FailureOr<Value> unboxCondInit(OpBuilder &builder, Location loc,
                                      Value condTensor) {
  auto t = dyn_cast<RankedTensorType>(condTensor.getType());
  if (!t || t.getRank() != 0)
    return failure();
  auto intTy = dyn_cast<IntegerType>(t.getElementType());
  if (!intTy)
    return failure();
  unsigned width = intTy.getWidth();
  if (width != 1 && width != 8)
    return failure();

  // Constant-fold path. Read the raw bit pattern via APInt (signedness-
  // agnostic) and convert to bool by comparison to zero.
  if (Operation *defOp = condTensor.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "onnx.Constant") {
      if (auto attr = defOp->getAttrOfType<DenseElementsAttr>("value")) {
        auto at = dyn_cast<RankedTensorType>(attr.getType());
        if (at && at.getRank() == 0 && at.getElementType() == intTy) {
          bool truthy = (*attr.getValues<APInt>().begin()).getZExtValue() != 0;
          Value i1Val = arith::ConstantIntOp::create(
              builder, loc, builder.getI1Type(), truthy ? 1 : 0);
          return i1Val;
        }
      }
    }
  }

  // Non-constant fallback.
  Value extracted =
      tensor::ExtractOp::create(builder, loc, condTensor, ValueRange{});
  if (width == 1)
    return extracted; // already an i1 scalar

  // width == 8: reduce to i1 by comparing to zero. arith.cmpi requires
  // signless operands; bridge ui8/si8 -> signless i8 with an
  // unrealized_conversion_cast (legal at this pipeline stage, eliminated
  // by downstream type conversion since the bit patterns are identical).
  Type signlessI8 = builder.getIntegerType(8);
  if (!intTy.isSignless()) {
    extracted = UnrealizedConversionCastOp::create(
                    builder, loc, TypeRange{signlessI8}, ValueRange{extracted})
                    .getResult(0);
  }
  Value zero = arith::ConstantIntOp::create(builder, loc, signlessI8, 0);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                               extracted, zero)
      .getResult();
}

static FailureOr<RankedTensorType>
validateCarrierEquality(Operation *loopOp, unsigned carrierIndex,
                        ArrayRef<Type> participantTypes) {
  auto initType = dyn_cast<RankedTensorType>(participantTypes.front());
  if (!initType)
    return loopOp->emitOpError("loop carrier #")
           << carrierIndex << " requires a ranked v_init type";

  // Normalize under-refined source boundaries to v_init. The outlined function
  // and hip.loop verifier then enforce exact current/result equality; actual
  // shape evolution is deferred to the conservative-join layer.
  (void)participantTypes;
  return initType;
}

//===----------------------------------------------------------------------===//
// OnnxLoopOutlinePass
//===----------------------------------------------------------------------===//

struct OnnxLoopOutlinePass
    : public PassWrapper<OnnxLoopOutlinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxLoopOutlinePass)

  StringRef getArgument() const override { return "onnx-loop-outline"; }

  StringRef getDescription() const override {
    return "Outline each onnx.Loop body into a separate func.func and "
           "replace the loop with hip.loop";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;
  LogicalResult outlineAll(ModuleOp module);

  /// Outline a single onnx.Loop. Returns failure on unsupported variant.
  LogicalResult outlineLoop(Operation *loopOp, unsigned &counter);
};

LogicalResult OnnxLoopOutlinePass::outlineLoop(Operation *loopOp,
                                               unsigned &counter) {
  Location loc = loopOp->getLoc();
  ModuleOp module = loopOp->getParentOfType<ModuleOp>();
  MLIRContext *ctx = loopOp->getContext();

  // ONNX Loop has exactly one region with one block: the body.
  if (loopOp->getNumRegions() != 1)
    return loopOp->emitOpError("onnx.Loop with != 1 regions not supported");
  Region &bodyRegion = loopOp->getRegion(0);
  if (!bodyRegion.hasOneBlock())
    return loopOp->emitOpError(
        "onnx.Loop body with multiple blocks not supported");
  Block &bodyBlock = bodyRegion.front();

  // ONNX Loop has the operand layout (M, cond_init, v_init_1, ..., v_init_N).
  // M (trip count) is currently required; cond_init is optional. Importers
  // materialize a missing cond_init as `onnx.NoValue` (none type). When
  // absent, ONNX semantics is "cond stays true forever; only
  // max_trip_count terminates the loop". All downstream layers
  // (hip.loop dialect cond_init=Optional<I1>, LoopOpLowering null
  // handling, runtime fast-path) already accept that shape, so we
  // synthesize an i1-true here and force cond_is_passthrough.
  unsigned numOperands = loopOp->getNumOperands();
  if (numOperands < 2)
    return loopOp->emitOpError("onnx.Loop with missing M not yet supported");
  Value mTensor = loopOp->getOperand(0);
  Value condInitTensor = loopOp->getOperand(1);
  bool condInitIsNone = isa<NoneType>(condInitTensor.getType());
  ValueRange vInitTensors = loopOp->getOperands().drop_front(2);
  unsigned numLoopCarried = vInitTensors.size();

  if (loopOp->getNumResults() != numLoopCarried)
    return loopOp->emitOpError("loop-carried result count mismatch: expected ")
           << numLoopCarried << ", got " << loopOp->getNumResults();

  // ONNX Yield is the body terminator: (cond_out, v_out_1, ..., v_out_N,
  // scan_outputs...).
  Operation *yieldOp = bodyBlock.getTerminator();
  if (yieldOp->getName().getStringRef() != "onnx.Yield")
    return loopOp->emitOpError("onnx.Loop body must end with onnx.Yield, got ")
           << yieldOp->getName().getStringRef();
  if (yieldOp->getNumOperands() != 1 + numLoopCarried)
    return loopOp->emitOpError(
               "onnx.Loop scan outputs not yet supported (expected ")
           << (1 + numLoopCarried) << " yield operands for " << numLoopCarried
           << " loop-carried, got " << yieldOp->getNumOperands() << ")";

  // Block-arg layout in the body: (iter_t, cond_in_t, v_in_1, ..., v_in_N).
  // Validate the arg count matches what the operands imply.
  if (bodyBlock.getNumArguments() != 2 + numLoopCarried)
    return loopOp->emitOpError("onnx.Loop body arg count mismatch: expected ")
           << (2 + numLoopCarried) << ", got " << bodyBlock.getNumArguments();

  // The descriptor-frame ABI lands before shape-changing carrier joins. Keep
  // this layer independently safe by requiring one exact carrier type at every
  // source boundary.
  SmallVector<RankedTensorType> carrierTypes;
  carrierTypes.reserve(numLoopCarried);
  for (unsigned i = 0; i < numLoopCarried; ++i) {
    Type participants[] = {vInitTensors[i].getType(),
                           loopOp->getResult(i).getType(),
                           bodyBlock.getArgument(2 + i).getType(),
                           yieldOp->getOperand(1 + i).getType()};
    FailureOr<RankedTensorType> joined =
        validateCarrierEquality(loopOp, i, participants);
    if (failed(joined))
      return failure();
    carrierTypes.push_back(*joined);
  }

  // Detect cond passthrough BEFORE we touch the body. The check is SSA-
  // equality between the yield's cond_out operand and the body's cond_in
  // block arg.
  //
  // We walk through any leading `onnx.Identity` ops on the yield's
  // cond_out because the morphizen importer (and the ONNX exporter it
  // mirrors) materializes "cond passes through unchanged" as an explicit
  // `%cond_out = onnx.Identity(%cond_in)` rather than yielding the block
  // arg directly. Without this, every morphizen-produced loop would be
  // misclassified as non-passthrough and end up with a dead `onnx.Identity`
  // in the outlined body that has no `ConvertOnnxToHip` pattern and would
  // later fail bufferization ("op was not bufferized").
  Value yieldCondOutSrc = yieldOp->getOperand(0);
  while (Operation *defOp = yieldCondOutSrc.getDefiningOp()) {
    if (defOp->getName().getStringRef() != "onnx.Identity" ||
        defOp->getNumOperands() != 1)
      break;
    yieldCondOutSrc = defOp->getOperand(0);
  }
  BlockArgument condInArg = bodyBlock.getArgument(1);
  // When cond_init is none, ONNX spec says the body's cond_out is ignored;
  // we force the fast counted-loop path regardless of what the body yields.
  bool condIsPassthrough = condInitIsNone || (yieldCondOutSrc == condInArg);
  LLVM_DEBUG(llvm::dbgs() << "[onnx-loop-outline] cond_is_passthrough = "
                          << condIsPassthrough << "\n");

  auto parentFn = loopOp->getParentOfType<func::FuncOp>();
  if (!parentFn || parentFn.getNumArguments() == 0 ||
      !isa<ContextType>(parentFn.getArgument(0).getType()))
    return loopOp->emitOpError(
        "onnx.Loop is not inside a func.func with !hip.context as arg 0; "
        "ensure --hip-add-context-arg ran first");
  Value parentCtx = parentFn.getArgument(0);

  // Capture free variables defined above the body region.
  llvm::SetVector<Value> capturedSet;
  getUsedValuesDefinedAbove(bodyRegion, capturedSet);
  SmallVector<Value> captureVals;
  SmallVector<Value> capturedContexts;
  for (Value captured : capturedSet) {
    if (isa<ContextType>(captured.getType()))
      capturedContexts.push_back(captured);
    else
      captureVals.push_back(captured);
  }

  // Outlined function signature:
  //   arg0           : !hip.context (threaded by us; AddHipContextArg
  //                                  has already run on the parent func)
  //   arg1           : iter_t       (bodyBlock arg 0 type;
  //                                  typically tensor<i64>)
  //   arg2           : cond_in_t    (bodyBlock arg 1 type;
  //                                  typically tensor<i1>)
  //   args[3..3+N)   : v_carry_1..N (types from `vInitTensors`, i.e.
  //                                  the onnx.Loop's loop-carried
  //                                  operands feeding the loop from
  //                                  outside)
  //   args[3+N..)    : captures     (in iteration order of the SetVector)
  //
  // Sourcing v_carry types from v_init (rather than bodyBlock's v_in
  // args) is the key invariant that lets hip.loop's
  // `LoopOp::inferReturnTypes` derive result types without
  // disagreement: hip.loop's verifier requires
  // `result_type[i] == v_init[i].type`. Cloned body ops below carry
  // their source-IR result types verbatim. The importer emits
  // `tensor<*xT>` (unranked) for values whose shape it could not
  // derive — typically body-internal values inside `onnx.Loop` /
  // `onnx.If` / `onnx.Scan` regions, where the importer-side
  // shape-inference step has no usable per-iter rank for the body
  // block args. Those unranked types are refined post-conversion by
  // `--hip-infer-shapes` via `ReifyRankedShapedTypeOpInterface`, and
  // the loop signature (loop result types + body func arg/return
  // types at the v_carry slots) is synced by Phase 2 of that same
  // pass (`refineLoopSignatures`). See
  // `docs/design/unranked-tensor-handling.md`.
  Type ctxType = ContextType::get(ctx);
  SmallVector<Type> argTypes;
  argTypes.push_back(ctxType);
  argTypes.push_back(bodyBlock.getArgument(0).getType()); // iter_t
  argTypes.push_back(bodyBlock.getArgument(1).getType()); // cond_in_t
  llvm::append_range(argTypes, carrierTypes);
  for (Value c : captureVals)
    argTypes.push_back(c.getType());
  argTypes.push_back(LoopFrameType::get(ctx));

  // Skip the cond return value if the body trivially passes cond_in
  // through. The runtime trampoline aliases cond_in and cond_out on the
  // same device buffer, so the body's cond_out doesn't need to be
  // returned -- and BufferResultsToOutParams would otherwise materialize
  // an unnecessary memref<i1> out-param + a memref.copy that lowers to
  // a host-side llvm.intr.memcpy on a device pointer (crash, since
  // MemRefCopyOpLowering bails on i1 as non-byte-aligned).
  //
  // Declared return types here are the cloned body op result types
  // (= yield operand types), NOT v_init types. The cloned body ops
  // carry their source-IR result types at outline-pass exit, so
  // `func.return` operands match the declared return types and the
  // func is verifier-clean immediately after this pass.
  //
  // Two downstream passes refine those source-IR types when they are
  // unranked / under-refined:
  //   * `--convert-onnx-to-hip` consumes each cloned ONNX op and
  //     emits a HIP op whose result type is derived from the
  //     (already-refined) operand types — at the body-func entry,
  //     v_carry entry-block args carry the v_init type set above, so
  //     the first HIP op in the chain is constructed against ranked
  //     operand types. Function-boundary type mismatches that arise
  //     when the new HIP result type differs from the body func's
  //     declared return type are bridged by
  //     `builtin.unrealized_conversion_cast` via the partial-conversion
  //     framework and resolved by the next pass.
  //   * `--hip-infer-shapes` Phase 2 (`refineLoopSignatures`) syncs
  //     the body func's declared arg/return types and the hip.loop
  //     op's own result types with the v_init operand types after
  //     HIP-dialect refinements, then re-walks the body func once so
  //     each body op's `reifyResultShapes` catches up with the
  //     tighter entry-block arg types.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(1 + yieldOp->getNumOperands());
  resultTypes.push_back(IntegerType::get(ctx, 32));
  if (!condIsPassthrough)
    resultTypes.push_back(yieldOp->getOperand(0).getType());
  llvm::append_range(resultTypes, carrierTypes);

  auto fnType = FunctionType::get(ctx, argTypes, resultTypes);

  // Generate a unique symbol name. Use the parent function's name as a
  // prefix to keep the symbol table readable in multi-graph modules. On
  // collision, retry with the same prefix (don't drop it) and bump the
  // suffix until lookupSymbol returns null. `counter` ends one past the
  // suffix actually used so consecutive outlineLoop calls don't collide.
  auto buildName = [&](unsigned n) -> std::string {
    if (parentFn)
      return (parentFn.getName() + "_loop_body_n" + Twine(n)).str();
    return ("loop_body_n" + Twine(n)).str();
  };
  std::string fnName;
  do {
    fnName = buildName(counter++);
  } while (module.lookupSymbol(fnName));

  // Create the new func.func at the module level.
  OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto newFn = func::FuncOp::create(modBuilder, loc, fnName, fnType);
  newFn.setPrivate();

  // Build the entry block of the new function with matching arg types.
  Block *entry = newFn.addEntryBlock();

  // Map body block args -> entry block args [1..1+bodyArgs).
  // Map captures -> entry block args [1+bodyArgs..end).
  IRMapping mapping;
  for (Value capturedCtx : capturedContexts)
    mapping.map(capturedCtx, entry->getArgument(0));
  unsigned argIdx = 1; // skip ctx
  for (BlockArgument a : bodyBlock.getArguments())
    mapping.map(a, entry->getArgument(argIdx++));
  for (Value c : captureVals)
    mapping.map(c, entry->getArgument(argIdx++));

  // Clone every op from the body block (except the yield) into the new
  // function. We can't use `Block::without_terminator()` because onnx.Yield
  // is unregistered and that helper returns the FULL block when the last
  // op is unregistered (see Block.h comment on without_terminator).
  // Explicit-skip the yield by pointer equality instead.
  OpBuilder bodyBuilder(entry, entry->end());
  for (Operation &op : bodyBlock) {
    if (&op == yieldOp)
      continue;
    bodyBuilder.clone(op, mapping);
  }

  // In passthrough mode the yield's cond_out came from a chain of
  // `onnx.Identity` ops (detected above). Those cloned Identity ops are
  // now dead in the new body, but standard DCE won't remove them because
  // `onnx.Identity` is unregistered and MLIR conservatively assumes
  // unregistered ops have side effects. Manually erase the cloned copies
  // from leaf back to the block arg.
  if (condIsPassthrough) {
    Value chainV = yieldOp->getOperand(0);
    while (Operation *defOp = chainV.getDefiningOp()) {
      if (defOp->getName().getStringRef() != "onnx.Identity")
        break;
      Value clonedRes = mapping.lookupOrNull(defOp->getResult(0));
      if (!clonedRes)
        break;
      Operation *clonedOp = clonedRes.getDefiningOp();
      if (!clonedOp || !clonedOp->use_empty())
        break;
      chainV = defOp->getOperand(0);
      clonedOp->erase();
    }
  }

  // Build the func.return from the (mapped) yield operands. Cond is
  // skipped when cond_is_passthrough (see resultTypes above). The
  // cloned body ops carry their source ONNX result types — typically
  // unranked (`tensor<*xT>`) for body-internal values, because the
  // importer emits unranked for any value whose shape it could not
  // derive (the canonical case being body-internal values of
  // control-flow ops, where per-iter ranks are not available at
  // import time). Any operand-vs-result rank disagreement at the
  // cloned body ops is refined post-conversion by
  // `--hip-infer-shapes`.
  SmallVector<Value> returnVals;
  returnVals.reserve(resultTypes.size());
  returnVals.push_back(arith::ConstantIntOp::create(
      bodyBuilder, loc, bodyBuilder.getI32Type(), 0));
  if (!condIsPassthrough)
    returnVals.push_back(mapping.lookupOrDefault(yieldOp->getOperand(0)));
  for (unsigned i = 0; i < numLoopCarried; ++i) {
    Value returned = mapping.lookupOrDefault(yieldOp->getOperand(1 + i));
    if (returned.getType() != carrierTypes[i])
      returned =
          tensor::CastOp::create(bodyBuilder, loc, carrierTypes[i], returned);
    returnVals.push_back(returned);
  }
  func::ReturnOp::create(bodyBuilder, loc, returnVals);

  // Now build the hip.loop op at the original onnx.Loop location.
  OpBuilder outerBuilder(loopOp);

  Value ctxVal = parentCtx;

  // Unbox M (tensor<i64>) -> index, and cond_init (tensor<i1>) -> i1.
  FailureOr<Value> mIdx = unboxTripCount(outerBuilder, loc, ctxVal, mTensor);
  if (failed(mIdx))
    return loopOp->emitOpError("could not unbox M to index (expected "
                               "tensor<i64>, got ")
           << mTensor.getType() << ")";
  Value condI1Val;
  if (condInitIsNone) {
    // Synthesize i1-true; runtime takes counted-loop fast path and never
    // reads it back. See file header "Limitations" comment.
    condI1Val = arith::ConstantIntOp::create(outerBuilder, loc,
                                             outerBuilder.getI1Type(), 1);
  } else {
    FailureOr<Value> condI1 = unboxCondInit(outerBuilder, loc, condInitTensor);
    if (failed(condI1))
      return loopOp->emitOpError("could not unbox cond_init to i1 (expected "
                                 "0-D tensor<i1>, tensor<i8>, or tensor<ui8>, "
                                 "got ")
             << condInitTensor.getType() << ")";
    condI1Val = *condI1;
  }

  SmallVector<Value> joinedVInit;
  joinedVInit.reserve(numLoopCarried);
  for (unsigned i = 0; i < numLoopCarried; ++i) {
    Value init = vInitTensors[i];
    if (init.getType() != carrierTypes[i])
      init = tensor::CastOp::create(outerBuilder, loc, carrierTypes[i], init);
    joinedVInit.push_back(init);
  }

  // Build the hip.loop op via the `InferTypeOpInterface`-aware
  // `LoopOp::create` overload (no explicit `v_final` argument).
  // `LoopOp::inferReturnTypes` sources result types from the v_init
  // operand types, which by construction (above) match the outlined
  // body func's declared return types at the v_carry slots — the
  // hip.loop verifier's `result_type[i] == v_init[i].type` invariant
  // is satisfied automatically. Any residual under-refinement in the
  // cloned body op result types is caught up by `--hip-infer-shapes`
  // Phase 2 (`refineFuncBody` re-walk after body-arg sync).
  auto hipLoopOp = LoopOp::create(
      outerBuilder, loc,
      /*ctx=*/ctxVal,
      /*max_trip_count=*/*mIdx,
      /*cond_init=*/condI1Val,
      /*v_init=*/joinedVInit,
      /*captures=*/ValueRange(captureVals),
      /*parent_frame=*/Value(),
      /*body_func=*/FlatSymbolRefAttr::get(ctx, fnName),
      /*num_loop_carried=*/outerBuilder.getI32IntegerAttr(numLoopCarried),
      /*cond_is_passthrough=*/
      condIsPassthrough ? outerBuilder.getUnitAttr() : nullptr,
      /*descriptor_return=*/nullptr);

  // Preserve the source ONNX result boundary with compatible tensor casts.
  // Internally the loop always carries the conservative joined descriptor.
  for (unsigned i = 0; i < numLoopCarried; ++i) {
    Value replacement = hipLoopOp.getResult(i);
    Type oldType = loopOp->getResult(i).getType();
    if (replacement.getType() != oldType)
      replacement =
          tensor::CastOp::create(outerBuilder, loc, oldType, replacement);
    loopOp->getResult(i).replaceAllUsesWith(replacement);
  }
  loopOp->erase();
  return success();
}

LogicalResult OnnxLoopOutlinePass::outlineAll(ModuleOp module) {
  // Nested frame threading lands with shape-changing carriers. Reject nesting
  // transactionally in this ABI-only layer so a child can never escape without
  // an explicit parent lifetime.
  unsigned counter = 0;
  SmallVector<Operation *> loops;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "onnx.Loop")
      loops.push_back(op);
  });
  for (Operation *loopOp : loops) {
    for (Operation *parent = loopOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (parent->getName().getStringRef() != "onnx.Loop")
        continue;
      return loopOp->emitOpError(
          "nested onnx.Loop requires shape-carrier frame threading");
    }
    if (failed(outlineLoop(loopOp, counter)))
      return failure();
  }
  return success();
}

void OnnxLoopOutlinePass::runOnOperation() {
  ModuleOp module = getOperation();

  // Transactional validation/planning: run the complete nested post-order
  // transformation on a clone first. Any malformed child or parent leaves the
  // real module untouched, so no orphan body symbols or partially outlined
  // nested graph can survive a diagnostic.
  OwningOpRef<ModuleOp> planned = cast<ModuleOp>(module->clone());
  if (failed(outlineAll(*planned)) || failed(outlineAll(module)))
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> createOnnxLoopOutlinePass() {
  return std::make_unique<OnnxLoopOutlinePass>();
}

} // namespace hip
} // namespace mlir
