/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include <limits>

using namespace mlir;
using namespace mlir::hip;

#include "hip/Dialect/IR/HipDialect.cpp.inc"

void HipDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hip/Dialect/IR/HipTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/IR/HipOps.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/IR/HipTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Non-DPS ops: memory effect declarations
//===----------------------------------------------------------------------===//

void AllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

void GetPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

void GetHostScratchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

void FreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // memref is operand #1 (operand #0 is ctx)
  effects.emplace_back(MemoryEffects::Free::get(),
                       &getOperation()->getOpOperand(1),
                       SideEffects::DefaultResource::get());
}

void GetConstantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

void AllocOutputOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Two requirements, both load-bearing:
  //   (1) NOT MemoryEffects::Allocate. The returned buffer is EP/runtime-owned
  //       (a graph output): hip-pool-allocs must never pool it, and a custom
  //       pipeline that enables ownership-based buffer deallocation must never
  //       free it. Ownership is keyed on the Allocate effect, so we
  //       deliberately omit it (unlike AllocOp/GetPoolOp above).
  //   (2) A generic Write effect (no associated value) marks the side effect of
  //       calling into the EP output allocator, which mutates external runtime
  //       state. This is what keeps the op alive: an op carrying a Write is
  //       never trivially dead (DCE / canonicalize) even when its result is
  //       unused, and CSE never merges side-effecting ops. (Contrast: an
  //       Allocate-on-result op IS removed when its result is unused -- a
  //       second reason to avoid Allocate here.)
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void LoopAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Frame-owned storage: deliberately omit Allocate so PoolAllocs and generic
  // ownership-based deallocation cannot claim the carrier bank. The generic
  // Write models mutation of the per-invocation frame and keeps the call alive.
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void LoopFrameStatusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Reads the same mutable frame resource that hip.loop_alloc writes. This
  // prevents CSE/hoisting from moving a status observation across allocation.
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoopAllocOp::verify() {
  auto type = dyn_cast<MemRefType>(getMemref().getType());
  if (!type || !type.hasRank())
    return emitOpError("requires a ranked memref result");
  if (getCarrierIndex() < 0)
    return emitOpError("carrier_index must be non-negative");
  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      type.getNumDynamicDims())
    return emitOpError("dynamic-size operand count must match the result type");
  if (!type.getLayout().isIdentity())
    return emitOpError("requires an identity-layout result memref");
  auto body = getOperation()->getParentOfType<func::FuncOp>();
  if (!body || body.empty())
    return emitOpError("must be nested in an outlined loop body");
  if (body.getNumArguments() == 0 || getFrame() != body.getArguments().back() ||
      !isa<LoopFrameType>(body.getArgumentTypes().back()))
    return emitOpError(
        "frame must be the outlined body's final !hip.loop_frame argument");
  ModuleOp module = body->getParentOfType<ModuleOp>();
  LoopOp owner;
  if (module)
    module.walk([&](LoopOp loop) {
      if (!owner && loop.getBodyFunc() == body.getName())
        owner = loop;
    });
  if (!owner)
    return emitOpError("cannot resolve owning hip.loop for body ")
           << body.getName();
  unsigned carrier = static_cast<unsigned>(getCarrierIndex());
  if (carrier >= owner.getNumLoopCarried())
    return emitOpError("carrier_index ")
           << carrier << " is outside [0, " << owner.getNumLoopCarried() << ")";
  if (body.getNumArguments() <= 3 + carrier)
    return emitOpError("owning body is missing current carrier argument #")
           << carrier;
  auto currentType = dyn_cast<MemRefType>(body.getArgumentTypes()[3 + carrier]);
  if (!currentType || currentType != type)
    return emitOpError("result type ")
           << type << " must match current carrier #" << carrier << " type "
           << body.getArgumentTypes()[3 + carrier];
  return success();
}

void LoopFrameDestroyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void CopyOutputOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(1),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       &getOperation()->getOpOperand(2),
                       SideEffects::DefaultResource::get());
}

LogicalResult CopyOutputOp::verify() {
  auto source = dyn_cast<MemRefType>(getSource().getType());
  auto target = dyn_cast<MemRefType>(getTarget().getType());
  if (!source || !target || source.getRank() != target.getRank())
    return emitOpError(
        "source and target must have the same ranked memref rank");
  if (source.getElementType() != target.getElementType())
    return emitOpError("source and target element types must match");
  if (!target.getLayout().isIdentity())
    return emitOpError("target must have identity layout");
  for (int64_t dim = 0; dim < source.getRank(); ++dim) {
    int64_t lhs = source.getDimSize(dim);
    int64_t rhs = target.getDimSize(dim);
    if (!ShapedType::isDynamic(lhs) && !ShapedType::isDynamic(rhs) &&
        lhs != rhs)
      return emitOpError(
                 "source and target static extent mismatch at dimension ")
             << dim;
  }
  return success();
}

LogicalResult AllocOutputOp::verify() {
  // Operand convention matches hip.alloc: exactly one Index per dynamic dim of
  // the result memref (static dims come from the type, dynamic dims from
  // operands). Guards against malformed alloc_output before lowering.
  auto memrefTy = cast<MemRefType>(getMemref().getType());
  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      memrefTy.getNumDynamicDims())
    return emitOpError("expected ")
           << memrefTy.getNumDynamicDims() << " dynamic size operand(s), got "
           << getDynamicSizes().size();
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp: policy-neutral externalizable constant carrier
//===----------------------------------------------------------------------===//

ConstantOp::SourceKind ConstantOp::getSourceKind() {
  if (getValueAttr())
    return SourceKind::Inline;
  return getMemoryAddressAttr() ? SourceKind::Memory : SourceKind::File;
}

LogicalResult ConstantOp::verify() {
  RankedTensorType resultType = getResult().getType();
  if (!resultType.hasStaticShape())
    return emitOpError("requires a statically shaped ranked tensor result");

  Type elementType = resultType.getElementType();
  unsigned elementBits = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32() &&
        !elementType.isF64())
      return emitOpError("has unsupported floating-point element type ")
             << elementType;
    elementBits = floatType.getWidth();
  } else if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    elementBits = integerType.getWidth();
    if (elementBits != 1 && elementBits != 8 && elementBits != 16 &&
        elementBits != 32 && elementBits != 64)
      return emitOpError("has unsupported integer element type ")
             << elementType;
  } else {
    return emitOpError("has unsupported element type ") << elementType;
  }

  int64_t numElements = 1;
  for (int64_t dim : resultType.getShape()) {
    if (llvm::MulOverflow(numElements, dim, numElements))
      return emitOpError("result element count overflows int64");
  }
  int64_t elementBytes = static_cast<int64_t>((elementBits + 7) / 8);
  int64_t expectedBytes = 0;
  if (llvm::MulOverflow(numElements, elementBytes, expectedBytes))
    return emitOpError("result byte size overflows int64");

  if (IntegerAttr order = getSerializationOrderAttr())
    if (order.getInt() < 0)
      return emitOpError("serialization order must be non-negative");

  bool hasValue = getValueAttr() != nullptr;
  bool hasLocation = getLocationAttr() != nullptr;
  bool hasOffset = getOffsetAttr() != nullptr;
  bool hasMemoryAddress = getMemoryAddressAttr() != nullptr;
  bool hasSize = getSizeAttr() != nullptr;
  if (hasValue) {
    if (hasLocation || hasOffset || hasMemoryAddress || hasSize)
      return emitOpError("inline source must contain only `value`");
    auto denseValue = dyn_cast<DenseElementsAttr>(getValueAttr());
    if (!denseValue)
      return emitOpError("inline `value` must be a DenseElementsAttr");
    if (denseValue.getType() != resultType)
      return emitOpError("inline `value` type ")
             << denseValue.getType() << " does not match result type "
             << resultType;
    return success();
  }

  bool hasAnyFileField = hasLocation || hasOffset;
  if (hasAnyFileField && hasMemoryAddress)
    return emitOpError("file and memory sources are mutually exclusive");
  if (!hasAnyFileField && !hasMemoryAddress && !hasSize)
    return emitOpError(
        "requires exactly one source: `value`, complete file source, or "
        "complete memory source");
  if (hasAnyFileField && (!hasLocation || !hasOffset || !hasSize))
    return emitOpError(
        "external source requires `location`, `offset`, and `size` together");
  if (hasMemoryAddress && !hasSize)
    return emitOpError(
        "memory source requires `memory_address` and `size` together");
  if (!hasAnyFileField && !hasMemoryAddress)
    return emitOpError(
        "external source requires a file location or memory address");

  int64_t size = getSizeAttr().getInt();
  if (size <= 0)
    return emitOpError("external source `size` must be positive");
  if (size != expectedBytes)
    return emitOpError("external source byte size ")
           << size << " does not match result byte size " << expectedBytes;

  if (hasMemoryAddress) {
    if (getMemoryAddressAttr().getInt() == 0)
      return emitOpError("memory-address source has null address");
  } else {
    StringRef location = getLocationAttr().getValue();
    int64_t offset = getOffsetAttr().getInt();
    if (location.empty())
      return emitOpError("external source `location` must not be empty");
    if (offset < 0)
      return emitOpError("file source `offset` must be non-negative");
    if (offset > std::numeric_limits<int64_t>::max() - size)
      return emitOpError("file source range overflows int64");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoopOp: outlined-body counted/conditional loop
//===----------------------------------------------------------------------===//

void LoopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // v_init and captures are read-only from the caller's perspective. Carrier
  // writes target frame-owned banks, represented by a generic Write effect.
  for (OpOperand &operand : getVInitMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
  for (OpOperand &operand : getCapturesMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoopOp::verify() {
  uint32_t numLoopCarried = getNumLoopCarried();

  // num_loop_carried must equal the v_init count (both modes).
  if (numLoopCarried != getVInit().size())
    return emitOpError("num_loop_carried (")
           << numLoopCarried << ") must equal the v_init operand count ("
           << getVInit().size() << ")";

  bool memrefMode = getDescriptorReturn();
  unsigned expectedResults = numLoopCarried + (memrefMode ? 1u : 0u);
  if (getNumResults() != expectedResults)
    return emitOpError(memrefMode ? "memref mode" : "tensor mode")
           << " expects " << expectedResults << " results, got "
           << getNumResults();
  if (memrefMode && !isa<LoopFrameType>(getResult(numLoopCarried).getType()))
    return emitOpError("descriptor-return mode final result must be "
                       "!hip.loop_frame");

  for (uint32_t i = 0; i < numLoopCarried; ++i) {
    Type initType = getVInit()[i].getType();
    Type resultType = getResult(i).getType();
    bool tensorPair = !memrefMode && isa<RankedTensorType>(initType) &&
                      isa<RankedTensorType>(resultType);
    bool memrefPair =
        memrefMode && isa<MemRefType>(initType) && isa<MemRefType>(resultType);
    if (!tensorPair && !memrefPair)
      return emitOpError("carrier #")
             << i << " must use matching ranked tensor or memref categories";
    if (initType != resultType)
      return emitOpError("result type #")
             << i << " (" << resultType << ") must match v_init type #" << i
             << " (" << initType << ")";
  }
  if (Value parent = getParentFrame()) {
    auto function = getOperation()->getParentOfType<func::FuncOp>();
    if (!function || function.getNumArguments() == 0 ||
        parent != function.getArguments().back() ||
        !isa<LoopFrameType>(parent.getType()))
      return emitOpError(
          "parent frame must be the enclosing body's final argument");
  }

  // body_func symbol resolution is checked in verifySymbolUses() so the
  // verifier driver can share a SymbolTableCollection across all
  // symbol-user ops in the module.
  return success();
}

LogicalResult LoopOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto bodyFunc = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
      *this, getBodyFuncAttr());
  if (!bodyFunc)
    return emitOpError("body_func '")
           << getBodyFunc() << "' does not reference a func.func";

  uint32_t numLoopCarried = getNumLoopCarried();
  unsigned numCaptures = getCaptures().size();
  unsigned expectedArgs = 4 + numLoopCarried + numCaptures;
  if (bodyFunc.getNumArguments() != expectedArgs)
    return emitOpError("body_func argument count mismatch: expected ")
           << expectedArgs
           << " (context, iter, cond, carriers, captures, frame), got "
           << bodyFunc.getNumArguments();
  ArrayRef<Type> args = bodyFunc.getArgumentTypes();
  if (!isa<ContextType>(args[0]))
    return emitOpError("body_func argument #0 must be !hip.context");
  bool descriptorMode = getDescriptorReturn();
  // A zero-carrier/no-capture loop gives One-Shot no tensor value on which to
  // invoke the external model. Its body function boundary is nevertheless
  // converted first; LoopBodyToOutParams materializes the frame token in the
  // immediately following pass.
  bool zeroCarrierBoundaryTransition = !descriptorMode && numLoopCarried == 0 &&
                                       getNumResults() == 0 &&
                                       isa<MemRefType>(args[1]);
  bool bodyDescriptorMode = descriptorMode || zeroCarrierBoundaryTransition;
  auto isModeRankZeroInteger = [bodyDescriptorMode](Type type, unsigned widthA,
                                                    unsigned widthB = 0) {
    ShapedType shaped;
    if (bodyDescriptorMode)
      shaped = dyn_cast<MemRefType>(type);
    else
      shaped = dyn_cast<RankedTensorType>(type);
    if (!shaped || shaped.getRank() != 0)
      return false;
    auto integer = dyn_cast<IntegerType>(shaped.getElementType());
    return integer && (integer.getWidth() == widthA ||
                       (widthB != 0 && integer.getWidth() == widthB));
  };
  if (!isModeRankZeroInteger(args[1], 64))
    return emitOpError("body_func argument #1 must be rank-zero i64 ")
           << (bodyDescriptorMode ? "memref" : "ranked tensor") << " iter";
  if (!isModeRankZeroInteger(args[2], 1, 8))
    return emitOpError("body_func argument #2 must be rank-zero i1/i8 ")
           << (bodyDescriptorMode ? "memref" : "ranked tensor") << " condition";
  if (!isa<LoopFrameType>(args.back()))
    return emitOpError("body_func final argument must be !hip.loop_frame");
  for (unsigned i = 0; i < numLoopCarried; ++i)
    if (args[3 + i] != getResult(i).getType())
      return emitOpError("body_func current carrier #")
             << i << " type " << args[3 + i]
             << " must match loop carrier result type "
             << getResult(i).getType();
  for (unsigned i = 0; i < numCaptures; ++i)
    if (args[3 + numLoopCarried + i] != getCaptures()[i].getType())
      return emitOpError("body_func capture #") << i << " type mismatch";
  for (unsigned i = 0; i < numCaptures; ++i) {
    Type capture = getCaptures()[i].getType();
    bool supported = bodyDescriptorMode ? isa<MemRefType>(capture)
                                        : isa<RankedTensorType>(capture);
    if (!supported)
      return emitOpError("capture #")
             << i << " must be a lowering-supported ranked "
             << (bodyDescriptorMode ? "memref" : "tensor")
             << "; context is threaded separately";
  }

  unsigned condResults = getCondIsPassthrough() ? 0u : 1u;
  unsigned expectedBodyResults = 1 + condResults + numLoopCarried;
  if (bodyFunc.getNumResults() != expectedBodyResults)
    return emitOpError("body_func result count mismatch: expected status")
           << (condResults ? ", condition" : "") << " and " << numLoopCarried
           << " carrier descriptors";
  ArrayRef<Type> results = bodyFunc.getResultTypes();
  if (!results[0].isInteger(32))
    return emitOpError("body_func result #0 must be i32 status");
  if (condResults && results[1] != args[2])
    return emitOpError(
        "body_func condition result must match condition argument type");
  unsigned carrierStart = 1 + condResults;
  for (unsigned i = 0; i < numLoopCarried; ++i) {
    Type bodyResult = results[carrierStart + i];
    Type contract = getResult(i).getType();
    if (bodyResult != contract)
      return emitOpError("body_func carrier result #")
             << i << " type " << bodyResult << " must match loop carrier type "
             << contract;
  }
  return success();
}

// Result types are mechanically the `v_init` operand types — matches the
// `LoopOp::verify` contract, so any caller that uses an InferType-aware
// builder (operand types only, no explicit result types) is verifier-clean
// by construction.
//
LogicalResult
LoopOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  LoopOpAdaptor adaptor(operands, attributes, properties, regions);
  auto vInit = adaptor.getVInit();
  inferredReturnTypes.reserve(vInit.size());
  for (Value v : vInit)
    if (isa<RankedTensorType, MemRefType>(v.getType()))
      inferredReturnTypes.push_back(v.getType());
  if (adaptor.getDescriptorReturn())
    inferredReturnTypes.push_back(LoopFrameType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp: outlined then/else conditional (DPS)
//===----------------------------------------------------------------------===//

MutableOperandRange IfOp::getDpsInitsMutable() { return getOInitMutable(); }

void IfOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getOInitMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
  for (OpOperand &operand : getCapturesMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
}

LogicalResult IfOp::verify() {
  uint32_t numOutputs = getNumOutputs();

  if (numOutputs != getOInit().size())
    return emitOpError("num_outputs (")
           << numOutputs << ") must equal the o_init operand count ("
           << getOInit().size() << ")";

  bool tensorMode = true;
  if (!getOInit().empty())
    tensorMode = isa<RankedTensorType>(getOInit()[0].getType());

  if (tensorMode) {
    if (numOutputs != getNumResults())
      return emitOpError("tensor mode: num_outputs (")
             << numOutputs << ") must equal the result count ("
             << getNumResults() << ")";
    for (uint32_t i = 0; i < numOutputs; ++i)
      if (getOInit()[i].getType() != getResult(i).getType())
        return emitOpError("result type #")
               << i << " (" << getResult(i).getType()
               << ") must match o_init type #" << i << " ("
               << getOInit()[i].getType() << ")";
  } else {
    if (getNumResults() != 0)
      return emitOpError("memref mode must have zero results, got ")
             << getNumResults();
  }
  return success();
}

LogicalResult IfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto thenFn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
      *this, getThenFuncAttr());
  if (!thenFn)
    return emitOpError("then_func '")
           << getThenFunc() << "' does not reference a func.func";
  auto elseFn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
      *this, getElseFuncAttr());
  if (!elseFn)
    return emitOpError("else_func '")
           << getElseFunc() << "' does not reference a func.func";
  return success();
}

LogicalResult
IfOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  IfOpAdaptor adaptor(operands, attributes, properties, regions);
  auto oInit = adaptor.getOInit();
  inferredReturnTypes.reserve(oInit.size());
  for (Value v : oInit) {
    if (isa<RankedTensorType>(v.getType()))
      inferredReturnTypes.push_back(v.getType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Helpers for DPS compute ops (custom parse/print, verify, interfaces)
//===----------------------------------------------------------------------===//

/// Emit memory effects for a DPS compute op: memref inputs read, memref inits
/// write. Non-memref operands (e.g. !hip.context, index scalars) are skipped.
static void emitDpsMemoryEffects(
    ArrayRef<OpOperand *> inputOperands, MutableOperandRange initOperands,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand *operand : inputOperands) {
    if (!isa<MemRefType>(operand->get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : initOperands) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// Generic DPS assembly format: parse / print
//
// Single-output ops:
//   Tensor:  %r = hip.op(%ctx) ins(%a, %b : T1, T2)
//                               outs(%c : T3) -> T3
//   Memref:  hip.op(%ctx) ins(%a, %b : M1, M2) outs(%c : M3)
//
// Multi-output ops use Variadic results:
//   Tensor:  %r:2 = hip.op(%ctx) ins(...) outs(%c, %d : T1, T2) -> T1, T2
//   Memref:  hip.op(%ctx) ins(...) outs(%c, %d : M1, M2)
//===----------------------------------------------------------------------===//

/// Parse a parenthesized comma-separated list of operands with types:
///   `(` ssa-use `,` ... `:` type `,` ... `)`
static ParseResult
parseOperandListWithTypes(OpAsmParser &parser,
                          SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
                          SmallVectorImpl<Type> &types) {
  if (parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  do {
    ops.emplace_back();
    if (parser.parseOperand(ops.back()))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseColon())
    return failure();
  do {
    types.emplace_back();
    if (parser.parseType(types.back()))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();
  if (ops.size() != types.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "operand/type count mismatch");
  return success();
}

/// Print a parenthesized comma-separated list of operands with types.
static void printOperandListWithTypes(OpAsmPrinter &p, ValueRange operands) {
  p << "(";
  llvm::interleaveComma(operands, p, [&](Value v) { p.printOperand(v); });
  p << " : ";
  llvm::interleaveComma(operands.getTypes(), p);
  p << ")";
}

//===----------------------------------------------------------------------===//
// Parse/print for single-init ops (ctx + ins + outs(1) [-> result])
//
// Format:  `(` ctx `)` `ins` `(` ... `)` `outs` `(` ... `)` [attr-dict]
//          [`->` type]
//===----------------------------------------------------------------------===//

/// Parse a single-init DPS compute op.
/// \p numIns       number of data operands in the ins(...) clause.
/// \p extraScalars operands between ctx and ins (e.g. dim0, dim1 for
///                 transpose). They are parsed as `(` ctx `,` scalar... `)`.
static ParseResult parseSingleInitDpsOp(OpAsmParser &parser,
                                        OperationState &result, unsigned numIns,
                                        unsigned extraScalars = 0) {
  OpAsmParser::UnresolvedOperand ctxOperand;
  Type ctxType;
  SmallVector<OpAsmParser::UnresolvedOperand> scalarOps;
  SmallVector<Type> scalarTypes;

  // `(` ctx [`,` scalar `,` scalar ...] `)`
  if (parser.parseLParen() || parser.parseOperand(ctxOperand))
    return failure();
  for (unsigned i = 0; i < extraScalars; ++i) {
    scalarOps.emplace_back();
    if (parser.parseComma() || parser.parseOperand(scalarOps.back()))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  // Resolve ctx as !hip.context
  ctxType = ContextType::get(parser.getContext());
  if (parser.resolveOperand(ctxOperand, ctxType, result.operands))
    return failure();
  // Resolve scalars as index
  for (auto &s : scalarOps)
    if (parser.resolveOperand(s, IndexType::get(parser.getContext()),
                              result.operands))
      return failure();

  // `ins` `(` operands `:` types `)`
  SmallVector<OpAsmParser::UnresolvedOperand> insOps;
  SmallVector<Type> insTypes;
  if (parser.parseKeyword("ins") ||
      parseOperandListWithTypes(parser, insOps, insTypes))
    return failure();
  if (insOps.size() != numIns)
    return parser.emitError(parser.getCurrentLocation(), "expected ")
           << numIns << " ins operand(s)";
  for (unsigned i = 0; i < insOps.size(); ++i)
    if (parser.resolveOperand(insOps[i], insTypes[i], result.operands))
      return failure();

  // `outs` `(` operand `:` type `)`
  SmallVector<OpAsmParser::UnresolvedOperand> outsOps;
  SmallVector<Type> outsTypes;
  if (parser.parseKeyword("outs") ||
      parseOperandListWithTypes(parser, outsOps, outsTypes))
    return failure();
  for (unsigned i = 0; i < outsOps.size(); ++i)
    if (parser.resolveOperand(outsOps[i], outsTypes[i], result.operands))
      return failure();

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Optional `->` result type(s) for tensor mode
  if (succeeded(parser.parseOptionalArrow())) {
    SmallVector<Type> resultTypes;
    do {
      resultTypes.emplace_back();
      if (parser.parseType(resultTypes.back()))
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    result.addTypes(resultTypes);
  }
  return success();
}

/// Print a single-init DPS compute op.
static void printSingleInitDpsOp(OpAsmPrinter &p, Operation *op, Value ctx,
                                 ValueRange scalarArgs, ValueRange ins,
                                 ValueRange outs) {
  p << "(";
  p.printOperand(ctx);
  for (Value s : scalarArgs) {
    p << ", ";
    p.printOperand(s);
  }
  p << ") ins";
  printOperandListWithTypes(p, ins);
  p << " outs";
  printOperandListWithTypes(p, outs);
  p.printOptionalAttrDict(op->getAttrs());
  if (op->getNumResults() > 0) {
    p << " -> ";
    llvm::interleaveComma(op->getResultTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// ConvOp: ins(input, weights, bias), outs(output)
// Uses declarative assemblyFormat - parse/print auto-generated by TableGen
//===----------------------------------------------------------------------===//

MutableOperandRange ConvOp::getDpsInitsMutable() { return getOutputMutable(); }

void ConvOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp: ins(input, weights, bias), outs(output)
// Uses declarative assemblyFormat - parse/print auto-generated by TableGen
//===----------------------------------------------------------------------===//

MutableOperandRange ConvTransposeOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ConvTransposeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// HipblasltMatmulOp: ins(A, B), outs(C)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatmulOp: ins(A, B), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange MatmulOp::getDpsInitsMutable() {
  // 0=ctx, 1=A, 2=B, 3=output
  return getOutputMutable();
}

void MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult MatmulOp::verify() {
  // First the cross-cutting DPS contract (all-tensor-or-all-memref +
  // result-count parity); failures here also rule out bogus operand types,
  // so the matmul shape check below can rely on getShapeOf().
  if (failed(verifyDpsComputeOp(*this, {getA(), getB(), getOutput()},
                                /*numInits=*/1)))
    return failure();

  ArrayRef<int64_t> aShape = detail::getShapeOf(getA());
  ArrayRef<int64_t> bShape = detail::getShapeOf(getB());
  if (failed(mlir::hip::verifyHipOpShape(*this, [&] {
        return mlir::hip::inferMatmulShape(aShape, bShape,
                                           [&] { return this->emitOpError(); });
      })))
    return failure();
  return mlir::hip::verifyStridedBatchMatmul(
      aShape, bShape, [&]() { return this->emitOpError(); });
}

// `MatmulOp::reifyResultShapes` lives in
// `lib/Dialect/IR/HipReifyResultShapesImpl.cpp`. See the header banner in
// that file and `docs/design/hip-shape-inference.md` for the rationale.

//===----------------------------------------------------------------------===//
// RmsNormOp: ins(input, scale), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange RmsNormOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void RmsNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult RmsNormOp::verify() {
  return verifyDpsComputeOp(*this, {getInput(), getScale(), getOutput()},
                            /*numInits=*/1);
}

//===----------------------------------------------------------------------===//
// SkipRmsNormOp: ins(input, skip, gamma, [bias])  outs(...variadic...)
//===----------------------------------------------------------------------===//

MutableOperandRange SkipRmsNormOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

void SkipRmsNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult SkipRmsNormOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// LayerNormOp: ins(input, scale, [bias]) outs(output, [mean, [inv_std]])
//===----------------------------------------------------------------------===//

MutableOperandRange LayerNormOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

void LayerNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult LayerNormOp::verify() {
  // Outputs cardinality must be 1, 2, or 3 (output, [mean], [inv_std]).
  unsigned numOutputs = getOutputs().size();
  if (numOutputs < 1 || numOutputs > 3)
    return emitOpError("expected 1 to 3 output buffers (output, [mean, "
                       "[inv_std]]), got ")
           << numOutputs;

  // Defer the all-tensor-or-all-memref consistency check to the shared helper
  // so this op behaves like the rest of the DPS family.
  SmallVector<Value> dataOperands;
  dataOperands.push_back(getInput());
  dataOperands.push_back(getScale());
  if (Value b = getBias())
    dataOperands.push_back(b);
  for (Value out : getOutputs())
    dataOperands.push_back(out);
  return verifyDpsComputeOp(*this, dataOperands, /*numInits=*/numOutputs);
}

//===----------------------------------------------------------------------===//
// RopeOp: ins(input, [position_ids], cos_cache, sin_cache), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange RopeOp::getDpsInitsMutable() {
  // position_ids is Optional, so the raw operand index of `output` shifts when
  // it is absent. Use the generated accessor instead of a hardcoded index.
  return getOutputMutable();
}

void RopeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// MiopenAddOp: ins(A, B), outs(C)
//===----------------------------------------------------------------------===//

MutableOperandRange MiopenAddOp::getDpsInitsMutable() { return getCMutable(); }

void MiopenAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

ParseResult MiopenAddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void MiopenAddOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{}, {getA(), getB()},
                       {getC()});
}

//===----------------------------------------------------------------------===//
// MulOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange MulOp::getDpsInitsMutable() { return getOutputMutable(); }

void MulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void MulOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{},
                       {getLhs(), getRhs()}, {getOutput()});
}

//===----------------------------------------------------------------------===//
// AddOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange AddOp::getDpsInitsMutable() { return getOutputMutable(); }

void AddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void AddOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{},
                       {getLhs(), getRhs()}, {getOutput()});
}

//===----------------------------------------------------------------------===//
// MiopenSoftmaxOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange MiopenSoftmaxOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void MiopenSoftmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult MiopenSoftmaxOp::verify() {
  if (failed(
          verifyDpsComputeOp(*this, {getInput(), getOutput()}, /*numInits=*/1)))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (inputType.getRank() == 0)
    return emitOpError("requires positive-rank input and output");
  if (inputType.getRank() != outputType.getRank())
    return emitOpError("input and output ranks must match");

  Type elementType = inputType.getElementType();
  if (outputType.getElementType() != elementType)
    return emitOpError("input and output element types must match");
  if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32())
    return emitOpError("unsupported element type ")
           << elementType << "; expected f16, bf16, or f32";

  for (int64_t axis : llvm::seq<int64_t>(0, inputType.getRank())) {
    int64_t inputExtent = inputType.getDimSize(axis);
    int64_t outputExtent = outputType.getDimSize(axis);
    if (!ShapedType::isDynamic(inputExtent) &&
        !ShapedType::isDynamic(outputExtent) && inputExtent != outputExtent)
      return emitOpError("input and output dimensions must match at axis ")
             << axis << ": " << inputExtent << " vs " << outputExtent;
  }
  return success();
}

ParseResult MiopenSoftmaxOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/1);
}

void MiopenSoftmaxOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{}, {getInput()},
                       {getOutput()});
}

//===----------------------------------------------------------------------===//
// TransposeOp: ins(input), outs(output), attrs: perm
//===----------------------------------------------------------------------===//

MutableOperandRange TransposeOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void TransposeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult TransposeOp::verify() {
  if (failed(
          verifyDpsComputeOp(*this, {getInput(), getOutput()}, /*numInits=*/1)))
    return failure();

  // Determine input rank when available (ranked tensor or memref).
  int64_t rank = -1;
  if (auto t = dyn_cast<RankedTensorType>(getInput().getType()))
    rank = t.getRank();
  else if (auto m = dyn_cast<MemRefType>(getInput().getType()))
    rank = m.getRank();

  ArrayAttr permAttr = getPerm();
  if (rank >= 0 && static_cast<int64_t>(permAttr.size()) != rank)
    return emitOpError("perm length (")
           << permAttr.size() << ") must match input rank (" << rank << ")";

  // perm must be a permutation of [0, rank).
  llvm::SmallVector<bool> seen(permAttr.size(), false);
  for (Attribute a : permAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(a);
    if (!intAttr)
      return emitOpError("perm must be a list of integers");
    int64_t v = intAttr.getValue().getSExtValue();
    if (v < 0 || v >= static_cast<int64_t>(permAttr.size()))
      return emitOpError("perm value ") << v << " is out of range";
    if (seen[v])
      return emitOpError("perm must be a permutation (duplicate value ")
             << v << ")";
    seen[v] = true;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp: ins(indices, table), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GatherOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// OneHotOp: ins(indices, depth, values), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange OneHotOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void OneHotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CompressOp: ins(input, condition), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange CompressOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void CompressOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ScatterElementsOp: ins(data, indices, updates), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ScatterElementsOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ScatterElementsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GatherElementsOp: ins(data, indices), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GatherElementsOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void GatherElementsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// TopKOp: ins(x, k), outs(values, indices)
//===----------------------------------------------------------------------===//

// Operand order is (ctx, x, k, values, indices); the two DPS inits are the
// trailing contiguous range.
MutableOperandRange TopKOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*start=*/3, /*length=*/2);
}

void TopKOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// RangeOp: ins(start, limit, delta), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange RangeOp::getDpsInitsMutable() { return getOutputMutable(); }

void RangeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SiluOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange SiluOp::getDpsInitsMutable() { return getOutputMutable(); }

void SiluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult SiluOp::verify() {
  return verifyDpsComputeOp(*this, {getInput(), getOutput()}, /*numInits=*/1);
}

ParseResult SiluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/1);
}

void SiluOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{}, {getInput()},
                       {getOutput()});
}

//===----------------------------------------------------------------------===//
// SigmoidOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SigmoidOp::getDpsInitsMutable() { return getYMutable(); }

void SigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// TanhOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange TanhOp::getDpsInitsMutable() { return getYMutable(); }

void TanhOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SoftplusOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SoftplusOp::getDpsInitsMutable() { return getYMutable(); }

void SoftplusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GeluOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GeluOp::getDpsInitsMutable() { return getOutputMutable(); }

void GeluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// BiasGeluOp: ins(data, bias), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange BiasGeluOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void BiasGeluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// FastGeluOp: ins(input, [bias]), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange FastGeluOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void FastGeluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// LeakyReluOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange LeakyReluOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void LeakyReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// PoolOp: ins(input), outs([output] or [output, indices])
//===----------------------------------------------------------------------===//

MutableOperandRange PoolOp::getDpsInitsMutable() { return getOutputsMutable(); }

void PoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ResizeOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ResizeOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ResizeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GlobalPoolOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GlobalPoolOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void GlobalPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReciprocalOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange ReciprocalOp::getDpsInitsMutable() { return getYMutable(); }

void ReciprocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SqrtOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SqrtOp::getDpsInitsMutable() { return getYMutable(); }

void SqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SubOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange SubOp::getDpsInitsMutable() { return getOutputMutable(); }

void SubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// MinOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange MinOp::getDpsInitsMutable() { return getOutputMutable(); }

void MinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// MaxOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange MaxOp::getDpsInitsMutable() { return getOutputMutable(); }

void MaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// WhereOp: ins(condition, x, y), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange WhereOp::getDpsInitsMutable() { return getOutputMutable(); }

void WhereOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CastOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange CastOp::getDpsInitsMutable() { return getOutputMutable(); }

void CastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceSumOp: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceSumOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceMeanOp: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceMeanOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceMeanOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceL2Op: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceL2Op::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceL2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceMaxOp: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceMaxOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceMinOp: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceMinOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// MatMulNBitsOp
//===----------------------------------------------------------------------===//

MutableOperandRange MatMulNBitsOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void MatMulNBitsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult MatMulNBitsOp::verify() {
  SmallVector<Value> operands = {getA(), getB(), getScales()};
  if (Value zeroPoints = getZeroPoints())
    operands.push_back(zeroPoints);
  if (Value gIdx = getGIdx())
    operands.push_back(gIdx);
  if (Value bias = getBias())
    operands.push_back(bias);
  operands.push_back(getOutput());
  if (failed(verifyDpsComputeOp(*this, operands, /*numInits=*/1)))
    return failure();

  auto aType = cast<ShapedType>(getA().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (outputType.getElementType() != aType.getElementType())
    return emitOpError("output element type must match A");
  if (getK() < 0 || getN() < 0 || getBlockSize() <= 0)
    return emitOpError(
        "K and N must be non-negative and block_size must be positive");
  if (aType.getRank() < 1)
    return emitOpError("A must have rank at least 1");
  int64_t contraction = aType.getDimSize(aType.getRank() - 1);
  if (!ShapedType::isDynamic(contraction) && contraction != getK())
    return emitOpError("A's trailing extent must equal K");

  FailureOr<SmallVector<int64_t>> expected =
      inferMatMulNBitsShape(aType.getShape(), getN());
  if (failed(expected))
    return emitOpError("could not infer MatMulNBits output shape");
  return verifyHipOpShape(
      *this, [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; });
}

//===----------------------------------------------------------------------===//
// QMoEOp
//===----------------------------------------------------------------------===//

MutableOperandRange QMoEOp::getDpsInitsMutable() { return getOutputMutable(); }

void QMoEOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GatherBlockQuantizedOp: ins(data, indices, scales, [zero_points])
//                          outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GatherBlockQuantizedOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void GatherBlockQuantizedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CausalConvWithStateOp: ins(input, weight, [bias], [past_state])
//                        outs(output, present_state)
//===----------------------------------------------------------------------===//

MutableOperandRange CausalConvWithStateOp::getDpsInitsMutable() {
  // DPS inits: output, present_state (always 2)
  // Count actual inputs before the inits
  unsigned numInputs = 1; // ctx
  numInputs++;            // input
  numInputs++;            // weight
  if (getBias())
    ++numInputs;
  if (getPastState())
    ++numInputs;

  return MutableOperandRange(*this, /*start=*/numInputs, /*length=*/2);
}

void CausalConvWithStateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

MutableOperandRange GemmOp::getDpsInitsMutable() { return getOutputMutable(); }

void GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult GemmOp::verify() {
  SmallVector<Value> dataOperands = {getInputA(), getInputB()};
  if (getInputC())
    dataOperands.push_back(getInputC());
  dataOperands.push_back(getOutput());
  // The cross-cutting DPS contract first (all-tensor-or-all-memref +
  // result-count parity); it also rules out non-shaped data operands, so the
  // shape check below can rely on getShapeOf().
  if (failed(verifyDpsComputeOp(*this, dataOperands, /*numInits=*/1)))
    return failure();

  std::optional<ArrayRef<int64_t>> cShape;
  if (getInputC())
    cShape = detail::getShapeOf(getInputC());
  return mlir::hip::verifyHipOpShape(*this, [&] {
    return mlir::hip::inferGemmShape(
        detail::getShapeOf(getInputA()), detail::getShapeOf(getInputB()),
        cShape, getTransA(), getTransB(), [&] { return this->emitOpError(); });
  });
}

//===----------------------------------------------------------------------===//
// GqaOp: Full MS spec implementation
//        ins(query, [key, value, past_key, past_value], seqlens_k,
//        total_seq_len,
//            [cos_cache, sin_cache, position_ids, attention_bias, head_sink,
//             k_scale, v_scale])
//        outs(output, present_state)
//===----------------------------------------------------------------------===//

MutableOperandRange LinearAttentionOp::getDpsInitsMutable() {
  // Operand segments:
  //   ctx(1), query(1), key(1), value(1),
  //   past_state(0|1), decay(0|1), beta(0|1),
  //   output(1), present_state(1)
  unsigned numInputs = 4; // ctx, query, key, value
  if (getPastState())
    ++numInputs;
  if (getDecay())
    ++numInputs;
  if (getBeta())
    ++numInputs;

  // DPS inits: output, present_state (always 2)
  return MutableOperandRange(*this, /*start=*/numInputs, /*length=*/2);
}

void LinearAttentionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// MultiHeadAttentionOp:
//   ins(query, [key, value, bias, key_padding_mask, attention_bias,
//                past_key, past_value, past_sequence_length,
//                cache_indirection])
//   outs(output, [present_key, present_value, qk])
//===----------------------------------------------------------------------===//

MutableOperandRange MultiHeadAttentionOp::getDpsInitsMutable() {
  // Count actual inputs (skip ctx which is always first)
  unsigned numInputs = 1; // ctx
  if (getQuery())
    ++numInputs;
  if (getKey())
    ++numInputs;
  if (getValue())
    ++numInputs;
  if (getBias())
    ++numInputs;
  if (getKeyPaddingMask())
    ++numInputs;
  if (getAttentionBias())
    ++numInputs;
  if (getPastKey())
    ++numInputs;
  if (getPastValue())
    ++numInputs;
  if (getPastSequenceLength())
    ++numInputs;
  if (getCacheIndirection())
    ++numInputs;

  // DPS inits: output (always), [present_key, present_value, qk] (optional)
  unsigned numInits = 1;
  if (getPresentKey())
    ++numInits;
  if (getPresentValue())
    ++numInits;
  if (getQk())
    ++numInits;

  return MutableOperandRange(*this, /*start=*/numInputs, /*length=*/numInits);
}

void MultiHeadAttentionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GqaOp: ins(query, [key, value, past_key, past_value,]
//             seqlens_k, total_seq_len, [cos_cache, ...])
//        outs(output, present_key, present_value, [output_qk])
//===----------------------------------------------------------------------===//

MutableOperandRange GqaOp::getDpsInitsMutable() {
  // DPS inits start after all inputs
  // Required inputs: ctx(0), query(1), seqlens_k(6), total_seq_len(7)
  // Optional inputs: key(2), value(3), past_key(4), past_value(5),
  //                  cos_cache(8-14: 7 optional inputs)
  // Then DPS outputs: output, present_key, present_value, [output_qk]

  // Count actual inputs (skip ctx which is always first)
  unsigned numInputs = 1; // ctx
  if (getQuery())
    ++numInputs;
  if (getKey())
    ++numInputs;
  if (getValue())
    ++numInputs;
  if (getPastKey())
    ++numInputs;
  if (getPastValue())
    ++numInputs;
  if (getSeqlensK())
    ++numInputs;
  if (getTotalSeqLen())
    ++numInputs;
  if (getCosCache())
    ++numInputs;
  if (getSinCache())
    ++numInputs;
  if (getPositionIds())
    ++numInputs;
  if (getAttentionBias())
    ++numInputs;
  if (getHeadSink())
    ++numInputs;
  if (getKScale())
    ++numInputs;
  if (getVScale())
    ++numInputs;

  // DPS inits: output, present_key, present_value, [output_qk]
  unsigned numInits = 3;
  if (getOutputQk())
    numInits = 4;

  return MutableOperandRange(*this, /*start=*/numInputs, /*length=*/numInits);
}

void GqaOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult GqaOp::verify() {
  // Verify quantization parameter consistency
  auto kQuantType = getKQuantType().str();
  auto vQuantType = getVQuantType().str();

  // If k_quant_type != "NONE", k_scale must be provided
  if (kQuantType != "NONE" && !getKScale()) {
    return emitOpError("k_quant_type is '")
           << kQuantType << "' but k_scale is not provided";
  }

  // If v_quant_type != "NONE", v_scale must be provided
  if (vQuantType != "NONE" && !getVScale()) {
    return emitOpError("v_quant_type is '")
           << vQuantType << "' but v_scale is not provided";
  }

  // Verify quant_type values
  if (kQuantType != "NONE" && kQuantType != "PER_TENSOR" &&
      kQuantType != "PER_CHANNEL") {
    return emitOpError("k_quant_type must be 'NONE', 'PER_TENSOR', or "
                       "'PER_CHANNEL', got '")
           << kQuantType << "'";
  }

  if (vQuantType != "NONE" && vQuantType != "PER_TENSOR" &&
      vQuantType != "PER_CHANNEL") {
    return emitOpError("v_quant_type must be 'NONE', 'PER_TENSOR', or "
                       "'PER_CHANNEL', got '")
           << vQuantType << "'";
  }

  // Verify bit width
  int64_t bitWidth = getKvCacheBitWidth();
  if (bitWidth != 4 && bitWidth != 8) {
    return emitOpError("kv_cache_bit_width must be 4 or 8, got ") << bitWidth;
  }

  // Verify qk_output mode
  int64_t qkOutput = getQkOutput();
  if (qkOutput < 0 || qkOutput > 2) {
    return emitOpError("qk_output must be 0, 1, or 2, got ") << qkOutput;
  }

  // If qk_output != 0, output_qk must be provided
  if (qkOutput != 0 && !getOutputQk()) {
    return emitOpError("qk_output is ")
           << qkOutput << " but output_qk buffer is not provided";
  }

  // Verify paired optional inputs: cos_cache/sin_cache must both be present or
  // both absent
  bool hasCosCache = getCosCache() != nullptr;
  bool hasSinCache = getSinCache() != nullptr;
  if (hasCosCache != hasSinCache) {
    return emitOpError("cos_cache and sin_cache must both be provided or both "
                       "be omitted (found cos_cache=")
           << (hasCosCache ? "present" : "absent")
           << ", sin_cache=" << (hasSinCache ? "present" : "absent") << ")";
  }

  // Verify paired optional inputs: key/value must both be present or both
  // absent
  bool hasKey = getKey() != nullptr;
  bool hasValue = getValue() != nullptr;
  if (hasKey != hasValue) {
    return emitOpError("key and value must both be provided or both be omitted "
                       "(found key=")
           << (hasKey ? "present" : "absent")
           << ", value=" << (hasValue ? "present" : "absent") << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// HipDNNGraphOp: ins(variadic), outs(variadic)
//===----------------------------------------------------------------------===//

MutableOperandRange HipDNNGraphOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

void HipDNNGraphOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// EqualOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange EqualOp::getDpsInitsMutable() { return getOutputMutable(); }

void EqualOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// DivOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange DivOp::getDpsInitsMutable() { return getOutputMutable(); }

void DivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// AbsOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange AbsOp::getDpsInitsMutable() { return getYMutable(); }

void AbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// NegOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange NegOp::getDpsInitsMutable() { return getYMutable(); }

void NegOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// NotOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange NotOp::getDpsInitsMutable() { return getYMutable(); }

void NotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// OrOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange OrOp::getDpsInitsMutable() { return getOutputMutable(); }

void OrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// AndOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange AndOp::getDpsInitsMutable() { return getOutputMutable(); }

void AndOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CosOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange CosOp::getDpsInitsMutable() { return getYMutable(); }

void CosOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SinOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SinOp::getDpsInitsMutable() { return getYMutable(); }

void SinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CeilOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange CeilOp::getDpsInitsMutable() { return getYMutable(); }

void CeilOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ExpOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange ExpOp::getDpsInitsMutable() { return getYMutable(); }

void ExpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// LogOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange LogOp::getDpsInitsMutable() { return getYMutable(); }

void LogOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// CumSumOp: ins(x, axis), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange CumSumOp::getDpsInitsMutable() { return getYMutable(); }

void CumSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// PadOp: ins(data, pads, [constant_value], [axes]), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange PadOp::getDpsInitsMutable() { return getOutputMutable(); }

void PadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// TileOp: ins(input, repeats), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange TileOp::getDpsInitsMutable() { return getOutputMutable(); }

void TileOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult TileOp::verify() {
  if (failed(verifyDpsComputeOp(*this, {getInput(), getRepeats(), getOutput()},
                                /*numInits=*/1)))
    return failure();
  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto repeatsType = dyn_cast<ShapedType>(getRepeats().getType());
  auto outputType = dyn_cast<ShapedType>(getOutput().getType());
  if (!inputType || !inputType.hasRank() || !repeatsType ||
      !repeatsType.hasRank() || !outputType || !outputType.hasRank())
    return emitOpError("input, repeats, and output must be ranked");
  if (repeatsType.getRank() != 1 ||
      !repeatsType.getElementType().isInteger(64) ||
      repeatsType.isDynamicDim(0) ||
      repeatsType.getDimSize(0) != inputType.getRank())
    return emitOpError(
        "repeats must be static-length rank-1 i64 matching input rank");
  if (outputType.getRank() != inputType.getRank())
    return emitOpError("output rank must match input rank");

  std::optional<ArrayRef<int64_t>> staticRepeats = getStaticRepeats();
  if (!staticRepeats)
    return success();
  FailureOr<SmallVector<int64_t>> expected =
      mlir::hip::inferTileShape(inputType.getShape(), *staticRepeats);
  if (failed(expected))
    return emitOpError(
        "static_repeats must match input rank and be non-negative");
  return mlir::hip::verifyHipOpShape(
      *this, [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; });
}

//===----------------------------------------------------------------------===//
// ExpandOp: ins(input, shape), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ExpandOp::getDpsInitsMutable() {
  return getOutputMutable();
}

LogicalResult ExpandOp::verify() {
  if (failed(verifyDpsComputeOp(getOperation(),
                                {getInput(), getShape(), getOutput()},
                                /*numInits=*/1)))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto shapeType = cast<ShapedType>(getShape().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (shapeType.getRank() != 1)
    return emitOpError("shape must be rank 1");
  if (shapeType.isDynamicDim(0))
    return emitOpError("shape length must be static");
  Type shapeElementType = shapeType.getElementType();
  if (!shapeElementType.isInteger(32) && !shapeElementType.isInteger(64))
    return emitOpError("shape element type must be i32 or i64");

  Type elementType = inputType.getElementType();
  if (outputType.getElementType() != elementType)
    return emitOpError("input and output element types must match");
  if (!elementType.isF16() && !elementType.isF32() &&
      !elementType.isInteger(32) && !elementType.isInteger(64) &&
      !elementType.isUnsignedInteger(8))
    return emitOpError("unsupported input element type ")
           << elementType << "; expected f16, f32, i32, i64, or ui8";

  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  int64_t targetRank = shapeType.getDimSize(0);
  if (outputRank != std::max(inputRank, targetRank))
    return emitOpError("output rank must equal max(input rank, shape length)");
  if (outputRank > 8)
    return emitOpError("output rank exceeds the runtime maximum of 8");

  int64_t inputPadding = outputRank - inputRank;
  for (int64_t axis : llvm::seq<int64_t>(inputPadding, outputRank)) {
    int64_t inputExtent = inputType.getDimSize(axis - inputPadding);
    int64_t outputExtent = outputType.getDimSize(axis);
    if (!ShapedType::isDynamic(inputExtent) &&
        !ShapedType::isDynamic(outputExtent) && inputExtent != 1 &&
        inputExtent != outputExtent)
      return emitOpError("statically incompatible input/output extent at axis ")
             << axis << ": " << inputExtent << " vs " << outputExtent;
  }

  SmallVector<int64_t> targetShape;
  if (!matchConstantIntTensor(getShape(), targetShape, /*expectedRank=*/1))
    return success();
  FailureOr<SmallVector<int64_t>> expected =
      inferExpandShape(inputType.getShape(), targetShape);
  if (failed(expected))
    return emitOpError("constant target is negative or broadcast-incompatible");
  for (int64_t axis : llvm::seq<int64_t>(0, outputRank)) {
    int64_t expectedExtent = (*expected)[axis];
    int64_t outputExtent = outputType.getDimSize(axis);
    if (!ShapedType::isDynamic(expectedExtent) &&
        !ShapedType::isDynamic(outputExtent) && expectedExtent != outputExtent)
      return emitOpError("output extent contradicts constant target at axis ")
             << axis << ": expected " << expectedExtent << ", got "
             << outputExtent;
  }
  return success();
}

void ExpandOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReduceProdOp: ins(data, axes), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ReduceProdOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ReduceProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// LessOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange LessOp::getDpsInitsMutable() { return getOutputMutable(); }

void LessOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// GatherNDOp: ins(data, indices), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange GatherNDOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void GatherNDOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SliceOp: ins(data, starts, ends, [axes], [steps]), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange SliceOp::getDpsInitsMutable() { return getOutputMutable(); }

void SliceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// SignOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SignOp::getDpsInitsMutable() { return getYMutable(); }

void SignOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ModOp: ins(lhs, rhs), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ModOp::getDpsInitsMutable() { return getOutputMutable(); }

void ModOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ScatterNDOp: ins(data, indices, updates), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ScatterNDOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void ScatterNDOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// NonZeroOp: ins(x), outs(y, count)
//===----------------------------------------------------------------------===//

// Operand order is (ctx, x, y, count); the two DPS inits (y, count) are the
// trailing contiguous range, as required by DestinationStyleOpInterface.
MutableOperandRange NonZeroOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*start=*/2, /*length=*/2);
}

void NonZeroOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ReadbackDimOp: ins(scalar) -> index
//===----------------------------------------------------------------------===//

// Reads the device `scalar` buffer (a Read effect on the operand) plus a stream
// synchronization. The Read-after-Write against the producing kernel's write
// keeps it correctly ordered (and, with no speculatable trait, un-hoistable).
// The known non-allocating effect also keeps this op valid if a custom pipeline
// enables ownership-based buffer deallocation.
void ReadbackDimOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Operand order is (ctx, scalar); attach the Read to the scalar operand.
  if (isa<MemRefType>(getScalar().getType()))
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(1),
                         SideEffects::DefaultResource::get());
}

void ReadbackScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Operand order is (ctx, scalar); attach the Read to the scalar operand.
  if (isa<MemRefType>(getScalar().getType()))
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(1),
                         SideEffects::DefaultResource::get());
}

void ReadbackShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Operand order is (ctx, vector); attach the Read to the vector operand.
  if (isa<MemRefType>(getVector().getType()))
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(1),
                         SideEffects::DefaultResource::get());
}

LogicalResult ReadbackShapeOp::verify() {
  auto vectorType = dyn_cast<ShapedType>(getVector().getType());
  if (!vectorType || !vectorType.hasRank() || vectorType.getRank() != 1)
    return emitOpError("vector must be rank 1");
  if (!vectorType.getElementType().isInteger(64))
    return emitOpError("vector element type must be i64");
  int64_t count = getCount();
  if (count < 0)
    return emitOpError("count must be non-negative");
  if (vectorType.isDynamicDim(0) || vectorType.getDimSize(0) != count)
    return emitOpError("vector length must be static and equal count");
  if (static_cast<int64_t>(getNumResults()) != count)
    return emitOpError("result count must equal count");
  if (llvm::any_of(getResultTypes(),
                   [](Type type) { return !isa<IndexType>(type); }))
    return emitOpError("all results must be index type");
  return success();
}

void ReadbackControlOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &source : getSourcesMutable()) {
    if (isa<MemRefType>(source.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &source,
                           SideEffects::DefaultResource::get());
  }
}

void CheckedExpandExtentOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult CheckedExpandExtentOp::verify() {
  if (getExpectedExtentAttr().getInt() < -1)
    return emitOpError("expected_extent must be -1 or non-negative");
  return success();
}

LogicalResult ReadbackControlOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ReadbackControlOpAdaptor adaptor(operands, attributes, properties, regions);
  SmallVector<Type> sourceTypes;
  llvm::transform(adaptor.getSources(), std::back_inserter(sourceTypes),
                  [](Value value) { return value.getType(); });
  FailureOr<ReadbackControlLayout> layout =
      getReadbackControlLayout(sourceTypes);
  if (failed(layout))
    return failure();

  inferredReturnTypes.push_back(IntegerType::get(context, 1));
  inferredReturnTypes.append(layout->totalCount, IntegerType::get(context, 64));
  return success();
}

LogicalResult ReadbackControlOp::verify() {
  SmallVector<Type> sourceTypes(getSources().getTypes());
  FailureOr<ReadbackControlLayout> layout =
      getReadbackControlLayout(sourceTypes);
  if (failed(layout))
    return emitOpError(
        "requires one or more statically-sized rank-0/rank-1 i32/i64 sources");

  for (auto [index, source] : llvm::enumerate(getSources())) {
    auto memref = dyn_cast<MemRefType>(source.getType());
    if (!memref || memref.getRank() == 0)
      continue;
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(memref.getStridesAndOffset(strides, offset)) ||
        strides.size() != 1 || strides.front() != 1)
      return emitOpError("memref source #")
             << index << " must be contiguous (unit rank-1 stride)";
  }

  if (!getValid().getType().isInteger(1))
    return emitOpError("valid result must be i1");
  if (static_cast<int64_t>(getValues().size()) != layout->totalCount)
    return emitOpError("expected ")
           << layout->totalCount << " i64 value results, got "
           << getValues().size();
  if (llvm::any_of(getValues().getTypes(),
                   [](Type type) { return !type.isInteger(64); }))
    return emitOpError("all flattened value results must be i64");
  return success();
}

//===----------------------------------------------------------------------===//
// SizeOp: ins(x), outs(y)
//===----------------------------------------------------------------------===//

MutableOperandRange SizeOp::getDpsInitsMutable() { return getYMutable(); }

void SizeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.cpp.inc"
