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
// LoopOp: outlined-body counted/conditional loop (DPS)
//===----------------------------------------------------------------------===//

MutableOperandRange LoopOp::getDpsInitsMutable() { return getVInitMutable(); }

void LoopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Match the DPS convention used by other Hip ops (see emitDpsMemoryEffects):
  //   - v_init are DPS inits  -> Write (the body accumulates into them across
  //     iterations; without Write, post-bufferization DCE drops the entire
  //     loop because it appears side-effect-free with no live results).
  //   - captures are data inputs -> Read.
  //   - ctx / index / i1 operands are skipped (not memref).
  // Pre-bufferization v_init are tensors (not memref) -- the isa<MemRefType>
  // filter naturally suppresses effects then, which is correct: the op's
  // tensor result use prevents DCE in tensor mode.
  for (OpOperand &operand : getVInitMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
  for (OpOperand &operand : getCapturesMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
}

LogicalResult LoopOp::verify() {
  uint32_t numLoopCarried = getNumLoopCarried();

  // num_loop_carried must equal the v_init count (both modes).
  if (numLoopCarried != getVInit().size())
    return emitOpError("num_loop_carried (")
           << numLoopCarried << ") must equal the v_init operand count ("
           << getVInit().size() << ")";

  // Determine mode from v_init type when present; otherwise default to
  // tensor mode (a num_loop_carried=0 op makes no sense, but allow it).
  bool tensorMode = true;
  if (!getVInit().empty())
    tensorMode = isa<RankedTensorType>(getVInit()[0].getType());

  if (tensorMode) {
    // Tensor mode: results count == num_loop_carried, types match v_init.
    if (numLoopCarried != getNumResults())
      return emitOpError("tensor mode: num_loop_carried (")
             << numLoopCarried << ") must equal the result count ("
             << getNumResults() << ")";
    for (uint32_t i = 0; i < numLoopCarried; ++i)
      if (getVInit()[i].getType() != getResult(i).getType())
        return emitOpError("result type #")
               << i << " (" << getResult(i).getType()
               << ") must match v_init type #" << i << " ("
               << getVInit()[i].getType() << ")";
  } else {
    // Memref mode (post-bufferization): no results, v_init carries writes.
    if (getNumResults() != 0)
      return emitOpError("memref mode must have zero results, got ")
             << getNumResults();
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
  return success();
}

// Result types are mechanically the `v_init` operand types — matches the
// `LoopOp::verify` contract, so any caller that uses an InferType-aware
// builder (operand types only, no explicit result types) is verifier-clean
// by construction.
//
// Memref-mode v_init produces 0 result types (DPS post-bufferization
// convention; the verifier rejects mixed mode anyway).
LogicalResult
LoopOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  LoopOpAdaptor adaptor(operands, attributes, properties, regions);
  auto vInit = adaptor.getVInit();
  inferredReturnTypes.reserve(vInit.size());
  for (Value v : vInit) {
    if (isa<RankedTensorType>(v.getType()))
      inferredReturnTypes.push_back(v.getType());
  }
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

LogicalResult ConvOp::verify() {
  SmallVector<Value> dataOperands = {getInput(), getWeights()};
  if (getBias())
    dataOperands.push_back(getBias());
  dataOperands.push_back(getOutput());
  if (failed(verifyDpsComputeOp(*this, dataOperands, /*numInits=*/1)))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto weightType = cast<ShapedType>(getWeights().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  Type elementType = inputType.getElementType();
  if (!isa<FloatType>(elementType) ||
      weightType.getElementType() != elementType ||
      outputType.getElementType() != elementType)
    return emitOpError(
        "input, weights, and output must have the same floating-point type");

  FailureOr<SmallVector<int64_t>> expected = mlir::hip::inferConvShape(
      detail::getShapeOf(getInput()), detail::getShapeOf(getWeights()),
      detail::getI64Array(getKernelShape()), detail::getI64Array(getStrides()),
      detail::getI64Array(getPads()), detail::getI64Array(getDilations()),
      getGroup(), [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();
  if (failed(mlir::hip::verifyHipOpShape(
          *this,
          [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; })))
    return failure();

  if (Value bias = getBias()) {
    auto biasType = cast<ShapedType>(bias.getType());
    if (biasType.getElementType() != elementType)
      return emitOpError("bias must have the same element type as input");
    if (!biasType.hasRank() || biasType.getRank() != 1)
      return emitOpError("bias must be rank 1");
    int64_t outputChannels = (*expected)[1];
    if (!ShapedType::isDynamic(outputChannels) && !biasType.isDynamicDim(0) &&
        biasType.getDimSize(0) != outputChannels)
      return emitOpError("bias length must match output channels");
  }
  return success();
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

LogicalResult ConvTransposeOp::verify() {
  SmallVector<Value> dataOperands = {getInput(), getWeights()};
  if (getBias())
    dataOperands.push_back(getBias());
  dataOperands.push_back(getOutput());
  if (failed(verifyDpsComputeOp(*this, dataOperands, /*numInits=*/1)))
    return failure();

  FailureOr<SmallVector<int64_t>> expected = mlir::hip::inferConvTransposeShape(
      detail::getShapeOf(getInput()), detail::getShapeOf(getWeights()),
      detail::getI64Array(getKernelShape()), detail::getI64Array(getStrides()),
      detail::getI64Array(getPads()), detail::getI64Array(getDilations()),
      detail::getI64Array(getOutputPadding()), getGroup(),
      [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();
  if (failed(mlir::hip::verifyHipOpShape(
          *this,
          [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; })))
    return failure();

  if (Value bias = getBias()) {
    auto biasType = dyn_cast<ShapedType>(bias.getType());
    if (!biasType || !biasType.hasRank() || biasType.getRank() != 1)
      return emitOpError("bias must be rank 1");
    int64_t outputChannels = (*expected)[1];
    if (!ShapedType::isDynamic(outputChannels) && !biasType.isDynamicDim(0) &&
        biasType.getDimSize(0) != outputChannels)
      return emitOpError("bias length must match output channels");
  }
  return success();
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

LogicalResult SkipRmsNormOp::verify() {
  unsigned numOutputs = getOutputs().size();
  if (numOutputs < 1 || numOutputs > 2)
    return emitOpError("expected 1 or 2 output buffers, got ") << numOutputs;

  SmallVector<Value> operands = {getInput(), getSkip(), getGamma()};
  if (Value bias = getBias())
    operands.push_back(bias);
  llvm::append_range(operands, getOutputs());
  if (failed(verifyDpsComputeOp(*this, operands, numOutputs)))
    return failure();

  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto skipType = dyn_cast<ShapedType>(getSkip().getType());
  auto gammaType = dyn_cast<ShapedType>(getGamma().getType());
  if (!inputType || !inputType.hasRank() || !skipType || !skipType.hasRank() ||
      !gammaType || !gammaType.hasRank())
    return emitOpError("input, skip, and gamma must be ranked");
  std::optional<ArrayRef<int64_t>> biasShape;
  if (Value bias = getBias()) {
    auto biasType = dyn_cast<ShapedType>(bias.getType());
    if (!biasType || !biasType.hasRank())
      return emitOpError("bias must be ranked");
    biasShape = biasType.getShape();
  }

  FailureOr<SmallVector<SmallVector<int64_t>>> expected =
      inferSkipRmsNormOutputShapes(inputType.getShape(), skipType.getShape(),
                                   gammaType.getShape(), biasShape, numOutputs,
                                   [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();

  Type elementType = inputType.getElementType();
  if (!elementType.isF16() && !elementType.isF32())
    return emitOpError("runtime supports only f16/f32 input");
  for (Value value : operands)
    if (cast<ShapedType>(value.getType()).getElementType() != elementType)
      return emitOpError("all input and output element types must match");
  if (getEpsilon().convertToDouble() < 0.0)
    return emitOpError("epsilon must be non-negative");

  for (unsigned index = 0; index < numOutputs; ++index) {
    if (failed(verifyHipOpShape(
            *this,
            [&]() -> FailureOr<SmallVector<int64_t>> {
              return (*expected)[index];
            },
            index)))
      return failure();
    if (getNumResults() &&
        getResult(index).getType() != getOutputs()[index].getType())
      return emitOpError("tensor result #")
             << index << " type must match its output buffer type";
  }
  return success();
}

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
  if (failed(verifyDpsComputeOp(*this, dataOperands, /*numInits=*/numOutputs)))
    return failure();

  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto scaleType = dyn_cast<ShapedType>(getScale().getType());
  if (!inputType || !inputType.hasRank() || inputType.getRank() < 1 ||
      !scaleType || !scaleType.hasRank())
    return emitOpError("input and scale must be ranked");
  int64_t rank = inputType.getRank();
  int64_t rawAxis = static_cast<int64_t>(getAxis());
  int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
  if (axis < 0 || axis >= rank)
    return emitOpError("axis must be in [-rank, rank)");

  int64_t normalizedRank = rank - axis;
  if (scaleType.getRank() != normalizedRank)
    return emitOpError(
        "runtime requires scale rank to equal the normalized suffix rank");
  for (int64_t i : llvm::seq<int64_t>(0, normalizedRank)) {
    int64_t inputDim = inputType.getDimSize(axis + i);
    int64_t scaleDim = scaleType.getDimSize(i);
    if (!ShapedType::isDynamic(inputDim) && !ShapedType::isDynamic(scaleDim) &&
        inputDim != scaleDim)
      return emitOpError("scale shape must equal input.shape[axis:]");
  }
  if (Value bias = getBias()) {
    auto biasType = dyn_cast<ShapedType>(bias.getType());
    if (!biasType || !biasType.hasRank() ||
        biasType.getShape() != scaleType.getShape())
      return emitOpError("bias shape must equal scale shape");
  }

  Type inputElementType = inputType.getElementType();
  if (!inputElementType.isF16() && !inputElementType.isF32())
    return emitOpError("runtime supports only f16/f32 input");
  if (scaleType.getElementType() != inputElementType)
    return emitOpError("scale element type must match input");
  if (Value bias = getBias())
    if (cast<ShapedType>(bias.getType()).getElementType() != inputElementType)
      return emitOpError("bias element type must match input");
  if (cast<ShapedType>(getOutputs().front().getType()).getElementType() !=
      inputElementType)
    return emitOpError("output element type must match input");
  FailureOr<Type> statsElementType =
      mlir::hip::inferLayerNormStatsType(getContext(), getStashType());
  if (failed(statsElementType))
    return emitOpError("runtime supports stash_type 0/1 (f32) or 10 (f16)");

  FailureOr<SmallVector<SmallVector<int64_t>>> expected =
      mlir::hip::inferLayerNormOutputShapes(inputType.getShape(), getAxis(),
                                            numOutputs);
  if (failed(expected))
    return emitOpError("could not infer output shapes");
  for (auto [index, output] : llvm::enumerate(getOutputs())) {
    auto outputType = dyn_cast<ShapedType>(output.getType());
    if (!outputType || !outputType.hasRank() ||
        outputType.getRank() != static_cast<int64_t>((*expected)[index].size()))
      return emitOpError("output #") << index << " has the wrong rank";
    for (int64_t dim : llvm::seq<int64_t>(0, outputType.getRank())) {
      int64_t actual = outputType.getDimSize(dim);
      int64_t wanted = (*expected)[index][dim];
      if (!ShapedType::isDynamic(actual) && !ShapedType::isDynamic(wanted) &&
          actual != wanted)
        return emitOpError("output #")
               << index << " dimension " << dim << " must be " << wanted;
    }
    if (index > 0 && outputType.getElementType() != *statsElementType)
      return emitOpError("stats output element type must match stash_type");
  }
  return success();
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
  return verifyDpsComputeOp(*this, {getInput(), getOutput()}, /*numInits=*/1);
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

LogicalResult CausalConvWithStateOp::verify() {
  SmallVector<Value> operands = {getInput(), getWeight()};
  if (Value bias = getBias())
    operands.push_back(bias);
  if (Value pastState = getPastState())
    operands.push_back(pastState);
  operands.push_back(getOutput());
  operands.push_back(getPresentState());
  if (failed(verifyDpsComputeOp(*this, operands, /*numInits=*/2)))
    return failure();

  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto weightType = dyn_cast<ShapedType>(getWeight().getType());
  if (!inputType || !inputType.hasRank() || !weightType ||
      !weightType.hasRank())
    return emitOpError("input and weight must be ranked");
  std::optional<ArrayRef<int64_t>> biasShape;
  if (Value bias = getBias()) {
    auto biasType = dyn_cast<ShapedType>(bias.getType());
    if (!biasType || !biasType.hasRank())
      return emitOpError("bias must be ranked");
    biasShape = biasType.getShape();
  }
  std::optional<ArrayRef<int64_t>> pastStateShape;
  if (Value pastState = getPastState()) {
    auto pastStateType = dyn_cast<ShapedType>(pastState.getType());
    if (!pastStateType || !pastStateType.hasRank())
      return emitOpError("past_state must be ranked");
    pastStateShape = pastStateType.getShape();
  }

  FailureOr<SmallVector<SmallVector<int64_t>>> expected =
      inferCausalConvWithStateOutputShapes(
          inputType.getShape(), weightType.getShape(), biasShape,
          pastStateShape, getNdim(), [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();
  if (getActivation() != "none" && getActivation() != "silu" &&
      getActivation() != "swish")
    return emitOpError("activation must be one of: none, silu, swish");

  Type elementType = inputType.getElementType();
  if (!elementType.isF16() && !elementType.isF32())
    return emitOpError("runtime supports only f16/f32 input");
  for (Value value : operands)
    if (cast<ShapedType>(value.getType()).getElementType() != elementType)
      return emitOpError("all input and output element types must match");

  for (unsigned index = 0; index < 2; ++index) {
    if (failed(verifyHipOpShape(
            *this,
            [&]() -> FailureOr<SmallVector<int64_t>> {
              return (*expected)[index];
            },
            index)))
      return failure();
    Value init = index == 0 ? getOutput() : getPresentState();
    if (getNumResults() && getResult(index).getType() != init.getType())
      return emitOpError("tensor result #")
             << index << " type must match its output buffer type";
  }
  return success();
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

LogicalResult MultiHeadAttentionOp::verify() {
  if (!getKey() || !getValue())
    return emitOpError(
        "default runtime requires separate query, key, and value inputs");
  if (getBias() || getKeyPaddingMask() || getAttentionBias() || getPastKey() ||
      getPastValue() || getPastSequenceLength() || getCacheIndirection())
    return emitOpError(
        "default runtime does not support bias, masks, past/cache inputs, or "
        "cache indirection");
  if (getPresentKey() || getPresentValue() || getQk())
    return emitOpError(
        "default runtime does not support present_key, present_value, or qk "
        "outputs");

  SmallVector<Value> operands = {getQuery(), getKey(), getValue(), getOutput()};
  if (failed(verifyDpsComputeOp(*this, operands, /*numInits=*/1)))
    return failure();

  auto queryType = cast<ShapedType>(getQuery().getType());
  auto keyType = cast<ShapedType>(getKey().getType());
  auto valueType = cast<ShapedType>(getValue().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!queryType.getElementType().isF16() ||
      keyType.getElementType() != queryType.getElementType() ||
      valueType.getElementType() != queryType.getElementType() ||
      outputType.getElementType() != queryType.getElementType())
    return emitOpError(
        "default runtime requires fp16 query, key, value, and output");
  if (getUnidirectional() != 0 && getUnidirectional() != 1)
    return emitOpError("unidirectional must be 0 or 1");
  if (getMaskFilterValue().convertToFloat() != -10000.0f)
    return emitOpError(
        "default runtime supports only mask_filter_value = -10000");

  FailureOr<SmallVector<int64_t>> expected = inferMultiHeadAttentionOutputShape(
      queryType.getShape(), keyType.getShape(), valueType.getShape(),
      getNumHeads(), [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();

  // An output template may keep a known query extent dynamic, but it cannot
  // make an unknown runtime query extent static.
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputShape.size() == expected->size()) {
    for (size_t dim : llvm::seq<size_t>(0, outputShape.size())) {
      if (ShapedType::isDynamic((*expected)[dim]) &&
          !ShapedType::isDynamic(outputShape[dim]))
        return emitOpError("output dimension ")
               << dim
               << " must remain dynamic because the corresponding "
                  "query extent is dynamic";
    }
  }
  return verifyHipOpShape(
      *this, [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; });
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

LogicalResult LinearAttentionOp::verify() {
  SmallVector<Value> operands = {getQuery(), getKey(), getValue()};
  if (Value past = getPastState())
    operands.push_back(past);
  if (Value decay = getDecay())
    operands.push_back(decay);
  if (Value beta = getBeta())
    operands.push_back(beta);
  operands.push_back(getOutput());
  operands.push_back(getPresentState());
  if (failed(verifyDpsComputeOp(*this, operands, /*numInits=*/2)))
    return failure();

  auto queryType = dyn_cast<ShapedType>(getQuery().getType());
  auto keyType = dyn_cast<ShapedType>(getKey().getType());
  auto valueType = dyn_cast<ShapedType>(getValue().getType());
  if (!queryType || !queryType.hasRank() || !keyType || !keyType.hasRank() ||
      !valueType || !valueType.hasRank())
    return emitOpError("query, key, and value must be ranked");
  if (queryType.getElementType() != keyType.getElementType() ||
      queryType.getElementType() != valueType.getElementType())
    return emitOpError("query, key, and value element types must match");

  FailureOr<SmallVector<SmallVector<int64_t>>> expected =
      inferLinearAttentionOutputShapes(queryType.getShape(), keyType.getShape(),
                                       valueType.getShape(), getQNumHeads(),
                                       getKvNumHeads(),
                                       [&]() { return this->emitOpError(); });
  if (failed(expected))
    return failure();

  auto verifyShape = [&](Value value, ArrayRef<int64_t> wanted,
                         StringRef name) -> LogicalResult {
    auto type = dyn_cast<ShapedType>(value.getType());
    if (!type || !type.hasRank() ||
        type.getRank() != static_cast<int64_t>(wanted.size()))
      return emitOpError() << name << " has the wrong rank";
    for (int64_t dim : llvm::seq<int64_t>(0, type.getRank())) {
      int64_t actual = type.getDimSize(dim);
      if (!ShapedType::isDynamic(actual) &&
          !ShapedType::isDynamic(wanted[dim]) && actual != wanted[dim])
        return emitOpError()
               << name << " dimension " << dim << " must be " << wanted[dim];
    }
    if (type.getElementType() != queryType.getElementType())
      return emitOpError() << name << " element type must match query";
    return success();
  };

  if (failed(verifyShape(getOutput(), (*expected)[0], "output")) ||
      failed(verifyShape(getPresentState(), (*expected)[1], "present_state")))
    return failure();
  if (Value past = getPastState())
    if (failed(verifyShape(past, (*expected)[1], "past_state")))
      return failure();
  return success();
}

LogicalResult GqaOp::verify() {
  SmallVector<Value> operands = {getQuery()};
  auto appendIfPresent = [&](Value value) {
    if (value)
      operands.push_back(value);
  };
  appendIfPresent(getKey());
  appendIfPresent(getValue());
  appendIfPresent(getPastKey());
  appendIfPresent(getPastValue());
  operands.push_back(getSeqlensK());
  operands.push_back(getTotalSeqLen());
  appendIfPresent(getCosCache());
  appendIfPresent(getSinCache());
  appendIfPresent(getPositionIds());
  appendIfPresent(getAttentionBias());
  appendIfPresent(getHeadSink());
  appendIfPresent(getKScale());
  appendIfPresent(getVScale());
  operands.push_back(getOutput());
  operands.push_back(getPresentKey());
  operands.push_back(getPresentValue());
  appendIfPresent(getOutputQk());
  unsigned numInits = getOutputQk() ? 4 : 3;
  if (failed(verifyDpsComputeOp(*this, operands, numInits)))
    return failure();

  bool hasPastKey = getPastKey() != nullptr;
  bool hasPastValue = getPastValue() != nullptr;
  if (hasPastKey != hasPastValue)
    return emitOpError(
        "past_key and past_value must both be provided or both be omitted");
  if (getNumHeads() <= 0 || getKvNumHeads() <= 0)
    return emitOpError("num_heads and kv_num_heads must be positive");
  if (getNumHeads() % getKvNumHeads() != 0)
    return emitOpError("num_heads must be divisible by kv_num_heads");

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

  if ((qkOutput != 0) != static_cast<bool>(getOutputQk()))
    return emitOpError(
        "output_qk must be present exactly when qk_output is nonzero");

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

  auto queryType = cast<ShapedType>(getQuery().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  auto presentKeyType = cast<ShapedType>(getPresentKey().getType());
  auto presentValueType = cast<ShapedType>(getPresentValue().getType());
  if (queryType.getRank() != 3 || outputType.getRank() != 3)
    return emitOpError("query and output must be rank-3");
  if (presentKeyType.getRank() != 4 || presentValueType.getRank() != 4)
    return emitOpError("present_key and present_value must be rank-4 BNSH");

  auto checkCompatible = [&](int64_t lhs, int64_t rhs,
                             const Twine &what) -> LogicalResult {
    if (!ShapedType::isDynamic(lhs) && !ShapedType::isDynamic(rhs) &&
        lhs != rhs)
      return emitOpError() << what << " (" << lhs << " vs " << rhs << ")";
    return success();
  };
  auto checkDim = [&](ShapedType lhsType, int64_t lhsDim, ShapedType rhsType,
                      int64_t rhsDim, const Twine &what) -> LogicalResult {
    return checkCompatible(lhsType.getDimSize(lhsDim),
                           rhsType.getDimSize(rhsDim), what);
  };
  auto checkStaticValue = [&](ShapedType type, int64_t dim, int64_t expected,
                              const Twine &what) -> LogicalResult {
    int64_t actual = type.getDimSize(dim);
    if (!ShapedType::isDynamic(actual) && actual != expected)
      return emitOpError() << what << " (expected " << expected << ", got "
                           << actual << ")";
    return success();
  };

  if (failed(checkDim(outputType, 0, queryType, 0,
                      "output batch must match query batch")) ||
      failed(checkDim(outputType, 1, queryType, 1,
                      "output sequence must match query sequence")) ||
      failed(checkDim(presentKeyType, 0, queryType, 0,
                      "present_key batch must match query batch")) ||
      failed(checkDim(presentValueType, 0, queryType, 0,
                      "present_value batch must match query batch")) ||
      failed(
          checkStaticValue(presentKeyType, 1, getKvNumHeads(),
                           "present_key head count must equal kv_num_heads")) ||
      failed(
          checkStaticValue(presentValueType, 1, getKvNumHeads(),
                           "present_value head count must equal kv_num_heads")))
    return failure();
  for (int64_t dim : llvm::seq<int64_t>(0, 4))
    if (failed(checkDim(presentKeyType, dim, presentValueType, dim,
                        "present_key and present_value shapes must match")))
      return failure();

  int64_t headDivisor =
      hasKey ? getNumHeads() : getNumHeads() + 2 * getKvNumHeads();
  int64_t queryHidden = queryType.getDimSize(2);
  std::optional<int64_t> knownHeadSize;
  if (!ShapedType::isDynamic(queryHidden)) {
    if (queryHidden % headDivisor != 0)
      return emitOpError("query hidden extent must be divisible by ")
             << (hasKey ? "num_heads" : "num_heads + 2 * kv_num_heads");
    knownHeadSize = queryHidden / headDivisor;
  }

  auto mergeKnownHeadSize = [&](int64_t candidate,
                                const Twine &name) -> LogicalResult {
    if (ShapedType::isDynamic(candidate))
      return success();
    if (knownHeadSize && *knownHeadSize != candidate)
      return emitOpError() << name
                           << " head size must match the query head size";
    knownHeadSize = candidate;
    return success();
  };
  auto mergeHeadSize = [&](int64_t hidden, int64_t heads,
                           const Twine &name) -> LogicalResult {
    if (ShapedType::isDynamic(hidden))
      return success();
    if (hidden % heads != 0)
      return emitOpError() << name << " hidden extent must be divisible by "
                           << heads;
    return mergeKnownHeadSize(hidden / heads, name);
  };

  if (hasKey) {
    auto keyType = cast<ShapedType>(getKey().getType());
    auto valueType = cast<ShapedType>(getValue().getType());
    if ((keyType.getRank() != 3 && keyType.getRank() != 4) ||
        valueType.getRank() != keyType.getRank())
      return emitOpError(
          "key and value must have the same rank, either rank-3 or rank-4");
    if (failed(checkDim(keyType, 0, queryType, 0,
                        "key batch must match query batch")) ||
        failed(checkDim(valueType, 0, queryType, 0,
                        "value batch must match query batch")))
      return failure();
    if (keyType.getRank() == 3) {
      if (failed(checkDim(keyType, 1, valueType, 1,
                          "key and value sequence extents must match")) ||
          failed(
              mergeHeadSize(keyType.getDimSize(2), getKvNumHeads(), "key")) ||
          failed(
              mergeHeadSize(valueType.getDimSize(2), getKvNumHeads(), "value")))
        return failure();
    } else {
      for (int64_t dim : llvm::seq<int64_t>(0, 4))
        if (failed(checkDim(keyType, dim, valueType, dim,
                            "key and value shapes must match")))
          return failure();
      if (failed(checkStaticValue(keyType, 1, getKvNumHeads(),
                                  "key head count must equal kv_num_heads")) ||
          failed(mergeKnownHeadSize(keyType.getDimSize(3), "key")))
        return failure();
    }
  }

  if (knownHeadSize) {
    if (failed(checkStaticValue(outputType, 2, getNumHeads() * *knownHeadSize,
                                "output hidden extent is incompatible with "
                                "num_heads")) ||
        failed(checkStaticValue(presentKeyType, 3, *knownHeadSize,
                                "present_key head size is incompatible")) ||
        failed(checkStaticValue(presentValueType, 3, *knownHeadSize,
                                "present_value head size is incompatible")))
      return failure();
  }

  if (hasPastKey) {
    auto pastKeyType = cast<ShapedType>(getPastKey().getType());
    auto pastValueType = cast<ShapedType>(getPastValue().getType());
    if (pastKeyType.getRank() != 4 || pastValueType.getRank() != 4)
      return emitOpError("past_key and past_value must be rank-4 BNSH");
    for (int64_t dim : llvm::seq<int64_t>(0, 4))
      if (failed(checkDim(pastKeyType, dim, pastValueType, dim,
                          "past_key and past_value shapes must match")))
        return failure();
    if (failed(checkDim(pastKeyType, 0, queryType, 0,
                        "past cache batch must match query batch")) ||
        failed(checkStaticValue(pastKeyType, 1, getKvNumHeads(),
                                "past cache head count must equal "
                                "kv_num_heads")) ||
        failed(checkDim(pastKeyType, 0, presentKeyType, 0,
                        "past and present cache batches must match")) ||
        failed(checkDim(pastKeyType, 1, presentKeyType, 1,
                        "past and present cache head counts must match")) ||
        failed(checkDim(pastKeyType, 3, presentKeyType, 3,
                        "past and present cache head sizes must match")))
      return failure();
    int64_t pastCapacity = pastKeyType.getDimSize(2);
    int64_t presentCapacity = presentKeyType.getDimSize(2);
    if (!ShapedType::isDynamic(pastCapacity) &&
        !ShapedType::isDynamic(presentCapacity) &&
        presentCapacity < pastCapacity)
      return emitOpError(
          "present cache capacity cannot be smaller than past capacity");
  }

  if (Value outputQk = getOutputQk()) {
    auto outputQkType = cast<ShapedType>(outputQk.getType());
    if (outputQkType.getRank() != 4)
      return emitOpError("output_qk must be rank-4");
    if (failed(checkDim(outputQkType, 0, queryType, 0,
                        "output_qk batch must match query batch")) ||
        failed(checkStaticValue(outputQkType, 1, getNumHeads(),
                                "output_qk head count must equal num_heads")) ||
        failed(checkDim(outputQkType, 2, queryType, 1,
                        "output_qk query sequence must match query")))
      return failure();
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

//===----------------------------------------------------------------------===//
// ExpandOp: ins(input, shape), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange ExpandOp::getDpsInitsMutable() {
  return getOutputMutable();
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
