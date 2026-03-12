/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

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

void FreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // memref is operand #1 (operand #0 is ctx)
  effects.emplace_back(MemoryEffects::Free::get(),
                       &getOperation()->getOpOperand(1),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Helpers for DPS compute ops (custom parse/print, verify, interfaces)
//===----------------------------------------------------------------------===//

static bool isTensorMode(Value v) { return isa<RankedTensorType>(v.getType()); }

/// Verify that all data operands (skipping ctx and Index args) are
/// uniformly tensor or memref, and that results match the mode.
static LogicalResult verifyDpsComputeOp(Operation *op,
                                        ArrayRef<Value> dataOperands,
                                        unsigned numInits) {
  if (dataOperands.empty())
    return op->emitOpError("expected at least one data operand");

  bool tensorMode = isTensorMode(dataOperands.front());
  for (Value v : dataOperands) {
    if (isTensorMode(v) != tensorMode)
      return op->emitOpError(
          "all data operands must be the same kind (all tensor or all memref)");
  }

  unsigned numResults = op->getNumResults();
  if (tensorMode) {
    if (numResults != numInits)
      return op->emitOpError("tensor mode requires ")
             << numInits << " result(s), got " << numResults;
    for (unsigned i = 0; i < numResults; ++i) {
      if (!isa<RankedTensorType>(op->getResult(i).getType()))
        return op->emitOpError("result #") << i << " must be a ranked tensor";
    }
  } else {
    if (numResults != 0)
      return op->emitOpError("memref mode must have zero results, got ")
             << numResults;
  }
  return success();
}

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
// HipblasltMatmulOp: ins(A, B), outs(C)
//===----------------------------------------------------------------------===//

MutableOperandRange HipblasltMatmulOp::getDpsInitsMutable() {
  return getCMutable();
}

void HipblasltMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult HipblasltMatmulOp::verify() {
  return verifyDpsComputeOp(*this, {getA(), getB(), getC()}, /*numInits=*/1);
}

ParseResult HipblasltMatmulOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void HipblasltMatmulOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{}, {getA(), getB()},
                       {getC()});
}

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
// SkipRmsNormOp: ins(input, skip, gamma), outs(output, skip_output)
//===----------------------------------------------------------------------===//

MutableOperandRange SkipRmsNormOp::getDpsInitsMutable() {
  // output and skip_output are operands #4 and #5
  // (0=ctx,1=input,2=skip,3=gamma)
  return MutableOperandRange(*this, /*start=*/4, /*length=*/2);
}

void SkipRmsNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult SkipRmsNormOp::verify() {
  return verifyDpsComputeOp(
      *this, {getInput(), getSkip(), getGamma(), getOutput(), getSkipOutput()},
      /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// MiopenRopeOp: ins(cos_cache, sin_cache), outs(q, k)
// Extra scalar: start_pos (Index)
//===----------------------------------------------------------------------===//

MutableOperandRange MiopenRopeOp::getDpsInitsMutable() {
  // 0=ctx, 1=start_pos, 2=cos_cache, 3=sin_cache, 4=q, 5=k
  return MutableOperandRange(*this, /*start=*/4, /*length=*/2);
}

void MiopenRopeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult MiopenRopeOp::verify() {
  return verifyDpsComputeOp(*this,
                            {getQ(), getK(), getCosCache(), getSinCache()},
                            /*numInits=*/2);
}

ParseResult MiopenRopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // `(` ctx `,` start_pos `)` `ins` ... `outs` ...
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2,
                              /*extraScalars=*/1);
}

void MiopenRopeOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), {getStartPos()},
                       {getCosCache(), getSinCache()}, {getQ(), getK()});
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

LogicalResult MiopenAddOp::verify() {
  return verifyDpsComputeOp(*this, {getA(), getB(), getC()}, /*numInits=*/1);
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

LogicalResult MulOp::verify() {
  return verifyDpsComputeOp(*this, {getLhs(), getRhs(), getOutput()},
                            /*numInits=*/1);
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void MulOp::print(OpAsmPrinter &p) {
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
// TransposeOp: ins(input), outs(output), extra scalars: dim0, dim1
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
  return verifyDpsComputeOp(*this, {getInput(), getOutput()}, /*numInits=*/1);
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/1,
                              /*extraScalars=*/2);
}

void TransposeOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), {getDim0(), getDim1()}, {getInput()},
                       {getOutput()});
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

LogicalResult GatherOp::verify() {
  return verifyDpsComputeOp(*this, {getIndices(), getTable(), getOutput()},
                            /*numInits=*/1);
}

ParseResult GatherOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSingleInitDpsOp(parser, result, /*numIns=*/2);
}

void GatherOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(), /*scalarArgs=*/{},
                       {getIndices(), getTable()}, {getOutput()});
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
// SigmoidOp: ins(input), outs(output)
//===----------------------------------------------------------------------===//

MutableOperandRange SigmoidOp::getDpsInitsMutable() {
  return getOutputMutable();
}

void SigmoidOp::getEffects(
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
// GqaOp: ins(q, k, v), outs(kv_cache, output)
// Extra scalars: layer, start_pos, seq_len
//===----------------------------------------------------------------------===//

MutableOperandRange GqaOp::getDpsInitsMutable() {
  // 0=ctx, 1=layer, 2=start_pos, 3=seq_len, 4=q, 5=k, 6=v,
  // 7=kv_cache, 8=output
  return MutableOperandRange(*this, /*start=*/7, /*length=*/2);
}

void GqaOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}

LogicalResult GqaOp::verify() {
  return verifyDpsComputeOp(*this,
                            {getQ(), getK(), getV(), getKvCache(), getOutput()},
                            /*numInits=*/2);
}

ParseResult GqaOp::parse(OpAsmParser &parser, OperationState &result) {
  // `(` ctx `,` layer `,` start_pos `,` seq_len `)`
  return parseSingleInitDpsOp(parser, result, /*numIns=*/3,
                              /*extraScalars=*/3);
}

void GqaOp::print(OpAsmPrinter &p) {
  printSingleInitDpsOp(p, *this, getCtx(),
                       {getLayer(), getStartPos(), getSeqLen()},
                       {getQ(), getK(), getV()}, {getKvCache(), getOutput()});
}

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.cpp.inc"
