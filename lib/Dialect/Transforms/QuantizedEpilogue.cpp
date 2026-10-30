/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "QuantizedEpilogue.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace mlir::hip {
namespace {

// One dequantized operand: (data - zeroPoint) * scale, per tensor.
struct DequantizedOperand {
  Value data;
  float scale;
  int32_t zeroPoint;
};

// q(add(dq(a), dq(b))) or q(mul(dq(a), dq(b))) with per-tensor parameters.
struct QuantizedBinary {
  bool isAdd;
  DequantizedOperand lhs, rhs;
  float outScale;
  int32_t outZeroPoint;
};

static DenseElementsAttr getScalarAttr(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  auto attr = dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
  if (!attr || attr.getNumElements() != 1)
    return {};
  return attr;
}

static std::optional<float> getScalarF32(Value value) {
  DenseElementsAttr attr = getScalarAttr(value);
  if (!attr || !attr.getElementType().isF32())
    return std::nullopt;
  float result = *attr.getValues<float>().begin();
  if (!std::isfinite(result))
    return std::nullopt;
  return result;
}

static std::optional<int32_t> getScalarInt(Value value) {
  DenseElementsAttr attr = getScalarAttr(value);
  if (!attr || !isa<IntegerType>(attr.getElementType()))
    return std::nullopt;
  APInt v = *attr.getValues<APInt>().begin();
  return static_cast<int32_t>(attr.getElementType().isUnsignedInteger()
                                  ? v.getZExtValue()
                                  : v.getSExtValue());
}

static std::optional<int32_t> getZeroPoint(Value zeroPoint) {
  if (!zeroPoint)
    return 0;
  return getScalarInt(zeroPoint);
}

static bool isByte(Type type) {
  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  return intType && intType.getWidth() == 8;
}

static std::optional<DequantizedOperand> matchDequantize(Value value) {
  auto dq = value.getDefiningOp<DequantizeLinearOp>();
  if (!dq || dq->getNumResults() != 1 || dq.getPackedInt4() ||
      dq.getBlockSize() != 0 || !isByte(dq.getInput().getType()))
    return std::nullopt;
  std::optional<float> scale = getScalarF32(dq.getScale());
  std::optional<int32_t> zeroPoint = getZeroPoint(dq.getZeroPoint());
  if (!scale || !zeroPoint)
    return std::nullopt;
  return DequantizedOperand{dq.getInput(), *scale, *zeroPoint};
}

static int64_t getNumElements(Value value) {
  return cast<RankedTensorType>(value.getType()).getNumElements();
}

static std::optional<QuantizedBinary> matchQuantizedBinary(QuantizeLinearOp q) {
  if (q->getNumResults() != 1 || q.getSaturate() == 0 || q.getPackedInt4() ||
      q.getBlockSize() != 0 || !isByte(q.getResult(0).getType()))
    return std::nullopt;
  Operation *def = q.getInput().getDefiningOp();
  if (!isa_and_present<AddOp, MulOp>(def) || def->getNumResults() != 1 ||
      !getElementTypeOrSelf(def->getResult(0).getType()).isF32())
    return std::nullopt;
  Value lhsValue =
      isa<AddOp>(def) ? cast<AddOp>(def).getLhs() : cast<MulOp>(def).getLhs();
  Value rhsValue =
      isa<AddOp>(def) ? cast<AddOp>(def).getRhs() : cast<MulOp>(def).getRhs();
  std::optional<DequantizedOperand> lhs = matchDequantize(lhsValue);
  std::optional<DequantizedOperand> rhs = matchDequantize(rhsValue);
  std::optional<float> outScale = getScalarF32(q.getScale());
  std::optional<int32_t> outZeroPoint = getZeroPoint(q.getZeroPoint());
  if (!lhs || !rhs || !outScale || !outZeroPoint)
    return std::nullopt;
  // QLinearAdd and QLinearMul take one 8-bit type for A, B, and C.
  Type outElement = getElementTypeOrSelf(q.getResult(0).getType());
  if (getElementTypeOrSelf(lhs->data.getType()) != outElement ||
      getElementTypeOrSelf(rhs->data.getType()) != outElement)
    return std::nullopt;
  bool isAdd = isa<AddOp>(def);
  // ORT adds a broadcast span either to a span of the same shape or to one
  // scalar; other broadcasts have no single QLinearAdd form to follow.
  if (isAdd && lhsValue.getType() != rhsValue.getType() &&
      (getNumElements(lhsValue) == 1) == (getNumElements(rhsValue) == 1))
    return std::nullopt;
  return QuantizedBinary{isAdd, *lhs, *rhs, *outScale, *outZeroPoint};
}

class Emitter {
public:
  Emitter(OpBuilder &builder, Location loc, Value ctx)
      : builder(builder), loc(loc), ctx(ctx) {}

  Value f32(float value) {
    auto type = RankedTensorType::get({}, builder.getF32Type());
    return ConstantOp::create(builder, loc, type,
                              DenseElementsAttr::get(type, value));
  }
  Value toFloat(Value data) {
    auto type =
        cast<RankedTensorType>(data.getType()).clone(builder.getF32Type());
    return CastOp::create(builder, loc, TypeRange{type}, ctx, data, empty(type),
                          builder.getI64IntegerAttr(/*FLOAT=*/1))
        .getResult(0);
  }
  template <typename OpTy>
  Value binary(Value lhs, Value rhs, RankedTensorType type) {
    return OpTy::create(builder, loc, TypeRange{type}, ctx, lhs, rhs,
                        empty(type))
        .getResult(0);
  }
  static RankedTensorType floatTypeOf(Value value) {
    return cast<RankedTensorType>(value.getType())
        .clone(Float32Type::get(value.getContext()));
  }

private:
  Value empty(RankedTensorType type) {
    return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                   type.getElementType());
  }

  OpBuilder &builder;
  Location loc;
  Value ctx;
};

// ORT CPU's QLinearAdd on x86:
//   rA = sA / sC, rB = sB / sC
//   fixed = zC - fma(rA, zA, rB * zB)
//   c = fma(a, rA, fma(b, rB, fixed))       tensor B
//   c = fma(a, rA, fixed + b * rB)          one-element B
// Operand A is the side that spans the broadcast. The kernel writes each fma as
// a multiply and an add, which rocMLIR contracts where it can. Where it cannot
// (a multiply that LLVM moves into the saturating select of an in-kernel
// quantize), a result that rounds at a half may differ by one unit.
static Value emitQLinearAdd(Emitter &emit, const QuantizedBinary &op,
                            RankedTensorType resultType) {
  bool lhsIsScalar =
      getNumElements(op.lhs.data) == 1 && getNumElements(op.rhs.data) != 1;
  const DequantizedOperand &a = lhsIsScalar ? op.rhs : op.lhs;
  const DequantizedOperand &b = lhsIsScalar ? op.lhs : op.rhs;
  bool scalarB = getNumElements(b.data) == 1 && getNumElements(a.data) != 1;

  float rA = a.scale / op.outScale;
  float rB = b.scale / op.outScale;
  float bZeroTerm = rB * static_cast<float>(b.zeroPoint);
  float fixed = static_cast<float>(op.outZeroPoint) -
                std::fma(rA, static_cast<float>(a.zeroPoint), bZeroTerm);

  Value aTerm = emit.binary<MulOp>(emit.toFloat(a.data), emit.f32(rA),
                                   Emitter::floatTypeOf(a.data));
  Value bPart;
  if (std::optional<int32_t> constantB = getScalarInt(b.data);
      scalarB && constantB) {
    float bTerm = static_cast<float>(*constantB) * rB;
    bPart = emit.f32(fixed + bTerm);
  } else {
    RankedTensorType bType = Emitter::floatTypeOf(b.data);
    Value bTerm = emit.binary<MulOp>(emit.toFloat(b.data), emit.f32(rB), bType);
    bPart = emit.binary<AddOp>(bTerm, emit.f32(fixed), bType);
  }
  return emit.binary<AddOp>(aTerm, bPart, resultType);
}

// ORT CPU's QLinearMul on x86, with an exact integer product:
//   c = (a - zA) * (b - zB) * (sA * sB / sC) + zC
static Value emitQLinearMul(Emitter &emit, const QuantizedBinary &op,
                            RankedTensorType resultType) {
  float scaleProduct = op.lhs.scale * op.rhs.scale;
  float ratio = scaleProduct / op.outScale;
  auto centered = [&](const DequantizedOperand &operand) {
    Value value = emit.toFloat(operand.data);
    if (operand.zeroPoint == 0)
      return value;
    return emit.binary<SubOp>(value,
                              emit.f32(static_cast<float>(operand.zeroPoint)),
                              Emitter::floatTypeOf(operand.data));
  };
  Value product =
      emit.binary<MulOp>(centered(op.lhs), centered(op.rhs), resultType);
  Value scaled = emit.binary<MulOp>(product, emit.f32(ratio), resultType);
  if (op.outZeroPoint == 0)
    return scaled;
  return emit.binary<AddOp>(
      scaled, emit.f32(static_cast<float>(op.outZeroPoint)), resultType);
}

static bool isDeadValueOp(Operation &op) {
  return op.use_empty() && isa<DequantizeLinearOp, AddOp, MulOp, SubOp, CastOp,
                               ConstantOp, tensor::EmptyOp>(op);
}

} // namespace

// Before:
//   %da = hip.dequantize_linear ins(%a, %sa) zero_point(%za)
//   %db = hip.dequantize_linear ins(%b, %sb) zero_point(%zb)
//   %s  = hip.add ins(%da, %db)
//   %c  = hip.quantize_linear ins(%s, %sc) zero_point(%zc)
// After:
//   %fa = hip.cast ins(%a)                 %fb = hip.cast ins(%b)
//   %ta = hip.mul ins(%fa, %rA)            %tb = hip.mul ins(%fb, %rB)
//   %u  = hip.add ins(%tb, %fixed)
//   %s  = hip.add ins(%ta, %u)
//   %c  = hip.quantize_linear ins(%s, %one) zero_point(%zero)
void rewriteQuantizedBinaryOps(func::FuncOp kernel) {
  SmallVector<QuantizeLinearOp> quantizes(kernel.getOps<QuantizeLinearOp>());
  bool rewrote = false;
  for (QuantizeLinearOp q : quantizes) {
    std::optional<QuantizedBinary> op = matchQuantizedBinary(q);
    if (!op)
      continue;
    OpBuilder builder(q);
    Emitter emit(builder, q.getLoc(), q.getCtx());
    auto resultType = cast<RankedTensorType>(q.getInput().getType());
    Value value = op->isAdd ? emitQLinearAdd(emit, *op, resultType)
                            : emitQLinearMul(emit, *op, resultType);
    Type outElement = getElementTypeOrSelf(q.getResult(0).getType());
    auto zeroType = RankedTensorType::get({}, outElement);
    Value zero =
        ConstantOp::create(builder, q.getLoc(), zeroType,
                           DenseElementsAttr::get(zeroType, APInt(8, 0)));
    q.getInputMutable().assign(value);
    q.getScaleMutable().assign(emit.f32(1.0f));
    q.getZeroPointMutable().assign(zero);
    rewrote = true;
  }
  if (!rewrote)
    return;
  Block &body = kernel.getBody().front();
  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body)))
      if (isDeadValueOp(op)) {
        op.erase();
        erased = true;
      }
  }
}

} // namespace mlir::hip
