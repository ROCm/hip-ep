/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipQuantTypes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hip;

IntegerType mlir::hip::getHipStorageType(quant::QuantizedType type) {
  unsigned width = type.getStorageTypeIntegralWidth();
  return IntegerType::get(type.getContext(), width,
                          type.isSigned() ? IntegerType::Signless
                                          : IntegerType::Unsigned);
}

bool mlir::hip::isQuantized(Type type) {
  return isa<quant::QuantizedType>(getElementTypeOrSelf(type));
}

Type mlir::hip::toHipStorageType(Type type) {
  auto quantType = dyn_cast<quant::QuantizedType>(getElementTypeOrSelf(type));
  if (!quantType)
    return type;
  Type storage = getHipStorageType(quantType);
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.clone(storage);
  return storage;
}

// wrap_qelementwise: per-tensor scales, one storage type for lhs, rhs and
// output (i8/ui8/i16/ui16/i32).
static bool supportsQElementwise(TypeRange inputs, Type result) {
  auto out = dyn_cast<quant::UniformQuantizedType>(result);
  if (!out || inputs.size() != 2)
    return false;
  IntegerType storage = getHipStorageType(out);
  unsigned width = storage.getWidth();
  if (width != 8 && width != 16 && !(width == 32 && storage.isSignless()))
    return false;
  return llvm::all_of(inputs, [&](Type type) {
    auto in = dyn_cast<quant::UniformQuantizedType>(type);
    return in && getHipStorageType(in) == storage;
  });
}

// wrap_qmatmul / wrap_qgemm: per-tensor scales. A and the result are 8 or 16
// bit, B is 8 bit (the kernel splits only A into byte planes), and gemm's bias
// C may also be 32 bit (the int32 an ONNX quantizer emits).
static bool isUniformOfWidth(Type type, ArrayRef<unsigned> widths) {
  auto quantType = dyn_cast<quant::UniformQuantizedType>(type);
  return quantType &&
         llvm::is_contained(widths, quantType.getStorageTypeIntegralWidth());
}

static bool supportsQMatmul(TypeRange inputs, Type result, bool allowBias) {
  size_t numInputs = inputs.size();
  if (numInputs < 2 || numInputs > (allowBias ? 3u : 2u))
    return false;
  if (!isUniformOfWidth(inputs[0], {8, 16}) ||
      !isUniformOfWidth(inputs[1], {8}) || !isUniformOfWidth(result, {8, 16}))
    return false;
  return numInputs == 2 || isUniformOfWidth(inputs[2], {8, 16, 32});
}

bool mlir::hip::isQuantizedOp(Operation *op) {
  if (auto attr = op->getAttrOfType<ArrayAttr>("quant_types"))
    return llvm::any_of(attr.getAsRange<TypeAttr>(), [](TypeAttr type) {
      return isa<quant::QuantizedType>(type.getValue());
    });
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return isQuantized(type); });
}

SmallVector<quant::UniformQuantizedType, 3>
mlir::hip::getQuantElementTypes(Operation *op) {
  SmallVector<Type, 4> types;
  if (auto attr = op->getAttrOfType<ArrayAttr>("quant_types")) {
    for (auto typeAttr : attr.getAsRange<TypeAttr>())
      types.push_back(typeAttr.getValue());
  } else if (auto quantOp = dyn_cast<HipQuantCapableOpInterface>(op)) {
    for (Value value : quantOp.getQuantOperands())
      types.push_back(getElementTypeOrSelf(value.getType()));
  }

  SmallVector<quant::UniformQuantizedType, 3> result;
  for (Type type : types) {
    auto quantType = dyn_cast<quant::UniformQuantizedType>(type);
    if (!quantType)
      return {};
    result.push_back(quantType);
  }
  return result;
}

LogicalResult mlir::hip::verifyQuantTypes(Operation *op, ValueRange operands) {
  auto attr = op->getAttrOfType<ArrayAttr>("quant_types");
  unsigned numQuantized = llvm::count_if(
      operands, [](Value value) { return isQuantized(value.getType()); });

  if (!attr) {
    if (numQuantized != 0 && numQuantized != operands.size())
      return op->emitOpError(
          "operands must be all quantized or all non-quantized");
    return success();
  }

  if (numQuantized != 0)
    return op->emitOpError(
        "'quant_types' is only valid on storage-typed operands");
  if (attr.size() != operands.size())
    return op->emitOpError("'quant_types' expects ")
           << operands.size() << " entries, got " << attr.size();

  size_t numQuantTypes =
      llvm::count_if(attr.getAsRange<TypeAttr>(), [](TypeAttr type) {
        return isa<quant::QuantizedType>(type.getValue());
      });
  if (numQuantTypes != 0 && numQuantTypes != attr.size())
    return op->emitOpError(
        "'quant_types' must be all quantized or all non-quantized");

  for (auto [value, typeAttr] :
       llvm::zip_equal(operands, attr.getAsRange<TypeAttr>())) {
    Type operandType = getElementTypeOrSelf(value.getType());
    Type entry = typeAttr.getValue();
    // A non-quantized entry is the operand's own element type.
    if (auto quantType = dyn_cast<quant::QuantizedType>(entry))
      entry = getHipStorageType(quantType);
    if (operandType != entry)
      return op->emitOpError("operand element type ")
             << operandType << " does not match the 'quant_types' entry "
             << typeAttr.getValue();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// HipQuantCapableOpInterface
//===----------------------------------------------------------------------===//

namespace mlir::hip {
#include "hip/Dialect/IR/HipQuantCapableOpInterface.cpp.inc"
} // namespace mlir::hip

SmallVector<Value> AddOp::getQuantOperands() {
  return {getLhs(), getRhs(), getOutput()};
}

bool AddOp::supportsQuantTypes(TypeRange inputs, Type result) {
  return supportsQElementwise(inputs, result);
}

SmallVector<Value> MatmulOp::getQuantOperands() {
  SmallVector<Value> operands = {getA(), getB()};
  if (getBias())
    operands.push_back(getBias());
  operands.push_back(getOutput());
  return operands;
}

bool MatmulOp::supportsQuantTypes(TypeRange inputs, Type result) {
  return supportsQMatmul(inputs, result, /*allowBias=*/true);
}

SmallVector<Value> GemmOp::getQuantOperands() {
  SmallVector<Value> operands = {getInputA(), getInputB()};
  if (getInputC())
    operands.push_back(getInputC());
  operands.push_back(getOutput());
  return operands;
}

bool GemmOp::supportsQuantTypes(TypeRange inputs, Type result) {
  return supportsQMatmul(inputs, result, /*allowBias=*/true);
}

//===----------------------------------------------------------------------===//
// StorageCastOp
//===----------------------------------------------------------------------===//

LogicalResult StorageCastOp::verify() {
  auto inType = cast<RankedTensorType>(getInput().getType());
  auto outType = cast<RankedTensorType>(getResult().getType());
  if (inType.getShape() != outType.getShape())
    return emitOpError("input and result shapes must match");

  auto inQuant = dyn_cast<quant::QuantizedType>(inType.getElementType());
  auto outQuant = dyn_cast<quant::QuantizedType>(outType.getElementType());
  if (static_cast<bool>(inQuant) == static_cast<bool>(outQuant))
    return emitOpError("exactly one side must be quantized");

  quant::QuantizedType quantType = inQuant ? inQuant : outQuant;
  Type storage = inQuant ? outType.getElementType() : inType.getElementType();
  if (storage != getHipStorageType(quantType))
    return emitOpError("storage type ")
           << storage << " does not match " << quantType << " (expected "
           << getHipStorageType(quantType) << ")";
  return success();
}

OpFoldResult StorageCastOp::fold(FoldAdaptor) {
  // cast(cast(x)) -> x when it restores x's type.
  if (auto producer = getInput().getDefiningOp<StorageCastOp>())
    if (producer.getInput().getType() == getType())
      return producer.getInput();
  return {};
}
