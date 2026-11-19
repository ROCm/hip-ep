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

bool mlir::hip::isQuantizedOp(Operation *op) {
  if (auto quantOp = dyn_cast<HipQuantCapableOpInterface>(op))
    return quantOp.isQuantized();
  auto quantized = [](Type type) { return isQuantized(type); };
  return llvm::any_of(op->getOperandTypes(), quantized) ||
         llvm::any_of(op->getResultTypes(), quantized);
}

SmallVector<Type> mlir::hip::getOperandQuantTypes(Operation *op) {
  if (auto quantOp = dyn_cast<HipQuantCapableOpInterface>(op))
    return quantOp.getOperandQuantTypes();
  SmallVector<Type> types;
  for (Value value : op->getOperands())
    if (isa<ShapedType>(value.getType()))
      types.push_back(getElementTypeOrSelf(value.getType()));
  for (Type type : op->getResultTypes())
    types.push_back(getElementTypeOrSelf(type));
  return types;
}

SmallVector<quant::UniformQuantizedType, 3>
mlir::hip::getQuantElementTypes(Operation *op) {
  auto quantOp = dyn_cast<HipQuantCapableOpInterface>(op);
  if (!quantOp)
    return {};
  SmallVector<quant::UniformQuantizedType, 3> result;
  for (Type type : quantOp.getOperandQuantTypes()) {
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
