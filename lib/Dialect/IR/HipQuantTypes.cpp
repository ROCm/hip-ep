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

// Integer operands (indices) are never quantized, so all-or-none skips them.
static bool isInteger(Type type) {
  return getElementTypeOrSelf(type).isIntOrIndex();
}

LogicalResult mlir::hip::verifyQuantTypes(Operation *op, ValueRange operands) {
  auto attr = op->getAttrOfType<ArrayAttr>("quant_types");
  unsigned numQuantized = llvm::count_if(
      operands, [](Value value) { return isQuantized(value.getType()); });

  if (!attr) {
    unsigned numQuantizable = llvm::count_if(
        operands, [](Value value) { return !isInteger(value.getType()); });
    if (numQuantized != 0 && numQuantized != numQuantizable)
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

  auto entries = attr.getAsRange<TypeAttr>();
  size_t numQuantTypes = llvm::count_if(
      entries, [](TypeAttr type) { return isQuantized(type.getValue()); });
  size_t numQuantizable = llvm::count_if(
      entries, [](TypeAttr type) { return !isInteger(type.getValue()); });
  if (numQuantTypes != 0 && numQuantTypes != numQuantizable)
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

SmallVector<Value> mlir::hip::getDefaultQuantOperands(Operation *op) {
  SmallVector<Value> operands;
  auto shaped = [](Value value) { return isa<ShapedType>(value.getType()); };
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    llvm::append_range(operands,
                       llvm::make_filter_range(dps.getDpsInputs(), shaped));
    llvm::append_range(operands, dps.getDpsInits());
    return operands;
  }
  llvm::append_range(operands,
                     llvm::make_filter_range(op->getOperands(), shaped));
  llvm::append_range(operands, op->getResults());
  return operands;
}

SmallVector<Type>
mlir::hip::getDefaultOperandQuantTypes(Operation *op,
                                       ValueRange quantOperands) {
  if (auto attr = op->getAttrOfType<ArrayAttr>("quant_types"))
    return llvm::map_to_vector(attr.getAsRange<TypeAttr>(),
                               [](TypeAttr type) { return type.getValue(); });
  return llvm::map_to_vector(quantOperands, [](Value value) {
    return getElementTypeOrSelf(value.getType());
  });
}

namespace {

/// The interface's defaults for ops that do not declare it.
struct GenericQuantCapableModel
    : HipQuantCapableOpInterface::FallbackModel<GenericQuantCapableModel> {
  SmallVector<Value> getQuantOperands(Operation *op) const {
    return getDefaultQuantOperands(op);
  }
  SmallVector<Type> getOperandQuantTypes(Operation *op) const {
    return getDefaultOperandQuantTypes(op, getQuantOperands(op));
  }
  bool isQuantized(Operation *op) const {
    return llvm::any_of(getOperandQuantTypes(op),
                        [](Type type) { return hip::isQuantized(type); });
  }
  Type getInputQuantType(Operation *op, unsigned index) const {
    SmallVector<Type> types = getOperandQuantTypes(op);
    assert(index + 1 < types.size() && "not an input");
    return types[index];
  }
  Type getResultQuantType(Operation *op) const {
    return getOperandQuantTypes(op).back();
  }
  quant::UniformQuantizedType getUniformQuantType(Operation *op,
                                                  unsigned index) const {
    return dyn_cast<quant::UniformQuantizedType>(
        getOperandQuantTypes(op)[index]);
  }
};

// DPS ops that must not take quant types.
constexpr llvm::StringLiteral kNotQuantCapable[] = {
    // Q/DQ and ops that are already quantized.
    "hip.quantize_linear", "hip.dequantize_linear", "hip.qadd", "hip.qmul",
    "hip.qmatmul", "hip.qgemm", "hip.qconv", "hip.qsigmoid",
    "hip.qlpnormalization", "hip.qlinear_conv", "hip.qlinear_matmul",
    "hip.quant_conv", "hip.quant_matmul", "hip.matmul_nbits", "hip.qmoe",
    "hip.qmoe_amd", "hip.dequant_bnb4", "hip.unpack_int4",
    "hip.gather_block_quantized",
    // Still reach a kernel through QMulFusion / QSigmoidFusion.
    "hip.mul", "hip.sigmoid",
    // Layout: run on storage ints (QdqRoundTrip), no requantization.
    "hip.transpose", "hip.expand", "hip.slice", "hip.pad", "hip.tile",
    // Element type conversion and graph containers.
    "hip.cast", "hip.hipdnn_graph", "hip.rocmlir"};

} // namespace

void mlir::hip::attachQuantCapableInterface(Dialect *dialect) {
  MLIRContext *ctx = dialect->getContext();
  // A stale name would silently stop excluding its op.
  assert(llvm::all_of(kNotQuantCapable,
                      [&](StringRef name) {
                        return RegisteredOperationName::lookup(name, ctx);
                      }) &&
         "unknown op in kNotQuantCapable");
  for (RegisteredOperationName name :
       ctx->getRegisteredOperationsByDialect(dialect->getNamespace())) {
    if (!name.hasInterface<DestinationStyleOpInterface>() ||
        name.hasInterface<HipQuantCapableOpInterface>() ||
        llvm::is_contained(kNotQuantCapable, name.getStringRef()))
      continue;
    name.attachInterface<GenericQuantCapableModel>();
  }
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
