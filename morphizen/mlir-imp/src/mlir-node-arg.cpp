/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-node-arg.hpp"
#include "./mlir-constants.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h" // for DenseElementsAttr
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include <cstring>
#include <glog/logging.h>
#include <numeric>
#include <variant>

namespace morphizen {
namespace mlir_impl {

namespace {
// Helper function to extract shape from MLIR value. Returns nullopt for
// UnrankedTensorType (tensor<*xT>) to keep the ranked vs unranked distinction
// across the Stage1 (TensorDesc) / Stage2 (mlir::Value) variant.
std::optional<MLIRNodeArg::shape_t> extractShapeFromValue(mlir::Value value) {
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
    auto mlirShape = tensorType.getShape();
    return MLIRNodeArg::shape_t(mlirShape.begin(), mlirShape.end());
  }
  return std::nullopt;
}

// Helper function to extract element type from MLIR value
int extractElementTypeFromValue(mlir::Value value) {
  mlir::Type type = value.getType();
  mlir::Type elementType;
  if (auto shapedType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    elementType = shapedType.getElementType();
  } else if (auto unrankedType =
                 mlir::dyn_cast<mlir::UnrankedTensorType>(type)) {
    elementType = unrankedType.getElementType();
  }
  if (!elementType) {
    LOG(WARNING) << "Value does not have a ranked or unranked tensor type, "
                    "using default shape and type";
    return 1; // TensorProto_DataType_FLOAT
  }
  return mlirElementTypeToOnnxType(elementType);
}
} // anonymous namespace

int mlirElementTypeToOnnxType(mlir::Type elementType) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    // Map MLIR integer types to our element type constants
    unsigned width = intType.getWidth();
    bool isSigned = intType.isSigned() || intType.isSignless();
    int mlir_type_code = -1;
    switch (width) {
    case 4:
      mlir_type_code = isSigned ? 22 : 21; // TensorProto_DataType_INT4 or UINT4
      break;
    case 8:
      mlir_type_code = isSigned ? 3 : 2; // INT8 : UINT8
      break;
    case 16:
      mlir_type_code = isSigned ? 5 : 4; // INT16 : UINT16
      break;
    case 32:
      mlir_type_code = isSigned ? 6 : 12; // INT32 : UINT32
      break;
    case 64:
      mlir_type_code = isSigned ? 7 : 13; // INT64 : UINT64
      break;
    default:
      LOG(WARNING) << "Unsupported integer width: " << width
                   << ", defaulting to INT32";
      mlir_type_code = 6; // Default to INT32
      break;
    }
    return mlir_type_code;
  } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    // Map MLIR float types to our element type constants
    unsigned width = floatType.getWidth();
    int mlir_type_code = -1;
    switch (width) {
    case 16:
      mlir_type_code = 10; // TensorProto_DataType_FLOAT16
      break;
    case 32:
      mlir_type_code = 1; // TensorProto_DataType_FLOAT
      break;
    case 64:
      mlir_type_code = 11; // TensorProto_DataType_DOUBLE
      break;
    default:
      LOG(WARNING) << "Unsupported float width: " << width
                   << ", defaulting to FLOAT";
      mlir_type_code = 1; // Default to FLOAT
      break;
    }
    return mlir_type_code;
  }
  LOG(WARNING) << "Unsupported MLIR element type, defaulting to FLOAT";
  return 1; // TensorProto_DataType_FLOAT
}

MLIRNodeArg::MLIRNodeArg(const std::string &name, const shape_t *shape,
                         int element_type)
    : name_{name},
      value_{TensorDesc{{name,
                         shape ? std::optional<shape_t>{*shape} : std::nullopt,
                         element_type},
                        std::nullopt}} {
  validateElementType(element_type);
}

MLIRNodeArg::MLIRNodeArg(const std::string &name, const shape_t &shape,
                         int element_type, const std::string &loc,
                         size_t offset, size_t size)
    : name_{name}, value_{TensorDesc{{name, shape, element_type},
                                     data_t{ExternalRef{loc, offset, size}}}} {
  validateElementType(element_type);
}

MLIRNodeArg::MLIRNodeArg(const std::string &name, const shape_t &shape,
                         int element_type, const void *data, size_t data_size)
    : name_{name},
      value_{TensorDesc{{name, shape, element_type}, std::nullopt}} {
  CHECK(!name.empty()) << "Argument name cannot be empty";
  validateElementType(element_type);

  if (data && data_size > 0) {
    std::vector<uint8_t> vec(data_size);
    std::memcpy(vec.data(), data, data_size);
    std::get<TensorDesc>(value_).data = data_t{std::move(vec)};
  }
}

MLIRNodeArg::MLIRNodeArg(const std::string &name, mlir::Value value)
    : name_{name}, value_{value} {
  CHECK(!name.empty()) << "Argument name cannot be empty";
  validateElementType(extractElementTypeFromValue(value));
}

const std::string &MLIRNodeArg::getName() const { return name_; }

std::optional<MLIRNodeArg::shape_t> MLIRNodeArg::getShape() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_))
    return desc->meta.shape;
  return extractShapeFromValue(std::get<mlir::Value>(value_));
}

void MLIRNodeArg::setShape(const MLIRNodeArg::shape_t &shape) {
  if (auto *val = std::get_if<mlir::Value>(&value_); val && *val) {
    LOG(FATAL) << "Currently , Once the Operation created , the shape can not "
                  "be changed";
  }
  std::get<TensorDesc>(value_).meta.shape = shape;
}

int MLIRNodeArg::getElementType() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_))
    return desc->meta.element_type;
  return extractElementTypeFromValue(std::get<mlir::Value>(value_));
}

void MLIRNodeArg::setElementType(int data_type) {
  if (auto *val = std::get_if<mlir::Value>(&value_); val && *val) {
    LOG(FATAL)
        << "Currently , Once the Operation created , the element type can not "
           "be changed";
  }
  std::get<TensorDesc>(value_).meta.element_type = data_type;
}

const mlir::Value &MLIRNodeArg::getValue() const {
  if (auto *val = std::get_if<mlir::Value>(&value_))
    return *val;
  static const mlir::Value null_value{nullptr};
  return null_value;
}

mlir::Value &MLIRNodeArg::getValue() {
  if (auto *val = std::get_if<mlir::Value>(&value_))
    return *val;
  value_ = mlir::Value(nullptr);
  return std::get<mlir::Value>(value_);
}

void MLIRNodeArg::setValue(mlir::Value value) const { value_ = value; }

mlir::Type MLIRNodeArg::getType(mlir::OpBuilder &builder) const {
  if (auto *val = std::get_if<mlir::Value>(&value_)) {
    if (*val)
      return val->getType();
  }

  // nullopt routes through onnxElementTypeToMlirType's nullptr branch, which
  // emits mlir::UnrankedTensorType.
  auto shape = getShape();
  return onnxElementTypeToMlirType(getElementType(), builder,
                                   shape ? &*shape : nullptr);
}

const void *MLIRNodeArg::getData() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_)) {
    if (desc->data.has_value()) {
      auto &data_var = desc->data.value();
      if (auto *ext = std::get_if<ExternalRef>(&data_var))
        return reinterpret_cast<void *>(ext->offset);
      if (auto *vec = std::get_if<std::vector<uint8_t>>(&data_var))
        return vec->data();
    }
    return nullptr;
  }

  auto &val = std::get<mlir::Value>(value_);
  if (val) {
    if (auto defining_op = val.getDefiningOp()) {
      if (defining_op->getName().getStringRef() == "onnx.Constant") {
        if (auto value_attr = defining_op->getAttr("value")) {
          if (auto dense_attr =
                  mlir::dyn_cast<mlir::DenseElementsAttr>(value_attr)) {
            return dense_attr.getRawData().data();
          }
        }
      }
    }
  }

  return nullptr;
}

size_t MLIRNodeArg::getDataSize() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_)) {
    if (desc->data.has_value()) {
      auto &data_var = desc->data.value();
      if (auto *ext = std::get_if<ExternalRef>(&data_var))
        return ext->size;
      if (auto *vec = std::get_if<std::vector<uint8_t>>(&data_var))
        return vec->size();
    }
    return 0;
  }

  auto &val = std::get<mlir::Value>(value_);
  if (val) {
    if (auto defining_op = val.getDefiningOp()) {
      if (defining_op->getName().getStringRef() == "onnx.Constant") {
        if (auto value_attr = defining_op->getAttr("value")) {
          if (auto dense_attr =
                  mlir::dyn_cast<mlir::DenseElementsAttr>(value_attr)) {
            return dense_attr.getRawData().size();
          }
        }
      }
    }
  }

  return 0;
}

bool MLIRNodeArg::hasData() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_))
    return desc->data.has_value();

  auto &val = std::get<mlir::Value>(value_);
  if (val) {
    if (auto defining_op = val.getDefiningOp())
      return defining_op->getName().getStringRef() == "onnx.Constant";
  }
  return false;
}

bool MLIRNodeArg::isExternalData() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_)) {
    return desc->data.has_value() &&
           std::holds_alternative<ExternalRef>(desc->data.value());
  }
  if (auto *val = std::get_if<mlir::Value>(&value_); val && *val) {
    if (auto defining_op = val->getDefiningOp())
      return defining_op->getName().getStringRef() == "onnx.Constant" &&
             defining_op->hasAttrOfType<mlir::StringAttr>("location");
  }
  return false;
}

int64_t MLIRNodeArg::getElementCount() const {
  auto shape = getShape();
  if (!shape || shape->empty())
    return 0;
  return std::accumulate(shape->begin(), shape->end(), 1LL,
                         std::multiplies<int64_t>());
}

bool MLIRNodeArg::isConstantValue() const {
  if (auto *val = std::get_if<mlir::Value>(&value_)) {
    if (!*val)
      return false;
    if (auto defining_op = val->getDefiningOp())
      return defining_op->getName().getStringRef() == "onnx.Constant";
    return false;
  }
  return hasData();
}

const MLIRNodeArg::TensorDesc &MLIRNodeArg::getDesc() const {
  return std::get<TensorDesc>(value_);
}

const MLIRNodeArg::TensorMeta &MLIRNodeArg::getMeta() const {
  return std::get<TensorDesc>(value_).meta;
}

const MLIRNodeArg::ExternalRef *MLIRNodeArg::getExternalRef() const {
  if (auto *desc = std::get_if<TensorDesc>(&value_)) {
    if (desc->data.has_value())
      return std::get_if<ExternalRef>(&desc->data.value());
  }
  return nullptr;
}

void MLIRNodeArg::validateElementType(int element_type) const {
  switch (element_type) {
  case 1:  // TensorProto_DataType_FLOAT
  case 2:  // TensorProto_DataType_UINT8
  case 3:  // TensorProto_DataType_INT8
  case 4:  // TensorProto_DataType_UINT16
  case 5:  // TensorProto_DataType_INT16
  case 6:  // TensorProto_DataType_INT32
  case 7:  // TensorProto_DataType_INT64
  case 9:  // TensorProto_DataType_BOOL
  case 10: // TensorProto_DataType_FLOAT16
  case 11: // TensorProto_DataType_DOUBLE
  case 12: // TensorProto_DataType_UINT32
  case 13: // TensorProto_DataType_UINT64
  case 16: // TensorProto_DataType_BFLOAT16
  case 21: // TensorProto_DataType_UINT4
  case 22: // TensorProto_DataType_INT4
    break; // Valid types
  default:
    LOG(WARNING) << "Unsupported element type: " << element_type;
    break;
  }
}

} // namespace mlir_impl
} // namespace morphizen
