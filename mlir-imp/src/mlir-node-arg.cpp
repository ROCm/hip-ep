/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-node-arg.hpp"
#include "./mlir-constants.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h" // for ConstantOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h" // for DenseElementsAttr
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include <cstring>
#include <glog/logging.h>
#include <numeric>

namespace morphizen {
namespace mlir_impl {

namespace {
// Helper function to extract shape from MLIR value
MLIRNodeArg::shape_t extractShapeFromValue(mlir::Value value) {
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
    auto mlirShape = tensorType.getShape();
    return MLIRNodeArg::shape_t(mlirShape.begin(), mlirShape.end());
  }
  return MLIRNodeArg::shape_t{};
}

// Helper function to extract element type from MLIR value
int extractElementTypeFromValue(mlir::Value value) {
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
    if (auto intType =
            mlir::dyn_cast<mlir::IntegerType>(tensorType.getElementType())) {
      // Map MLIR integer types to our element type constants
      unsigned width = intType.getWidth();
      bool isSigned = intType.isSigned() || intType.isSignless();

      if (width == 4) {
        return isSigned ? 21 : 22; // TensorProto_DataType_INT4 :
                                   // TensorProto_DataType_UINT4
      } else if (width == 8) {
        return isSigned ? 3 : 2; // TensorProto_DataType_INT8 :
                                 // TensorProto_DataType_UINT8
      } else if (width == 16) {
        return isSigned ? 5 : 4; // TensorProto_DataType_INT16 :
                                 // TensorProto_DataType_UINT16
      } else if (width == 32) {
        return isSigned ? 6 : 12; // TensorProto_DataType_INT32 :
                                  // TensorProto_DataType_UINT32
      } else if (width == 64) {
        return isSigned ? 7 : 13; // TensorProto_DataType_INT64 :
                                  // TensorProto_DataType_UINT64
      } else {
        LOG(WARNING) << "Unsupported integer width: " << width
                     << ", defaulting to INT32";
        return 6; // TensorProto_DataType_INT32
      }
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(
                   tensorType.getElementType())) {
      // Map MLIR float types to our element type constants
      unsigned width = floatType.getWidth();

      if (width == 16) {
        return 10; // TensorProto_DataType_FLOAT16
      } else if (width == 32) {
        return 1; // TensorProto_DataType_FLOAT
      } else if (width == 64) {
        return 11; // TensorProto_DataType_DOUBLE
      } else {
        LOG(WARNING) << "Unsupported float width: " << width
                     << ", defaulting to FLOAT";
        return 1; // TensorProto_DataType_FLOAT
      }
    } else {
      LOG(WARNING) << "Unsupported element type, defaulting to FLOAT";
      return 1; // TensorProto_DataType_FLOAT
    }
  } else {
    LOG(WARNING) << "Value does not have a ranked tensor type, using default "
                    "shape and type";
    return 1; // TensorProto_DataType_FLOAT
  }
}
} // anonymous namespace

MLIRNodeArg::MLIRNodeArg(const std::string& name, const shape_t& shape,
                         int element_type)
    : name_(name), shape_(shape), element_type_(element_type), value_(nullptr) {
  CHECK(!name.empty()) << "Argument name cannot be empty";
  validateElementType(element_type);
}

MLIRNodeArg::MLIRNodeArg(const std::string& name, const shape_t& shape,
                         int element_type, const void* data, size_t data_size)
    : name_(name), shape_(shape), element_type_(element_type), value_(nullptr) {
  CHECK(!name.empty()) << "Argument name cannot be empty";
  validateElementType(element_type);

  if (data && data_size > 0) {
    data_store_ = std::vector<uint8_t>();
    copyData(data, data_size);
  }
}

MLIRNodeArg::MLIRNodeArg(const std::string& name, mlir::Value value)
    : name_(name), shape_(extractShapeFromValue(value)),
      element_type_(extractElementTypeFromValue(value)), value_(value) {
  CHECK(!name.empty()) << "Argument name cannot be empty";
  validateElementType(element_type_);
  // DenseElementsAttr with splat values need special handling to expand data
  // clang-format off
  // eg : %3 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> {node.outputs = ["conv_bias"]} : () -> tensor<64xf32> loc(#loc) // user: %4
  // clang-format on
  // We cache the full data in data_store_ when
  // DenseElementsAttr with splat
  //  For getData() and getDataSize() return actual data and data_size
  if (auto defining_op = value_.getDefiningOp()) {
    if (auto const_op = mlir::dyn_cast<mlir::arith::ConstantOp>(defining_op)) {
      auto attr = const_op.getValueAttr();
      if (auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        // For dense tensor constants, use the actual raw data size
        // shape empty means scalar, skip it
        if (dense_attr.isSplat() && !shape_.empty()) {
          // For splat constants, we need to expand the single value to fill the
          // entire tensor
          auto raw_data = dense_attr.getRawData();
          if (!raw_data.empty()) {
            int64_t element_size = raw_data.size(); // TODO : i4, u4 ?
            int64_t element_count = getElementCount();
            data_store_ = std::vector<uint8_t>(element_count * element_size);
            const char* src_data = raw_data.data();
            uint8_t* dest_data = data_store_->data();

            for (int64_t i = 0; i < element_count; ++i) {
              std::memcpy(dest_data + i * element_size, src_data, element_size);
            }
          }
        }
      }
    }
  }
}

const std::string& MLIRNodeArg::getName() const { return name_; }

const MLIRNodeArg::shape_t& MLIRNodeArg::getShape() const { return shape_; }

void MLIRNodeArg::setShape(const MLIRNodeArg::shape_t& shape) {
  if (value_) {
    LOG(FATAL) << "Currently , Once the Operation created , the shape can not "
                  "be changed";
  }
  shape_ = shape;
};

int MLIRNodeArg::getElementType() const { return element_type_; }

void MLIRNodeArg::setElementType(int data_type) {
  if (value_) {
    LOG(FATAL)
        << "Currently , Once the Operation created , the element type can not "
           "be changed";
  }
  element_type_ = data_type;
}

const mlir::Value& MLIRNodeArg::getValue() const { return value_; }
mlir::Value& MLIRNodeArg::getValue() { return value_; }

void MLIRNodeArg::setValue(mlir::Value value) const { value_ = value; }

mlir::Type MLIRNodeArg::getType(mlir::OpBuilder& builder) const {
  // If we have a valid MLIR value, return its type
  if (value_) {
    return value_.getType();
  }

  // Use the shared utility function to convert element type and create tensor
  // type
  auto& shape = getShape();
  return onnxElementTypeToMlirType(getElementType(), builder, &shape);
}

const void* MLIRNodeArg::getData() const {
  // First check if we have stored data
  if (data_store_.has_value()) {
    return data_store_->data();
  }

  // If no stored data, try to extract from MLIR value if it's a constant
  if (value_) {
    if (auto defining_op = value_.getDefiningOp()) {
      if (auto const_op =
              mlir::dyn_cast<mlir::arith::ConstantOp>(defining_op)) {
        // Get the constant value attribute
        auto attr = const_op.getValue();

        // Handle different attribute types
        if (auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
          // For dense tensor constants (handles all dense<...> cases including
          // binary data) This covers: dense<0>, dense<[1, -1]>,
          // dense<"0xFD09..."> etc.
          auto raw_data = dense_attr.getRawData();
          if (raw_data.empty()) {
            LOG(WARNING) << "DenseElementsAttr has empty raw data for: "
                         << name_;
            return nullptr;
          }
          return raw_data.data();
        }
      }
    }
  }

  return nullptr;
}

size_t MLIRNodeArg::getDataSize() const {
  // First check if we have stored data
  if (data_store_.has_value()) {
    return data_store_->size();
  }

  // If no stored data, try to calculate size from MLIR value if it's a constant
  if (value_) {
    if (auto defining_op = value_.getDefiningOp()) {
      if (auto const_op =
              mlir::dyn_cast<mlir::arith::ConstantOp>(defining_op)) {
        auto attr = const_op.getValue();

        if (auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
          // For dense tensor constants, use the actual raw data size
          // This is more reliable than calculating from shape/element type
          auto raw_data = dense_attr.getRawData();
          return raw_data.size();
        }
      }
    }
  }

  return 0;
}

bool MLIRNodeArg::hasData() const {
  // First check if we have stored data
  if (data_store_.has_value() && !data_store_->empty()) {
    return true;
  }

  // If no stored data, check if we have a constant MLIR value
  if (value_) {
    if (auto defining_op = value_.getDefiningOp()) {
      if (auto const_op =
              mlir::dyn_cast<mlir::arith::ConstantOp>(defining_op)) {
        return true; // Constant operations always have data
      }
    }
  }

  return false;
}

int64_t MLIRNodeArg::getElementCount() const {
  if (shape_.empty()) {
    return 0;
  }
  return std::accumulate(shape_.begin(), shape_.end(), 1LL,
                         std::multiplies<int64_t>());
}

size_t MLIRNodeArg::getElementSize() const {
  switch (element_type_) {
  case 1: // TensorProto_DataType_FLOAT
    return sizeof(float);
  case 2: // TensorProto_DataType_UINT8
    return sizeof(uint8_t);
  case 3: // TensorProto_DataType_INT8
    return sizeof(int8_t);
  case 4: // TensorProto_DataType_UINT16
    return sizeof(uint16_t);
  case 5: // TensorProto_DataType_INT16
    return sizeof(int16_t);
  case 6: // TensorProto_DataType_INT32
    return sizeof(int32_t);
  case 7: // TensorProto_DataType_INT64
    return sizeof(int64_t);
  case 9: // TensorProto_DataType_BOOL
    return sizeof(bool);
  case 10: // TensorProto_DataType_FLOAT16
    return sizeof(uint16_t);
  case 11: // TensorProto_DataType_DOUBLE
    return sizeof(double);
  case 12: // TensorProto_DataType_UINT32
    return sizeof(uint32_t);
  case 13: // TensorProto_DataType_UINT64
    return sizeof(uint64_t);
  case 16: // TensorProto_DataType_BFLOAT16
    return sizeof(uint16_t);
  case 21:                      // TensorProto_DataType_INT4
    return sizeof(uint8_t) / 2; // 4 bits
  case 22:                      // TensorProto_DataType_UINT4
    return sizeof(uint8_t) / 2; // 4 bits
  default:
    return sizeof(float);
  }
}

bool MLIRNodeArg::isConstantValue() const {
  if (!value_) {
    return hasData();
  }
  if (auto defining_op = value_.getDefiningOp()) {
    return mlir::isa<mlir::arith::ConstantOp>(defining_op);
  }
  return false;
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
  case 21: // TensorProto_DataType_INT4
  case 22: // TensorProto_DataType_UINT4
    break; // Valid types
  default:
    LOG(WARNING) << "Unsupported element type: " << element_type;
    break;
  }
}

void MLIRNodeArg::copyData(const void* data, size_t size) {
  CHECK(data != nullptr) << "Data pointer cannot be null";
  CHECK(size > 0) << "Data size must be positive";
  CHECK(data_store_.has_value()) << "Data storage not initialized";

  data_store_->resize(size);
  std::memcpy(data_store_->data(), data, size);
}

} // namespace mlir_impl
} // namespace morphizen
