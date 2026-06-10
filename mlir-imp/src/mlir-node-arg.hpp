/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace morphizen {
namespace mlir_impl {

// MLIRNodeArg — represents a named tensor value in the graph.
//
// Internally it holds:
//
//   name_    — the tensor name (persists across both stages)
//   value_   — variant<mlir::Value, TensorDesc>
//
// where TensorDesc is:
//
//   { meta: {name, shape, element_type},
//     data: optional< vector<uint8_t> | ExternalRef > }
//
// Lifecycle
// ---------
// A NodeArg starts in Stage 1 (TensorDesc) and transitions to Stage 2
// (mlir::Value) once the corresponding MLIR operation is created.
//
//   Stage 1 — value_ holds TensorDesc.
//             Shape, type, and optional data are stored directly.
//
//   Stage 2 — value_ holds mlir::Value.
//             Shape and type are embedded in the Value's tensor type.
//             Inline data lives in the op's DenseElementsAttr.
//             External data info lives in flat op attrs:
//             location (StringAttr), offset (I64), size (I64).
//
// MLIR mapping by NodeArg kind
// ----------------------------
// graph input      →  func.func block argument (%arg0 : tensor<...>)
//                      has name, shape, type; no data.
//
// graph output     →  onnx.Return %result
//                      shape and type come from the result's tensor type.
//
// node output      →  %result = onnx.XXXX {node_name = "..."}
//                      node_name is non-standard (MLIR ops are unnamed).
//
// constant (inline)→  onnx.Constant {value = dense<...>}
//                      vector<uint8_t> becomes DenseElementsAttr.
//
// constant (ext.)  →  onnx.Constant {location = "...", offset = N, size = N}
//                      non-standard extension (onnx-mlir has no such attr).
//
class alignas(8) MLIRNodeArg {
public:
  using shape_t = llvm::SmallVector<int64_t>;

  struct TensorMeta {
    std::string name;
    // nullopt = unranked tensor (ORT HasShape() == false). Empty value =
    // rank-0 scalar. Both lower to tensor<*xT> vs tensor<T> respectively.
    std::optional<shape_t> shape;
    int element_type;
  };

  struct ExternalRef {
    std::string location;
    size_t offset = 0;
    size_t size = 0;
  };

  using data_t = std::optional<std::variant<std::vector<uint8_t>, ExternalRef>>;

  struct TensorDesc {
    TensorMeta meta;
    data_t data;
  };

  using value_t = std::variant<mlir::Value, TensorDesc>;

  // Non-copyable but movable
  MLIRNodeArg(const MLIRNodeArg&) = delete;
  MLIRNodeArg& operator=(const MLIRNodeArg&) = delete;
  MLIRNodeArg(MLIRNodeArg&&) = default;
  MLIRNodeArg& operator=(MLIRNodeArg&&) = default;

  /// Constructor for tensor argument (no data). nullptr shape signals an
  /// unranked tensor (mapped to tensor<*xT>); non-null shape is taken as-is
  /// (empty = rank-0 scalar, non-empty = ranked).
  MLIRNodeArg(const std::string& name, const shape_t* shape, int element_type);

  /// Constructor for tensor argument (external data)
  MLIRNodeArg(const std::string& name, const shape_t& shape, int element_type,
              const std::string& loc, size_t offset, size_t size);

  /// Constructor for concrete tensor (with data)
  MLIRNodeArg(const std::string& name, const shape_t& shape, int element_type,
              const void* data, size_t data_size);

  /// Constructor from MLIR Value (extracts shape and type from value)
  MLIRNodeArg(const std::string& name, mlir::Value value);

  /// Get the argument name
  const std::string& getName() const;

  /// Get the shape. Mirrors Ort::TensorTypeAndShapeInfo: nullopt for unranked
  /// tensors (mapped to tensor<*xT> at the ORT boundary), Some({}) for
  /// rank-0 scalars, Some({dims...}) for ranked tensors.
  std::optional<shape_t> getShape() const;

  void setShape(const shape_t& shape);

  /// Get the element type
  int getElementType() const;
  void setElementType(int data_type);

  /// Get the MLIR value (mutable)
  const mlir::Value& getValue() const;
  mlir::Value& getValue();

  /// Set the MLIR value (mutable)
  void setValue(mlir::Value value) const;

  /// Get the MLIR type based on element type or value type
  mlir::Type getType(mlir::OpBuilder& builder) const;

  // === Data access methods (for concrete tensors) ===

  /// Get raw data pointer (returns nullptr for tensor arguments)
  const void* getData() const;

  /// Get data size in bytes
  size_t getDataSize() const;

  /// Check if this tensor has data storage
  bool hasData() const;

  /// Check if data is a non-owning external reference (zero-copy path)
  bool isExternalData() const;

  // === Utility methods ===

  /// Get total number of elements
  int64_t getElementCount() const;

  /// Check if the held mlir::Value is a constant
  /// Returns true if the value is produced by a constant operation
  /// (e.g., onnx.Constant)
  bool isConstantValue() const;

  // === Template methods for typed data access ===

  // === Structured access ===
  const TensorDesc& getDesc() const;
  const TensorMeta& getMeta() const;
  const ExternalRef* getExternalRef() const;

private:
  std::string name_;
  mutable value_t value_;

  void validateElementType(int element_type) const;
};

// === Type alias for backward compatibility ===
// This allows existing MLIRTensor code to work unchanged
using MLIRTensor = MLIRNodeArg;

} // namespace mlir_impl
} // namespace morphizen
