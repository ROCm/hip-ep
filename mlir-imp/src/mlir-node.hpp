/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>
#include <vector>

namespace morphizen {
namespace mlir_impl {

// Forward declarations
class MLIRNodeArgIndex;
class MLIRNodeIndex;
class MLIRNodeAttributes;
class MLIRGraph;

/**
 * @brief MLIR Node wrapper providing ONNX-like API
 *
 * This class wraps mlir::Operation* and provides a rich API similar to ONNX
 * Node, allowing users to access node properties, inputs, outputs, and
 * attributes in a familiar way.
 */
class MLIRNode : public mlir::Op<MLIRNode> {
public:
  using Op::Op;
  /**
   * @brief Get the name of this node
   * @return Node name, or "<unnamed>" if not available
   */
  mlir::StringRef getName() const;

  /**
   * @brief Get the operation domain
   * @return Domain string extracted from operation name
   */
  mlir::StringRef getDomain() const;

  /**
   * @brief Get the operation type
   * @return Operation type extracted from operation name
   */
  mlir::StringRef getOpType() const;

  /**
   * @brief Get the description of this node
   * @return Node description, or "<no description>" if not available
   */
  mlir::StringRef getDescription() const;

  /**
   * @brief Get input NodeArg pointers from morphizen.node_inputs attribute
   * @return Vector of NodeArg pointers stored in the node's input attribute
   */
  std::vector<MLIRNodeArgIndex> getInputNodeArgs() const;

  /**
   * @brief Get implicit input NodeArg pointers from
   *        morphizen.node_implicit_inputs attribute. Empty for region-less ops.
   * @return Vector of NodeArg pointers for outer-scope captures
   */
  std::vector<MLIRNodeArgIndex> getImplicitInputNodeArgs() const;

  /**
   * @brief Get output NodeArg pointers from morphizen.node_outputs attribute
   * @return Vector of NodeArg pointers stored in the node's output attribute
   */
  std::vector<MLIRNodeArgIndex> getOutputNodeArgs() const;

  /// Setters: serialize a vector of MLIRNodeArgIndex into the corresponding
  /// morphizen.node_inputs / node_implicit_inputs / node_outputs attribute.
  /// Each element is encoded as an i64 IntegerAttr (round-trips via
  /// MLIRNodeArgIndex::to_uint64 / from_uint64).
  void setInputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args);
  void setImplicitInputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args);
  void setOutputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args);

  /// `morphizen.*` attributes are runtime-only (encode raw process
  /// pointers); they must be stripped before the module is serialized to
  /// disk, then put back. Caller holds the snapshot across the write.
  llvm::SmallVector<mlir::NamedAttribute> backupAndClearMorphizenAttrs();
  void restoreMorphizenAttrs(llvm::ArrayRef<mlir::NamedAttribute> snapshot);

  /**
   * @brief Check if this node represents a fused operation
   * @return true if the node is a fused operation, false otherwise
   */
  bool isFused() const;

  /**
   * @brief Get the function body (sub-graph) for this node if it exists
   * @return Pointer to the MLIRGraph representing the function body, or nullptr
   * if none
   */
  const MLIRGraph* getFunctionBody();
};
static_assert(sizeof(MLIRNode) == sizeof(mlir::Operation*));
} // namespace mlir_impl
} // namespace morphizen
