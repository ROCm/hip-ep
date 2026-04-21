/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/Builders.h"

namespace morphizen {
namespace mlir_impl {
// clang-format off
//===----------------------------------------------------------------------===//
// ONNX → MLIR Mapping
//
//   ONNX GraphProto Entity   | MLIR Representation
//   -------------------------+--------------------------------------------
//   GraphProto.name          | FuncOp attribute "onnx.graph.name" [string]
//   GraphProto.input         | FuncOp.block[0].block_argument attribute "onnx.name" [string]
//   GraphProto.output        | returnOp
//   NodeProto.name           | Operation attribute "onnx_node_name" [string]
//   NodeProto.output         | Operation Attribute "node.outputs" [array<string>]
//   node's inputs (runtime)  | Operation Attribute "morphizen.node_inputs" [array<MLIRNodeArgIndex>]
//   node's outputs (runtime) | Operation Attribute "morphizen.node_outputs" [array<MLIRNodeArgIndex>]
//   GraphProto.initializer   | onnx.Constant
//   AttributeProto           | `morphizen.placeholder` Operation
//===----------------------------------------------------------------------===//
// clang-format on

// Not all mlir::Operation* are morphizen::Node*;
// `onnx.Constant` and `onnx.Return` (or `func.return`) are special cases.

// MLIR/ONNX attribute name constants to avoid typos
namespace attr_names {

// Attribute names defined in morphizen-mlir framework
// "onnx.graph.name" to represent GraphProto.name
constexpr const char* ONNX_GRAPH_NAME = "onnx.graph.name";
// "node.outputs" to represent NodeProto.output (outputs NodeArg Name)
constexpr const char* NODE_OUTPUTS = "node.outputs";
// "morphizen.node_inputs" to store node inputs pointers for runtime access
constexpr const char* MORPHIZEN_NODE_INPUTS = "morphizen.node_inputs";
// "morphizen.node_outputs" to store node outputs pointers for runtime access
constexpr const char* MORPHIZEN_NODE_OUTPUTS = "morphizen.node_outputs";
// "morphizen.placeholder" for placeholder operations in attributes
constexpr const char* MORPHIZEN_PLACEHOLDER = "morphizen.placeholder";

// onnx.Custom convention: when an op cannot be expressed by a native MLIR op
// name (custom domains like com.microsoft, or unregistered ONNX ops), it is
// wrapped as `onnx.Custom` with the original op_type stored in
// `function_name` and the original domain stored in `domain_name`. These two
// attributes are MLIR-internal bookkeeping; the corresponding ONNX-level
// information lives in Node.op_type / Node.domain. They must never be
// re-emitted as user-visible ONNX node attributes (see
// MLIRNode::getOpType / getDomain for the reverse-decode).
constexpr const char* CUSTOM_OP_FUNCTION_NAME = "function_name";
constexpr const char* CUSTOM_OP_DOMAIN_NAME = "domain_name";

// Attribute names defined in onnx-mlir project
// The "onnx_node_name"  attribute to represent NodeProto.name
// This attribute is attached to operations in the onnx-mlir
// Example :
// ONNX node:         name = "QuantizeLinear_2"
// MLIR operation:    %936 = "onnx.QuantizeLinear"(...)
//                        {..., onnx_node_name = "QuantizeLinear_2"}
// Implementation:
//   - Set by `Operation.setAttr("onnx_node_name", ...)`
constexpr const char* ONNX_NODE_NAME = "onnx_node_name";
//===--------------------------------------===//
// The "onnx.name" attribute
// This attribute is attached to function arguments and results in the onnx-mlir
// function generated from an ONNX model. It preserves the original ONNX
// NodeArg name from the graph input/output.
// Example (resnet50.mlir):
// func.func @main_graph(%arg0: tensor<1x3x224x224xf32> {onnx.name = "blob.1"})
// -> (tensor<1x1000xf32> {onnx.name = "1327"}) {
//
// Implementation:
//   - Set on arguments via `func_.setArgAttr(i, "onnx.name", ...)`
//   - Set on results via  `func_.setResultAttr(i, "onnx.name", ...)`
//===--------------------------------------===//
constexpr const char* ONNX_NAME = "onnx.name";

} // namespace attr_names

namespace onnx_mlir {
// "onnx.NoValue" for representing optional/empty operands in ONNX operations
// MLIR has semantic differences from ONNX: MLIR cannot create values out of
// thin air; values must either be operation results or graph inputs. The
// onnx.NoValue operation is adopted from the onnx-mlir project to handle
// optional operands consistently.
constexpr const char* ONNX_NONE = "onnx.NoValue";

// "onnx.Return" is the function return operation in the ONNX dialect.
// It terminates a func::FuncOp in the ONNX dialect and is semantically
// equivalent to func::ReturnOp but allows shape refinement between operands
// and function signature result types. This is adopted from onnx-mlir for
// compatibility with onnx-mlir's MLIR representation.
constexpr const char* ONNX_RETURN = "onnx.Return";

// Helper function to check if an operation is an onnx.Return operation
inline bool isOnnxReturn(mlir::Operation* op) {
  return op && op->getName().getStringRef() == ONNX_RETURN;
}

// Helper function to check if an operation is either onnx.Return or func.return
// This is useful for transitional code that may encounter either terminator
inline bool isReturnOp(mlir::Operation* op) {
  if (!op)
    return false;
  auto name = op->getName().getStringRef();
  return name == ONNX_RETURN || name == "func.return";
}
} // namespace onnx_mlir

// Utility function to convert ONNX element types to MLIR types
// If shape is provided, creates a tensor type (ranked or unranked)
// If shape is not provided, returns just the element type
mlir::Type
onnxElementTypeToMlirType(int element_type, mlir::OpBuilder& builder,
                          const llvm::SmallVector<int64_t>* shape = nullptr);

} // namespace mlir_impl
} // namespace morphizen
