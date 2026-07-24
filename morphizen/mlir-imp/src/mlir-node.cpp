/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-node.hpp"
#include "./mlir-constants.hpp"
#include "./mlir-graph-store.hpp"
#include "./mlir-graph.hpp"
#include "./mlir-model.hpp"
#include "./mlir-named-attribute.hpp"
#include "./mlir-node-arg-index.hpp"
#include "./mlir-node-attributes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <glog/logging.h>

namespace morphizen {
namespace mlir_impl {

// File-private attribute names. These are runtime-pointer-encoded indices
// that should not leak outside MLIRNode -- consumers go through
// MLIRNode::get/set/backupAndClear/restore methods.
namespace {
constexpr const char *MORPHIZEN_NODE_INPUTS = "morphizen.node_inputs";
constexpr const char *MORPHIZEN_NODE_IMPLICIT_INPUTS =
    "morphizen.node_implicit_inputs";
constexpr const char *MORPHIZEN_NODE_OUTPUTS = "morphizen.node_outputs";
} // namespace

// === Property Accessors ===

mlir::StringRef MLIRNode::getName() const {
  if (auto attr = (*this)->getAttrOfType<mlir::StringAttr>(
          attr_names::ONNX_NODE_NAME)) {
    return attr.getValue();
  }
  return "<unnamed>";
}

mlir::StringRef MLIRNode::getDomain() const {
  auto op_name = (*this)->getName();
  auto full_op_name = op_name.getStringRef().split('.').second;
  CHECK(!full_op_name.empty());

  // Check if this is an onnx.Custom operation
  if (full_op_name == "Custom") {
    // For onnx.Custom, domain is stored in the "domain_name" attribute
    if (auto attr = (*this)->getAttrOfType<mlir::StringAttr>("domain_name")) {
      return attr.getValue();
    }
    return "";
  }

  // For regular operations, parse domain from operation name (e.g.,
  // "com.microsoft:Op")
  auto x = full_op_name.split(':');
  return x.second.empty() ? "" : x.first;
}

mlir::StringRef MLIRNode::getOpType() const {
  auto op_name = (*this)->getName();
  auto full_op_name = op_name.getStringRef().split('.').second;
  CHECK(!full_op_name.empty());

  // Check if this is an onnx.Custom operation
  if (full_op_name == "Custom") {
    // For onnx.Custom, op_type is stored in the "function_name" attribute
    if (auto attr = (*this)->getAttrOfType<mlir::StringAttr>("function_name")) {
      return attr.getValue();
    }
    return "Custom";
  }

  // For regular operations, parse op_type from operation name
  auto x = full_op_name.split(':');
  return x.second.empty() ? x.first : x.second;
}

mlir::StringRef MLIRNode::getDescription() const {
  // TODO: is "description" defined in onnx-mlir, and put it into the constant
  // header file.
  if (auto attr = (*this)->getAttrOfType<mlir::StringAttr>("description")) {
    return attr.getValue();
  }
  return "<no description>";
}

namespace {

// Decode a morphizen.node_inputs / node_implicit_inputs / node_outputs
// attribute (ArrayAttr of i64 IntegerAttr storing raw uint64 payloads
// via bit-pattern reinterpret) into a vector of indices.
std::vector<MLIRNodeArgIndex> decode_node_arg_array_attr(mlir::Attribute attr) {
  std::vector<MLIRNodeArgIndex> result;
  auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr);
  if (!arrayAttr) {
    return result;
  }
  result.reserve(arrayAttr.size());
  for (auto e : arrayAttr) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(e)) {
      result.push_back(MLIRNodeArgIndex::from_uint64(
          static_cast<uint64_t>(intAttr.getInt())));
    }
  }
  return result;
}

// Encode a vector of indices into an ArrayAttr of i64 IntegerAttrs for the
// matching attribute slot.
mlir::ArrayAttr
encode_node_arg_array_attr(mlir::MLIRContext *ctx,
                           llvm::ArrayRef<MLIRNodeArgIndex> args) {
  mlir::OpBuilder builder(ctx);
  llvm::SmallVector<mlir::Attribute> entries;
  entries.reserve(args.size());
  for (const auto &a : args) {
    entries.push_back(
        builder.getI64IntegerAttr(static_cast<int64_t>(a.to_uint64())));
  }
  return builder.getArrayAttr(entries);
}

} // namespace

std::vector<MLIRNodeArgIndex> MLIRNode::getInputNodeArgs() const {
  return decode_node_arg_array_attr((*this)->getAttr(MORPHIZEN_NODE_INPUTS));
}

std::vector<MLIRNodeArgIndex> MLIRNode::getImplicitInputNodeArgs() const {
  return decode_node_arg_array_attr(
      (*this)->getAttr(MORPHIZEN_NODE_IMPLICIT_INPUTS));
}

std::vector<MLIRNodeArgIndex> MLIRNode::getOutputNodeArgs() const {
  return decode_node_arg_array_attr((*this)->getAttr(MORPHIZEN_NODE_OUTPUTS));
}

void MLIRNode::setInputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args) {
  (*this)->setAttr(MORPHIZEN_NODE_INPUTS,
                   encode_node_arg_array_attr((*this)->getContext(), args));
}

void MLIRNode::setImplicitInputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args) {
  (*this)->setAttr(MORPHIZEN_NODE_IMPLICIT_INPUTS,
                   encode_node_arg_array_attr((*this)->getContext(), args));
}

void MLIRNode::setOutputNodeArgs(llvm::ArrayRef<MLIRNodeArgIndex> args) {
  (*this)->setAttr(MORPHIZEN_NODE_OUTPUTS,
                   encode_node_arg_array_attr((*this)->getContext(), args));
}

llvm::SmallVector<mlir::NamedAttribute>
MLIRNode::backupAndClearMorphizenAttrs() {
  llvm::SmallVector<mlir::NamedAttribute> snapshot;
  // Collect first, then remove: removeAttr invalidates the attr-dict
  // iteration order.
  for (auto &named : (*this)->getAttrs()) {
    if (named.getName().strref().starts_with("morphizen.")) {
      snapshot.push_back(named);
    }
  }
  for (auto &named : snapshot) {
    (*this)->removeAttr(named.getName());
  }
  return snapshot;
}

void MLIRNode::restoreMorphizenAttrs(
    llvm::ArrayRef<mlir::NamedAttribute> snapshot) {
  for (auto &named : snapshot) {
    (*this)->setAttr(named.getName(), named.getValue());
  }
}

bool MLIRNode::isFused() const {
  // fused nodes are implemented as func.call
  return mlir::isa<mlir::func::CallOp>(*this);
}

const MLIRGraph *MLIRNode::getFunctionBody() {
  if (auto call_op = mlir::dyn_cast<mlir::func::CallOp>(getOperation())) {
    auto callee_name = call_op.getCallee().str();
    if (auto *graph = GraphStore::get_graph_by_symbol_name(callee_name)) {
      return graph;
    }
    auto &graph = const_cast<MLIRGraph &>(getOutputNodeArgs()[0].get_graph());
    auto &model = const_cast<MLIRModel &>(graph.get_model());
    auto func_ops = model.getModule().getOps<mlir::func::FuncOp>();
    for (auto func_op : func_ops) {
      if (func_op.getSymName() == callee_name) {
        return graph.add_subgraph(std::make_unique<MLIRGraph>(model, func_op));
      }
    }
    LOG(WARNING) << "getFunctionBody: Could not find function '" << callee_name
                 << "' for fused node '" << getName().str() << "'";
  } else {
    LOG(WARNING) << "getFunctionBody: Node '" << getName().str()
                 << "' is not a fused operation (not a func.call)";
  }
  return nullptr;
}
} // namespace mlir_impl
} // namespace morphizen
