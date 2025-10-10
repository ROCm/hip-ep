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
#include <glog/logging.h>

namespace morphizen {
namespace mlir_impl {

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
  auto x = full_op_name.split(':');
  return x.second.empty() ? "" : x.first;
}

mlir::StringRef MLIRNode::getOpType() const {
  auto op_name = (*this)->getName();
  auto full_op_name = op_name.getStringRef().split('.').second;
  CHECK(!full_op_name.empty());
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

std::vector<MLIRNodeArgIndex> MLIRNode::getInputNodeArgs() const {
  std::vector<MLIRNodeArgIndex> inputs;

  // Read the "morphizen.node_inputs" attribute
  if (auto inputsAttr = (*this)->getAttr(attr_names::MORPHIZEN_NODE_INPUTS)) {
    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(inputsAttr)) {
      inputs.reserve(arrayAttr.size());
      for (auto attr : arrayAttr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
          inputs.push_back(MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(
              reinterpret_cast<const void*>(intAttr.getInt())));
        }
      }
    }
  }

  return inputs;
}

std::vector<MLIRNodeArgIndex> MLIRNode::getOutputNodeArgs() const {
  std::vector<MLIRNodeArgIndex> outputs;

  // Read the "morphizen.node_outputs" attribute
  if (auto outputsAttr = (*this)->getAttr(attr_names::MORPHIZEN_NODE_OUTPUTS)) {
    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(outputsAttr)) {
      outputs.reserve(arrayAttr.size());
      for (auto attr : arrayAttr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
          outputs.push_back(MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(
              reinterpret_cast<const void*>(intAttr.getInt())));
        }
      }
    }
  }

  return outputs;
}

bool MLIRNode::isFused() const {
  // fused nodes are implemented as func.call
  return mlir::isa<mlir::func::CallOp>(*this);
}

const MLIRGraph* MLIRNode::getFunctionBody() {
  if (auto call_op = mlir::dyn_cast<mlir::func::CallOp>(getOperation())) {
    auto callee_name = call_op.getCallee().str();
    if (auto* graph = GraphStore::get_graph_by_symbol_name(callee_name)) {
      return graph;
    }
    auto& graph = const_cast<MLIRGraph&>(getOutputNodeArgs()[0].get_graph());
    auto& model = const_cast<MLIRModel&>(graph.get_model());
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
