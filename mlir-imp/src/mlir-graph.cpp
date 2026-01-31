/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-graph.hpp"
#include "./mlir-constants.hpp"
#include "./mlir-graph-store.hpp"
#include "./mlir-node-arg.hpp"

#include "./mlir-node.hpp"
#include "mlir-constants.hpp"
#include "mlir-model.hpp"
#include "mlir-named-attribute.hpp"
#include "mlir-node-attributes.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"   // for ConstantOp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // for EmptyOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "morphizen-utils/env_config.hpp"
#include "llvm/ADT/STLExtras.h"       // for map_range, to_vector
#include "llvm/ADT/SmallSet.h"        // for SmallSet
#include "llvm/ADT/SmallVector.h"     // for SmallVector
#include "llvm/Support/raw_ostream.h" // for raw_fd_ostream
#include <algorithm>                  // for std::sort
#include <glog/logging.h>
#include <iomanip>                    // for std::setprecision
#include <system_error>               // for std::error_code
#include <unordered_set>              // for std::unordered_set
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH, "0")
DEF_ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_GENERIC, "0")
DEF_ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_DEBUG_INFO, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH) >= n)
namespace morphizen {
namespace mlir_impl {

static mlir::Operation* get_or_create_terminator(mlir::func::FuncOp func) {
  // Get the function's entry block
  auto& entryBlock = func.getBody().front();
  mlir::Operation* terminator = nullptr;
  // Find the existing terminator (should be a func.return operation)
  auto has_terminator = entryBlock.mightHaveTerminator();

  if (!has_terminator) {
    // No terminator exists, create a new func.return operation
    auto* context = func->getContext();
    mlir::OpBuilder builder(context);
    builder.setInsertionPointToEnd(&entryBlock);

    // Convert gsl::span to SmallVector for MLIR API
    llvm::SmallVector<mlir::Value> outputValues({});
    terminator = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                                      outputValues);
  } else {
    terminator = entryBlock.getTerminator();
  }
  return terminator;
}

static mlir::Operation* get_or_create_none(mlir::func::FuncOp func) {
  // First, try to find an existing NoneOp in the function
  mlir::Operation* existing_none = nullptr;
  func.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      existing_none = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (existing_none) {
    return existing_none;
  }

  // No existing NoneOp found, create a new one
  mlir::OpBuilder builder(func->getContext());
  builder.setInsertionPointToStart(&func.getBody().front());
  mlir::OperationState state(builder.getUnknownLoc(), onnx_mlir::ONNX_NONE);
  state.addTypes(builder.getNoneType());
  return builder.create(state);
}
MLIRGraph::MLIRGraph(MLIRModel& model, mlir::func::FuncOp func,
                     uint32_t proposed_graph_id)
    : model_(model), func_(func),
      graph_id_(proposed_graph_id > 0
                    ? GraphStore::allocate_graph_id(this, proposed_graph_id)
                    : GraphStore::allocate_graph_id(this)),
      terminator_{get_or_create_terminator(func)},
      none_{get_or_create_none(func)}, value_map_() {
  initialize();
}

MLIRGraph::~MLIRGraph() { GraphStore::release_graph_id(this, graph_id_); }

std::string MLIRGraph::extract_value_name(const mlir::Value value) {
  // As defined in the language reference,
  // each Value is either a BlockArgument or the result of exactly one
  // Operation (an Operation can have multiple results, each of them is a
  // separate Value).

  // Try to get the name from onnx.name attribute if it's a block argument
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // Get name from onnx.name attribute on function argument
    if (auto nameAttr =
            func_.getArgAttr(blockArg.getArgNumber(), attr_names::ONNX_NAME)) {
      if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr)) {
        return stringAttr.getValue().str();
      }
    }
    return "input_" + std::to_string(blockArg.getArgNumber());
  } else if (auto definingOp = value.getDefiningOp()) {
    // Get name from node.outputs attribute of the defining operation
    auto defining_op_attrs = MLIRNodeAttributes(definingOp);
    if (defining_op_attrs.has_attribute(attr_names::NODE_OUTPUTS)) {
      auto node_output_names =
          defining_op_attrs.get_attribute_as_strings(attr_names::NODE_OUTPUTS);
      unsigned ret_index = mlir::cast<mlir::OpResult>(value).getResultNumber();
      if (ret_index < node_output_names.size()) {
        return node_output_names[ret_index];
      }
    }
    return "unnamed_" + std::to_string(reinterpret_cast<uintptr_t>(
                            value.getAsOpaquePointer()));
  } else {
    LOG(FATAL) << "Value is neither BlockArgument nor has defining op.";
    return "_undefined";
  }
}

void MLIRGraph::initialize() {
  MY_LOG(1) << "Initializing MLIRGraph";
  node_args_map_.clear();
  all_node_args_.clear();
  graph_inputs_.clear();
  graph_outputs_.clear();
  constant_initializers_.clear();
  initialized_tensors_cache_.clear();
  staging_nodes_.clear();

  initialize_graph_inputs();
  initialize_constant_initializers();
  initialize_node_args_map();
  initialize_graph_outputs();
  // Maintain morphizen attributes after all node args are initialized
  maintain_morphizen_attributes();
}

void MLIRGraph::initialize_node_args_map() {
  MY_LOG(1) << "Initializing node args map and constant initializers";
  // Walk all operations and extract their results
  func_.walk([&](mlir::Operation* op) {
    // Skip the function operation itself
    if (op == func_.getOperation()) {
      return;
    }
    // Skip return operations (terminators)
    if (mlir::isa<mlir::func::ReturnOp>(op)) {
      return;
    }
    // Skip arith::ConstantOp, arith::ConstantOp mapping onnx
    // GraphProto.initializer
    if (mlir::isa<mlir::arith::ConstantOp>(op)) {
      // already processed in initialize_constant_initializers
      return;
    }
    if (op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      return;
    }

    // Process all results of this operation
    for (auto result : op->getResults()) {
      // Try to get the name from node.outputs attribute or generate one
      std::string name = extract_value_name(result);
      // Create MLIRNodeArg from the operation result
      all_node_args_.push_back(std::make_unique<MLIRNodeArg>(name, result));
      node_args_map_.emplace(name, MLIRNodeArgIndex::node_output(
                                       (int32_t)all_node_args_.size() - 1,
                                       GraphId::create_main_graph(graph_id_)));
      // MY_LOG(1) << "Extracted node arg from operation result: " << name;
    }
  });
}

void MLIRGraph::initialize_constant_initializers() {
  MY_LOG(1) << "Initializing constant initializers";
  // use mlir filter iterator to find all arith.ConstantOp
  auto& block = func_.getBody().front();
  auto constant_ops = block.getOps<mlir::arith::ConstantOp>();
  for (auto op : constant_ops) {
    // attr :  NODE_OUTPUTS
    // constant op has a single result
    auto value = op.getResult();
    auto value_name = extract_value_name(value);
    all_node_args_.push_back(std::make_unique<MLIRNodeArg>(value_name, value));
    node_args_map_.emplace(
        value_name,
        MLIRNodeArgIndex::node_output((int32_t)all_node_args_.size() - 1,
                                      GraphId::create_main_graph(graph_id_)));

    constant_initializers_.push_back(node_args_map_[value_name]);
    initialized_tensors_cache_[value_name] =
        &node_args_map_[value_name].get_const_data_as_tensor();
  }
}

void MLIRGraph::initialize_graph_inputs() {
  MY_LOG(1) << "Initializing graph inputs";

  // Also extract from function arguments
  auto arguments = func_.getArguments();
  for (auto arg : arguments) {
    // Get the argument name from the onnx.name attribute
    auto name = extract_value_name(arg);
    // Create MLIRNodeArg from the function argument
    all_node_args_.push_back(std::make_unique<MLIRNodeArg>(name, arg));
    node_args_map_.emplace(name, MLIRNodeArgIndex::node_output(
                                     (int32_t)all_node_args_.size() - 1,
                                     GraphId::create_main_graph(graph_id_)));
    graph_inputs_.push_back(node_args_map_[name]);
    MY_LOG(2) << "Added graph input: " << name;
  }
  MY_LOG(1) << "Initialized " << graph_inputs_.size() << " graph inputs";
}

void MLIRGraph::initialize_graph_outputs() {
  MY_LOG(1) << "Initializing graph outputs";

  // Clear existing outputs
  graph_outputs_.clear();

  CHECK(terminator_) << "Terminator must not be null when initializing outputs";
  // Get the return operation to find graph outputs
  if (terminator_ && mlir::isa<mlir::func::ReturnOp>(terminator_)) {
    auto return_op = mlir::cast<mlir::func::ReturnOp>(terminator_);
    auto operands = return_op.getOperands();

    for (auto operand : operands) {
      // Find the corresponding NodeArg in our collection
      for (size_t i = 0; i < all_node_args_.size(); ++i) {
        const auto& node_arg = all_node_args_[i];
        if (node_arg && node_arg->getValue() == operand) {
          graph_outputs_.push_back(node_args_map_[node_arg->getName()]);
          MY_LOG(2) << "Added graph output: " << node_arg->getName()
                    << " at index " << i;
          break;
        }
      }
    }
  }

  MY_LOG(1) << "Initialized " << graph_outputs_.size() << " graph outputs";
}

void MLIRGraph::maintain_morphizen_attributes() {
  MY_LOG(1) << "Maintaining morphizen.node_inputs and morphizen.node_outputs "
               "attributes";

  // Get MLIR context and builder
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);
  func_.walk([&](mlir::Operation* op) {
    // Skip function operation itself, return operations, constant operations,
    // and our custom onnx.None operations
    if (op != func_.getOperation() && !mlir::isa<mlir::func::ReturnOp>(op) &&
        !mlir::isa<mlir::arith::ConstantOp>(op) &&
        op->getName().getStringRef() != onnx_mlir::ONNX_NONE) {
      // Collect input NodeArg pointers for this operation
      llvm::SmallVector<mlir::Attribute> inputIndexes;
      for (mlir::Value operand : op->getOperands()) {
        auto value_name = extract_value_name(operand);
        inputIndexes.push_back(builder.getI64IntegerAttr(
            reinterpret_cast<int64_t>(get_node_arg_index(value_name)
                                          .to_morphizen_core_node_arg_ptr())));
      }

      op->setAttr(attr_names::MORPHIZEN_NODE_INPUTS,
                  builder.getArrayAttr(inputIndexes));

      // Collect output NodeArg pointers for this operation
      llvm::SmallVector<mlir::Attribute> outputIndexes;
      for (auto result : op->getResults()) {
        auto value_name = extract_value_name(result);
        outputIndexes.push_back(builder.getI64IntegerAttr(
            reinterpret_cast<int64_t>(get_node_arg_index(value_name)
                                          .to_morphizen_core_node_arg_ptr())));
      }

      op->setAttr(attr_names::MORPHIZEN_NODE_OUTPUTS,
                  builder.getArrayAttr(outputIndexes));
    }
  });

  MY_LOG(1) << "Completed maintaining morphizen attributes";
}

const std::string& MLIRGraph::get_name() const {
  // get attribute attr_names::ONNX_GRAPH_NAME from func_
  CHECK(func_) << "func_ must not be null when retrieving graph name";
  if (auto attr = func_->getAttr(attr_names::ONNX_GRAPH_NAME)) {
    if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
      static thread_local std::string graph_name;
      graph_name = string_attr.getValue().str();
      return graph_name;
    }
  }

  // Fallback to default name if attribute is not found
  static const std::string default_name = "mlir_main_graph";
  return default_name;
}

void MLIRGraph::set_name(const char* name) {
  // Set the onnx.graph.name attribute on the function
  CHECK(func_) << "func_ must not be null when setting graph name";
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);
  func_->setAttr(attr_names::ONNX_GRAPH_NAME, builder.getStringAttr(name));
  MY_LOG(1) << "Set graph name to: " << name;
}
const MLIRModel& MLIRGraph::get_model() const { return model_; }

std::string MLIRGraph::get_symbol_name() const {
  return const_cast<mlir::func::FuncOp&>(func_).getSymName().str();
}

std::vector<mlir::Operation*> MLIRGraph::nodes_unsafe() const {
  std::vector<mlir::Operation*> nodes;

  // Iterate over all operations in the function body
  const_cast<mlir::func::FuncOp&>(func_).walk([&](mlir::Operation* op) {
    // Skip function operation itself, return operations, constant operations,
    // and our custom onnx.None operations
    if (op != const_cast<mlir::func::FuncOp&>(func_).getOperation() &&
        !mlir::isa<mlir::func::ReturnOp>(op) &&
        !mlir::isa<mlir::arith::ConstantOp>(op) &&
        op->getName().getStringRef() != onnx_mlir::ONNX_NONE) {
      // Add operation as node
      nodes.push_back(op);
    }
  });

  MY_LOG(1) << "Found " << nodes.size() << " nodes in MLIR graph";
  return nodes;
}

llvm::SmallVector<MLIRNodeArgIndex> MLIRGraph::get_inputs() const {
  return graph_inputs_;
}

llvm::SmallVector<MLIRNodeArgIndex> MLIRGraph::get_outputs() const {
  return graph_outputs_;
}

void MLIRGraph::set_outputs(
    const llvm::SmallVector<MLIRNodeArgIndex>& outputs) {

  graph_outputs_ = outputs;

  CHECK(func_) << "func_ must not be null when setting outputs";
  // Convert MLIRNodeArgIndex to mlir::Value
  llvm::SmallVector<mlir::Value> mlir_outputs;
  mlir_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    // Get the mlir::Value from the NodeArg
    auto* nodeArg = get_node_arg(output);
    CHECK(nodeArg->getValue()) << "NodeArg must have a value for outputs";
    mlir_outputs.push_back(nodeArg->getValue());
  }

  if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(terminator_)) {
    // Replace the operands of the existing return operation
    returnOp->setOperands(
        llvm::ArrayRef<mlir::Value>(mlir_outputs.data(), mlir_outputs.size()));

    MY_LOG(1) << "Updated existing func.return terminator with "
              << mlir_outputs.size() << " outputs";
  } else {
    LOG(FATAL) << "Unexpected terminator type: "
               << terminator_->getName().getStringRef().str()
               << ", expected func.return";
  }

  // Update func_ type to reflect the new output types
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);

  auto argTypes = func_.getArgumentTypes();
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(mlir_outputs.size());
  for (const auto& output : mlir_outputs) {
    resultTypes.push_back(output.getType());
  }

  func_.setType(builder.getFunctionType(argTypes, resultTypes));
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto name = outputs[i].get_name();
    func_.setResultAttr((unsigned int)i, mlir_impl::attr_names::ONNX_NAME,
                        mlir::StringAttr::get(func_.getContext(), name));
  }

  MY_LOG(1) << "Updated function type with " << resultTypes.size()
            << " output types";
}

void MLIRGraph::set_inputs(const llvm::SmallVector<MLIRNodeArgIndex>& inputs) {
  graph_inputs_ = inputs;
  // for now, we only support set_inputs when the graph has no inputs.
  auto arguments = func_.getArguments();
  if (!arguments.empty()) {
    LOG(FATAL) << "Graph already has inputs, cannot set new inputs.";
  }
  auto builder = mlir::OpBuilder(func_.getContext());
  // Get the function's entry block and add arguments to it
  auto& entryBlock = func_.getBody().front();
  auto loc = builder.getUnknownLoc();
  for (auto& input : inputs) {
    // For inputs, we typically don't have shape information at this point
    // so we create unranked tensor types (empty shape)
    auto node_arg = get_node_arg(input);
    node_arg->setValue(entryBlock.addArgument(node_arg->getType(builder), loc));
  }

  arguments = func_.getArguments();
  auto retTys = func_.getResultTypes();
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (size_t i = 0; i < inputs.size(); ++i) {
    argTypes.push_back(arguments[i].getType());
  }
  func_.setType(builder.getFunctionType(argTypes, retTys));
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto name = inputs[i].get_name();
    func_.setArgAttr((unsigned int)i, mlir_impl::attr_names::ONNX_NAME,
                     mlir::StringAttr::get(func_.getContext(), name));
  }
  return;
}

const mlir::Operation* MLIRGraph::get_node(size_t index) const {
  static_assert(sizeof(mlir::Operation*) == sizeof(size_t));
  return reinterpret_cast<const mlir::Operation*>(index);
}

mlir::Operation*
MLIRGraph::producer_node(const std::string& node_arg_name) const {
  if (auto node_arg_index = get_node_arg_index(node_arg_name)) {
    if (const auto* node_arg = get_node_arg(node_arg_index)) {
      auto value = node_arg->getValue();
      if (mlir::Operation* defining_op = value.getDefiningOp()) {
        if (mlir::isa<mlir::arith::ConstantOp>(defining_op)) {
          // For constant initlalizer we skip it , for same with onnx
          return nullptr;
        }

        if (defining_op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
          return nullptr;
        }
        return defining_op;
      }
    }
  }
  MY_LOG(2) << "Node arg not found: " << node_arg_name;
  return nullptr;
}

MLIRNodeArgIndex MLIRGraph::get_node_arg_index(const std::string& name) const {
  // Look up the MLIR value by name in our SymbolTable
  auto node_arg = node_args_map_.find(name);
  if (node_arg != node_args_map_.end()) {
    return node_arg->second;
  }
  return MLIRNodeArgIndex::invalid();
}

MLIRNodeArgIndex
MLIRGraph::node_arg_new(const std::string& name,
                        const llvm::SmallVector<int64_t>* shape,
                        int element_type) {
  CHECK(shape != nullptr);
  // the name must not exists in `node_args_`
  if (node_args_map_.count(name) > 0) {
    auto node_arg_index = node_args_map_[name];
    // in onnxruntime implementation,
    // add_initialized_tensor updates name_to_initial_tensor_
    // InitializedTensorSet name_to_initial_tensor_
    // and node_arg_new_ update node_args_
    // std::unordered_map<std::string, std::unique_ptr<NodeArg>> node_args_;
    // so it is OK if we create a node arg more than once for constant
    // intializers.
    auto& node_arg = node_arg_index.get_node_arg();
    if (node_arg.isConstantValue() && *shape == node_arg.getShape() &&
        element_type == node_arg.getElementType()) {
      return node_arg_index;
    }
    LOG(FATAL) << "symbol \"" << name << "\" already exists.";
    // Return a default-constructed MLIRNodeArgIndex (this won't be reached due
    // to LOG(FATAL))
    return MLIRNodeArgIndex::invalid();
  }
  // Create MLIR type for the node argument
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);
  auto nodeArg = std::make_unique<MLIRNodeArg>(name, *shape, element_type);
  all_node_args_.push_back(std::move(nodeArg));
  auto ret =
      MLIRNodeArgIndex::node_output((int32_t)all_node_args_.size() - 1,
                                    GraphId::create_main_graph(graph_id_));
  node_args_map_.emplace(name, ret);
  MY_LOG(1) << "Created new node argument: " << name
            << " with element type: " << element_type;
  // Return MLIRNodeArgIndex wrapping the MLIRNodeArg pointer
  return ret;
}

const mlir::Operation*
MLIRGraph::add_node(const std::string& name, const std::string& op_type,
                    const std::string& description,
                    const std::vector<MLIRNodeArgIndex>& input_args,
                    const std::vector<MLIRNodeArgIndex>& output_args,
                    const MLIRNodeAttributes& attributes,
                    const std::string& domain) {
  (void)description; // description is not used in this implementation

  // Get MLIR context and builder
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);
  CHECK(terminator_ != nullptr);

  // Set insertion point - default to before terminator
  builder.setInsertionPoint(terminator_);

  mlir::Operation* latestInputOp = nullptr;
  bool hasConstantInput = false;

  if (input_args.empty()) {
    // Insert at the beginning of the function body when no input arguments
    auto& entryBlock = func_.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);
  } else {
    for (const auto& arg : input_args) {
      if (auto* input_node_arg = get_node_arg(arg)) {
        if (auto& value = input_node_arg->getValue()) {
          if (auto* definingOp = value.getDefiningOp()) {
            if (mlir::isa<mlir::arith::ConstantOp>(definingOp)) {
              hasConstantInput = true;
            }

            if (!latestInputOp || latestInputOp->isBeforeInBlock(definingOp)) {
              latestInputOp = definingOp;
              builder.setInsertionPointAfter(latestInputOp);
            }
          } else {
            // means graph input
            if (!latestInputOp) {
              // Only handle cases where InsertionPoint is not set
              auto& entryBlock = func_.getBody().front();
              builder.setInsertionPointToStart(&entryBlock);
            }
          }
        }
      }
    }

    // If any input is from arith.constant, find the last arith.constant in the
    // block and use it as reference point to ensure all constants stay at the
    // beginning
    if (hasConstantInput) {
      auto& entryBlock = func_.getBody().front();
      mlir::Operation* lastConstantOp = nullptr;

      for (auto& op : entryBlock.getOperations()) {
        if (mlir::isa<mlir::arith::ConstantOp>(op)) {
          lastConstantOp = &op;
        }
      }

      // Use the last constant operation as reference if it's after our current
      // insertion point
      if (lastConstantOp &&
          (!latestInputOp || latestInputOp->isBeforeInBlock(lastConstantOp))) {
        latestInputOp = lastConstantOp;
        builder.setInsertionPointAfter(latestInputOp);
      }
    }
  }

  // Convert MLIRNodeArgIndex to mlir::Value for input arguments
  llvm::SmallVector<mlir::Value> mlir_input_args;
  for (const auto& input : input_args) {
    if (auto* input_node_arg = get_node_arg(input)) {
      if (auto mlir_value = input_node_arg->getValue()) {
        mlir_input_args.push_back(mlir_value);
        continue;
      }
    }
    mlir_input_args.push_back(none_->getResult(0));
  }

  // For now, create a generic ONNX operation
  // In a real implementation, this would need to map op_type to specific MLIR
  // operations

  // Create operation name with domain
  std::string full_op_name = domain.empty() ? op_type : domain + ":" + op_type;

  // Convert output types from mlir::Values (if they have types)
  llvm::SmallVector<mlir::Type> result_types;
  for (const auto& output : output_args) {
    result_types.push_back(get_node_arg(output)->getType(builder));
  }

  // Create operation location
  auto loc = builder.getUnknownLoc();

  // Get attributes as MLIR dictionary
  mlir::DictionaryAttr mlir_attrs = attributes.get_mlir_dictionary();

  // Create the operation
  mlir::OperationState state(loc, "onnx." + full_op_name);
  state.addOperands(mlir_input_args);
  state.addTypes(result_types);
  state.attributes = mlir_attrs;

  mlir::Operation* op = builder.create(state);

  {
    // Store NodeArg pointer references as MLIR attributes for runtime access
    llvm::SmallVector<mlir::Attribute> inputIndexes;
    for (const auto& input : input_args) {
      inputIndexes.push_back(builder.getI64IntegerAttr(
          reinterpret_cast<int64_t>(input.to_morphizen_core_node_arg_ptr())));
    }
    op->setAttr(attr_names::MORPHIZEN_NODE_INPUTS,
                builder.getArrayAttr(inputIndexes));

    llvm::SmallVector<mlir::Attribute> outputIndexes;
    for (const auto& output : output_args) {
      outputIndexes.push_back(builder.getI64IntegerAttr(
          reinterpret_cast<int64_t>(output.to_morphizen_core_node_arg_ptr())));
    }
    op->setAttr(attr_names::MORPHIZEN_NODE_OUTPUTS,
                builder.getArrayAttr(outputIndexes));
  }

  // Collect output names for the "node.outputs" attribute
  llvm::SmallVector<mlir::Attribute> outputNames;
  for (const auto& output : output_args) {
    outputNames.push_back(builder.getStringAttr(output.get_name()));
  }

  // Set the "node.outputs" attribute on the operation
  if (!outputNames.empty()) {
    op->setAttr(attr_names::NODE_OUTPUTS, builder.getArrayAttr(outputNames));
  }
  // Set the "onnx_node_name" attribute on the operation
  op->setAttr(attr_names::ONNX_NODE_NAME, builder.getStringAttr(name));

  // Register the operation results in the symbol table
  mlir::IRRewriter rewriter(func_->getContext());
  for (const auto& [output, result] :
       llvm::zip(output_args, op->getResults())) {
    auto& node_arg = output.get_node_arg();
    if (auto& value = node_arg.getValue()) {
      rewriter.replaceAllUsesWith(value, result);
    }
    node_arg.setValue(result);
  }

  // Value replacement diagram:
  // Before:  Old Operation (e.g., Relu)    After:  New Operation
  //              ^                                      ^
  //              |                                      |
  //          old_value          ----replace---->    new_value
  //
  // This replaces all uses of the old operation's output value with the new
  // operation's result

  // For now, return a placeholder - in a real implementation this would
  // be converted to a proper morphizen::Node representation
  // This requires more infrastructure to map MLIR operations to Node objects
  MY_LOG(1) << "Added MLIR node: " << name << " (" << op_type << ")";
  staging_nodes_.insert(op);
  // Return the operation directly
  return op;
}

void MLIRGraph::add_constant_initialized_tensor(
    const mlir_impl::MLIRNodeArg* tensor1) {
  CHECK(tensor1 != nullptr) << "Cannot add null tensor to graph";

  std::string name = tensor1->getName();

  // Check if tensor with this name already exists
  auto node_arg_it = node_args_map_.find(name);
  if (node_arg_it == node_args_map_.end()) {
    auto node_arg_1 = std::make_unique<MLIRNodeArg>(
        std::move(const_cast<mlir_impl::MLIRNodeArg&>(*tensor1)));
    all_node_args_.push_back(std::move(node_arg_1));

    auto node_arg_index =
        MLIRNodeArgIndex::node_output((int32_t)all_node_args_.size() - 1,
                                      GraphId::create_main_graph(graph_id_));
    constant_initializers_.push_back(node_arg_index);
    node_arg_it = node_args_map_.emplace(name, node_arg_index).first;
    MY_LOG(1) << "Created new node argument for constant tensor: " << name;
  }
  auto node_arg_ptr = node_arg_it->second;
  auto node_arg = get_node_arg(node_arg_ptr);
  auto& node_arg_value = node_arg->getValue();
  CHECK(name == node_arg->getName());
  CHECK(!node_arg_value) << "node arg \"" << name << "\" is already in use";
  MY_LOG(1) << "Adding constant initialized tensor: " << name
            << " with data type: " << node_arg->getElementType()
            << " and data size: " << node_arg->getDataSize();

  // Get MLIR context and builder
  auto* context = func_->getContext();
  mlir::OpBuilder builder(context);

  // Find the last arith.constant operation to insert after it
  // Order should be: onnx.None -> arith.constant ops -> other ops -> return
  auto& block = func_.getBody().front();
  mlir::Operation* lastConstantOp = nullptr;
  mlir::Operation* noneOp = nullptr;

  for (auto& op : block.getOperations()) {
    // Track onnx.None operation
    if (op.getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      noneOp = &op;
    }
    // Track arith.constant operations
    else if (mlir::isa<mlir::arith::ConstantOp>(op)) {
      lastConstantOp = &op;
    } else if (!mlir::isa<mlir::func::ReturnOp>(op)) {
      // Stop searching when we hit the first non-constant, non-return, non-None
      // operation
      break;
    }
  }

  // Set insertion point: after last constant, or after None, or at start
  if (lastConstantOp) {
    builder.setInsertionPointAfter(lastConstantOp);
  } else if (noneOp) {
    // If no constants exist but None exists, insert after None
    builder.setInsertionPointAfter(noneOp);
  } else {
    builder.setInsertionPointToStart(&block);
  }

  mlir::Location loc = builder.getUnknownLoc();
  // Convert element_type (ONNX TensorProto::DataType) to MLIR tensor type
  auto tensorType = node_arg->getType(builder);
  auto shapedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensorType);
  if (!shapedTensorType) {
    // scalar
    if (auto scalar_tensor_type =
            mlir::dyn_cast<mlir::UnrankedTensorType>(tensorType)) {
      shapedTensorType =
          mlir::RankedTensorType::get({}, scalar_tensor_type.getElementType());
    } else {
      LOG(FATAL) << "Expected ShapedType for constant tensor: " << name;
    }
  }
  // Create dense element attribute from tensor data
  mlir::DenseElementsAttr denseAttr;
  const void* data = node_arg->getData();
  size_t dataSize = node_arg->getDataSize();

  // Create the dense elements attribute based on element type
  // TODO : i4, u4 ?
  auto rawData = llvm::ArrayRef<char>(static_cast<const char*>(data), dataSize);
  denseAttr =
      mlir::DenseElementsAttr::getFromRawBuffer(shapedTensorType, rawData);
  // Create arith.constant operation with the dense attribute
  auto constantOp = builder.create<mlir::arith::ConstantOp>(loc, denseAttr);
  constantOp->setAttr(attr_names::NODE_OUTPUTS,
                      builder.getArrayAttr({builder.getStringAttr(name)}));
  // update value in MLIRTensor object.
  node_arg->setValue(constantOp.getResult());

  MY_LOG(1) << " constant op \"" << name
            << "\":" << node_arg->getValue().getAsOpaquePointer() << " = "
            << " result=" << constantOp.getResult().getAsOpaquePointer()
            << " op=" << (void*)constantOp.getOperation();
  MY_LOG(1) << "node arg=" << MLIRNodeArgIndex(node_arg_ptr).to_string();
  MY_LOG(1) << "Successfully created constant tensor '" << name
            << "' as arith.constant operation";
}

const std::unordered_map<std::string, const void*>&
MLIRGraph::get_all_initialized_tensors() const {

  MY_LOG(1) << "Retrieved " << initialized_tensors_cache_.size()
            << " initialized tensors";
  return initialized_tensors_cache_;
}

void MLIRGraph::save(const std::string& filename,
                     const std::string& dat_filename,
                     size_t external_data_threshold) const {
  (void)dat_filename;            // Currently unused in MLIR implementation
  (void)external_data_threshold; // Currently unused in MLIR implementation

  MY_LOG(1) << "Saving MLIR graph to file: " << filename;

  // Get the parent module containing this function
  mlir::ModuleOp module = func_->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    LOG(ERROR) << "Cannot save graph: function is not contained in a module";
    return;
  }

  // Temporarily backup and remove internal morphizen attributes before saving
  struct AttributeBackup {
    mlir::Operation* op;
    mlir::Attribute inputsAttr;
    mlir::Attribute outputsAttr;
  };

  std::vector<AttributeBackup> backups;

  module.walk([&](mlir::Operation* op) {
    // Skip module and function operations themselves
    if (mlir::isa<mlir::ModuleOp>(op) || mlir::isa<mlir::func::FuncOp>(op)) {
      return;
    }

    // Backup and temporarily remove morphizen internal attributes
    AttributeBackup backup;
    backup.op = op;
    backup.inputsAttr = op->getAttr(attr_names::MORPHIZEN_NODE_INPUTS);
    backup.outputsAttr = op->getAttr(attr_names::MORPHIZEN_NODE_OUTPUTS);

    if (backup.inputsAttr || backup.outputsAttr) {
      backups.push_back(backup);

      // Temporarily remove the attributes
      if (backup.inputsAttr) {
        op->removeAttr(attr_names::MORPHIZEN_NODE_INPUTS);
      }
      if (backup.outputsAttr) {
        op->removeAttr(attr_names::MORPHIZEN_NODE_OUTPUTS);
      }
    }
  });

  // Open the output file
  std::error_code error_code;
  llvm::raw_fd_ostream output(filename, error_code);
  if (error_code) {
    LOG(ERROR) << "Failed to open file for writing: " << filename << " - "
               << error_code.message();
    return;
  }

  // Print the MLIR module to the file
  mlir::OpPrintingFlags flags;
  if (ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_GENERIC)) {
    flags.printGenericOpForm();
  }
  if (ENV_PARAM(MORPHIZEN_MLIR_SAVE_WITH_DEBUG_INFO)) {
    flags.enableDebugInfo();
    flags.printValueUsers();
  }
  module.print(output, flags);
  output.close();

  // Restore the backed up morphizen attributes
  for (const auto& backup : backups) {
    if (backup.inputsAttr) {
      backup.op->setAttr(attr_names::MORPHIZEN_NODE_INPUTS, backup.inputsAttr);
    }
    if (backup.outputsAttr) {
      backup.op->setAttr(attr_names::MORPHIZEN_NODE_OUTPUTS,
                         backup.outputsAttr);
    }
  }

  MY_LOG(1) << "Successfully saved MLIR graph to: " << filename << " (restored "
            << backups.size() << " operations with morphizen attributes)";
}
const MLIRNodeArg*
MLIRGraph::get_node_arg(MLIRNodeArgIndex node_arg_index) const {
  if (!node_arg_index.is_valid())
    return nullptr;
  return all_node_args_.size() > node_arg_index.get_index()
             ? all_node_args_.at(node_arg_index.get_index()).get()
             : nullptr;
}

std::vector<const mlir::Operation*>
MLIRGraph::get_consumer_nodes(const std::string& node_arg_name) const {
  std::vector<const mlir::Operation*> consumers;
  auto node_arg = get_node_arg(get_node_arg_index(node_arg_name));
  for (mlir::Operation* userOp : node_arg->getValue().getUsers()) {
    // Skip func.return operations to avoid treating function outputs as regular
    // consumers. Bug scenario: When a node's output (e.g.,
    // "linear_img_add_DequantizeLinear") is directly returned by the function,
    // including func.return as a consumer causes incorrect dependency analysis
    // in graph partitioning, leading to wrong subgraph boundaries or the
    // Partitioner incorrectly treating graph outputs as having real successors.
    if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
      continue;
    }
    consumers.push_back(userOp);
  }
  MY_LOG(1) << "Found " << consumers.size()
            << " consumer nodes for: " << node_arg_name;
  return consumers;
}

void MLIRGraph::reverse_dfs_from_preemp(
    gsl::span<const mlir::Operation* const> from,
    const std::function<bool(const mlir::Operation*)>& enter,
    const std::function<bool(const mlir::Operation*)>& leave,
    const std::function<bool(const mlir::Operation*, const mlir::Operation*)>&
        comp,
    const std::function<bool(const mlir::Operation* /*from*/,
                             const mlir::Operation* /*to*/)>& stop) const {

  // Implement preemptive reverse DFS traversal for MLIR backend
  using WorkEntry =
      std::pair<const mlir::Operation*, bool>; // bool represents leave or not
  std::vector<WorkEntry> stack;
  stack.reserve(from.size());

  // Initialize stack with starting nodes
  for (const auto* node_ptr : from) {
    stack.emplace_back(node_ptr, false);
  }

  // Track visited nodes to avoid cycles
  std::unordered_set<const mlir::Operation*> visited;

  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    const mlir::Operation* node_ptr = last_entry.first;

    // Skip null operations
    if (!node_ptr) {
      continue;
    }

    if (last_entry.second) {
      // leave node
      if (leave) {
        bool stop_processing = leave(node_ptr);
        // For preemp version, respect return value
        if (stop_processing) {
          break; // Stop processing if leave returns true
        }
      }
      continue;
    }

    // Check if already visited
    if (visited.count(node_ptr) > 0) {
      continue;
    }

    visited.insert(node_ptr);

    // Enter node
    if (enter) {
      bool stop_processing = enter(node_ptr);
      // For preemp version, respect return value
      if (stop_processing) {
        break; // Stop processing if enter returns true
      }
    }

    // Add leave operation to stack if needed
    if (leave) {
      stack.emplace_back(node_ptr, true);
    }

    // Use MLIR native interfaces to get input operands and their defining
    // operations This is more efficient than creating MLIRNode wrapper
    std::vector<const mlir::Operation*> producer_nodes;
    producer_nodes.reserve(
        const_cast<mlir::Operation*>(node_ptr)->getNumOperands());

    // Iterate through all operands of the current operation
    for (mlir::Value operand :
         const_cast<mlir::Operation*>(node_ptr)->getOperands()) {
      // Get the defining operation for this operand
      mlir::Operation* producer_op = operand.getDefiningOp();

      if (!producer_op) {
        // This operand is a block argument (graph input), skip it
        continue;
      }

      // Skip constant operations (initializers) in traversal
      if (mlir::isa<mlir::arith::ConstantOp>(producer_op)) {
        continue;
      }
      if (producer_op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
        continue;
      }
      if (staging_nodes_.count(producer_op)) {
        continue;
      }
      // Check stop condition using MLIR operations directly
      if (stop && stop(node_ptr, producer_op)) {
        continue;
      }

      // Add to collection if not visited
      if (visited.count(producer_op) == 0) {
        producer_nodes.push_back(producer_op);
      }
    }

    // Sort producer nodes if comparison function is provided
    if (comp && !producer_nodes.empty()) {
      std::sort(producer_nodes.begin(), producer_nodes.end(), comp);
    }

    // Add producer nodes to stack in reverse order for correct traversal order
    for (auto i = producer_nodes.rbegin(); i != producer_nodes.rend(); ++i) {
      stack.emplace_back(*i, false);
    }
  }
}

int MLIRGraph::resolve(bool force) {
  MY_LOG(1) << "MLIRGraph::resolve called with force=" << force;
  // In MLIR, the IR is always in a consistent state
  // Unlike ONNX which has staging graphs, MLIR operations are immediately
  // consistent when created. However, we can still perform validation
  // and optimization passes if needed.

  // For MLIR, we typically always run optimization passes when resolving
  // since MLIR is designed to be optimized incrementally, for now

  // Perform MLIR-specific graph validation and consistency checks
  try {
    // Verify that the function is well-formed
    if (failed(mlir::verify(func_))) {
      if (ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH))
        save("resolve_fatal.onnx", "", 10000);
      LOG(FATAL) << "MLIRGraph verification failed";
      return -1; // Verification failure
    }

    // TODO : shape inference

    // Re-initialize internal structures to ensure consistency
    initialize();

    MY_LOG(1) << "MLIRGraph::resolve completed successfully";
    return 0; // Success
  } catch (const std::exception& e) {
    LOG(FATAL) << "MLIRGraph::resolve failed with exception: " << e.what();
    return -1; // Error
  }
}
const MLIRGraph* MLIRGraph::add_subgraph(std::unique_ptr<MLIRGraph> graph) {
  subgraphs_cache_.push_back(std::move(graph));
  return subgraphs_cache_.rbegin()->get();
}

void MLIRGraph::remove_node(mlir::Operation* op) {
  if (op) {
    // Note: The operation will only be removed from the parent function, not
    // from all users.
    //       If it still has users, use remove() to detach, otherwise erase() to
    //       fully delete.
    if (op->use_empty()) {
      op->erase();
    } else {
      op->remove();
    }
  }
  // Node replacement requires two steps: 1. add new node; 2. remove old node
  // In ONNX: deletion order doesn't matter
  // In MLIR: must delete from back to front (reverse dependency order)
  // Example: Conv->Relu dependency chain requires deleting Relu first, then
  // Conv
}

void MLIRGraph::remove_initialized_tensor(const std::string& name) {
  func_.walk([&](mlir::Operation* op) -> mlir::WalkResult {
    if (!mlir::isa<mlir::arith::ConstantOp>(op)) {
      return mlir::WalkResult::advance();
    }
    if (op && op->use_empty()) {
      auto node_attr = MLIRNodeAttributes(op);
      auto node_output_names =
          node_attr.get_attribute_as_strings(attr_names::NODE_OUTPUTS);
      for (auto output_name : node_output_names) {
        if (output_name == name) {
          op->erase();
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });
}

mlir::Operation*
MLIRGraph::fuse(const std::string& name, const std::string& /* op_type*/,
                const std::vector<const mlir::Operation*>& nodes,
                const std::vector<MLIRNodeArgIndex>& inputs,
                const std::vector<MLIRNodeArgIndex>& outputs,
                const std::vector<MLIRNodeArgIndex>& constant_initializers) {
  // Core implementation of fusion operation: fuse multiple nodes into a single
  // func.call operation
  // Step 1: Create a fused subgraph function (func.func) containing all nodes
  // to be fused
  auto [func, cloned_ops_cache] =
      create_func_func(inputs, outputs, nodes, constant_initializers);

  // Step 2: Create a function call operation (func.call) in the main graph to
  // replace original nodes
  auto res = create_func_call(name, inputs, outputs, func, cloned_ops_cache);
  remove_func_ops(cloned_ops_cache);
  return res;
}

std::pair<mlir::func::FuncOp, std::stack<mlir::Operation*>>
MLIRGraph::create_func_func(
    const std::vector<MLIRNodeArgIndex>& inputs,
    const std::vector<MLIRNodeArgIndex>& outputs,
    const std::vector<const mlir::Operation*>& nodes,
    const std::vector<MLIRNodeArgIndex>& constant_initializers) {
  auto context = func_->getContext();
  mlir::OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();

  auto input_types =
      llvm::to_vector(llvm::map_range(inputs, [&](const auto& input) {
        return input.get_node_arg().getValue().getType();
      }));
  auto output_types =
      llvm::to_vector(llvm::map_range(outputs, [&](const auto& output) {
        return output.get_node_arg().getValue().getType();
      }));

  auto func_type = mlir::FunctionType::get(context, input_types, output_types);

  auto parent_module = func_->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTable symbolTable(parent_module);

  // Generate unique function name using MLIR SymbolTable
  unsigned int suffix = 0;
  auto func_name = mlir::SymbolTable::generateSymbolName<15>(
                       "sub_graph",
                       [&](llvm::StringRef name) -> bool {
                         return symbolTable.lookup(name) != nullptr;
                       },
                       suffix)
                       .str();
  auto temp_func = mlir::func::FuncOp::create(loc, func_name, func_type);

  { // Add the subgraph function to the same module as the main function
    mlir::OpBuilder module_builder(parent_module.getContext());
    module_builder.setInsertionPoint(parent_module.getBody(),
                                     std::prev(parent_module.getBody()->end()));
    module_builder.insert(temp_func);
  }

  auto& block = temp_func.getBody().emplaceBlock();

  for (const auto& input_type : input_types) {
    block.addArgument(input_type, loc);
  }

  builder.setInsertionPointToStart(&block);

  mlir::IRMapping value_mapping;

  // Map inputs to block arguments
  auto block_args = block.getArguments();
  for (const auto& [input, block_arg] : llvm::zip(inputs, block_args)) {
    value_mapping.map(input.get_node_arg().getValue(), block_arg);
  }
  std::stack<mlir::Operation*> cloned_ops_cache;
  // Clone constant initializers
  for (const auto& const_node_arg : constant_initializers) {
    if (auto producer_node = const_node_arg.get_producer_node()) {
      builder.setInsertionPointToEnd(&block);
      if (const_node_arg.is_constant()) {
        cloned_ops_cache.push(producer_node);
        builder.clone(*producer_node, value_mapping);
      }
    }
  }
  std::vector<mlir::Operation*> sorted_ops;
  std::unordered_set<const mlir::Operation*> node_set(nodes.begin(),
                                                      nodes.end());

  // Walk through the function in order and collect operations that need to be
  // fused
  func_.walk([&](mlir::Operation* walk_op) {
    if (node_set.count(walk_op)) {
      sorted_ops.push_back(walk_op);
    }
  });

  // Clone operations
  for (auto& op : sorted_ops) {
    // Handle missing constant dependencies
    for (auto operand : op->getOperands()) {
      if (!value_mapping.contains(operand)) {
        if (auto* defining_op = operand.getDefiningOp()) {
          if (mlir::isa<mlir::arith::ConstantOp>(operand.getDefiningOp())) {
            builder.setInsertionPointToEnd(&block);
            cloned_ops_cache.push(defining_op);
            builder.clone(*defining_op, value_mapping);
          }
        }
      }
    }
    builder.setInsertionPointToEnd(&block);
    cloned_ops_cache.push(op);
    builder.clone(*op, value_mapping);
  }

  // Create return operation with mapped outputs
  auto output_values =
      llvm::to_vector(llvm::map_range(outputs, [&](const auto& output) {
        return value_mapping.lookup(output.get_node_arg().getValue());
      }));
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output_values);

  return std::make_pair(temp_func, cloned_ops_cache);
}

mlir::Operation* MLIRGraph::create_func_call(
    const std::string& name, const std::vector<MLIRNodeArgIndex>& inputs,
    const std::vector<MLIRNodeArgIndex>& outputs, mlir::func::FuncOp fused_func,
    const std::stack<mlir::Operation*>& /* cloned_ops_cache */) {
  mlir::IRRewriter rewriter(func_->getContext());

  // Set insertion point based on inputs
  mlir::Operation* latestInputOp = nullptr;
  bool hasConstantInput = false;

  for (const auto& input : inputs) {
    auto value = input.get_node_arg().getValue();
    if (auto* definingOp = value.getDefiningOp()) {
      if (mlir::isa<mlir::arith::ConstantOp>(definingOp)) {
        hasConstantInput = true;
      }

      if (!latestInputOp || latestInputOp->isBeforeInBlock(definingOp)) {
        latestInputOp = definingOp;
      }
    }
  }

  // If any input is from arith.constant, find the last arith.constant in the
  // block
  if (hasConstantInput) {
    auto& entryBlock = func_.getBody().front();
    mlir::Operation* lastConstantOp = nullptr;

    for (auto& op : entryBlock.getOperations()) {
      if (mlir::isa<mlir::arith::ConstantOp>(op)) {
        lastConstantOp = &op;
      } else if (!mlir::isa<mlir::func::ReturnOp>(op)) {
        break;
      }
    }

    if (lastConstantOp &&
        (!latestInputOp || latestInputOp->isBeforeInBlock(lastConstantOp))) {
      latestInputOp = lastConstantOp;
    }
  }

  if (latestInputOp) {
    rewriter.setInsertionPointAfter(latestInputOp);
  } else {
    rewriter.setInsertionPoint(terminator_);
  }

  auto mlir_input_args =
      llvm::to_vector(llvm::map_range(inputs, [](const auto& input) {
        return input.get_node_arg().getValue();
      }));
  auto result_types =
      llvm::to_vector(llvm::map_range(outputs, [](const auto& output) {
        return output.get_node_arg().getValue().getType();
      }));

  mlir::OperationState state(rewriter.getUnknownLoc(), "func.call");
  state.addOperands(mlir_input_args);
  state.addTypes(result_types);

  // Add callee attribute for func.call operation - use actual function symbol
  // name
  auto calleeAttr = fused_func.getSymNameAttr();
  state.addAttribute("callee", mlir::FlatSymbolRefAttr::get(calleeAttr));

  auto* fuse_node = rewriter.create(state);

  auto output_names = llvm::to_vector(
      llvm::map_range(outputs, [&](const auto& output) -> mlir::Attribute {
        return rewriter.getStringAttr(output.get_name());
      }));
  fuse_node->setAttr(attr_names::NODE_OUTPUTS,
                     rewriter.getArrayAttr(output_names));
  fuse_node->setAttr(attr_names::ONNX_NODE_NAME, rewriter.getStringAttr(name));
  fuse_node->setAttr(
      attr_names::MORPHIZEN_NODE_INPUTS,
      rewriter.getArrayAttr(llvm::to_vector(
          llvm::map_range(inputs, [&](const auto& input) -> mlir::Attribute {
            return rewriter.getI64IntegerAttr(reinterpret_cast<int64_t>(
                input.to_morphizen_core_node_arg_ptr()));
          }))));
  fuse_node->setAttr(
      attr_names::MORPHIZEN_NODE_OUTPUTS,
      rewriter.getArrayAttr(llvm::to_vector(
          llvm::map_range(outputs, [&](const auto& output) -> mlir::Attribute {
            return rewriter.getI64IntegerAttr(reinterpret_cast<int64_t>(
                output.to_morphizen_core_node_arg_ptr()));
          }))));

  for (const auto& [output, result] :
       llvm::zip(outputs, fuse_node->getResults())) {
    rewriter.replaceAllUsesWith(output.get_node_arg().getValue(), result);
    output.get_node_arg().setValue(result);
  }

  // After creating func.call, move all operations that use its outputs after it
  llvm::SmallVector<mlir::Operation*> opsToMove;
  llvm::SmallSet<mlir::Operation*, 16> visited;

  std::function<void(mlir::Operation*)> collectDependentOps =
      [&](mlir::Operation* userOp) {
        if (visited.count(userOp) || userOp == fuse_node ||
            mlir::isa<mlir::func::ReturnOp>(userOp)) {
          return;
        }
        visited.insert(userOp);

        if (!userOp->isBeforeInBlock(fuse_node) && userOp != fuse_node) {
          return;
        }

        opsToMove.push_back(userOp);

        for (auto result : userOp->getResults()) {
          for (auto& use : result.getUses()) {
            collectDependentOps(use.getOwner());
          }
        }
      };

  for (auto result : fuse_node->getResults()) {
    for (auto& use : result.getUses()) {
      collectDependentOps(use.getOwner());
    }
  }

  std::sort(opsToMove.begin(), opsToMove.end(),
            [](mlir::Operation* a, mlir::Operation* b) {
              return a->isBeforeInBlock(b);
            });

  mlir::Operation* lastMovedOp = fuse_node;
  for (auto* opToMove : opsToMove) {
    opToMove->moveAfter(lastMovedOp);
    lastMovedOp = opToMove;
  }

  return fuse_node;
}
void MLIRGraph::remove_func_ops(
    std::stack<mlir::Operation*>& cloned_ops_cache) {
  // Delete cloned operations in LIFO order using stack
  // This ensures proper deletion order since later operations may depend on
  // earlier ones
  while (!cloned_ops_cache.empty()) {
    mlir::Operation* op = cloned_ops_cache.top();
    cloned_ops_cache.pop();
    if (op && op->use_empty()) {
      op->erase();
    }
  }
}
} // namespace mlir_impl
} // namespace morphizen
