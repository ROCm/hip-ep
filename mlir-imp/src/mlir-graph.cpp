/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir-graph.hpp"
#include "./mlir-constants.hpp"
#include "./mlir-context-manager.hpp"
#include "./mlir-graph-store.hpp"
#include "./mlir-node-arg.hpp"

#include "./mlir-node.hpp"
#include "mlir-model.hpp"
#include "mlir-named-attribute.hpp"
#include "mlir-node-attributes.hpp"
#include "mlir/Bytecode/BytecodeWriter.h"  // for writeBytecodeToFile
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // for EmptyOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"           // for IRMapping
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h" // for getUsedValuesDefinedAbove
#include "morphizen-foundation/env_config.hpp"
#include "llvm/ADT/STLExtras.h"          // for map_range, to_vector
#include "llvm/ADT/SetVector.h"          // for SetVector
#include "llvm/ADT/SmallPtrSet.h"        // for SmallPtrSet
#include "llvm/ADT/SmallSet.h"           // for SmallSet
#include "llvm/ADT/SmallVector.h"        // for SmallVector
#include "llvm/Support/raw_ostream.h"    // for raw_fd_ostream
#include <algorithm>                     // for std::sort
#include <glog/logging.h>
#include <iomanip>                       // for std::setprecision
#include <system_error>                  // for std::error_code
#include <unordered_map>                 // for std::unordered_map
#include <unordered_set>                 // for std::unordered_set
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH, "0")
DEF_ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH) >= n)
namespace morphizen {
namespace mlir_impl {

// Resolve the MLIRContext for a (possibly orphan) Block. Orphan blocks
// have no parent op; fall back to the shared MLIRContextManager singleton.
static mlir::MLIRContext* context_for(mlir::Block& block) {
  auto* parent_op = block.getParentOp();
  return parent_op != nullptr ? parent_op->getContext()
                              : &MLIRContextManager::getInstance().getContext();
}

static mlir::Operation* get_or_create_terminator(mlir::Block& entryBlock,
                                                 bool is_subgraph) {
  if (entryBlock.mightHaveTerminator()) {
    return entryBlock.getTerminator();
  }
  mlir::OpBuilder builder(context_for(entryBlock));
  builder.setInsertionPointToEnd(&entryBlock);

  llvm::StringRef op_name =
      is_subgraph ? onnx_mlir::ONNX_YIELD : onnx_mlir::ONNX_RETURN;
  mlir::OperationState state(builder.getUnknownLoc(), op_name);
  // onnx.Return / onnx.Yield have Terminator trait, no results needed
  return builder.create(state);
}

static mlir::Operation* get_or_create_none(mlir::Block& entryBlock) {
  // First, try to find an existing NoneOp in the block (was func.walk).
  // Block::walk does not include the parent op, so we walk only block contents.
  mlir::Operation* existing_none = nullptr;
  entryBlock.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      existing_none = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (existing_none) {
    return existing_none;
  }

  mlir::OpBuilder builder(context_for(entryBlock));
  builder.setInsertionPointToStart(&entryBlock);
  mlir::OperationState state(builder.getUnknownLoc(), onnx_mlir::ONNX_NONE);
  state.addTypes(builder.getNoneType());
  // onnx.NoValue requires a 'value' UnitAttr
  state.addAttribute("value", builder.getUnitAttr());
  return builder.create(state);
}
MLIRGraph::MLIRGraph(MLIRModel& model, mlir::func::FuncOp func,
                     uint32_t proposed_graph_id)
    : model_(model), entry_block_(&func.getBody().front()),
      graph_id_(proposed_graph_id > 0
                    ? GraphStore::allocate_graph_id(this, proposed_graph_id)
                    : GraphStore::allocate_graph_id(this)),
      terminator_{
          get_or_create_terminator(*entry_block_, /*is_subgraph=*/false)},
      none_{get_or_create_none(*entry_block_)}, value_map_() {
  initialize();
}

MLIRGraph::MLIRGraph(MLIRModel& model, mlir::Block& entry_block,
                     uint32_t proposed_graph_id)
    : model_(model), entry_block_(&entry_block),
      graph_id_(proposed_graph_id > 0
                    ? GraphStore::allocate_graph_id(this, proposed_graph_id)
                    : GraphStore::allocate_graph_id(this)),
      terminator_{
          get_or_create_terminator(*entry_block_, /*is_subgraph=*/true)},
      none_{get_or_create_none(*entry_block_)}, value_map_() {
  // No initialize() here -- body ops / initializers / outputs do not exist
  // yet. The caller is responsible for:
  //   1. setting parent_graph_ via set_parent_graph (so is_subgraph()
  //      returns true and the get_node_arg_index fallback can walk up),
  //   2. filling the body via emit_body / convert_graph, and
  //   3. for the orphan-block path (new_subgraph), having add_node
  //      transplant entry_block_ into a parent op region before calling
  //      resolve(true).
}

MLIRGraph* MLIRGraph::parent_graph() const { return parent_graph_; }

void MLIRGraph::set_parent_graph(MLIRGraph* p) { parent_graph_ = p; }

mlir::func::FuncOp MLIRGraph::func() const {
  return llvm::dyn_cast<mlir::func::FuncOp>(entry_block_->getParentOp());
}

bool MLIRGraph::is_subgraph() const { return parent_graph_ != nullptr; }

mlir::Block* MLIRGraph::take_orphan_block() {
  CHECK(entry_block_->getParent() == nullptr)
      << "take_orphan_block: block is not orphan (already transplanted?)";
  return entry_block_;
}

MLIRGraph& MLIRGraph::new_subgraph() {
  auto* orphan_block = new mlir::Block();
  // Block& ctor is private; std::make_unique cannot reach it.
  auto sub = std::unique_ptr<MLIRGraph>(new MLIRGraph(model_, *orphan_block));
  sub->set_parent_graph(this);
  // unique_ptr keeps MLIRGraph's address stable across vector growth, so
  // the raw pointer that attr_proto_new_graph encodes stays valid.
  auto* raw = sub.get();
  add_subgraph(std::move(sub));
  return *raw;
}

MLIRGraph::~MLIRGraph() {
  // Free the orphan block if graph_new_subgraph created one and add_node
  // never transplanted it (error / exception path). Transplanted blocks
  // and top-level FuncOp blocks both have a non-null parent region.
  if (entry_block_ != nullptr && entry_block_->getParent() == nullptr) {
    delete entry_block_;
  }
  GraphStore::release_graph_id(this, graph_id_);
}

void MLIRGraph::register_captured_alias(const std::string& name,
                                        mlir::Value outer_value) {
  CHECK(is_subgraph()) << "register_captured_alias is subgraph-only";
  CHECK(outer_value) << "register_captured_alias: null Value for '" << name
                     << "'";
  CHECK(node_args_map_.find(name) == node_args_map_.end())
      << "register_captured_alias: name '" << name
      << "' already registered in subgraph node_args_map_";

  // Shadow NodeArg aliasing the outer Value; in-body op emission picks
  // it up by name and uses outer_value directly as the operand.
  auto shadow = std::make_unique<MLIRNodeArg>(name, outer_value);
  all_node_args_.push_back(std::move(shadow));
  auto idx =
      MLIRNodeArgIndex::node_output((int32_t)all_node_args_.size() - 1,
                                    GraphId::create_main_graph(graph_id_));
  node_args_map_.emplace(name, idx);

  // Also seed value_map_ for extract_value_name's Value -> name reverse
  // lookup; in-body ops that consume the outer value will resolve back
  // to the captured name through this map.
  value_map_.insert(name, outer_value);
}

std::string MLIRGraph::extract_value_name(const mlir::Value value) {
  // As defined in the language reference,
  // each Value is either a BlockArgument or the result of exactly one
  // Operation (an Operation can have multiple results, each of them is a
  // separate Value).

  // Try to get the name from onnx.name attribute if it's a block argument
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // Subgraph BlockArguments have no FuncOp arg-attr owner; names live in
    // value_map_ (populated by register_captured_alias / set_inputs).
    if (is_subgraph()) {
      if (auto name = value_map_.lookup(value)) {
        return *name;
      }
      return "input_" + std::to_string(blockArg.getArgNumber());
    }
    // Top-level: name lives on the FuncOp via setArgAttr (see ONNX_NAME).
    if (auto nameAttr =
            func().getArgAttr(blockArg.getArgNumber(), attr_names::ONNX_NAME)) {
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
  populate_node_arg_indexes();
}

void MLIRGraph::initialize_node_args_map() {
  MY_LOG(1) << "Initializing node args map and constant initializers";
  // Walk all operations and extract their results. Block::walk does not
  // include the parent op, so no need to skip the FuncOp self-visit anymore.
  entry_block_->walk([&](mlir::Operation* op) {
    // Skip return operations (terminators) - both onnx.Return and func.return
    if (onnx_mlir::isReturnOp(op)) {
      return;
    }
    // Skip onnx.Constant, onnx.Constant mapping onnx
    // GraphProto.initializer
    if (op->getName().getStringRef() == "onnx.Constant") {
      // already processed in initialize_constant_initializers
      return;
    }
    if (op->getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      return;
    }

    // Process all results of this operation
    for (auto result : op->getResults()) {
      // Skip optional/unused outputs that were assigned NoneType
      // (e.g. unused outputs of SkipSimplifiedLayerNormalization)
      if (mlir::isa<mlir::NoneType>(result.getType()))
        continue;
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
  // Filter operations by name to find all onnx.Constant
  for (auto& op : entry_block_->getOperations()) {
    if (op.getName().getStringRef() == "onnx.Constant") {
      // attr :  NODE_OUTPUTS
      // constant op has a single result
      auto value = op.getResult(0);
      auto value_name = extract_value_name(value);
      all_node_args_.push_back(
          std::make_unique<MLIRNodeArg>(value_name, value));
      node_args_map_.emplace(
          value_name,
          MLIRNodeArgIndex::node_output((int32_t)all_node_args_.size() - 1,
                                        GraphId::create_main_graph(graph_id_)));

      constant_initializers_.push_back(node_args_map_[value_name]);
      initialized_tensors_cache_[value_name] =
          &node_args_map_[value_name].get_const_data_as_tensor();
    }
  }
}

void MLIRGraph::initialize_graph_inputs() {
  MY_LOG(1) << "Initializing graph inputs";

  // Extract entry block arguments. For top-level this is equivalent to
  // func().getArguments(); for subgraph regions this picks up the
  // block args created lazily by the subgraph branch of set_inputs as
  // ort-bridge drives convert_graph_inputs -> node_arg_new + set_inputs.
  auto arguments = entry_block_->getArguments();
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

  graph_outputs_.clear();

  CHECK(terminator_) << "Terminator must not be null when initializing outputs";
  // Get the return operation to find graph outputs
  // Handle both onnx.Return and func.return for compatibility
  if (terminator_ && onnx_mlir::isReturnOp(terminator_)) {
    auto operands = terminator_->getOperands();

    for (auto operand : operands) {
      for (size_t i = 0; i < all_node_args_.size(); ++i) {
        const auto& node_arg = all_node_args_[i];
        if (node_arg && node_arg->getValue() == operand) {
          graph_outputs_.push_back(node_args_map_[node_arg->getName()]);
          if (!model_output_names_frozen_) {
            model_output_names_.insert(node_arg->getName());
          }
          MY_LOG(2) << "Added graph output: " << node_arg->getName()
                    << " at index " << i;
          break;
        }
      }
    }
  }

  model_output_names_frozen_ = true;
  MY_LOG(1) << "Initialized " << graph_outputs_.size() << " graph outputs"
            << " (model_output_names: " << model_output_names_.size() << ")";
}

void MLIRGraph::populate_node_arg_indexes() {
  MY_LOG(1) << "Populating per-op NodeArg index attributes";

  auto value_to_index = [&](mlir::Value value) {
    return get_node_arg_index(extract_value_name(value));
  };

  // Pre-order so the nested-region skip below can stop descent.
  entry_block_->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    // Skip return operations, constant operations, and our custom onnx.None
    // operations
    if (!onnx_mlir::isReturnOp(op) &&
        op->getName().getStringRef() != "onnx.Constant" &&
        op->getName().getStringRef() != onnx_mlir::ONNX_NONE) {
      auto node = MLIRNode(op);

      llvm::SmallVector<MLIRNodeArgIndex> explicit_inputs;
      explicit_inputs.reserve(op->getNumOperands());
      for (mlir::Value operand : op->getOperands()) {
        explicit_inputs.push_back(value_to_index(operand));
      }
      node.setInputNodeArgs(explicit_inputs);

      // An outer Value may appear as both an explicit operand AND a body
      // capture (e.g. onnx-mlir's loopfix-promoted Loop where outer-scope
      // refs aren't fully rewritten to block args); dedup against the
      // explicit set so it records once.
      if (op->getNumRegions() > 0) {
        llvm::SetVector<mlir::Value> captures;
        mlir::getUsedValuesDefinedAbove(op->getRegions(), captures);
        llvm::SmallPtrSet<mlir::Value, 8> explicit_operands(
            op->getOperands().begin(), op->getOperands().end());

        llvm::SmallVector<MLIRNodeArgIndex> implicit_inputs;
        implicit_inputs.reserve(captures.size());
        for (mlir::Value v : captures) {
          if (!explicit_operands.contains(v)) {
            implicit_inputs.push_back(value_to_index(v));
          }
        }
        node.setImplicitInputNodeArgs(implicit_inputs);
      }

      llvm::SmallVector<MLIRNodeArgIndex> outputs;
      outputs.reserve(op->getNumResults());
      for (auto result : op->getResults()) {
        outputs.push_back(value_to_index(result));
      }
      node.setOutputNodeArgs(outputs);
    }
    // Region contents belong to their own subgraph MLIRGraph; skip descent.
    // Use getNumRegions() rather than an op-name allow-list so future
    // region-bearing ONNX ops cannot slip through.
    if (op->getNumRegions() > 0) {
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  MY_LOG(1) << "Completed populating NodeArg index attributes";
}

const std::string& MLIRGraph::get_name() const {
  if (is_subgraph()) {
    // Regions have no name; parent op encodes the slot ("then_branch"/...).
    // Sentinel keeps log call sites that touch graph_.name() safe.
    static const std::string subgraph_name = "subgraph_region";
    return subgraph_name;
  }
  auto f = func();
  if (auto attr = f->getAttr(attr_names::ONNX_GRAPH_NAME)) {
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
  if (is_subgraph()) {
    // No-op: subgraph regions have no FuncOp symbol to attach onnx.graph.name
    // to. ort-bridge calls set_name unconditionally inside convert_graph;
    // silently accept and move on.
    MY_LOG(1) << "Skipping set_name on subgraph region: " << name;
    return;
  }
  auto f = func();
  auto* context = f->getContext();
  mlir::OpBuilder builder(context);
  f->setAttr(attr_names::ONNX_GRAPH_NAME, builder.getStringAttr(name));
  MY_LOG(1) << "Set graph name to: " << name;
}
const MLIRModel& MLIRGraph::get_model() const { return model_; }

std::string MLIRGraph::get_symbol_name() const {
  CHECK(!is_subgraph()) << "get_symbol_name is top-level only "
                           "(subgraph regions have no symbol)";
  return func().getSymName().str();
}

std::vector<mlir::Operation*> MLIRGraph::nodes_unsafe() const {
  std::vector<mlir::Operation*> nodes;

  // Top-level only: matches ORT::Graph::Nodes() "this view only"
  // contract. Body ops of region-bearing nodes (Loop / If / Scan) live
  // in the subgraph's own MLIRGraph instance and are walked by it.
  // Block::walk() is recursive in MLIR; using it here would surface
  // body ops as top-level fuse candidates and break the EP invariant
  // supported_nodes.size() == ep_supported_outputs.size().
  for (mlir::Operation& op : *entry_block_) {
    if (!onnx_mlir::isReturnOp(&op) &&
        op.getName().getStringRef() != "onnx.Constant" &&
        op.getName().getStringRef() != onnx_mlir::ONNX_NONE) {
      nodes.push_back(&op);
    }
  }

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
  llvm::SmallVector<MLIRNodeArgIndex> valid_outputs;
  llvm::SmallVector<mlir::Value> mlir_outputs;
  valid_outputs.reserve(outputs.size());
  mlir_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    auto* nodeArg = get_node_arg(output);
    if (!nodeArg || !nodeArg->getValue() ||
        mlir::isa<mlir::NoneType>(nodeArg->getValue().getType())) {
      MY_LOG(1) << "Skipping NoneType output: " << output.get_name();
      continue;
    }
    // model_output_names_ is populated only by initialize_graph_outputs()
    // from the model's existing onnx.Return. Use it to filter spurious
    // intermediate outputs that the ORT bridge may include beyond the
    // model's actual graph outputs. When empty (graph built from scratch),
    // no filtering is applied.
    if (!model_output_names_.empty() &&
        !model_output_names_.count(output.get_name())) {
      MY_LOG(1) << "Skipping non-model output: " << output.get_name();
      continue;
    }
    valid_outputs.push_back(output);
    mlir_outputs.push_back(nodeArg->getValue());
  }
  graph_outputs_ = valid_outputs;

  if (is_subgraph()) {
    // Subgraph terminator is onnx.Yield; rewrite its operands. Parent op's
    // result types are already fixed when add_node consumed the marker
    // AttributeProto -- do NOT mutate them via setType (no FuncOp to
    // mutate anyway).
    CHECK(onnx_mlir::isReturnOrYieldOp(terminator_))
        << "subgraph terminator must be onnx.Yield (or onnx.Return for "
           "top-level), got: "
        << (terminator_ ? terminator_->getName().getStringRef().str()
                        : std::string("<null>"));
    terminator_->setOperands(
        llvm::ArrayRef<mlir::Value>(mlir_outputs.data(), mlir_outputs.size()));
    MY_LOG(1) << "Updated subgraph "
              << terminator_->getName().getStringRef().str()
              << " terminator with " << mlir_outputs.size() << " outputs";
    return;
  }

  // Handle both onnx.Return and func.return terminators
  if (onnx_mlir::isReturnOp(terminator_)) {
    terminator_->setOperands(
        llvm::ArrayRef<mlir::Value>(mlir_outputs.data(), mlir_outputs.size()));

    MY_LOG(1) << "Updated existing "
              << terminator_->getName().getStringRef().str()
              << " terminator with " << mlir_outputs.size() << " outputs";
  } else {
    LOG(FATAL) << "Unexpected terminator type: "
               << terminator_->getName().getStringRef().str()
               << ", expected onnx.Return or func.return";
  }

  // Update the FuncOp type to reflect the new output types.
  auto f = func();
  auto* context = f->getContext();
  mlir::OpBuilder builder(context);

  auto argTypes = f.getArgumentTypes();
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(mlir_outputs.size());
  for (const auto& output : mlir_outputs) {
    resultTypes.push_back(output.getType());
  }

  f.setType(builder.getFunctionType(argTypes, resultTypes));
  for (size_t i = 0; i < valid_outputs.size(); ++i) {
    auto name = valid_outputs[i].get_name();
    f.setResultAttr((unsigned int)i, mlir_impl::attr_names::ONNX_NAME,
                    mlir::StringAttr::get(f.getContext(), name));
  }

  MY_LOG(1) << "Updated function type with " << resultTypes.size()
            << " output types";
}

void MLIRGraph::set_inputs(const llvm::SmallVector<MLIRNodeArgIndex>& inputs) {
  graph_inputs_ = inputs;

  if (is_subgraph()) {
    // Bind slot i to its block_arg, growing a new one if the orphan block
    // doesn't have one yet. Safe to grow at this point: the parent op's
    // signature (set later by add_node) does not track block_arg count.
    mlir::OpBuilder builder(context_for(*entry_block_));
    auto loc = builder.getUnknownLoc();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto* nodeArg = get_node_arg(inputs[i]);
      CHECK(nodeArg != nullptr)
          << "subgraph set_inputs: null NodeArg at slot " << i;
      mlir::Value bound_value;
      if (i < entry_block_->getNumArguments()) {
        bound_value = entry_block_->getArgument(static_cast<unsigned>(i));
      } else {
        bound_value = entry_block_->addArgument(nodeArg->getType(builder), loc);
      }
      nodeArg->setValue(bound_value);
      // value_map_ insert silently no-ops on duplicate name; safe even if
      // a prior call already bound this name (idempotent contract).
      value_map_.insert(nodeArg->getName(), bound_value);
    }
    return;
  }

  // for now, we only support set_inputs when the graph has no inputs.
  auto f = func();
  auto arguments = f.getArguments();
  if (!arguments.empty()) {
    LOG(FATAL) << "Graph already has inputs, cannot set new inputs.";
  }
  auto builder = mlir::OpBuilder(f.getContext());
  // Get the function's entry block and add arguments to it
  auto& entryBlock = *entry_block_;
  auto loc = builder.getUnknownLoc();
  for (auto& input : inputs) {
    // For inputs, we typically don't have shape information at this point
    // so we create unranked tensor types (empty shape)
    auto node_arg = get_node_arg(input);
    node_arg->setValue(entryBlock.addArgument(node_arg->getType(builder), loc));
  }

  arguments = f.getArguments();
  auto retTys = f.getResultTypes();
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (size_t i = 0; i < inputs.size(); ++i) {
    argTypes.push_back(arguments[i].getType());
  }
  f.setType(builder.getFunctionType(argTypes, retTys));
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto name = inputs[i].get_name();
    f.setArgAttr((unsigned int)i, mlir_impl::attr_names::ONNX_NAME,
                 mlir::StringAttr::get(f.getContext(), name));
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
        if (defining_op->getName().getStringRef() == "onnx.Constant") {
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
  // Local hit: top-level graphs and subgraphs both go through this path
  // first.
  auto it = node_args_map_.find(name);
  if (it != node_args_map_.end()) {
    return it->second;
  }
  // Mirror ORT::Graph::GetNodeArgIncludingParentGraphs: first ancestor that
  // knows the name wins. On hit we lazy-register a captured alias locally so
  // the local map answers next time and resolve(true) sees the binding.
  // const_cast: the lazy alias mutates node_args_map_ / value_map_ behind a
  // logically-const view.
  if (parent_graph_ != nullptr) {
    auto outer = parent_graph_->get_node_arg_index(name);
    if (outer.is_valid()) {
      auto outer_value = outer.get_node_arg().getValue();
      const_cast<MLIRGraph*>(this)->register_captured_alias(name, outer_value);
      return node_args_map_.at(name);
    }
  }
  return MLIRNodeArgIndex::invalid();
}

MLIRNodeArgIndex
MLIRGraph::node_arg_new(const std::string& name,
                        const llvm::SmallVector<int64_t>* shape,
                        int element_type) {
  // shape == nullptr signals an unranked tensor at the ORT boundary (ORT's
  // TensorTypeAndShapeInfo::HasShape() == false). Lowered to
  // mlir::UnrankedTensorType by MLIRNodeArg::getType().
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
    auto existing_shape = node_arg.getShape();
    if (node_arg.isConstantValue() && shape != nullptr &&
        existing_shape.has_value() && *shape == *existing_shape &&
        element_type == node_arg.getElementType()) {
      return node_arg_index;
    }
    LOG(FATAL) << "symbol \"" << name << "\" already exists.";
    return MLIRNodeArgIndex::invalid();
  }
  auto nodeArg = std::make_unique<MLIRNodeArg>(name, shape, element_type);
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

  // Pick out the variant subgraph-ref markers; they encode the regions
  // to create and must not land on the real op's attribute dict.
  mlir::DictionaryAttr mlir_attrs_full = attributes.get_mlir_dictionary();
  std::vector<std::pair<std::string, MLIRGraph*>> sub_refs;
  for (const auto& na : mlir_attrs_full) {
    if (auto* sub =
            static_cast<const MLIRNamedAttribute&>(na).get_subgraph_ref()) {
      sub_refs.emplace_back(na.getName().str(), sub);
    }
  }
  // DictionaryAttr is alphabetically sorted by name, but onnx.If wants
  // region(0)=then_branch, region(1)=else_branch -- force that order.
  if (op_type == "If") {
    std::stable_partition(sub_refs.begin(), sub_refs.end(), [](const auto& p) {
      return p.first == "then_branch";
    });
  }
  const size_t total_regions = sub_refs.size();

  mlir::OpBuilder builder(context_for(*entry_block_));
  CHECK(terminator_ != nullptr);

  // Anchor the insertion point after every entry_block_-defined value this op
  // depends on: explicit operands, the none_ placeholder used for missing
  // optional inputs, and the last onnx.Constant. Operands from an enclosing
  // block or block args already dominate, so they don't constrain placement;
  // restricting the comparison to entry_block_ keeps isBeforeInBlock
  // well-defined (it is UB across blocks). Keeping the op after the last
  // constant holds the constants contiguous at the block top -- interleaving
  // other ops among them changes the constant-streaming backend's results.
  // Loop/If/Scan body captures are folded in after transplant (below).
  {
    mlir::Operation* anchor = nullptr;
    for (const auto& arg : input_args) {
      auto* input_node_arg = get_node_arg(arg);
      if (!input_node_arg)
        continue;
      mlir::Value value = input_node_arg->getValue();
      mlir::Operation* def = value ? value.getDefiningOp() : nullptr;
      if (!def || def->getBlock() != entry_block_)
        continue;
      if (!anchor || anchor->isBeforeInBlock(def))
        anchor = def;
    }
    if (none_ && none_->getBlock() == entry_block_ &&
        (!anchor || anchor->isBeforeInBlock(none_)))
      anchor = none_;
    mlir::Operation* lastConstantOp = nullptr;
    for (auto& blockOp : entry_block_->getOperations()) {
      if (blockOp.getName().getStringRef() == "onnx.Constant")
        lastConstantOp = &blockOp;
    }
    if (lastConstantOp && (!anchor || anchor->isBeforeInBlock(lastConstantOp)))
      anchor = lastConstantOp;
    if (anchor)
      builder.setInsertionPointAfter(anchor);
    else
      builder.setInsertionPointToStart(entry_block_);
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

  // Convert output types from mlir::Values (if they have types).
  // Outputs with empty names or no registered NodeArg are treated as
  // optional/unused (NoneType). Empty names follow the ONNX convention for
  // unconnected optional outputs. The NoneType results are then filtered
  // out of graph outputs by set_outputs().
  llvm::SmallVector<mlir::Type> result_types;
  for (unsigned idx = 0; idx < output_args.size(); ++idx) {
    if (!output_args[idx].is_valid() || output_args[idx].get_name().empty()) {
      result_types.push_back(builder.getNoneType());
      continue;
    }
    if (auto* output_node_arg = get_node_arg(output_args[idx])) {
      result_types.push_back(output_node_arg->getType(builder));
    } else {
      result_types.push_back(builder.getNoneType());
    }
  }

  // Create operation location
  auto loc = builder.getUnknownLoc();

  // Strip subgraph-ref markers from the attribute dictionary that lands on
  // the real op. When no markers are present this is the no-op same-pointer
  // assignment; only the variant-AttributeProto path pays the rebuild cost.
  mlir::DictionaryAttr mlir_attrs;
  if (sub_refs.empty()) {
    mlir_attrs = mlir_attrs_full;
  } else {
    llvm::SmallVector<mlir::NamedAttribute> filtered;
    filtered.reserve(mlir_attrs_full.size() - sub_refs.size());
    for (const auto& na : mlir_attrs_full) {
      if (static_cast<const MLIRNamedAttribute&>(na).get_subgraph_ref()) {
        continue;
      }
      filtered.push_back(na);
    }
    mlir_attrs = mlir::DictionaryAttr::get(builder.getContext(), filtered);
  }

  // Determine if this operation should use onnx.Custom representation.
  // Two cases require onnx.Custom:
  // 1. Non-standard domain ops (e.g., com.microsoft)
  // 2. Standard-domain ops not registered in onnx-mlir's ONNX dialect.
  //    The hip-compiler uses a specific onnx-mlir fork that may lack newer
  //    ONNX ops (e.g. opset 21+). These must be emitted as onnx.Custom so
  //    the compiler can parse them and match its own conversion patterns.
  static const std::unordered_set<std::string> unregistered_onnx_ops = {
      "SimplifiedLayerNormalization",
  };

  bool is_custom_domain =
      !domain.empty() && domain != "ai.onnx" && domain != "onnx";
  bool use_custom_op = is_custom_domain || unregistered_onnx_ops.count(op_type);

  mlir::Operation* op = nullptr;
  if (use_custom_op) {
    // For custom domain operations (e.g., com.microsoft), use onnx.Custom
    // This follows the onnx-mlir convention for custom operators
    mlir::OperationState state(loc, "onnx.Custom");
    state.addOperands(mlir_input_args);
    state.addTypes(result_types);
    for (size_t r = 0; r < total_regions; ++r) {
      state.addRegion();
    }

    // Add function_name and domain_name as properties (inside <{}>)
    llvm::SmallVector<mlir::NamedAttribute> props;
    props.push_back(
        builder.getNamedAttr("function_name", builder.getStringAttr(op_type)));

    // Copy existing attributes and add domain_name
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("domain_name", builder.getStringAttr(domain)));
    for (auto& attr : mlir_attrs) {
      attrs.push_back(attr);
    }

    // Set properties and attributes
    state.addAttribute("function_name", builder.getStringAttr(op_type));
    for (auto& attr : attrs) {
      state.addAttribute(attr.getName(), attr.getValue());
    }

    op = builder.create(state);
    MY_LOG(1) << "Created onnx.Custom operation for " << domain << ":"
              << op_type;
  } else {
    // For standard ONNX operations, use the normal naming convention
    std::string full_op_name = op_type;
    mlir::OperationState state(loc, "onnx." + full_op_name);
    state.addOperands(mlir_input_args);
    state.addTypes(result_types);
    for (size_t r = 0; r < total_regions; ++r) {
      state.addRegion();
    }

    if (op_type == "Cast") {
      // onnx.Cast requires 'to' as a TypeAttr property, not an IntegerAttr.
      // The EP receives 'to' as an ONNX TensorProto_DataType enum (integer),
      // but the registered ONNX dialect in the compiler expects TypeAttr.
      llvm::SmallVector<mlir::NamedAttribute> new_attrs;
      for (auto& attr : mlir_attrs) {
        if (attr.getName() == "to") {
          if (auto int_attr =
                  mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
            int onnx_type = static_cast<int>(int_attr.getSInt());
            mlir::Type mlir_type =
                onnxElementTypeToMlirElementType(onnx_type, builder);
            new_attrs.push_back(
                builder.getNamedAttr("to", mlir::TypeAttr::get(mlir_type)));
            continue;
          }
        }
        new_attrs.push_back(attr);
      }
      state.attributes =
          mlir::DictionaryAttr::get(builder.getContext(), new_attrs);
    } else {
      state.attributes = mlir_attrs;
    }

    op = builder.create(state);
  }

  {
    auto node = MLIRNode(op);
    node.setInputNodeArgs(input_args);
    node.setOutputNodeArgs(output_args);
  }

  // Collect output names for the "node.outputs" attribute
  llvm::SmallVector<mlir::Attribute> outputNames;
  for (const auto& output : output_args) {
    if (output.is_valid() && get_node_arg(output)) {
      outputNames.push_back(builder.getStringAttr(output.get_name()));
    } else {
      outputNames.push_back(builder.getStringAttr(""));
    }
  }

  // Set the "node.outputs" attribute on the operation
  if (!outputNames.empty()) {
    op->setAttr(attr_names::NODE_OUTPUTS, builder.getArrayAttr(outputNames));
  }
  // Set the "onnx_node_name" attribute on the operation
  op->setAttr(attr_names::ONNX_NODE_NAME, builder.getStringAttr(name));

  // Register the operation results in the symbol table.
  mlir::IRRewriter rewriter(context_for(*entry_block_));
  for (const auto& [output, result] :
       llvm::zip(output_args, op->getResults())) {
    if (!output.is_valid() || !get_node_arg(output))
      continue;
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

  // Transplant each sub's orphan block into the matching parent region
  // (slot i = sub_refs[i]). The sub is already owned by subgraphs_cache_.
  for (size_t i = 0; i < sub_refs.size(); ++i) {
    MLIRGraph* sub = sub_refs[i].second;
    CHECK(sub != nullptr) << "subgraph-ref marker for attr '"
                          << sub_refs[i].first << "' decoded to null";
    mlir::Region& region = op->getRegion(static_cast<unsigned>(i));
    mlir::Block* orphan = sub->take_orphan_block();
    region.push_back(orphan);
    sub->resolve(/*force=*/true);
  }

  // Region captures: a Loop/If/Scan body may use outer-scope values that are
  // not operands of the op, so the anchoring above cannot see them. With the
  // body attached, enumerate them via getUsedValuesDefinedAbove; the producer
  // of a captured value can appear later in graph order than the region-bearing
  // op, so if a capture is defined after the op in entry_block_, sink the op
  // past it to keep the body's use dominated.
  if (op->getNumRegions() > 0) {
    llvm::SetVector<mlir::Value> captured;
    for (mlir::Region& region : op->getRegions())
      mlir::getUsedValuesDefinedAbove(region, captured);
    mlir::Operation* anchor = nullptr;
    for (mlir::Value value : captured) {
      mlir::Operation* def = value.getDefiningOp();
      if (!def || def->getBlock() != entry_block_)
        continue;
      if (!anchor || anchor->isBeforeInBlock(def))
        anchor = def;
    }
    if (anchor && op->isBeforeInBlock(anchor))
      op->moveAfter(anchor);
  }

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

  mlir::OpBuilder builder(context_for(*entry_block_));

  // Find the last onnx.Constant operation to insert after it
  // Order should be: onnx.None -> onnx.Constant ops -> other ops -> return
  auto& block = *entry_block_;
  mlir::Operation* lastConstantOp = nullptr;
  mlir::Operation* noneOp = nullptr;

  for (auto& op : block.getOperations()) {
    // Track onnx.None operation
    if (op.getName().getStringRef() == onnx_mlir::ONNX_NONE) {
      noneOp = &op;
    }
    // Track onnx.Constant operations
    else if (op.getName().getStringRef() == "onnx.Constant") {
      lastConstantOp = &op;
    } else if (!onnx_mlir::isReturnOp(&op)) {
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
  const void* data = node_arg->getData();
  size_t dataSize = node_arg->getDataSize();

  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(shapedTensorType);

  if (node_arg->isExternalData()) {
    auto* ext = node_arg->getExternalRef();
    CHECK(ext) << "ExternalRef must exist for external data: " << name;
    state.addAttribute("location", builder.getStringAttr(ext->location));
    state.addAttribute(
        "offset", builder.getI64IntegerAttr(static_cast<int64_t>(ext->offset)));
    state.addAttribute(
        "size", builder.getI64IntegerAttr(static_cast<int64_t>(ext->size)));
  } else {
    auto rawData =
        llvm::ArrayRef<char>(static_cast<const char*>(data), dataSize);
    auto denseAttr =
        mlir::DenseElementsAttr::getFromRawBuffer(shapedTensorType, rawData);
    state.addAttribute("value", denseAttr);
  }

  mlir::Operation* op = builder.create(state);
  op->setAttr(attr_names::NODE_OUTPUTS,
              builder.getArrayAttr({builder.getStringAttr(name)}));
  // update value in MLIRTensor object.
  node_arg->setValue(op->getResult(0));

  MY_LOG(1) << " constant op \"" << name
            << "\":" << node_arg->getValue().getAsOpaquePointer() << " = "
            << " result=" << op->getResult(0).getAsOpaquePointer()
            << " op=" << (void*)op;
  MY_LOG(1) << "node arg=" << MLIRNodeArgIndex(node_arg_ptr).to_string();
  MY_LOG(1) << "Successfully created constant tensor '" << name
            << "' as onnx.Constant operation";
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
  CHECK(!is_subgraph())
      << "save is top-level only "
         "(subgraph embedded in parent op; dump module instead)";
  (void)dat_filename;            // Currently unused in MLIR implementation
  (void)external_data_threshold; // Currently unused in MLIR implementation

  MY_LOG(1) << "Saving MLIR graph to file: " << filename;

  // Get serialized string
  std::string mlir_string = save_string();
  if (mlir_string.empty()) {
    LOG(ERROR) << "Failed to serialize MLIR graph";
    return;
  }

  // Write to file
  std::error_code error_code;
  llvm::raw_fd_ostream output(filename, error_code);
  if (error_code) {
    LOG(ERROR) << "Failed to open file for writing: " << filename << " - "
               << error_code.message();
    return;
  }

  output << mlir_string;
  output.close();

  MY_LOG(1) << "Successfully saved MLIR graph to: " << filename;
}

std::string MLIRGraph::save_string() const {
  CHECK(!is_subgraph())
      << "save_string is top-level only "
         "(subgraph embedded in parent op; dump module instead)";
  MY_LOG(1) << "Serializing MLIR graph to string";

  // Get the parent module containing this function
  mlir::ModuleOp module = func()->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    LOG(ERROR)
        << "Cannot serialize graph: function is not contained in a module";
    return "";
  }

  std::vector<
      std::pair<mlir::Operation*, llvm::SmallVector<mlir::NamedAttribute>>>
      backups;
  module.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ModuleOp>(op) || mlir::isa<mlir::func::FuncOp>(op)) {
      return;
    }
    backups.emplace_back(op, MLIRNode(op).backupAndClearMorphizenAttrs());
  });

  auto restore_backups = [&]() {
    for (auto& [op, snapshot] : backups) {
      MLIRNode(op).restoreMorphizenAttrs(snapshot);
    }
  };

  // Serialize the MLIR module (text or bytecode based on env var)
  std::string result;
  llvm::raw_string_ostream stream(result);

  if (ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT)) {
    // Text format for human readability
    module.print(stream);
  } else {
    // Bytecode format (default, more compact)
    mlir::BytecodeWriterConfig config;
    if (failed(mlir::writeBytecodeToFile(module, stream, config))) {
      LOG(ERROR) << "Failed to write MLIR bytecode";
      restore_backups();
      return "";
    }
  }
  stream.flush();

  restore_backups();

  MY_LOG(1) << "Successfully serialized MLIR graph to string ("
            << backups.size() << " ops snapshotted, format="
            << (ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT) ? "text" : "bytecode")
            << ")";
  return result;
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
    // Skip return operations (onnx.Return or func.return) to avoid treating
    // function outputs as regular consumers. Bug scenario: When a node's output
    // (e.g., "linear_img_add_DequantizeLinear") is directly returned by the
    // function, including return ops as a consumer causes incorrect dependency
    // analysis in graph partitioning, leading to wrong subgraph boundaries or
    // the Partitioner incorrectly treating graph outputs as having real
    // successors.
    if (onnx_mlir::isReturnOp(userOp)) {
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
      if (producer_op->getName().getStringRef() == "onnx.Constant") {
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

void MLIRGraph::canonicalize_optional_outputs() {
  // Align with onnx-mlir behavior: for multi-result operations, any result
  // that has no users (not consumed by other ops, not in onnx.Return) is an
  // unused optional output and should be typed as `none`.
  // This avoids allocating dummy buffers for unused intermediate values.
  auto* ctx = entry_block_->getParentOp()->getContext();
  auto noneType = mlir::NoneType::get(ctx);
  mlir::OpBuilder builder(ctx);

  // Collect the set of values used by the terminator (onnx.Return / onnx.Yield)
  llvm::DenseSet<mlir::Value> terminator_operands;
  if (terminator_) {
    for (auto operand : terminator_->getOperands())
      terminator_operands.insert(operand);
  }

  llvm::SmallVector<mlir::Operation*> ops_to_process;
  // Block::walk does not visit the parent op, so no FuncOp self-skip needed.
  entry_block_->walk([&](mlir::Operation* op) {
    if (op == terminator_ || op == none_)
      return;
    if (op->getNumResults() <= 1)
      return;
    bool has_unused = false;
    for (auto result : op->getResults()) {
      if (result.use_empty() && !terminator_operands.contains(result) &&
          !mlir::isa<mlir::NoneType>(result.getType())) {
        has_unused = true;
        break;
      }
    }
    if (has_unused)
      ops_to_process.push_back(op);
  });

  for (auto* op : ops_to_process) {
    llvm::SmallVector<mlir::Type> newTypes;
    llvm::SmallVector<unsigned> unused_indices;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      if (result.use_empty() && !terminator_operands.contains(result) &&
          !mlir::isa<mlir::NoneType>(result.getType())) {
        newTypes.push_back(noneType);
        unused_indices.push_back(i);
      } else {
        newTypes.push_back(result.getType());
      }
    }

    // Rebuild the operation with updated result types
    builder.setInsertionPoint(op);
    mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(op->getOperands());
    state.addTypes(newTypes);
    state.addAttributes(op->getAttrs());
    // Copy inherent/discardable attributes and properties
    if (op->getPropertiesStorageSize())
      state.propertiesAttr = op->getPropertiesAsAttribute();

    auto* newOp = builder.create(state);

    // Replace all uses of old results with new results
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      op->getResult(i).replaceAllUsesWith(newOp->getResult(i));

    op->erase();

    for (unsigned idx : unused_indices) {
      MY_LOG(1) << "Canonicalized unused result #" << idx << " of "
                << newOp->getName().getStringRef().str() << " to none";
    }
  }
}

int MLIRGraph::resolve(bool force) {
  MY_LOG(1) << "MLIRGraph::resolve called with force=" << force;
  // The entry block must be attached to a region by the time resolve runs
  // (top-level FuncOp blocks are always attached; sub graphs created by
  // graph_new_subgraph become attached when add_node transplants them).
  CHECK(entry_block_ != nullptr && entry_block_->getParent() != nullptr)
      << "MLIRGraph::resolve called with an orphan entry block: "
         "graph_new_subgraph must be paired with attr_proto_new_graph + "
         "graph_add_node before resolve(true) runs.";

  // In MLIR, the IR is always in a consistent state
  // Unlike ONNX which has staging graphs, MLIR operations are immediately
  // consistent when created. However, we can still perform validation
  // and optimization passes if needed.

  // For MLIR, we typically always run optimization passes when resolving
  // since MLIR is designed to be optimized incrementally, for now

  // Perform MLIR-specific graph validation and consistency checks
  try {
    // Subgraphs have no FuncOp; verification is the parent (top-level)
    // MLIRGraph::resolve()'s responsibility (mlir::verify on the top-level
    // func walks nested regions transitively). Skip the FuncOp-typed check
    // here for subgraphs and proceed straight to canonicalize + re-init.
    if (!is_subgraph()) {
      // Verify that the function is well-formed
      if (failed(mlir::verify(func()))) {
        if (ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH))
          save("resolve_fatal.onnx", "", 10000);
        LOG(FATAL) << "MLIRGraph verification failed";
        return -1; // Verification failure
      }
    }

    // TODO : shape inference

    // Canonicalize unused optional outputs to none (align with onnx-mlir)
    canonicalize_optional_outputs();

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
  entry_block_->walk([&](mlir::Operation* op) -> mlir::WalkResult {
    if (op->getName().getStringRef() != "onnx.Constant") {
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
  CHECK(!is_subgraph()) << "fuse is top-level only "
                           "(EP partition does not descend into subgraphs)";
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
  CHECK(!is_subgraph()) << "create_func_func is top-level only";
  auto parent_func = func();
  auto context = parent_func->getContext();
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

  auto parent_module = parent_func->getParentOfType<mlir::ModuleOp>();
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
  parent_func.walk([&](mlir::Operation* walk_op) {
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
          if (defining_op->getName().getStringRef() == "onnx.Constant") {
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

  // Create onnx.Return operation with mapped outputs
  auto output_values =
      llvm::to_vector(llvm::map_range(outputs, [&](const auto& output) {
        return value_mapping.lookup(output.get_node_arg().getValue());
      }));
  mlir::OperationState returnState(builder.getUnknownLoc(),
                                   onnx_mlir::ONNX_RETURN);
  returnState.addOperands(output_values);
  builder.create(returnState);

  return std::make_pair(temp_func, cloned_ops_cache);
}

mlir::Operation* MLIRGraph::create_func_call(
    const std::string& name, const std::vector<MLIRNodeArgIndex>& inputs,
    const std::vector<MLIRNodeArgIndex>& outputs, mlir::func::FuncOp fused_func,
    const std::stack<mlir::Operation*>& /* cloned_ops_cache */) {
  CHECK(!is_subgraph()) << "create_func_call is top-level only";
  mlir::IRRewriter rewriter(func()->getContext());

  // Set insertion point based on inputs
  mlir::Operation* latestInputOp = nullptr;
  bool hasConstantInput = false;

  for (const auto& input : inputs) {
    auto value = input.get_node_arg().getValue();
    if (auto* definingOp = value.getDefiningOp()) {
      if (definingOp->getName().getStringRef() == "onnx.Constant") {
        hasConstantInput = true;
      }

      if (!latestInputOp || latestInputOp->isBeforeInBlock(definingOp)) {
        latestInputOp = definingOp;
      }
    }
  }

  // If any input is from onnx.Constant, find the last onnx.Constant in the
  // block
  if (hasConstantInput) {
    auto& entryBlock = func().getBody().front();
    mlir::Operation* lastConstantOp = nullptr;

    for (auto& op : entryBlock.getOperations()) {
      if (op.getName().getStringRef() == "onnx.Constant") {
        lastConstantOp = &op;
      } else if (!onnx_mlir::isReturnOp(&op)) {
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
  auto fused = MLIRNode(fuse_node);
  fused.setInputNodeArgs(inputs);
  fused.setOutputNodeArgs(outputs);

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
        if (userOp == fuse_node || onnx_mlir::isReturnOp(userOp)) {
          return;
        }

        // userOp can live inside a nested region (e.g. Loop/If/Scan body)
        // when fuse_node's outputs are implicitly captured by a region-
        // bearing op. The naive `userOp->isBeforeInBlock(fuse_node)` is
        // UB across blocks (mlir/lib/IR/Operation.cpp:386). Translate
        // userOp to its ancestor that lies in fuse_node's block: that
        // ancestor is the actual ordering proxy, because fuse_node must
        // dominate the region-bearing op for the capture to be valid SSA.
        // Same idiom as
        // mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp.
        mlir::Operation* anchor =
            fuse_node->getBlock()->findAncestorOpInBlock(*userOp);
        if (!anchor || anchor == fuse_node || visited.count(anchor)) {
          return;
        }
        visited.insert(anchor);

        // Only ops preceding fuse_node need to be moved after it to
        // preserve SSA dominance when fuse_node's results replace the
        // original outputs.
        if (!anchor->isBeforeInBlock(fuse_node)) {
          return;
        }

        opsToMove.push_back(anchor);

        // Recurse on the anchor's results (the in-block representative),
        // not userOp's, since transitive in-block users come through the
        // anchor in fuse_node's block.
        for (auto result : anchor->getResults()) {
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
  CHECK(!is_subgraph()) << "remove_func_ops is top-level only";
  // Delete cloned ops in LIFO order so later ops are erased before their
  // producers. dropAllReferences() is load-bearing for region-bearing
  // ops (onnx.Loop / If / Scan) whose body holds inter-op SSA uses (e.g.
  // onnx.Add result -> onnx.Yield); without it, ~Operation walks the
  // body and trips its use_empty assert. For region-less ops it is a
  // no-op.
  while (!cloned_ops_cache.empty()) {
    mlir::Operation* op = cloned_ops_cache.top();
    cloned_ops_cache.pop();
    if (op && op->use_empty()) {
      op->dropAllReferences();
      op->erase();
    }
  }
}
} // namespace mlir_impl
} // namespace morphizen
