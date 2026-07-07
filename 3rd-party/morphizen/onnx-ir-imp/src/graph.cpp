/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
#include "./graph.hpp"
#include "./graph-id.hpp"
#include "./graph-resolver.hpp"
#include "./graph-store.hpp"
#include "./model.hpp"
#include "./node-arg-index.hpp"
#include "./node-index.hpp"
#include "./staging-graph.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <algorithm>
#include <fstream>
#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_GRAPH, "0");
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_GRAPH) >= (n))

namespace morphizen {
// this function create a main graph for a model proto, to represent in memory
// the main graph of the model.
std::unique_ptr<Graph>
Graph::create_main_graph(morphizen_onnx::ModelProto& model_proto,
                         const Model* parent_model) {
  return create_graph(*model_proto.mutable_graph(), parent_model, 0, nullptr);
}
// this is the private constructor for creating a graph from a GraphProto.
std::unique_ptr<Graph>
Graph::create_graph(morphizen_onnx::GraphProto& graph_proto,
                    const Model* parent_model, uint32_t proposed_graph_index,
                    const Graph* parent_graph) {
  if (parent_graph == nullptr) {
    // ignore shape inference for subgraphs, because many shape inference
    // functions for com.xilinx op are ready yet.
    morphizen_onnx::shape_inference::InferShapes(
        &graph_proto, parent_model->get_opset_imports());
  }
  return std::make_unique<Graph>(PrivateTag{}, graph_proto, parent_model,
                                 proposed_graph_index, parent_graph);
}

Graph::Graph(PrivateTag, morphizen_onnx::GraphProto& graph_proto,
             const Model* parent_model, uint32_t proposed_graph_id,
             const Graph* parent_graph)
    : graph_proto_(graph_proto), parent_model_(parent_model),
      parent_graph_(parent_graph),
      graph_index_(proposed_graph_id > 0
                       ? GraphStore::allocate_graph_id(this, proposed_graph_id)
                       : GraphStore::allocate_graph_id(this)),
      producer_map_{get_graph_id()} {
  initialize();
}

Graph::~Graph() { GraphStore::release_graph_id(this, graph_index_); }
// === Graph API Method Implementations ===

const std::string& Graph::get_name() const { return graph_proto_.name(); }

const Model& Graph::get_model() const {
  if (!parent_model_) {
    throw std::runtime_error("Graph has no parent model");
  }
  return *parent_model_;
}
const std::filesystem::path& Graph::get_model_path() const {
  return get_model().get_model_path();
}
GraphId Graph::get_graph_id() const {
  return GraphId::create_main_graph(graph_index_);
}

const morphizen_onnx::GraphProto& Graph::get_graph_proto() const {
  return graph_proto_;
}
morphizen_onnx::GraphProto& Graph::get_graph_proto() { return graph_proto_; }
// staging graph is mutable.
StagingGraph* Graph::get_staging_graph() const { return staging_graph_.get(); }

std::vector<NodeIndex> Graph::nodes_unsafe() const {
  std::vector<NodeIndex> result;
  result.reserve(nodes_.size()); // Create NodeIndex for each node in the graph
  for (size_t i = 0; i < nodes_.size(); ++i) {
    result.emplace_back(
        NodeIndex(static_cast<unsigned int>(i), get_graph_id()));
  }
  return result;
}

std::vector<NodeArgIndex> Graph::get_inputs_unsafe() const {
  std::vector<NodeArgIndex> result;

  // Get the input count from the graph proto
  const auto& inputs = graph_proto_.input();
  result.reserve(inputs.size());

  // Convert each input to a NodeArgIndex
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_name = inputs[i].name();
    auto node_arg = get_node_arg(input_name);
    CHECK(node_arg.is_valid()) << "Input node_arg not found: " << input_name
                               << ", graph ID: " << get_graph_id().to_string();
    CHECK(node_arg.is_graph_input())
        << "Input node_arg is not a graph input: " << input_name
        << ", graph ID: " << get_graph_id().to_string();
    result.push_back(node_arg);
  }
  return result;
}

std::vector<NodeArgIndex> Graph::get_outputs_unsafe() const {
  std::vector<NodeArgIndex> result;

  // Get the output count from the graph proto
  const auto& outputs = staging_graph_ ? staging_graph_->graph_proto_.output()
                                       : graph_proto_.output();
  result.reserve(outputs.size());

  // Graph outputs are typically node outputs, so we use NODE_OUTPUT type
  // The index should map to the value_info or be resolved through the
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_name = outputs[i].name();
    auto node_arg = get_node_arg(output_name);
    CHECK(node_arg.is_valid()) << "Output node_arg not found: " << output_name
                               << ", graph ID: " << get_graph_id().to_string();
    // remove this checking, for Model-PSI-QDQ-v3_0.onnx, it is possible that
    // a constant initializer is a group output due to the ORT constant
    // folding optimization
    /*
    CHECK(node_arg.is_graph_output())
        << "Output node_arg is not a graph output: " << output_name
        << ", graph ID: " << get_graph_id().to_string();
    */
    result.push_back(node_arg);
  }
  return result;
}

const NodeIndex Graph::producer_node(const std::string& node_arg_name) const {
  return get_node_arg(node_arg_name).get_producer_node();
}

const NodeArgIndex Graph::get_node_arg(const std::string& name) const {
  // Check if the name exists in the node_args_map_
  if (staging_graph_) {
    auto it = staging_graph_->node_args_map_.find(name);
    if (it != staging_graph_->node_args_map_.end()) {
      return it->second; // Return the NodeArgIndex if found
    }
  }
  {
    auto it = node_args_map_.find(name);
    if (it != node_args_map_.end()) {
      return it->second; // Return the NodeArgIndex if found
    }
  }
  if (parent_graph_) {
    return parent_graph_->get_node_arg(name);
  }
  // If not found, return an invalid NodeArgIndex
  return NodeArgIndex::invalid();
}

// std::unordered_map<std::string, const TensorProto*> ==  const
// morphizen::InitializedTensorSet&
const std::unordered_map<std::string, const morphizen_onnx::TensorProto*>&
Graph::get_all_initialized_tensors() const {
  // Ensure the initializers map is populated
  return initializers_map_;
}

void Graph::remove_node(NodeIndex node_index) const {
  CHECK(node_index.is_valid(*this));
  // Enter inconsistent state when modifying the graph
  ensure_enter_into_inconsistent_state();
  staging_graph_->remove_node(node_index);
}

NodeIndex Graph::add_node(
    const std::string& name, const std::string& op_type,
    const std::string& description, const std::vector<NodeArgIndex>& input_args,
    const std::vector<NodeArgIndex>& output_args,
    ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*
        attributes,
    const std::string& domain) const {
  // 1. Input validation
  validate_add_node_parameters(name, op_type, input_args, output_args);

  // 2. Enter inconsistent state
  ensure_enter_into_inconsistent_state();

  return staging_graph_->add_node(name, op_type, description, input_args,
                                  output_args, attributes, domain);
}

void Graph::save(const std::string& filename, const std::string& dat_filename,
                 size_t external_data_threshold) const {
  // Create a temporary model proto containing this graph for saving
  morphizen_onnx::ModelProto temp_model;
  temp_model.set_ir_version(parent_model_->ir_version());
  temp_model.set_producer_name(parent_model_->producer_name());
  temp_model.set_producer_version(parent_model_->producer_version());
  temp_model.set_domain(parent_model_->domain());
  temp_model.set_model_version(parent_model_->model_version());

  // Copy opset imports
  for (const auto& opset_import : parent_model_->model_proto().opset_import()) {
    *temp_model.add_opset_import() = opset_import;
  }

  // Copy the graph
  *temp_model.mutable_graph() = graph_proto_;

  // Save to file
  std::ofstream output_file(filename, std::ios::binary);
  if (!output_file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  if (!temp_model.SerializeToOstream(&output_file)) {
    throw std::runtime_error("Failed to serialize model to file: " + filename);
  }

  output_file.close();

  // TODO: Handle external data saving with dat_filename and
  // external_data_threshold
  (void)dat_filename;
  (void)external_data_threshold;
}
std::string Graph::save_string() const {
  // Create a temporary model proto containing this graph for saving
  morphizen_onnx::ModelProto temp_model;
  temp_model.set_ir_version(parent_model_->ir_version());
  temp_model.set_producer_name(parent_model_->producer_name());
  temp_model.set_producer_version(parent_model_->producer_version());
  temp_model.set_domain(parent_model_->domain());
  temp_model.set_model_version(parent_model_->model_version());
  // Copy opset imports
  for (const auto& opset_import : parent_model_->model_proto().opset_import()) {
    *temp_model.add_opset_import() = opset_import;
  }
  // Copy the graph
  *temp_model.mutable_graph() = graph_proto_;
  // Serialize to string
  std::string output_string;
  output_string.reserve(temp_model.ByteSizeLong());
  if (!temp_model.SerializeToString(&output_string)) {
    throw std::runtime_error("Failed to serialize model to string");
  }
  return output_string;
}

NodeIndex
Graph::fuse(const std::string& name, const std::string& op_type,
            const std::vector<size_t>& nodes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const std::vector<std::string>& /*constant_initializers*/) {
  if (staging_graph_ != nullptr) {
    LOG(FATAL) << "Fusing nodes in staging graph, entering inconsistent state";
    // resolve(true); // Force resolve to ensure staging graph is consistent
  }
  CHECK(!inputs.empty()) << "empty inputs";
  CHECK(!outputs.empty()) << "empty outputs";
  auto fused_graph = morphizen_onnx::GraphProto();
  auto left_nodes =
      google::protobuf::RepeatedPtrField<morphizen_onnx::NodeProto>();
  auto moved_nodes =
      google::protobuf::RepeatedPtrField<morphizen_onnx::NodeProto>();
  auto nodes_set = std::set<size_t>();
  for (auto node_index : nodes) {
    auto ni = NodeIndex::from_morphizen_core_node_index(node_index);
    CHECK_EQ(ni.get_graph_id().get_raw(), get_graph_id().get_raw())
        << " invalid reference";
    nodes_set.insert(ni.get_index());
  }

  MY_LOG(1) << "Fusing nodes: " << nodes.size() << " inputs: " << inputs.size()
            << " outputs: " << outputs.size()
            << " into a new node with name: " << name
            << " and op_type: " << op_type;
  for (const auto& input : inputs) {
    MY_LOG(1) << "Input " << input;
  }
  for (const auto& output : outputs) {
    MY_LOG(1) << "Output " << output;
  }
  // exclude graph inputs
  auto input_set =
      std::unordered_set<std::string>(inputs.begin(), inputs.end());
  morphizen_onnx::NodeProto* inserted_fused_node_proto = nullptr;
  int fused_node_index = -1;
  auto find_insert_position =
      [&input_set, &inserted_fused_node_proto, &fused_node_index, &left_nodes](
          const ::google::protobuf::RepeatedPtrField<std::string>& names)
      -> bool {
    if (inserted_fused_node_proto != nullptr) {
      // already inserted, no need to find position
      return true;
    }
    for (auto& out : names) {
      if (input_set.count(out) > 0) {
        input_set.erase(out);
        if (input_set.empty()) {
          inserted_fused_node_proto = left_nodes.Add();
          fused_node_index = left_nodes.size() - 1;
        }
      }
    }
    return (inserted_fused_node_proto != nullptr);
  };
  auto convert_value_infos_to_strings =
      [](const ::google::protobuf::RepeatedPtrField<
          morphizen_onnx::ValueInfoProto>& value_infos)
      -> ::google::protobuf::RepeatedPtrField<std::string> {
    ::google::protobuf::RepeatedPtrField<std::string> result;
    for (const auto& value_info : value_infos) {
      *result.Add() = (value_info.name());
    }
    return result;
  };

  find_insert_position(convert_value_infos_to_strings(graph_proto_.input()));
  for (auto i = 0; i < graph_proto_.node_size(); ++i) {
    auto is_fused_node = nodes_set.count(i) > 0;
    auto new_node = is_fused_node ? moved_nodes.Add() : left_nodes.Add();
    MY_LOG(2) << "Processing node " << i << ": "
              << (is_fused_node ? " (fused node)\n" : "\n")
              << graph_proto_.node(i).DebugString();
    MY_LOG(2) << "moved_nodes size: " << moved_nodes.size()
              << ", left_nodes size: " << left_nodes.size();
    { // for topological order
      if (!is_fused_node) {
        // all inputs are available, we can insert the fused node right after
        // it.
        find_insert_position(graph_proto_.node(i).output());
      }
    }
    new_node->Swap(graph_proto_.mutable_node(i));
  }
  if (left_nodes.empty()) {
    CHECK(inserted_fused_node_proto == nullptr);
    CHECK(fused_node_index == -1);
    fused_node_index = 0;
    inserted_fused_node_proto = left_nodes.Add();
  }
  CHECK(inserted_fused_node_proto != nullptr)
      << "inserted_fused_node_proto must be set to a valid position in "
         "left_nodes";
  auto& fused_node = *inserted_fused_node_proto;
  fused_node.mutable_input()->Assign(inputs.begin(), inputs.end());
  fused_node.mutable_output()->Assign(outputs.begin(), outputs.end());
  fused_graph.set_name(name);
  for (auto& input : inputs) {
    auto* input_arg = fused_graph.add_input();
    input_arg->CopyFrom(get_node_arg(input).get_value_info());
  }
  for (auto& output : outputs) {
    auto* output_arg = fused_graph.add_output();
    output_arg->CopyFrom(get_node_arg(output).get_value_info());
  }
  for (auto node_index : nodes) {
    auto ni = NodeIndex::from_morphizen_core_node_index(node_index);
    auto output_node_args = ni.get_output_node_args();
    for (auto node_arg : output_node_args) {
      auto& value_info = node_arg.get_value_info();
      auto is_output = std::find(outputs.begin(), outputs.end(),
                                 value_info.name()) != outputs.end();
      if (is_output) {
        continue;
      }
      fused_graph.add_value_info()->CopyFrom(value_info);
    }
  }

  fused_node.set_name(name);
  fused_node.set_op_type(op_type);
  fused_node.set_domain("com.xilinx");
  fused_node.set_doc_string("Fused node for " + name);
  auto attr = fused_node.add_attribute();
  attr->set_name("body");
  CHECK(!moved_nodes.empty());
  fused_graph.mutable_node()->Swap(&moved_nodes);
  attr->mutable_g()->Swap(&fused_graph);
  auto fused_graph_id = const_cast<Model*>(parent_model_)
                            ->create_subgraph(*this, *attr->mutable_g());
  // set attribute["fused_node_index"] to the index of the fused node
  // IMPORTANT: Must be done BEFORE swapping fused_node into the graph
  auto* attr_fused_node_index = fused_node.add_attribute();
  attr_fused_node_index->set_name("fused_node_index");
  attr_fused_node_index->set_type(morphizen_onnx::AttributeProto::INT);
  attr_fused_node_index->set_i(static_cast<int64_t>(fused_node_index));
  // IMPORTANT: Must be done BEFORE swapping fused_node into the graph

  auto* attr_fused_graph_id = fused_node.add_attribute();
  attr_fused_graph_id->set_name("fused_graph_id");
  attr_fused_graph_id->set_type(morphizen_onnx::AttributeProto::INT);
  attr_fused_graph_id->set_i(static_cast<int64_t>(fused_graph_id.get_raw()));
  inserted_fused_node_proto->Swap(&fused_node);

  graph_proto_.mutable_node()->Swap(&left_nodes);
  GraphStore::release_graph_id(this, graph_index_);
  graph_index_ = GraphStore::allocate_graph_id(this, graph_index_);
  // std::cout << "Fused node index: " << fused_node_index << std::endl;
  // std::cout << "Graph \n " << graph_proto_.DebugString() << std::endl;
  initialize();
  // Return the NodeIndex of the inserted fused node
  return NodeIndex(static_cast<unsigned int>(fused_node_index), get_graph_id());
}

int Graph::resolve(bool force) {

  MY_LOG(1) << "Graph::resolve called with force=" << force;
  // `force` does not make any sense here, as we always resolve the graph
  if (!need_resolve()) {
    MY_LOG(1) << "Graph does not need resolution, skipping.";
    return 0; // No resolution needed
  }
  //
  // Use GraphResolver to handle the resolution process
  GraphResolver resolver;

  // allocate a new graph ID for the resolved graph
  // the new graph ID must be larger than the original graph id.
  // to make sure all old node arg index and node index are properly
  // invalidated.
  // it must before staging_graph_.reset();, otherwise the
  // staging_graph_.graph_index_ will be in use and causes troubles.
  CHECK(staging_graph_ != nullptr)
      << "Staging graph must not be released before allocating new graph ID";
  uint32_t old_graph_index = graph_index_;
  uint32_t new_graph_index = GraphStore::allocate_graph_id(this, graph_index_);
  CHECK_GT(new_graph_index, old_graph_index)
      << "New graph ID must be greater than the old graph ID";

  // Get the new graph ID first
  auto new_graph_id = GraphId::create_main_graph(graph_index_);

  auto resolved_proto =
      resolver.resolve(*this, new_graph_id,
                       const_cast<Model*>(parent_model_)->get_opset_imports());
  // we must reset graph_index after resolve, otherwise, we cannot get constant
  // initializers.
  graph_index_ = new_graph_index;
  // Release the old graph ID and invalidate references
  GraphStore::release_graph_id(this, old_graph_index);
  // Update our graph proto with the resolved version
  graph_proto_ = std::move(resolved_proto);

  // Graph ID was already allocated above, no need to call
  // allocate_new_graph_index_and_release_old() again

  // Clear the staging graph and proto to return to consistent state
  staging_graph_.reset(); // graph_id of staging_graph_ will be released

  // Re-initialize the graph to ensure all internal structures are consistent
  initialize();
  MY_LOG(1) << "Graph::resolve completed successfully";
  return 0; // Success
}

bool Graph::need_resolve() const {
  // Return true if staging_graph_ is not nullptr, indicating we're in
  // inconsistent state, or if staging graph has different number of
  // nodes
  return staging_graph_ != nullptr;
}

std::vector<NodeIndex>
Graph::get_consumer_nodes(const std::string& node_arg_name) const {
  auto node_arg_index = get_node_arg(node_arg_name);
  // find the node node index
  if (!node_arg_index.is_valid()) {
    // if not found, return empty vector
    MY_LOG(1) << "NodeArg name not found: " << node_arg_name;
    return {};
  }
  auto id = node_arg_index.get_graph_id();
  if (id.get_index() != graph_index_) {
    LOG(WARNING) << "NodeArg name: " << node_arg_name
                 << " has graph ID: " << id.get_index()
                 << ", but current graph ID is: " << graph_index_;
    return {}; // Return empty vector if graph ID does not match
  }
  if (!id.is_staging()) {
    auto consumer_it = consumer_map_.find(node_arg_index);
    if (consumer_it == consumer_map_.end()) {
      // if not found, return empty vector
      MY_LOG(1) << "No consumer nodes found for NodeArg: " << node_arg_name;
      return {};
    }
    return consumer_it->second; // Return the vector of NodeIndex
  }
  return {};
}

void Graph::add_initialized_tensor(
    const morphizen_onnx::TensorProto& tensor) const {
  // Add the tensor to the staging graph
  auto name = tensor.name();
  if (name.empty()) {
    // throw an error if the tensor name is empty
    throw std::invalid_argument(
        "TensorProto name cannot be empty when adding to graph");
  }
  ensure_enter_into_inconsistent_state();
  return staging_graph_->add_initialized_tensor(tensor);
}

void Graph::set_inputs(gsl::span<NodeArgIndex> inputs) const {
  // Enter inconsistent state when modifying the graph
  ensure_enter_into_inconsistent_state();
  return staging_graph_->set_inputs(inputs);
}
void Graph::set_outputs(gsl::span<const NodeArgIndex> outputs) const {
  // Enter inconsistent state when modifying the graph
  ensure_enter_into_inconsistent_state();
  return staging_graph_->set_outputs(outputs);
}
void Graph::remove_initialized_tensor(const std::string& tensor_name) const {
  // it is only used by
  // morphizen_vaiml_common/graph_update_initializer.cpp:56:
  // actually `MORPHIZEN_ORT_API(graph_remove_node)(graph, {nullptr,
  // node_arg});` is the same thing, but it is obscured and to be deprecacted.
  //
  // in morphizen pass, create-const-op, we need to remove original initializer
  // otherwise ORT graph resolver will fail because of duplicated node arg
  // names.
  auto get_node_arg_local = [this](const std::string& name) -> NodeArgIndex {
    auto it = node_args_map_.find(name);
    if (it != node_args_map_.end()) {
      return it->second; // Return the NodeArgIndex if found
    }
    return NodeArgIndex::invalid();
  };
  // do not search for the constant initializer recursively
  auto node_arg_index = get_node_arg_local(tensor_name);
  if (!node_arg_index.is_valid()) {
    LOG(ERROR) << "NodeArg name not found: " << tensor_name;
    return; // If not found, do nothing
  }
  CHECK(node_arg_index.is_valid_initializer())
      << "NodeArgIndex must be a valid initializer: "
      << node_arg_index.to_string();
  auto graph_id = node_arg_index.get_graph_id();
  CHECK_EQ(graph_id.get_index(), get_graph_id().get_index())
      << "Graph ID mismatch: " << graph_id.to_string() << " vs "
      << get_graph_id().to_string();
  CHECK(!graph_id.is_staging())
      << "Cannot remove initializer from staging graph: "
      << graph_id.to_string();
  auto name = node_arg_index.get_name_unsafe();
  CHECK(name != nullptr && !name->empty())
      << "NodeArgIndex name cannot be empty when removing initializer: "
      << node_arg_index.to_string();
  auto index = node_arg_index.get_index();
  CHECK_LT(index, get_graph_proto().initializer_size())
      << "NodeArgIndex index out of bounds: " << index
      << " for initializer size: " << get_graph_proto().initializer_size();
  // Enter inconsistent state when modifying the graph
  ensure_enter_into_inconsistent_state();
  staging_graph_->remove_initialized_tensor(index, tensor_name);
}

void Graph::reverse_dfs_from_preemp(
    gsl::span<const NodeIndex> from,
    const std::function<bool(const NodeIndex&)>& enter,
    const std::function<bool(const NodeIndex&)>& leave,
    const std::function<bool(const NodeIndex&, const NodeIndex&)>& comp,
    const std::function<bool(const NodeIndex&, const NodeIndex&)>& stop,
    bool include_staging_graph) const {

  // Call common implementation with sorting and return value handling
  reverse_dfs_from_impl(from, enter, leave, comp, stop, include_staging_graph);
}

void Graph::set_graph_name(const char* name) const {
  graph_proto_.set_name(name);
}

void* Graph::node_arg_clone(const NodeArg& node_arg,
                            const std::string& name) const {
  // it is only used by this pass
  // to be removed.
  // clang-format off
/*
morphizen_pass_graph_output_add_node/src/graph_output_add_node.cpp:71:            MORPHIZEN_ORT_API(node_arg_clone)(*graph, *output.node_arg, name);
*/
 LOG(FATAL) << "not implemented yet, please use staging graph to clone node arg";
  // clang-format on
  // ensure_enter_into_inconsistent_state();
  // return staging_graph_->node_arg_clone(node_arg, name);
  (void)node_arg; // Suppress unused parameter warning
  (void)name;     // Suppress unused parameter warning
  return nullptr; // Placeholder return value, should not reach here
}

NodeArgIndex Graph::node_arg_new(const std::string& name,
                                 const std::vector<int64_t>* shape,
                                 int element_type) const {

  if (name.empty()) {
    // for now, we use empty string to represent optional argument, and it is an
    // invalid node arg. we don't distinguish invalid node arg and optional node
    // arg.
    return NodeArgIndex::invalid();
  }
  // check if the name already exists in the original graph
  if (node_args_map_.find(name) != node_args_map_.end()) {
    throw std::invalid_argument(
        "NodeArg name already exists in original graph");
  }
  // check if the name already exists in the staging graph
  ensure_enter_into_inconsistent_state();
  return staging_graph_->node_arg_new(name, shape, element_type);
}

void Graph::initialize() {
  initialize_map();
  initialize_nodes();
  initialize_consumer_map();
  initialize_initializers();
}

void Graph::initialize_map() {
  // Initialize the node_args_map_ and producer_map_ here
  // This is a placeholder for actual implementation
  // In a real implementation, you would populate these maps based on the
  // graph_proto_ For now, we leave them empty
  node_args_map_.clear();
  producer_map_.clear();
  producer_map_.set_graph_id(get_graph_id());
  producer_map_.reserve((unsigned int)graph_proto_.value_info_size(),
                        (unsigned int)graph_proto_.output_size());
  auto node_arg_name_to_value_info_index =
      std::unordered_map<std::string, unsigned int>();
  {
    // Initialize value_info_map_ to map node_arg names to value_info indices
    // This is used to track the value_info for each node_arg
    unsigned int value_info_counter = 0;
    for (auto& value_info : graph_proto_.value_info()) {
      node_arg_name_to_value_info_index[value_info.name()] =
          value_info_counter++;
    }
  }
  auto find_value_info_index =
      [&node_arg_name_to_value_info_index](const std::string& name) -> int {
    auto it = node_arg_name_to_value_info_index.find(name);
    if (it == node_arg_name_to_value_info_index.end()) {
      MY_LOG(1)
          << "NodeArg name not found in value_info_map_, please run shape "
             "infer first: "
          << name;
      return -1;
    }
    return it->second;
  };
  for (unsigned int node_index_0 = 0;
       node_index_0 < (unsigned int)graph_proto_.node_size(); ++node_index_0) {
    auto& node = graph_proto_.node(node_index_0);
    NodeIndex node_idx(node_index_0, get_graph_id());
    for (auto& output : node.output()) {
      auto output_index = get_graph_output_index(output);
      NodeArgIndex node_arg_index = NodeArgIndex::invalid();
      if (output_index >= 0) {
        node_arg_index = NodeArgIndex::graph_output((unsigned int)output_index,
                                                    get_graph_id());
      } else {
        auto value_info_index =
            find_value_info_index(output); // Find the value_info index
        if (value_info_index < 0) {
          // If the output is not found in value_info, create a new one.
          // ONNX::ShapeInfer seems to be a best effort.
          value_info_index = graph_proto_.value_info_size();
          auto new_value_info =
              graph_proto_.add_value_info(); // Create a new value_info
          new_value_info->set_name(output);
          node_arg_name_to_value_info_index[output] = value_info_index;
        }
        node_arg_index =
            NodeArgIndex::node_output(value_info_index, get_graph_id());
      }
      node_args_map_[output] = node_arg_index;
      producer_map_[node_arg_index] = node_idx;
    }
  }
  for (unsigned int const_index_0 = 0u;
       const_index_0 < (unsigned int)graph_proto_.initializer_size();
       ++const_index_0) {
    auto& initializer = graph_proto_.initializer(const_index_0);
    NodeArgIndex index =
        NodeArgIndex::initializer(const_index_0, get_graph_id());
    node_args_map_[initializer.name()] = index;
  }
  for (unsigned int graph_input_index_0 = 0u;
       graph_input_index_0 < (unsigned int)graph_proto_.input_size();
       ++graph_input_index_0) {
    NodeArgIndex index =
        NodeArgIndex::graph_input(graph_input_index_0, get_graph_id());
    node_args_map_[graph_proto_.input(graph_input_index_0).name()] = index;
  }
}
void Graph::initialize_nodes() {
  auto size = graph_proto_.node_size();
  nodes_.clear();
  nodes_.reserve((size_t)size);
  for (unsigned int i = 0; i < (unsigned int)size; ++i) {
    NodeIndex self(i, get_graph_id());
    auto& node_proto = graph_proto_.node(i);
    auto input_args = std::vector<NodeArgIndex>();
    auto output_args = std::vector<NodeArgIndex>();
    for (auto& node_input : node_proto.input()) {
      auto node_arg = get_node_arg(node_input);
      input_args.push_back(node_arg);
    }
    for (auto& node_output : node_proto.output()) {
      auto node_arg = get_node_arg(node_output);
      output_args.push_back(node_arg);
    }
    nodes_.emplace_back(Node::create_node(self, input_args, output_args));
  }
}

void Graph::initialize_consumer_map() {
  // Initialize the consumer_map_ to track which nodes consume each node
  // argument
  consumer_map_.clear();

  // Initialize empty vectors for all known node arguments
  for (const auto& [name, node_arg_index] : node_args_map_) {
    consumer_map_[node_arg_index] = std::vector<NodeIndex>();
  }

  // Iterate through all nodes and populate consumer relationships
  for (unsigned int node_index_0 = 0;
       node_index_0 < (unsigned int)graph_proto_.node_size(); ++node_index_0) {
    auto& node = graph_proto_.node(node_index_0);
    NodeIndex node_idx(node_index_0, get_graph_id());

    // For each input of this node, mark this node as a consumer
    for (const auto& input_name : node.input()) {
      auto node_arg_index = get_node_arg(input_name);

      if (node_arg_index.is_valid()) {
        node_arg_index.get_graph_id()
            .get_graph()
            ->consumer_map_[node_arg_index]
            .push_back(node_idx);
      } else {
        // throw std::runtime_error("Input not found in node_args_map: " +
        //                          input_name);
        // invalid input means optional input.
      }
      // Note: If input is not found in node_args_map_, it might be
      // 1. an optional input
      // or an error case that should be handled at a higher level
    }
  }
}

void Graph::reverse_dfs_from_impl(
    gsl::span<const NodeIndex> from,
    const std::function<bool(const NodeIndex&)>& enter,
    const std::function<bool(const NodeIndex&)>& leave,
    const std::function<bool(const NodeIndex&, const NodeIndex&)>& comp,
    const std::function<bool(const NodeIndex&, const NodeIndex&)>& stop,
    bool include_staging_graph) const {
  constexpr bool use_return_values = true;
  using WorkEntry = std::pair<NodeIndex, bool>; // bool represents leave or not
  std::vector<WorkEntry> stack;
  stack.reserve(from.size());

  // Initialize stack with starting nodes
  for (const auto& node_idx : from) {
    stack.emplace_back(node_idx, false);
  }

  // Track visited nodes to avoid cycles
  std::unordered_set<NodeIndex> visited;

  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    const NodeIndex& node_idx = last_entry.first;

    // Skip invalid nodes
    if (!node_idx.is_valid()) {
      continue;
    }

    if (last_entry.second) {
      // leave node
      if (leave) {
        bool stop_processing = leave(node_idx);
        // For preemp version, respect return value; for regular version,
        // ignore it
        if (use_return_values && stop_processing) {
          break; // Stop processing if leave returns true;
        }
      }
      continue;
    }

    // Check if already visited
    if (visited.count(node_idx) > 0) {
      continue;
    }

    visited.insert(node_idx);

    // Enter node
    if (enter) {
      bool stop_processing = enter(node_idx);
      // For preemp version, respect return value; for regular version, ignore
      // it
      if (use_return_values && stop_processing) {
        break; // Stop processing if enter returns true;
      }
    }

    // Add leave operation to stack if needed
    if (leave) {
      stack.emplace_back(node_idx, true);
    }

    // Get the node and traverse its inputs
    if (node_idx.is_valid()) {
      const auto& input_args = node_idx.get_input_node_args();

      // Collect producer nodes
      std::vector<NodeIndex> producer_nodes;

      for (const auto& input_arg_idx : input_args) {
        if (!input_arg_idx.is_valid()) {
          // when input_arg_idex.is_valid() is false, it means an optional
          // argument.
          continue;
        } else if (input_arg_idx.is_initializer()) {
          // continue;
        } else if (input_arg_idx.is_graph_input()) {
          continue;
        }

        auto producer_idx = input_arg_idx.get_producer_node();
        // Skip invalid producers (graph inputs, initializers)
        if (!producer_idx.is_valid()) {
          continue;
        }
        // do not include staging graph for dfs
        // For PSS fuse_transpose pass time :  31,726ms -> 9994ms
        if (!include_staging_graph) {
          if (producer_idx.get_graph_id().is_staging()) {
            continue;
          }
        }

        // Check stop condition
        if (stop && stop(node_idx, producer_idx)) {
          continue;
        }

        // Add to collection if not visited
        if (visited.count(producer_idx) == 0) {
          producer_nodes.push_back(producer_idx);
        }
      }

      // Sort producer nodes if comparison function is provided
      if (comp && !producer_nodes.empty()) {
        std::sort(producer_nodes.begin(), producer_nodes.end(), comp);
      } // Add producer nodes to stack (sorted if comp was provided)
      // Note: we use rbegin & rend to reverse the order of processing (same
      // with onnxruntime) . because we want to process the last producer node
      // first, which is important for certain optimizations.
      // Usually, in models there are many Q/DQ ops. The inputs of a Q/DQ op are
      // (X, scale, zero_point), where scale and zero_point are typically
      // initializers. Traversing the X node first is more effective for pattern
      // matching.
      // e.g. PSS fuse_transpose pass profiling :
      // The first transpose matching , visited_counter is 23 -> 13
      // The fuse_transpose pass time : 9994ms -> 228ms
      for (auto i = producer_nodes.rbegin(); i != producer_nodes.rend(); ++i) {
        stack.emplace_back(*i, false); // OK.
      }
    }
  }
}

void Graph::initialize_initializers() {
  // Clear and repopulate the initializers map
  initializers_map_.clear();

  // Get all initialized tensors from the graph proto
  const auto& initializers = graph_proto_.initializer();

  // Populate the map with tensor name -> tensor proto pointer
  for (int i = 0; i < initializers.size(); ++i) {
    const auto& tensor = initializers[i];
    initializers_map_[tensor.name()] = &tensor;
  }
}

void Graph::ensure_enter_into_inconsistent_state() const {
  // If staging_graph_ is already created, we're already in inconsistent state
  if (staging_graph_ != nullptr) {
    return;
  }

  // Create a staging graph that owns its GraphProto
  // The staging graph is used to track modifications that will be applied
  // when resolve() is called.
  staging_graph_ = StagingGraph::create_from_graph(*this);
}

int Graph::get_graph_output_index(const std::string& name) const {
  auto& output = graph_proto_.output();
  auto it = std::find_if(output.begin(), output.end(),
                         [&name](const morphizen_onnx::ValueInfoProto& output) {
                           return output.name() == name;
                         });
  if (it == output.end()) {
    return -1; // Not a graph output
  }
  return static_cast<int>(it - output.begin());
}

// Helper method implementations for add_node refactoring

void Graph::validate_add_node_parameters(
    const std::string& name, const std::string& op_type,
    const std::vector<NodeArgIndex>& input_args,
    const std::vector<NodeArgIndex>& output_args) const {
  (void)name;
  (void)op_type;

  auto is_valid_node_arg_index = [this](const NodeArgIndex& arg) -> bool {
    auto id = arg.get_graph_id();
    auto ret = true;
    ret = ret && arg.is_valid();
    ret = ret && id.get_index() == graph_index_;
    if (id.is_staging()) {
      ret = ret && staging_graph_ != nullptr;
    }
    return ret;
  };

  for (const auto& input_arg : input_args) {
    if (!input_arg.is_valid()) {
      continue; // invalid input arg means optional input.
    }
    if (!is_valid_node_arg_index(input_arg)) {
      throw std::runtime_error("Invalid input argument: " +
                               input_arg.to_string());
    }
  }
  for (const auto& output_arg : output_args) {
    if (!output_arg.is_valid()) {
      continue; // invalid output arg means optional output.
    }
    if (!is_valid_node_arg_index(output_arg)) {
      throw std::runtime_error("Invalid output argument: " +
                               output_arg.to_string());
    }
  }
}

GraphId Graph::allocate_new_graph_index_and_release_old() {
  return GraphId::from_raw(0);
}

} // namespace morphizen
