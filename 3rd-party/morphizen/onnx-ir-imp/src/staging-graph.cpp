/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "staging-graph.hpp"
#include "graph-resolver.hpp"
#include "graph.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <atomic>
#include <glog/logging.h>
#include <sstream>
#include <stdexcept>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_GRAPH, "0");
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_GRAPH) >= (n))

namespace morphizen {
std::unique_ptr<StagingGraph>
StagingGraph::create_from_graph(const Graph& main_graph) {
  auto staging_graph = std::make_unique<StagingGraph>(PrivateTag{}, main_graph);
  return staging_graph;
}
StagingGraph::StagingGraph(PrivateTag, const Graph& main_graph)
    : main_graph_(main_graph), producer_map_{GraphId::create_staging_graph(
                                   main_graph.get_graph_id().get_index())} {
  graph_proto_.Clear();
  auto& proto = main_graph.get_graph_proto();
  *graph_proto_.mutable_input() = proto.input();
  *graph_proto_.mutable_output() = proto.output();
  node_args_map_.clear();
  initializers_map_.clear();
  nodes_.clear();
  log_messages_.clear();
}

StagingGraph::~StagingGraph() {}

GraphId StagingGraph::get_graph_id() const {
  return GraphId::create_staging_graph(main_graph_.get_graph_id().get_index());
}

const Graph& StagingGraph::get_main_graph() const { return main_graph_; }

const morphizen_onnx::GraphProto& StagingGraph::get_graph_proto() const {
  return graph_proto_;
}
morphizen_onnx::GraphProto& StagingGraph::get_graph_proto() {
  return graph_proto_;
}
void StagingGraph::remove_node(NodeIndex node_index) {
  CHECK(node_index.is_valid(get_main_graph()));
  // This function is const member function as it only update staging_graph_
  // to remove a node, we create meta node in staging_graph_,
  //   - op type: "delete"
  //   - op domain: GRAPH_META_DOMAIN
  //   - input: []
  //   - output: as same as the deleted node's outputs
  // `graph_resolve` will eventually remove the node from the graph_proto_
  auto graph_id = node_index.get_graph_id();
  CHECK_EQ(graph_id.get_index(), get_graph_id().get_index())
      << "Graph ID mismatch: " << graph_id.to_string() << " vs "
      << get_graph_id().to_string();
  if (graph_id.is_staging()) {
    CHECK_LT(node_index.get_index(), nodes_.size())
        << "NodeIndex is not valid in staging graph: "
        << node_index.to_string();
    CHECK_EQ(nodes_.size(), (size_t)graph_proto_.node_size())
        << "Staging graph nodes size mismatch with graph_proto_ node size: "
        << graph_proto_.node_size();
    // find node index on the original graph.
    auto& staging_node_proto = graph_proto_.node(node_index.get_index());
    auto& output = staging_node_proto.output();
    CHECK(!output.empty())
        << "Output of the node to be deleted must not be empty: "
        << node_index.to_string();
    // find it on the original graph. it only makes sense to remove the node on
    // the original graph.
    auto it = main_graph_.node_args_map_.find(output[0]);
    CHECK(it != main_graph_.node_args_map_.end())
        << "NodeArg not found in node_args_map_: " << output[0];
    node_index = main_graph_.producer_map_[it->second];
    CHECK(node_index.is_valid())
        << "Producer node index not found for NodeArg: "
        << it->second.to_string();
  }
  // Get the outputs of the node to be deleted
  // it only make sense to delete node on the original graph.
  CHECK_EQ(node_index.get_graph_id().get_raw(),
           main_graph_.get_graph_id().get_raw())
      << "NodeIndex graph ID mismatch: "
      << node_index.get_graph_id().to_string() << " vs "
      << main_graph_.get_graph_id().to_string();
  auto& node_proto = main_graph_.get_graph_proto().node(node_index.get_index());
  // Log the deletion operation
  log_remove_node(node_index, node_proto);
  // Create a meta node in staging graph to mark deletion
  auto* delete_node = graph_proto_.add_node();

  // Set meta node properties for deletion
  delete_node->set_name("__delete_" + node_index.to_string());
  delete_node->set_op_type("delete_node");
  delete_node->set_domain(GRAPH_META_DOMAIN);

  // No inputs for delete meta node
  // (inputs are left empty as specified)

  // Add outputs matching the deleted node's outputs
  *delete_node->mutable_output() = node_proto.output();
  // Add a special attribute to identify which node to delete
  auto* delete_attr = delete_node->add_attribute();
  delete_attr->set_name("target_node_index");
  delete_attr->set_type(morphizen_onnx::AttributeProto::INT);
  delete_attr->set_i(static_cast<int64_t>(node_index.get_index()));
  { // update nodes_
    auto meta_node_index =
        NodeIndex((unsigned int)(nodes_.size()), get_graph_id());
    nodes_.emplace_back(Node::create_node(meta_node_index, {}, {}));
  }
}
void StagingGraph::add_initialized_tensor(
    const morphizen_onnx::TensorProto& tensor) {
  // Add the tensor to the staging graph
  auto name = tensor.name();
  if (name.empty()) {
    // throw an error if the tensor name is empty
    throw std::invalid_argument(
        "TensorProto name cannot be empty when adding to graph");
  }
  log_add_initialized_tensor(tensor);
  auto it = initializers_map_.find(name);
  if (it != initializers_map_.end()) {
    // sorry, for efficiency purpose, we const cast it and move the data from
    // tensor into the graph this is not a good design, but we need to do it for
    // now
    LOG(WARNING) << "Replacing existing initializer tensor: " << name;
    const_cast<morphizen_onnx::TensorProto&>(*it->second)
        .Swap(const_cast<morphizen_onnx::TensorProto*>(&tensor));
  } else {
    auto* new_tensor = graph_proto_.add_initializer();
    new_tensor->Swap(const_cast<morphizen_onnx::TensorProto*>(&tensor));
    // Store the tensor in the staging graph's initializers_map_ for quick
    // access
    auto index = static_cast<unsigned int>(graph_proto_.initializer_size() - 1);
    node_args_map_[name] = NodeArgIndex::initializer(index, get_graph_id());
    initializers_map_[name] = new_tensor;
  }
  // for historical reasons, we do not
  // return anything here, but we should return a NodeArgIndex.
  return;
}

void StagingGraph::remove_initialized_tensor(unsigned int index,
                                             const std::string& tensor_name) {

  log_remove_initialized_tensor(index, tensor_name);
  // create a new meta node to mark the removal
  auto* remove_initialized_tensor = graph_proto_.add_node();
  remove_initialized_tensor->set_name("__remove_initializer_" + tensor_name);
  remove_initialized_tensor->set_op_type("remove_initializer");
  remove_initialized_tensor->set_domain(GRAPH_META_DOMAIN);
  // add attribute["index"] = index
  auto* attr = remove_initialized_tensor->add_attribute();
  attr->set_name("index");
  attr->set_type(morphizen_onnx::AttributeProto::INT);
  attr->set_i(static_cast<int64_t>(index));
  // add attribute["name"] = name
  auto* name_attr = remove_initialized_tensor->add_attribute();
  name_attr->set_name("name");
  name_attr->set_type(morphizen_onnx::AttributeProto::STRING);
  name_attr->set_s(tensor_name);
  // add output as the removed initializer
  // add a new nodes
  auto meta_node_index =
      NodeIndex(static_cast<unsigned int>(nodes_.size()), get_graph_id());
  nodes_.emplace_back(Node::create_node(meta_node_index, {}, {}));
}

void StagingGraph::log_add_initialized_tensor(
    const morphizen_onnx::TensorProto& tensor) {
  auto& tensor_name = tensor.name();
  // Log the addition of an initialized tensor
  std::ostringstream log_stream;
  log_stream << "Adding initialized tensor: " << tensor_name
             << " to staging graph ID: " << get_graph_id().to_string();

  // Add element type information
  log_stream << " | element_type: " << tensor.data_type();

  // Add shape information
  if (tensor.dims_size() > 0) {
    log_stream << " | shape: [";
    for (int i = 0; i < tensor.dims_size(); ++i) {
      if (i > 0)
        log_stream << ", ";
      log_stream << tensor.dims(i);
    }
    log_stream << "]";
  } else {
    log_stream << " | shape: scalar";
  }

  // Add data size information
  if (tensor.has_raw_data()) {
    log_stream << " | raw_data_size: " << tensor.raw_data().size() << " bytes";
  } else {
    // Count elements based on specific data type arrays
    int element_count = 0;
    if (tensor.float_data_size() > 0)
      element_count = tensor.float_data_size();
    else if (tensor.int32_data_size() > 0)
      element_count = tensor.int32_data_size();
    else if (tensor.int64_data_size() > 0)
      element_count = tensor.int64_data_size();
    else if (tensor.double_data_size() > 0)
      element_count = tensor.double_data_size();
    else if (tensor.uint64_data_size() > 0)
      element_count = tensor.uint64_data_size();

    if (element_count > 0) {
      log_stream << " | element_count: " << element_count;
    }
  }

  log_messages_.push_back(log_stream.str());
}

void StagingGraph::log_remove_initialized_tensor(
    unsigned int index, const std::string& tensor_name) {
  // Log the removal of an initialized tensor
  std::ostringstream log_stream;
  log_stream << "Removing initialized tensor: " << tensor_name
             << " from staging graph ID: " << get_graph_id().to_string();
  auto& tensor = main_graph_.get_graph_proto().initializer(index);
  // Add element type information
  log_stream << " | element_type: " << tensor.data_type();

  // Add shape information
  if (tensor.dims_size() > 0) {
    log_stream << " | shape: [";
    for (int i = 0; i < tensor.dims_size(); ++i) {
      if (i > 0)
        log_stream << ", ";
      log_stream << tensor.dims(i);
    }
    log_stream << "]";
  } else {
    log_stream << " | shape: scalar";
  }

  // Add data size information
  if (tensor.has_raw_data()) {
    log_stream << " | raw_data_size: " << tensor.raw_data().size() << " bytes";
  } else {
    // Count elements based on specific data type arrays
    int element_count = 0;
    if (tensor.float_data_size() > 0)
      element_count = tensor.float_data_size();
    else if (tensor.int32_data_size() > 0)
      element_count = tensor.int32_data_size();
    else if (tensor.int64_data_size() > 0)
      element_count = tensor.int64_data_size();
    else if (tensor.double_data_size() > 0)
      element_count = tensor.double_data_size();
    else if (tensor.uint64_data_size() > 0)
      element_count = tensor.uint64_data_size();

    if (element_count > 0) {
      log_stream << " | element_count: " << element_count;
    }
  }
  log_messages_.push_back(log_stream.str());
}

void StagingGraph::set_inputs(gsl::span<NodeArgIndex> inputs) {
  // Log the operation
  log_set_inputs(inputs);

  graph_proto_.mutable_input()->Clear();
  // remove all node arg index from node_args_map_ if its name
  // is one of inputs.
  for (auto it = node_args_map_.begin(); it != node_args_map_.end();) {
    if (std::find_if(inputs.begin(), inputs.end(),
                     [&it](const NodeArgIndex& input) {
                       return it->first == *input.get_name_unsafe();
                     }) != inputs.end()) {
      // Remove the node arg index from the map
      MY_LOG(1) << "Removing NodeArgIndex: " << it->first
                << " from staging_graph_ node_args_map_";
      it = node_args_map_.erase(it);
    } else {
      ++it;
    }
  }
  unsigned int input_index = 0;
  for (auto input : inputs) {
    input_index = input_index + 1;
    // Ensure the input is valid
    if (!input.is_valid()) {
      LOG(FATAL) << "Invalid NodeArgIndex provided for input: "
                 << input.to_string();
      continue;
    }
    auto* name = input.get_name_unsafe();
    if (name == nullptr || name->empty()) {
      LOG(FATAL) << "NodeArgIndex name cannot be empty when setting inputs";
      continue;
    }
    auto gi = graph_proto_.add_input();
    *gi = input.get_value_info();
    node_args_map_[*name] =
        NodeArgIndex::graph_input(input_index - 1, get_graph_id());
  }
}
void StagingGraph::set_outputs(gsl::span<const NodeArgIndex> outputs) {
  // Log the operation
  log_set_outputs(outputs);

  // Clear existing outputs
  graph_proto_.mutable_output()->Clear();
  // Remove all node arg index from node_args_map_ if its name
  // is one of outputs.
  for (auto it = node_args_map_.begin(); it != node_args_map_.end();) {
    if (std::find_if(outputs.begin(), outputs.end(),
                     [&it](const NodeArgIndex& output) {
                       return it->first == *output.get_name_unsafe();
                     }) != outputs.end()) {
      // Remove the node arg index from the map
      MY_LOG(1) << "Removing NodeArgIndex: " << it->first
                << " from staging_graph_ node_args_map_";
      it = node_args_map_.erase(it);
    } else {
      ++it;
    }
  }
  // Iterate through the provided outputs
  unsigned int output_index = 0;
  for (const auto& output : outputs) {
    output_index = output_index + 1;
    CHECK(output.is_valid())
        << "Invalid NodeArgIndex provided for output: " << output.to_string();
    const auto* name = output.get_name_unsafe();
    CHECK(name != nullptr && !name->empty())
        << "NodeArgIndex name cannot be empty when setting outputs: "
        << output.to_string();
    auto* go = graph_proto_.add_output();
    *go = output.get_value_info();
    auto graph_id = get_graph_id();
    auto new_node_arg_index =
        NodeArgIndex::graph_output(output_index - 1, graph_id);
    node_args_map_[*name] = new_node_arg_index;
    auto output_graph_id = output.get_graph_id();
    if (output_graph_id.is_staging()) {
      // Graph::get_outputs will returns `new_node_arg_index`, producer_map_
      // must be updated properly, otherwise we cannot get a complete graph from
      // graph outputs to graph inputs. However it is possible to create a node
      // arg first and then set it as a graph output, so that we cannot get the
      // output node until Graph::add_node
      auto v = output.get_producer_node();
      if (v.is_valid()) {
        producer_map_[new_node_arg_index] = v;
      }
    } else {
      producer_map_[output] =
          output.get_producer_node(); // to redirect node arg from original
                                      // graph into the staging graph.
      producer_map_[new_node_arg_index] =
          output.get_producer_node(); // for a fresh
                                      // get_all_nodes_topolocial_order.
    }
  }
}
void StagingGraph::log_set_inputs(gsl::span<const NodeArgIndex> inputs) {
  std::ostringstream oss;
  oss << "Setting inputs for graph ID: " << get_graph_id().to_string()
      << " with inputs: [";
  for (auto input : inputs) {
    // Ensure the input is valid
    CHECK(input.is_valid())
        << "Invalid NodeArgIndex provided for input: " << input.to_string();
    auto* name = input.get_name_unsafe();
    CHECK(name != nullptr && !name->empty())
        << "NodeArgIndex name cannot be empty when setting inputs: "
        << input.to_string();
    oss << *name << " ";
  }
  oss << "]";
  log_messages_.push_back(oss.str());
}

void StagingGraph::log_set_outputs(gsl::span<const NodeArgIndex> outputs) {
  std::ostringstream oss;
  oss << "Setting outputs for graph ID: " << get_graph_id().to_string()
      << " with outputs: [";
  for (const auto& output : outputs) {
    // Ensure the output is valid
    CHECK(output.is_valid())
        << "Invalid NodeArgIndex provided for output: " << output.to_string();
    const auto* name = output.get_name_unsafe();
    CHECK(name != nullptr && !name->empty())
        << "NodeArgIndex name cannot be empty when setting outputs: "
        << output.to_string();
    oss << *name << " ";
  }
  oss << "]";
  log_messages_.push_back(oss.str());
}
void StagingGraph::log_remove_node(
    const NodeIndex& node_index, const morphizen_onnx::NodeProto& node_proto) {
  // Log the deletion operation
  std::ostringstream log_stream;
  log_stream << "Marking node for deletion: " << node_index.to_string()
             << " with "
             << " op_type: " << node_proto.op_type()
             << " domain: " << node_proto.domain();
  log_stream << " inputs: [";
  for (const auto& input : node_proto.input()) {
    log_stream << input << " ";
  }
  log_stream << "], outputs: [";
  for (const auto& output : node_proto.output()) {
    log_stream << output << " ";
  }
  log_stream << "]";
  log_messages_.push_back(log_stream.str());
}
void StagingGraph::log_add_node(const std::string& name,
                                const std::string& op_type,
                                const std::string& description,
                                const std::string& domain,
                                const std::vector<NodeArgIndex>& input_args,
                                const std::vector<NodeArgIndex>& output_args) {
  // This function generates detailed log messages for node additions.
  // Sample outputs:
  // "Adding node: conv1 (op_type: Conv) with 2 inputs and 1 outputs in
  // domain: ai.onnx - First convolution layer | Inputs: [input_tensor,
  // conv1_weight] | Outputs: [conv1_output]" "Adding node: relu1 (op_type:
  // Relu) with 1 inputs and 1 outputs | Inputs: [conv1_output] | Outputs:
  // [relu1_output]" "Adding node: matmul (op_type: MatMul) with 2 inputs and
  // 1 outputs - Matrix multiplication | Inputs: [feature_vector,
  // weight_matrix] | Outputs: [matmul_result]"

  std::ostringstream log_stream;
  log_stream << "Adding node: " << name << " (op_type: " << op_type << ")";
  log_stream << " with " << input_args.size() << " inputs and "
             << output_args.size() << " outputs";

  if (!domain.empty()) {
    log_stream << " in domain: " << domain;
  }

  if (!description.empty()) {
    log_stream << " - " << description;
  }

  // Add detailed input information
  if (!input_args.empty()) {
    log_stream << " | Inputs: [";
    for (size_t i = 0; i < input_args.size(); ++i) {
      if (i > 0)
        log_stream << ", ";
      std::string input_name = input_args[i].get_name();
      log_stream << input_name;
    }
    log_stream << "]";
  }

  // Add detailed output information
  if (!output_args.empty()) {
    log_stream << " | Outputs: [";
    for (size_t i = 0; i < output_args.size(); ++i) {
      if (i > 0)
        log_stream << ", ";
      std::string output_name = output_args[i].get_name();
      log_stream << output_name;
    }
    log_stream << "]";
  }

  log_messages_.push_back(log_stream.str());
}

NodeIndex StagingGraph::add_node(
    const std::string& name, const std::string& op_type,
    const std::string& description, const std::vector<NodeArgIndex>& input_args,
    const std::vector<NodeArgIndex>& output_args,
    ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*
        attributes,
    const std::string& domain) {
  // 3. Log the operation
  log_add_node(name, op_type, description, domain, input_args, output_args);

  // 4. Create and configure the node
  auto* new_node = graph_proto_.add_node();
  configure_node_proto(new_node, name, op_type, description, domain,
                       attributes);

  // 5. Process inputs and outputs
  process_input_arguments(new_node, input_args);
  process_output_arguments(new_node, output_args);

  // 6. Update nodes_
  auto output_args1 =
      output_args; // output_args1 might be changed by
                   // update_staging_nodes_structures to create a new node and
                   // replace constant initializers or graph input.
  auto node_index = update_staging_nodes_structures(input_args, output_args1);
  // no need to update name_args_map_ as it is already done in
  // node_arg_new or get_node_arg, i.e. all input node args and output node
  // args must be valid.
  update_producers_for_new_node(node_index, output_args);
  update_consumers_for_new_node(node_index, input_args);
  // 7. Return the new node index (calculated at the end)
  return node_index;
}
void StagingGraph::update_producers_for_new_node(
    NodeIndex node_index, gsl::span<const NodeArgIndex> output_node_args) {
  // we don't need to update consumers for output node arg.
  // if it is on the original graph, it means the original node arg is replaced.
  // If it is on the staging graph, it means the node arg is newly created, it
  // is not possible to have any consumer on the original graph. if any consumer
  // exists on the staging graph, i.e. the consumer node is created before the
  // producer node, the the consumer map is updated by
  // update_consumers_for_new_node.
  for (const auto& output_arg : output_node_args) {
    NodeIndex producer = producer_map_[output_arg];
    CHECK(!producer.is_valid())
        << "Overwriting existing producer for output node arg: "
        << output_arg.to_string()
        << " from node index: " << output_arg.to_string()
        << " to new node index: " << node_index.to_string();
    auto graph_id = node_index.get_graph_id();
    CHECK_EQ(graph_id.get_index(), get_graph_id().get_index())
        << "New node index graph ID mismatch: "
        << node_index.get_graph_id().to_string() << " vs "
        << graph_id.to_string();
    producer_map_[output_arg] = node_index;
    // potentially update graph outputs, otherwise Graph::resolve cannot get all
    // nodes.
    {
      auto name = output_arg.get_name_unsafe();
      CHECK(name != nullptr && !name->empty());
      auto graph_output_node_arg = this->node_args_map_.find(*name);
      if (graph_output_node_arg != this->node_args_map_.end()) {
        if (graph_output_node_arg->second.is_graph_output()) {
          producer_map_[graph_output_node_arg->second] = node_index;
        }
      }
    }
    // If not found, insert the new producer
    MY_LOG(1) << "Adding new producer for output node arg: "
              << output_arg.to_string()
              << " with node index: " << node_index.to_string();
  }
}

void StagingGraph::update_consumers_for_new_node(
    NodeIndex node_index, gsl::span<const NodeArgIndex> input_node_args) {
  (void)node_index;
  (void)input_node_args;
  // Update consumers for the new node
  // TODO: please review the implementation.
  /* for (const node_index& input_arg : input_node_args) {
    auto it = consumer_map_.find(input_arg);
    if (it != consumer_map_.end()) {
      // If the input_arg already has consumers, add the new node index
      it->second.push_back(node_index);
      MY_LOG(1) << "Adding consumer node index: " << node_index.to_string()
                << " for input node arg: " << input_arg.to_string();
    } else {
      // If not found, create a new consumer entry
      consumer_map_[input_arg] = {node_index};
      MY_LOG(1) << "Creating new consumer entry for input node arg: "
                << input_arg.to_string()
                << " with consumer node index: " << node_index.to_string();
    }
  }*/
}
void StagingGraph::configure_node_proto(
    morphizen_onnx::NodeProto* new_node, const std::string& name,
    const std::string& op_type, const std::string& description,
    const std::string& domain,
    ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*
        attributes) const {
  new_node->set_name(name);
  new_node->set_op_type(op_type);

  if (!description.empty()) {
    new_node->set_doc_string(description);
  }

  if (!domain.empty()) {
    new_node->set_domain(domain);
  }
  if (attributes != nullptr) {
    new_node->mutable_attribute()->Swap(attributes);
  }
}
void StagingGraph::process_input_arguments(
    morphizen_onnx::NodeProto* new_node,
    const std::vector<NodeArgIndex>& input_args) {
  std::vector<NodeArgIndex> input_node_arg_indices;
  for (const auto& input_arg : input_args) {
    // input_arg might be on the original graph or the staging graph,
    // or it is an optional input
    std::string input_name;
    // invalid input_arg means optional input, so we check if it is valid
    if (input_arg.is_valid()) {
      auto unsafe_name = input_arg.get_name_unsafe();
      // if input arg is a valid input argument, it must have a name
      // and it must not be empty
      CHECK(unsafe_name != nullptr && !unsafe_name->empty())
          << "Input argument name cannot be empty";
      input_name = *unsafe_name;
    } else {
      // it is an optional input, ONNX specification uses empty string for
      // optional inputs
      input_name = "";
    }
    new_node->add_input(input_name);
  }
  return;
}
void StagingGraph::process_output_arguments(
    morphizen_onnx::NodeProto* new_node,
    const std::vector<NodeArgIndex>& output_args) {
  std::vector<NodeArgIndex> output_node_arg_indices;

  for (const auto& output_arg : output_args) {
    // Resolve NodeArgIndex to actual name
    std::string output_name = "";
    // If the output_arg is valid, it must have a name
    // and it must not be empty.
    if (output_arg.is_valid()) {
      auto unsafe_name = output_arg.get_name_unsafe();
      // if output arg is a valid output argument, it must have a name
      // and it must not be empty
      CHECK(unsafe_name != nullptr && !unsafe_name->empty())
          << "Output argument name cannot be empty";
      output_name = *unsafe_name;
    } else {
      // If the output_arg is invalid, it is an optional output
      // We can either skip it or assign a default name
      output_name = "";
    }

    new_node->add_output(output_name);
  }
}

NodeIndex StagingGraph::update_staging_nodes_structures(
    const std::vector<NodeArgIndex>& input_node_arg_indices,
    std::vector<NodeArgIndex>& output_node_arg_indices) {
  for (auto& output_node_arg_index : output_node_arg_indices) {
    CHECK(output_node_arg_index.is_valid_node_output() ||
          output_node_arg_index.is_valid_graph_output() ||
          output_node_arg_index.is_initializer())
        // it is OK to override a
        // constant initializer, for
        // example, convert a constant
        // initializer to Constant Node.
        << "Output NodeArgIndex must be a valid node output or graph output"
        << " but got: " << output_node_arg_index.to_string();
    auto output_name = output_node_arg_index.get_name_unsafe();
    CHECK(output_name != nullptr && !output_name->empty())
        << "Output argument name cannot be empty";
    auto ouput_node_arg_graph_id = output_node_arg_index.get_graph_id();
    CHECK_EQ(ouput_node_arg_graph_id.get_index(), get_graph_id().get_index())
        << "Output NodeArgIndex graph ID mismatch: "
        << ouput_node_arg_graph_id.to_string() << " vs "
        << get_graph_id().to_string();
    if (ouput_node_arg_graph_id.is_staging()) {
      // for staging graph, we must create a node arg before adding a node.
      CHECK(node_args_map_.count(*output_name) != 0)
          << "Output NodeArgIndex must be created before adding a node: "
          << *output_name;
    } else {
      // if the output_node_arg_index is on the original graph,
      // we must create a node arg for it, so that we can connect the
      // original graph to the staging graph.
      //
      // in onnxruntime implementation, the node_args_map_ is used to
      // map node arg name to node arg index, so we can use it to connect
      // the original graph to the staging graph.
      //
      // in this case, output_node_arg_index is a valid graph output,
      // so we need to create a node arg for it.
      CHECK(node_args_map_.count(*output_name) == 0)
          << "Output NodeArgIndex already exists: " << *output_name;
    }
    // so it is possible that staging_graph->node_args_maps_ holds a node arg
    // on the original graph.
    //
    // then node_arg_index.get_producer could return a node index on the staging
    // graph, in this way, we connect the original graph to the staging graph
    if (output_node_arg_index.is_initializer()) {
      // convert initializer to value info
      // when convert initializer to a const op.
      auto initializer =
          output_node_arg_index.get_const_data_as_tensor(this->main_graph_);
      auto new_value_info = graph_proto_.add_value_info();
      new_value_info->set_name(initializer->name());
      new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
          initializer->data_type());
      auto shape = new_value_info->mutable_type()
                       ->mutable_tensor_type()
                       ->mutable_shape();
      for (auto dim : initializer->dims()) {
        shape->mutable_dim()->Add()->set_dim_value(dim);
      }
      // replace initializer node arg to node output node arg
    }
    if (output_node_arg_index.is_node_output() ||
        output_node_arg_index.is_graph_output()) {
      node_args_map_[*output_name] = output_node_arg_index;
    } else if (output_node_arg_index.is_initializer()) {
      // NOTE: output_node_arg_index is a reference, the output_node_arg_indices
      // is changed.
      // output_node_arg_index = output_node_arg_index;
      auto insert_results = node_args_map_[*output_name] =
          output_node_arg_index;
    } else {
      LOG(FATAL) << "TODO: create a new node to replace graph inputs";
    }
  }
  auto node_index = NodeIndex((unsigned int)(nodes_.size()), get_graph_id());
  nodes_.emplace_back(Node::create_node(node_index, input_node_arg_indices,
                                        output_node_arg_indices));

  CHECK_EQ(nodes_.size(), (size_t)graph_proto_.node_size())
      << "Node count mismatch: " << nodes_.size() << " vs "
      << graph_proto_.node_size();
  return node_index;
}

NodeArgIndex StagingGraph::node_arg_new(const std::string& name,
                                        const std::vector<int64_t>* shape,
                                        int element_type) {
  // check if the name already exists in the staging graph
  auto it = node_args_map_.find(name);
  if (it != node_args_map_.end()) {
    // in onnxruntime implementation,
    // add_initialized_tensor updates name_to_initial_tensor_
    // InitializedTensorSet name_to_initial_tensor_
    // and node_arg_new_ update node_args_
    // std::unordered_map<std::string, std::unique_ptr<NodeArg>> node_args_;
    // so it is OK if we create a node arg more than once for constant
    // intializers.
    if (it->second.is_initializer()) {
      auto shape2 = it->second.get_shape_i64_unsafe();
      CHECK(shape != nullptr);
      CHECK(shape2 != nullptr);
      // check shape and shape2 are same
      CHECK_EQ(shape->size(), shape2->size())
          << "Shape mismatch for existing initializer: " + name;
      auto size = shape->size();
      for (auto i = 0u; i < size; ++i) {
        CHECK_EQ(shape->at(i), shape2->at(i))
            << "Shape mismatch at dimension " << i
            << " for initializer: " << name;
      }
      // check element type is same
      CHECK_EQ(it->second.get_element_type(), element_type)
          << "Element type mismatch for existing initializer: " + name;
      return it->second;
    }
    throw std::invalid_argument("NodeArg name already exists");
  }
  log_node_arg_new(name, shape, element_type);
  // Create a new value_info entry in staging graph
  auto* value_info = graph_proto_.add_value_info();
  value_info->set_name(name);

  // Set up the type information if provided
  if (shape != nullptr) {
    auto* type = value_info->mutable_type();
    auto* tensor_type = type->mutable_tensor_type();
    tensor_type->set_elem_type(element_type);

    auto* tensor_shape = tensor_type->mutable_shape();
    for (int64_t dim : *shape) {
      auto* dimension = tensor_shape->add_dim();
      dimension->set_dim_value(dim);
    }
  }

  // Calculate the index for the new value_info entry
  unsigned int value_info_index =
      static_cast<unsigned int>(graph_proto_.value_info_size() - 1);

  // Create and return a NodeArgIndex for the new node argument
  auto graph_output_index = get_graph_output_index(name);

  NodeArgIndex node_arg_index =
      graph_output_index >= 0
          ? NodeArgIndex::graph_output(
                static_cast<unsigned int>(graph_output_index), get_graph_id())
          : NodeArgIndex::node_output(value_info_index, get_graph_id());
  // Update the staging graph's node_args_map_
  node_args_map_[name] = node_arg_index;
  return node_arg_index;
}
void StagingGraph::log_node_arg_new(const std::string& name,
                                    const std::vector<int64_t>* shape,
                                    int element_type) {
  // Log the creation of a new NodeArg
  std::ostringstream log_stream;
  log_stream << "Creating new NodeArg: " << name
             << " with element_type: " << element_type;

  // Add shape information if provided
  if (shape != nullptr && !shape->empty()) {
    log_stream << " | shape: [";
    for (size_t i = 0; i < shape->size(); ++i) {
      if (i > 0)
        log_stream << ", ";
      log_stream << shape->at(i);
    }
    log_stream << "]";
  } else {
    log_stream << " | shape: scalar";
  }

  log_messages_.push_back(log_stream.str());
}
int StagingGraph::get_graph_output_index(const std::string& name) const {
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

} // namespace morphizen
