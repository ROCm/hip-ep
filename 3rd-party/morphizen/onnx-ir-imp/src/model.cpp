/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./model.hpp"
#include "./graph.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <algorithm>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <stdexcept>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_ORT_MODEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ORT_MODEL) >= n)
namespace morphizen {

std::unique_ptr<Model>
Model::create_model(morphizen_onnx::ModelProto &&model_proto) {
  return std::make_unique<Model>(PrivateTag{}, std::move(model_proto));
}

std::unique_ptr<Model> Model::load(const std::string &file) {
  auto model_proto = morphizen_onnx::ModelProto();

  // First, try to parse as binary protobuf
  {
    std::ifstream input_stream(file, std::ios::binary);
    if (!input_stream.is_open()) {
      LOG(ERROR) << "Failed to open file: " << file;
      return nullptr;
    }

    if (model_proto.ParseFromIstream(&input_stream)) {
      // Successfully parsed as binary format
      input_stream.close();
    } else {
      // Binary parsing failed, try text format
      MY_LOG(1) << "Failed to parse as binary ONNX, trying text format: "
                << file;
      input_stream.close();

      // Re-open file in text mode and read entire content
      std::ifstream text_stream(file, std::ios::in);
      if (!text_stream.is_open()) {
        LOG(ERROR) << "Failed to re-open file for text parsing: " << file;
        return nullptr;
      }

      std::string file_content((std::istreambuf_iterator<char>(text_stream)),
                               std::istreambuf_iterator<char>());
      text_stream.close();

      if (file_content.empty()) {
        LOG(ERROR) << "File is empty: " << file;
        return nullptr;
      }

      // Try parsing as text format
      if (!google::protobuf::TextFormat::ParseFromString(file_content,
                                                         &model_proto)) {
        LOG(ERROR) << "Failed to parse ONNX model from file (both binary and "
                      "text formats): "
                   << file;
        return nullptr;
      }
    }
  }

  // Create and validate the model
  auto model = std::make_unique<Model>(PrivateTag{}, std::move(model_proto));

  // Set the model path
  model->set_model_path(file);

  if (!model->is_valid()) {
    LOG(ERROR) << "Invalid model loaded from file: " << file;
    auto errors = model->get_validation_errors();
    for (const auto &error : errors) {
      LOG(ERROR) << "Validation error: " << error;
    }
    return nullptr;
  }

  return model;
}

Model::Model(PrivateTag /*tag*/, morphizen_onnx::ModelProto &&model_proto)
    : model_proto_(std::move(model_proto)) {
  initialize();
}

Model::Model(Model &&other) noexcept
    : model_proto_(std::move(other.model_proto_)),
      main_graph_(std::move(other.main_graph_)),
      opset_imports_(std::move(other.opset_imports_)),
      metadata_props_(std::move(other.metadata_props_)),
      model_path_(std::move(other.model_path_)),
      subgraphs_(std::move(other.subgraphs_)) {}

Model &Model::operator=(Model &&other) noexcept {
  if (this != &other) {
    model_proto_ = std::move(other.model_proto_);
    main_graph_ = std::move(other.main_graph_);
    opset_imports_ = std::move(other.opset_imports_);
    metadata_props_ = std::move(other.metadata_props_);
    model_path_ = std::move(other.model_path_);
    subgraphs_ = std::move(other.subgraphs_);
  }
  return *this;
}

void Model::initialize() {
  // Clear existing caches
  main_graph_.reset();
  opset_imports_.clear();
  metadata_props_.clear();
  subgraphs_.clear();

  // Build opset imports cache
  for (int i = 0; i < model_proto_.opset_import_size(); ++i) {
    const auto &opset = model_proto_.opset_import(i);
    opset_imports_[opset.domain()] = (int)opset.version();
  }

  // Build metadata properties cache
  for (int i = 0; i < model_proto_.metadata_props_size(); ++i) {
    const auto &prop = model_proto_.metadata_props(i);
    metadata_props_[prop.key()] = prop.value();
  }
}

const std::unordered_map<std::string, int> &Model::get_opset_imports() const {
  return opset_imports_;
}
std::unordered_map<std::string, int> &Model::get_opset_imports() {
  return opset_imports_;
}

int64_t Model::get_opset_version(const std::string &domain) const {
  auto it = opset_imports_.find(domain);
  return (it != opset_imports_.end()) ? it->second : -1;
}

bool Model::has_opset_import(const std::string &domain) const {
  return opset_imports_.find(domain) != opset_imports_.end();
}

const std::map<std::string, std::string> &Model::get_metadata_props() const {
  return metadata_props_;
}

std::string Model::get_metadata_prop(const std::string &key) const {
  auto it = metadata_props_.find(key);
  return (it != metadata_props_.end()) ? it->second : std::string();
}

bool Model::has_metadata_prop(const std::string &key) const {
  return metadata_props_.find(key) != metadata_props_.end();
}

void Model::set_metadata_prop(const std::string &key,
                              const std::string &value) {
  // Update the cache
  metadata_props_[key] = value;

  // Update the actual ModelProto
  // Find existing property or create new one
  for (int i = 0; i < model_proto_.metadata_props_size(); ++i) {
    auto *prop = model_proto_.mutable_metadata_props(i);
    if (prop->key() == key) {
      prop->set_value(value);
      return;
    }
  }

  // Property doesn't exist, add new one
  auto *new_prop = model_proto_.add_metadata_props();
  new_prop->set_key(key);
  new_prop->set_value(value);
}

bool Model::remove_metadata_prop(const std::string &key) {
  // Remove from cache
  auto cache_it = metadata_props_.find(key);
  if (cache_it == metadata_props_.end()) {
    return false; // Property didn't exist
  }
  metadata_props_.erase(cache_it);

  // Remove from ModelProto
  auto *metadata_props = model_proto_.mutable_metadata_props();
  for (int i = 0; i < metadata_props->size(); ++i) {
    if (metadata_props->Get(i).key() == key) {
      metadata_props->erase(metadata_props->begin() + i);
      return true;
    }
  }

  return false;
}

bool Model::is_valid() const {
  // Basic validation checks

  // Must have IR version
  if (model_proto_.ir_version() <= 0) {
    return false;
  }
  // Must have a graph
  if (!model_proto_.has_graph()) {
    return false;
  }

  // Basic graph validation - check if main graph has nodes
  const auto &main_graph = graph();
  if (main_graph.node_size() == 0) {
    return false;
  }

  // Must have at least one opset import
  if (opset_imports_.empty()) {
    return false;
  }

  // Producer name should not be empty (recommended)
  if (producer_name().empty()) {
    return false;
  }

  return true;
}

std::vector<std::string> Model::get_validation_errors() const {
  std::vector<std::string> errors;

  // Check IR version
  if (model_proto_.ir_version() <= 0) {
    errors.push_back("Invalid IR version: " +
                     std::to_string(model_proto_.ir_version()));
  }
  // Check graph presence
  if (!model_proto_.has_graph()) {
    errors.push_back("Model has no graph");
  } else {
    // Check if graph has nodes
    const auto &main_graph = graph();
    if (main_graph.node_size() == 0) {
      errors.push_back("Graph has no nodes");
    }
  }

  // Check opset imports
  if (opset_imports_.empty()) {
    errors.push_back("Model has no opset imports");
  }

  // Check producer name
  if (producer_name().empty()) {
    errors.push_back("Producer name is empty (recommended to set)");
  }

  // Check for main opset version
  if (!has_opset_import("ai.onnx") && !has_opset_import("")) {
    errors.push_back(
        "Model does not import main ONNX opset (ai.onnx or empty domain)");
  }

  return errors;
}

std::unique_ptr<Model> Model::clone(int64_t external_data_threshold) const {
  // Create a deep copy of the ModelProto
  morphizen_onnx::ModelProto cloned_proto;
  cloned_proto.CopyFrom(model_proto_);

  // Note: external_data_threshold is passed in but not currently used in the
  // basic clone This parameter could be used for future optimization to control
  // when to externalize large tensor data during cloning
  (void)external_data_threshold; // Suppress unused parameter warning

  // Create new model with the cloned proto
  auto cloned_model =
      std::make_unique<Model>(PrivateTag{}, std::move(cloned_proto));

  // Copy the model path
  cloned_model->set_model_path(model_path_);

  return cloned_model;
}

const morphizen::Graph &Model::main_graph() const {
  if (!model_proto_.has_graph()) {
    throw std::runtime_error("Model has no graph");
  }

  // Lazily initialize the main graph wrapper
  if (!main_graph_) {
    // We need to cast away const to create the Graph wrapper since
    // Graph::create_main_graph expects a non-const reference
    auto &non_const_proto =
        const_cast<morphizen_onnx::ModelProto &>(model_proto_);
    main_graph_ = morphizen::Graph::create_main_graph(non_const_proto, this);
  }

  return *main_graph_;
}

const morphizen_onnx::GraphProto &Model::graph() const {
  if (!model_proto_.has_graph()) {
    throw std::runtime_error("Model has no graph");
  }
  return model_proto_.graph();
}

GraphId Model::create_subgraph(const Graph &parent_graph,
                               morphizen_onnx::GraphProto &subgraph_proto) {
  auto graph_index = static_cast<int>(subgraphs_.size());
  if (0)
    std::cout << "subgraph is \n" << subgraph_proto.DebugString() << std::endl;
  auto subgraph =
      Graph::create_graph(subgraph_proto, this, graph_index, &parent_graph);
  auto ret = subgraph->get_graph_id();
  // Store the subgraph in our collection
  subgraphs_.push_back(std::move(subgraph));
  // Create and return a GraphId for the new subgraph
  return ret;
}

} // namespace morphizen
