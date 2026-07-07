/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./graph-id.hpp"
#include "./graph.hpp"
#include <filesystem>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>

namespace morphizen {
class Graph;
/**
 * @brief In-memory representation of ONNX ModelProto
 *
 * This class provides efficient operations on ONNX model structure,
 * including access to the graph, metadata, and opset information.
 * It wraps the ModelProto and provides convenient methods for
 * model inspection and manipulation.
 *
 * The Model class contains a Graph instance for efficient graph
 * operations while also providing model-level functionality.
 */
class Model {
private:
  // === Private Tag for Constructor ===

  /**
   * @brief Private tag to prevent direct construction
   */
  struct PrivateTag {};

public: /**
         * @brief Create a Model instance from ONNX ModelProto
         * @param model_proto The ONNX ModelProto to take ownership of
         * @return Unique pointer to Model instance
         */
  static std::unique_ptr<Model>
  create_model(morphizen_onnx::ModelProto &&model_proto);

  /**
   * @brief Load a Model from file
   * @param file Path to the ONNX model file
   * @return Unique pointer to Model instance, or nullptr if loading fails
   */
  static std::unique_ptr<Model> load(const std::string &file);

  /**
   * @brief Construct from ONNX ModelProto using move semantics (private)
   * @param tag Private tag to prevent direct construction
   * @param model_proto The ONNX ModelProto to take ownership of
   */
  explicit Model(PrivateTag tag, morphizen_onnx::ModelProto &&model_proto);

  /**
   * @brief Deleted copy constructor - Model is non-copyable
   */
  Model(const Model &other) = delete;

  /**
   * @brief Deleted copy assignment operator - Model is non-copyable
   */
  Model &operator=(const Model &other) = delete;

  /**
   * @brief Move constructor
   */
  Model(Model &&other) noexcept;

  /**
   * @brief Move assignment operator
   */
  Model &operator=(Model &&other) noexcept;

  /**
   * @brief Get the underlying ModelProto
   */
  const morphizen_onnx::ModelProto &model_proto() const { return model_proto_; }

  // === Model Information ===

  /**
   * @brief Get model IR version
   */
  int64_t ir_version() const { return model_proto_.ir_version(); }

  /**
   * @brief Get producer name
   */
  const std::string &producer_name() const {
    return model_proto_.producer_name();
  }

  /**
   * @brief Get producer version
   */
  const std::string &producer_version() const {
    return model_proto_.producer_version();
  }

  /**
   * @brief Get model domain
   */
  const std::string &domain() const { return model_proto_.domain(); }

  /**
   * @brief Get model version
   */
  int64_t model_version() const { return model_proto_.model_version(); }
  /**
   * @brief Get model documentation string
   */
  const std::string &doc_string() const { return model_proto_.doc_string(); }

  /**
   * @brief Get the file path this model was loaded from
   * @return The filesystem path (empty if not loaded from file)
   */
  const std::filesystem::path &get_model_path() const { return model_path_; }

  /**
   * @brief Set the model path
   * @param path The filesystem path to set
   */
  void set_model_path(const std::filesystem::path &path) { model_path_ = path; }

  // === Graph Access ===
  /**
   * @brief Get the main graph as morphizen::Graph   * @return Reference to the
   * main Graph
   */
  const morphizen::Graph &main_graph() const;

  /**
   * @brief Get the main graph proto (for compatibility)
   * @return Reference to the main GraphProto
   */
  const morphizen_onnx::GraphProto &graph() const;

  /**
   * @brief Create a new subgraph from a parent graph
   * @param parent_graph The parent graph to create a subgraph from
   * @return GraphId of the newly created subgraph
   */
  GraphId create_subgraph(const Graph &parent_graph,
                          morphizen_onnx::GraphProto &subgraph_proto);

  // === Opset Information ===
  /**
   * @brief Get opset imports as a map
   * @return Const reference to map from domain to version
   */
  const std::unordered_map<std::string, int> &get_opset_imports() const;
  std::unordered_map<std::string, int> &get_opset_imports();

  /**
   * @brief Get opset version for a specific domain
   * @param domain Domain name (default: "ai.onnx")
   * @return Opset version, or -1 if not found
   */
  int64_t get_opset_version(const std::string &domain = "ai.onnx") const;

  /**
   * @brief Check if model has opset import for domain
   * @param domain Domain name
   * @return true if domain is in opset imports
   */
  bool has_opset_import(const std::string &domain) const;

  // === Metadata ===
  /**
   * @brief Get metadata properties as a map
   * @return Const reference to map from property name to value
   */
  const std::map<std::string, std::string> &get_metadata_props() const;

  /**
   * @brief Get metadata property value
   * @param key Property name
   * @return Property value, or empty string if not found
   */
  std::string get_metadata_prop(const std::string &key) const;

  /**
   * @brief Set metadata property value
   * @param key Property name
   * @param value Property value
   */
  void set_metadata_prop(const std::string &key, const std::string &value);

  /**
   * @brief Remove metadata property
   * @param key Property name
   * @return true if property was removed, false if it didn't exist
   */
  bool remove_metadata_prop(const std::string &key);

  /**
   * @brief Check if metadata property exists
   * @param key Property name
   * @return true if property exists
   */
  bool has_metadata_prop(const std::string &key) const;

  /**
   * @brief Create a deep copy of this model
   * @param external_data_threshold Threshold for external data (default: 64)
   * @return Unique pointer to a new Model instance that is a deep copy
   */
  std::unique_ptr<Model> clone(int64_t external_data_threshold = 64) const;

  // === Model Validation ===

  /**
   * @brief Check if model appears to be valid (basic checks)
   * @return true if model passes basic validation
   */
  bool is_valid() const;

  /**
   * @brief Get model validation errors (if any)
   * @return Vector of error messages
   */
  std::vector<std::string> get_validation_errors() const;

private:
  // === Private Methods ===

  /**
   * @brief Initialize the model (build caches, etc.)
   */
  void initialize(); // === Private Data Members ===

  /// The underlying ONNX ModelProto
  morphizen_onnx::ModelProto model_proto_;

  /// The main graph wrapper (lazily initialized)
  mutable std::unique_ptr<morphizen::Graph> main_graph_;

  /// Cached opset imports for fast lookup
  std::unordered_map<std::string, int> opset_imports_;
  /// Cached metadata properties for fast lookup
  std::map<std::string, std::string> metadata_props_;
  /// The file path this model was loaded from (if any)
  std::filesystem::path model_path_;
  std::vector<std::unique_ptr<morphizen::Graph>> subgraphs_;
};

} // namespace morphizen
