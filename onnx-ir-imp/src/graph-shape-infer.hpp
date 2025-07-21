/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "./graph-inference-context.hpp"
#include "./onnx-deps.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace morphizen {

// Forward declarations
class Graph;

/**
 * @brief Shape inference utility class for ONNX graphs
 *
 * This class provides shape inference capabilities similar to ONNX's own
 * implementation but allows for customization and extension.
 */
class GraphShapeInfer {
public:
  /**
   * @brief Default constructor
   */
  GraphShapeInfer();

  /**
   * @brief Constructor that takes a Graph pointer for direct updates
   * @param graph The graph to perform shape inference on
   */
  explicit GraphShapeInfer(Graph* graph);

  /**
   * @brief Destructor
   */
  ~GraphShapeInfer();

  /**
   * @brief Perform shape inference on a graph
   * @param graph The graph to perform shape inference on
   * @return true if shape inference was successful, false otherwise
   */
  bool infer_shapes(Graph& graph);

  /**
   * @brief Perform shape inference on the graph passed to constructor
   * @return true if shape inference was successful, false otherwise
   */
  bool infer_shapes();

  /**
   * @brief Perform shape inference on a graph proto directly
   * @param graph_proto The graph proto to perform shape inference on
   * @param opset_imports The opset imports as a map from domain to version
   * @return true if shape inference was successful, false otherwise
   */
  bool infer_shapes(morphizen_onnx::GraphProto& graph_proto,
                    const std::unordered_map<std::string, int>& opset_imports);

  /**
   * @brief Get the last error message from shape inference
   * @return The last error message, empty if no error
   */
  const std::string& get_last_error() const;

  /**
   * @brief Enable or disable verbose logging during shape inference
   * @param verbose True to enable verbose logging, false to disable
   */
  void set_verbose(bool verbose);

  /**
   * @brief Set whether to check model for validity before shape inference
   * @param check True to enable model checking, false to disable
   */
  void set_check_model(bool check);

  /**
   * @brief Create a custom inference context for a node
   * @param node The node to create context for
   * @param valueTypesByName Map of value names to type information
   * @param inputDataByName Map of input names to tensor data
   * @param inputSparseDataByName Map of input names to sparse tensor data
   * @param options Shape inference options
   * @param generatedShapeData Optional data propagation map
   * @param graphInferenceContext Optional graph-level inference context
   * @return Unique pointer to the inference context
   */
  std::unique_ptr<GraphInferenceContextImpl> createInferenceContext(
      morphizen_onnx::NodeProto& node,
      const std::unordered_map<std::string, morphizen_onnx::TypeProto*>&
          valueTypesByName,
      const std::unordered_map<std::string, const morphizen_onnx::TensorProto*>&
          inputDataByName,
      const std::unordered_map<std::string,
                               const morphizen_onnx::SparseTensorProto*>&
          inputSparseDataByName,
      const morphizen_onnx::ShapeInferenceOptions& options,
      morphizen_onnx::shape_inference::DataValueMap* generatedShapeData =
          nullptr,
      morphizen_onnx::shape_inference::GraphInferenceContext*
          graphInferenceContext = nullptr) const;

  /**
   * @brief Demonstrate custom shape inference using ONNX InferenceContext
   * interface
   * @param graph_proto The graph to process
   * @return True if demonstration was successful, false otherwise
   */
  bool custom_infer_shapes(morphizen_onnx::GraphProto& graph_proto);

private:
  /**
   * @brief Validate the graph proto before shape inference
   * @param graph_proto The graph proto to validate
   * @return true if valid, false otherwise
   */
  bool validate_graph_proto(const morphizen_onnx::GraphProto& graph_proto);

  /**
   * @brief Setup symbol table for shape inference
   * @return A symbol table implementation for shape inference
   */
  std::unique_ptr<morphizen_onnx::shape_inference::SymbolTableImpl>
  create_symbol_table();

  /**
   * @brief Log shape inference progress
   * @param message The message to log
   */
  void log_progress(const std::string& message);

private:
  std::string last_error_; ///< Last error message
  bool verbose_;           ///< Enable verbose logging
  bool check_model_;       ///< Enable model validation before inference
  Graph* target_graph_; ///< The graph to perform shape inference on (optional)
};

} // namespace morphizen
