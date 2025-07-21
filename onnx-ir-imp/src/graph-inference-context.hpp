/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "./onnx-deps.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace morphizen {

/**
 * @brief MorphiZen-specific implementation of ONNX InferenceContext
 *
 * This class provides a concrete implementation of ONNX's InferenceContext
 * interface that's adapted for use within MorphiZen's shape inference system.
 * It follows the same design patterns as ONNX's InferenceContextImpl but uses
 * MorphiZen's type aliases.
 */
class GraphInferenceContextImpl : public morphizen_onnx::InferenceContext {
public:
  /**
   * @brief Construct a new Graph Inference Context Impl object
   *
   * @param node The node for which to create the inference context
   * @param valueTypesByName Map of value names to their type information
   * @param inputDataByName Map of input names to their tensor data (for
   * constants)
   * @param inputSparseDataByName Map of input names to their sparse tensor data
   * @param options Shape inference options
   * @param generatedShapeData Optional data propagation map
   * @param graphInferenceContext Optional graph-level inference context
   */
  GraphInferenceContextImpl(
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
          graphInferenceContext = nullptr);

  // InferenceContext interface implementation
  const morphizen_onnx::AttributeProto*
  getAttribute(const std::string& name) const override;
  size_t getNumInputs() const override;
  const morphizen_onnx::TypeProto* getInputType(size_t index) const override;
  bool hasInput(size_t index) const override;
  const morphizen_onnx::TensorProto* getInputData(size_t index) const override;
  const morphizen_onnx::SparseTensorProto*
  getInputSparseData(size_t index) const override;
  const morphizen_onnx::TensorShapeProto*
  getSymbolicInput(size_t index) const override;
  size_t getNumOutputs() const override;
  morphizen_onnx::TypeProto* getOutputType(size_t index) override;
  bool hasOutput(size_t index) override;
  morphizen_onnx::GraphInferencer*
  getGraphAttributeInferencer(const std::string& attribute_name) override;
  std::string getDisplayName() const override;

private:
  // Context data
  morphizen_onnx::shape_inference::GraphInferenceContext*
      graphInferenceContext_;
  morphizen_onnx::ShapeInferenceOptions options_;
  morphizen_onnx::NodeProto* node_;

  // Attribute access
  std::unordered_map<std::string, const morphizen_onnx::AttributeProto*>
      attributesByName_;
  std::unordered_map<std::string, morphizen_onnx::GraphProto*>
      graphProtoAttributesByName_;

  // Input/output type and data storage
  std::vector<const morphizen_onnx::TypeProto*> allInputTypes_;
  std::vector<morphizen_onnx::TypeProto> allOutputTypes_;
  std::vector<const morphizen_onnx::TensorProto*> allInputData_;
  std::vector<const morphizen_onnx::SparseTensorProto*> allInputSparseData_;
  std::vector<const morphizen_onnx::TensorShapeProto*> allShapeInputData_;

  // Graph attribute inferencers cache
  mutable std::unordered_map<std::string,
                             std::unique_ptr<morphizen_onnx::GraphInferencer>>
      graphAttributeInferencers_;
};

} // namespace morphizen
