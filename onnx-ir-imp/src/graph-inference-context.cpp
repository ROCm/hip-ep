/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./graph-inference-context.hpp"
#include <glog/logging.h>
#include <onnx/shape_inference/implementation.h>

namespace morphizen {

GraphInferenceContextImpl::GraphInferenceContextImpl(
    morphizen_onnx::NodeProto& node,
    const std::unordered_map<std::string, morphizen_onnx::TypeProto*>&
        valueTypesByName,
    const std::unordered_map<std::string, const morphizen_onnx::TensorProto*>&
        inputDataByName,
    const std::unordered_map<std::string,
                             const morphizen_onnx::SparseTensorProto*>&
        inputSparseDataByName,
    const morphizen_onnx::ShapeInferenceOptions& options,
    morphizen_onnx::shape_inference::DataValueMap* generatedShapeData,
    morphizen_onnx::shape_inference::GraphInferenceContext*
        graphInferenceContext)
    : graphInferenceContext_(graphInferenceContext), options_(options),
      node_(&node) {

  // Build attribute maps (adapted from ONNX InferenceContextImpl)
  for (auto& attr : *node.mutable_attribute()) {
    attributesByName_[attr.name()] = &attr;
    if (attr.has_g()) {
      // need a mutable GraphProto to run inferencing on this attribute
      graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
    }
  }

  // Build input type and data maps (adapted from ONNX InferenceContextImpl)
  for (const auto& input : node.input()) {
    // Find input type
    auto valueTypesIter = valueTypesByName.find(input);
    if (valueTypesIter != valueTypesByName.end()) {
      allInputTypes_.push_back(valueTypesIter->second);
    } else {
      allInputTypes_.push_back(nullptr);
    }

    // input data can be in 1 of the 3 containers (adapted from ONNX)
    // inputDataByName - this is when input is TensorProto
    // inputSparseDataByName - this is when input is SparseTensorProto
    // generatedShapeData - this is when input was generated as part of partial
    // data propagation
    const auto inputDataIter = inputDataByName.find(input);
    if (inputDataIter != inputDataByName.cend()) {
      allInputData_.push_back(inputDataIter->second);
      allInputSparseData_.push_back(nullptr);
      allShapeInputData_.push_back(nullptr);
    } else {
      allInputData_.push_back(nullptr);
      const auto inputSparseDataIter = inputSparseDataByName.find(input);
      if (inputSparseDataIter != inputSparseDataByName.cend()) {
        allInputSparseData_.push_back(inputSparseDataIter->second);
        allShapeInputData_.push_back(nullptr);
      } else {
        allInputSparseData_.push_back(nullptr);
        if (generatedShapeData != nullptr) {
          const auto inputShapeDataIter = generatedShapeData->find(input);
          if (inputShapeDataIter != generatedShapeData->cend()) {
            allShapeInputData_.push_back(&inputShapeDataIter->second);
          } else {
            allShapeInputData_.push_back(nullptr);
          }
        } else {
          allShapeInputData_.push_back(nullptr);
        }
      }
    }
  }

  // Initialize output types
  allOutputTypes_.resize(node.output_size());
}

const morphizen_onnx::AttributeProto*
GraphInferenceContextImpl::getAttribute(const std::string& name) const {
  auto iter = attributesByName_.find(name);
  if (iter == attributesByName_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

size_t GraphInferenceContextImpl::getNumInputs() const {
  return allInputTypes_.size();
}

const morphizen_onnx::TypeProto*
GraphInferenceContextImpl::getInputType(size_t index) const {
  if (index >= allInputTypes_.size()) {
    ONNX_THROW("Input " + std::to_string(index) + " is out of bounds.");
  }
  return allInputTypes_[index];
}

bool GraphInferenceContextImpl::hasInput(size_t index) const {
  // The default implementation below is used for backward-compatibility
  // for implementations of InferenceContext that don't provide an explicit
  // implementation. This works for normal usage, but may be imprecise in
  // the edge-case where an input is supplied but has no known type.
  // However, inference-methods work only under the assumption that the
  // input-types of all inputs are known.
  return ((index < getNumInputs()) && (getInputType(index) != nullptr));
}

const morphizen_onnx::TensorProto*
GraphInferenceContextImpl::getInputData(size_t index) const {
  if (index >= allInputData_.size()) {
    ONNX_THROW("Input " + std::to_string(index) + " is out of bounds.");
  }
  return allInputData_[index];
}

const morphizen_onnx::SparseTensorProto*
GraphInferenceContextImpl::getInputSparseData(size_t index) const {
  if (index >= allInputSparseData_.size()) {
    ONNX_THROW("Input " + std::to_string(index) + " is out of bounds.");
  }
  return allInputSparseData_[index];
}

const morphizen_onnx::TensorShapeProto*
GraphInferenceContextImpl::getSymbolicInput(size_t index) const {
  if (index >= allShapeInputData_.size()) {
    ONNX_THROW("Input " + std::to_string(index) + " is out of bounds.");
  }
  return allShapeInputData_[index];
}

size_t GraphInferenceContextImpl::getNumOutputs() const {
  return allOutputTypes_.size();
}

morphizen_onnx::TypeProto*
GraphInferenceContextImpl::getOutputType(size_t index) {
  if (index >= allOutputTypes_.size()) {
    ONNX_THROW("Output " + std::to_string(index) + " is out of bounds.");
  }
  return &allOutputTypes_[index];
}

bool GraphInferenceContextImpl::hasOutput(size_t index) {
  return (index < getNumOutputs() && (getOutputType(index) != nullptr));
}

morphizen_onnx::GraphInferencer*
GraphInferenceContextImpl::getGraphAttributeInferencer(
    const std::string& attr_name) {
  if (!graphInferenceContext_) {
    fail_type_inference("GraphProto attribute inferencing is not enabled in "
                        "this GraphInferenceContextImpl instance.");
  }

  morphizen_onnx::GraphInferencer* inferencer = nullptr;

  auto entry = graphAttributeInferencers_.find(attr_name);
  if (entry == graphAttributeInferencers_.cend()) {
    // create GraphInferencer instance
    auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
    if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
      fail_type_inference("Attribute ", attr_name,
                          " does not contain a graph.");
    }

    auto new_inferencer =
        std::make_unique<morphizen_onnx::shape_inference::GraphInferencerImpl>(
            *attrNameToGraphProto->second, *graphInferenceContext_, options_);
    inferencer = new_inferencer.get();
    graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
  } else {
    inferencer = entry->second.get();
  }

  return inferencer;
}

std::string GraphInferenceContextImpl::getDisplayName() const {
  // Adapted from ONNX InferenceContextImpl::getDisplayName()
  if (node_ == nullptr)
    return "";
  if (node_->domain().empty()) {
    if (node_->name().empty())
      return std::string("node ") + node_->op_type();
    return std::string("node ") + node_->op_type() + " (" + node_->name() + ")";
  }
  if (node_->name().empty())
    return std::string("node ") + node_->op_type() + "[" + node_->domain() +
           "]";
  return std::string("node ") + node_->op_type() + "[" + node_->domain() + "]" +
         " (" + node_->name() + ")";
}

} // namespace morphizen
