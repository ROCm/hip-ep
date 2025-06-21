/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include "./graph.hpp"
#include <onnx/onnx_pb.h>

namespace morphizen {
class IRConverter : public ApiPtrs {

public:
  static ModelUniquePtr
  to_onnx_model(const ApiPtrs& api_ptrs,
                const OrtGraph& graph); // Instance method for converting graph
                                        // to ONNX model
  OrtStatus* convert_to_model_proto(
      const Graph& graph,
      ONNX_NAMESPACE::ModelProto*
          model_proto); // Instance method for converting to ONNX graph
private:
  OrtStatus* convert_graph_proto(
      const Graph& graph,
      ONNX_NAMESPACE::GraphProto*
          graph_proto); // Convert graph inputs and outputs to ONNX format
  OrtStatus* convert_graph_inputs(const Graph& graph,
                                  ONNX_NAMESPACE::GraphProto* graph_proto);
  OrtStatus* convert_graph_outputs(const Graph& graph,
                                   ONNX_NAMESPACE::GraphProto* graph_proto);

  // Convert graph nodes to ONNX format
  OrtStatus* convert_nodes(const Graph& graph,
                           ONNX_NAMESPACE::GraphProto* graph_proto);

  // Convert graph initializers to ONNX format
  OrtStatus*
  convert_graph_initializers(const Graph& graph,
                             ONNX_NAMESPACE::GraphProto* graph_proto);

  // Convert ORT ValueInfo to ONNX ValueInfoProto
  OrtStatus* convert_value_info_proto(
      const Ort::ConstValueInfo& value_info,
      ONNX_NAMESPACE::ValueInfoProto*
          value_info_proto); // Convert ORT type information to ONNX TypeProto
  OrtStatus* convert_type_proto(const Ort::ConstTypeInfo& type_info,
                                ONNX_NAMESPACE::TypeProto* type_proto);

  // Convert ORT tensor data to ONNX TensorProto using external data references
  OrtStatus* convert_tensor_proto(const Ort::ConstValue& tensor_value,
                                  ONNX_NAMESPACE::TensorProto* tensor_proto);

  // Save converted model to file for debugging purposes
  void
  save_model_for_debugging(const ONNX_NAMESPACE::ModelProto* model_proto) const;

public:
  // class std::basic_string<char, struct std::char_traits<char>,
  //                         class std::allocator<char>> __cdecl Ort::detail::

public:
  IRConverter(const ApiPtrs& api_ptrs, const OrtGraph& graph);

private:
  Graph graph_;
  ONNX_NAMESPACE::ModelProto model_;
};
} // namespace morphizen
