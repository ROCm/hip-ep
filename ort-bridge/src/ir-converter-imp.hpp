/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ir-converter.hpp"
#include "ort-graph-wrapper.hpp"
namespace morphizen {
class IRConverterImp : public ApiPtrs {

public:
  static ModelUniquePtr
  to_onnx_model(const ApiPtrs& api_ptrs,
                const OrtGraph& graph); // Instance method for converting graph

public:
  IRConverterImp(const ApiPtrs& api_ptrs, const OrtGraph& graph);

  OrtStatus* convert_to_model(morphizen::Model& model)
      const; // Instance method for converting to ONNX graph
  OrtStatus* convert_metadata(morphizen::Graph& graph,
                              morphizen::Model& model) const;
  OrtStatus* convert_graph(morphizen::Graph& graph) const;

  void save_model_for_debugging(const morphizen::Model& model) const;

  OrtStatus* convert_graph_inputs(morphizen::Graph& graph) const;
  OrtStatus* convert_graph_outputs(morphizen::Graph& graph) const;
  OrtStatus* convert_graph_initializers(morphizen::Graph& graph) const;
  OrtStatus* convert_graph_nodes(morphizen::Graph& vaip_graph) const;

  OrtStatus* convert_value_info_proto(const Ort::ConstValueInfo& value_info,
                                      morphizen::Graph& graph,
                                      morphizen::NodeArg** node_arg) const;
  OrtStatus* convert_type_proto(const Ort::ConstTypeInfo& type_info,
                                int* element_type,
                                std::vector<int64_t>* shape) const;
  std::vector<morphizen::NodeArg*>
  guess_missing_output(std::vector<morphizen::NodeArg*> outputs,
                       morphizen::Graph& graph) const;

private:
  OrtGraphWrapper graph_;
};
} // namespace morphizen
