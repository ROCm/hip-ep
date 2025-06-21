/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./ir-converter.hpp"
#include "./graph.hpp"
#include <fstream>
#include <glog/logging.h>
#include <morphizen-utils/morphizen-utils.hpp>
#include <onnx/onnx_pb.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER, "0")
DEF_ENV_PARAM_2(MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE,
                "VitisAI-EP-IR-Converter.onnx", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER) >= n)
namespace morphizen {

// Constructor definition
IRConverter::IRConverter(const ApiPtrs& api_ptrs, const OrtGraph& graph)
    : ApiPtrs(api_ptrs), graph_(*this, graph) {}

ModelUniquePtr IRConverter::to_onnx_model(const ApiPtrs& api_ptrs,
                                          const OrtGraph& graph) {
  // Forward call to instance method
  IRConverter converter(api_ptrs, graph);
  auto model_proto = std::make_unique<ONNX_NAMESPACE::ModelProto>();
  converter.throw_if_error(
      converter.convert_to_model_proto(converter.graph_, model_proto.get()));

  // Convert to ModelUniquePtr (placeholder implementation)
  return ModelUniquePtr(nullptr, nullptr);
}

OrtStatus*
IRConverter::convert_to_model_proto(const Graph& graph,
                                    ONNX_NAMESPACE::ModelProto* model_proto) {
  // This is a stub implementation. Replace with actual conversion logic.
  MY_LOG(1) << "graph name =" << graph.name();
  model_proto->Clear(); // Clear the model to start fresh

  // Initialize the model with basic information
  model_proto->set_ir_version(graph.ir_version());
  model_proto->set_producer_name("Morphizen IR Converter");
  model_proto->set_producer_version("1.0.0");
  model_proto->set_domain("ai.morphizen");
  model_proto->set_model_version(1);
  model_proto->set_doc_string("Converted from OrtGraph to ONNX Model");

  // Set opset using the Graph member function
  auto* opset = model_proto->mutable_opset_import();
  for (const auto& [domain, version] : graph.guess_opset()) {
    auto* op = opset->Add();
    op->set_domain(domain);
    op->set_version(version);
    MY_LOG(3) << "Final opset: " << domain << " with version: " << version;
  } // Create the graph within the model
  auto graph_proto = model_proto->mutable_graph();
  throw_if_error(convert_graph_proto(graph, graph_proto));

  // Save converted model to file for debugging if enabled
  save_model_for_debugging(model_proto);

  return nullptr;
}

OrtStatus*
IRConverter::convert_graph_proto(const Graph& graph,
                                 ONNX_NAMESPACE::GraphProto* graph_proto) {
  MY_LOG(2) << "Converting ORT graph '" << graph.name() << "' to ONNX graph";

  // Set basic graph properties
  graph_proto->set_name(graph.name());
  graph_proto->set_doc_string(
      "Converted from OrtGraph to ONNX Graph"); // TODO: Add actual graph
                                                // conversion logic here  //
                                                // This would include:  // -
                                                // Converting nodes from ORT
                                                // format to ONNX format
  throw_if_error(convert_nodes(graph, graph_proto));
  // - Converting inputs/outputs
  throw_if_error(convert_graph_inputs(graph, graph_proto));
  throw_if_error(convert_graph_outputs(graph, graph_proto));
  // - Converting value infos
  // - Converting initializers
  throw_if_error(convert_graph_initializers(graph, graph_proto));
  MY_LOG(2) << "Graph conversion completed";
  return nullptr;
}

OrtStatus*
IRConverter::convert_graph_inputs(const Graph& graph,
                                  ONNX_NAMESPACE::GraphProto* graph_proto) {
  MY_LOG(2) << "Converting graph inputs to ONNX format";

  // Get inputs from the ORT graph
  auto inputs = graph.inputs();

  for (const OrtValueInfo* input : inputs) {
    // Create ValueInfo wrapper for the input
    auto value_info =
        Ort::ConstValueInfo(input); // Create ONNX ValueInfoProto for the input
    auto* input_proto =
        graph_proto->add_input(); // Convert the ValueInfo to ValueInfoProto
    throw_if_error(convert_value_info_proto(value_info, input_proto));
    MY_LOG(3) << "Added input: " << value_info.Name();
  }

  MY_LOG(2) << "Converted " << inputs.size() << " inputs";
  return nullptr;
}

OrtStatus*
IRConverter::convert_graph_outputs(const Graph& graph,
                                   ONNX_NAMESPACE::GraphProto* graph_proto) {
  MY_LOG(2) << "Converting graph outputs to ONNX format";

  // Get outputs from the ORT graph
  auto outputs = graph.outputs();

  for (const OrtValueInfo* output : outputs) {
    auto value_info = Ort::ConstValueInfo(
        output); // Create ONNX ValueInfoProto for the output
    auto* output_proto =
        graph_proto->add_output(); // Convert the ValueInfo to ValueInfoProto
    throw_if_error(convert_value_info_proto(value_info, output_proto));

    MY_LOG(3) << "Added output: " << value_info.Name();
  }
  MY_LOG(2) << "Converted " << outputs.size() << " outputs";
  return nullptr;
}

OrtStatus* IRConverter::convert_nodes(const Graph& graph,
                                      ONNX_NAMESPACE::GraphProto* graph_proto) {
  MY_LOG(2) << "Converting graph nodes to ONNX format";

  // Get nodes from the ORT graph
  auto nodes = graph.nodes();

  for (const OrtNode* node : nodes) {
    // Create ONNX NodeProto for each ORT node
    auto* node_proto = graph_proto->add_node();

    // Get node information from ORT API
    const char* op_type = nullptr;
    const char* name = nullptr;
    const char* domain = nullptr;
    OrtArrayOfConstObjects* inputs = nullptr;
    OrtArrayOfConstObjects* outputs = nullptr;
    size_t node_id = 0;
    throw_if_error(ort_api.Node_GetOperatorType(node, &op_type));
    throw_if_error(ort_api.Node_GetName(node, &name));
    throw_if_error(ort_api.Node_GetDomain(node, &domain));
    throw_if_error(ort_api.Node_GetId(node, &node_id));
    throw_if_error(ort_api.Node_GetInputs(node, &inputs));
    throw_if_error(ort_api.Node_GetOutputs(node, &outputs));
    // Set basic node properties
    node_proto->set_op_type(op_type);
    if (name == nullptr || name[0] == '\0') {
      node_proto->set_name(std::string("N") + std::to_string(node_id));
    }
    if (domain && domain[0] != '\0') {
      node_proto->set_domain(domain);
    }
    for (auto input : make_array_span<const OrtValueInfo* const>(inputs)) {
      node_proto->add_input(Ort::ConstValueInfo(input).Name());
    }
    for (auto output : make_array_span<const OrtValueInfo* const>(outputs)) {
      node_proto->add_output(Ort::ConstValueInfo(output).Name());
    }

    MY_LOG(3) << "Added node: " << node_proto->name()
              << " (op_type: " << node_proto->op_type() << ")";
  }

  MY_LOG(2) << "Converted " << nodes.size() << " nodes";
  return nullptr;
}

OrtStatus* IRConverter::convert_value_info_proto(
    const Ort::ConstValueInfo& value_info,
    ONNX_NAMESPACE::ValueInfoProto* value_info_proto) {
  MY_LOG(3) << "Converting ORT ValueInfo to ONNX ValueInfoProto";

  // Set the name
  value_info_proto->set_name(value_info.Name());
  // Get type information and convert it
  auto type_info = value_info.TypeInfo();
  throw_if_error(
      convert_type_proto(type_info, value_info_proto->mutable_type()));

  // TODO: Add doc_string support if available in the future
  // value_info_proto->set_doc_string(value_info.GetDocString());

  MY_LOG(3) << "ValueInfoProto conversion completed for: " << value_info.Name();
  return nullptr;
}
OrtStatus* IRConverter::convert_graph_initializers(
    const Graph& graph, ONNX_NAMESPACE::GraphProto* graph_proto) {
  MY_LOG(2) << "Converting graph initializers to ONNX format";

  // Get initializers from the ORT graph
  auto initializers = graph.initializers();

  for (const OrtValueInfo* initializer : initializers) {
    // Create ValueInfo wrapper for the initializer
    auto value_info = Ort::ConstValueInfo(initializer);
    MY_LOG(3) << "Added initializer: " << value_info.Name();
    const OrtValue* ort_value = nullptr;
    throw_if_error(
        ort_api.ValueInfo_GetInitializerValue(value_info, &ort_value));
    if (ort_value == nullptr) {
      MY_LOG(2) << "Initializer value is null for: " << value_info.Name();
      throw_error(std::string("cannot get OrtValue from OrtValueInfo: name=") +
                  value_info.Name());
    }
    // Create ONNX TensorProto for the initializer
    auto* initializer_proto = graph_proto->add_initializer();
    convert_tensor_proto(Ort::ConstValue(ort_value), initializer_proto);
  }

  MY_LOG(2) << "Converted " << initializers.size() << " initializers";
  return nullptr;
}
OrtStatus*
IRConverter::convert_type_proto(const Ort::ConstTypeInfo& type_info,
                                ONNX_NAMESPACE::TypeProto* type_proto) {
  MY_LOG(3) << "Converting ORT TypeInfo to ONNX TypeProto";

  try {
    // Get the ONNX type from the type info
    auto onnx_type = type_info.GetONNXType();

    switch (onnx_type) {
    case ONNX_TYPE_TENSOR: {
      MY_LOG(4) << "Converting tensor type";
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto* tensor_type = type_proto->mutable_tensor_type();

      // Set element type
      auto element_type = tensor_info.GetElementType();
      tensor_type->set_elem_type(static_cast<int32_t>(element_type));

      // Set shape information
      auto* shape = tensor_type->mutable_shape();
      auto shape_info = tensor_info.GetShape();

      for (size_t i = 0; i < shape_info.size(); ++i) {
        auto* dim = shape->add_dim();
        int64_t dim_value = shape_info[i];
        if (dim_value >= 0) {
          dim->set_dim_value(dim_value);
        } else {
          // Negative values typically indicate dynamic dimensions
          dim->set_dim_param("dynamic_" + std::to_string(i));
        }
      }
      break;
    }

    case ONNX_TYPE_SEQUENCE: {
      MY_LOG(4) << "Converting sequence type";
      return ort_api.CreateStatus(
          ORT_NOT_IMPLEMENTED,
          "Sequence type conversion is not implemented yet.");
    }

    case ONNX_TYPE_MAP: {
      MY_LOG(4) << "Converting map type";
      return ort_api.CreateStatus(
          ORT_NOT_IMPLEMENTED, "Map type conversion is not implemented yet.");
    }

    default:
      MY_LOG(2) << "Unknown or unsupported ONNX type: " << onnx_type;
      return ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "Unsupported ONNX type encountered during conversion");
    }

    MY_LOG(3) << "TypeProto conversion completed";
    return nullptr;
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}

OrtStatus*
IRConverter::convert_tensor_proto(const Ort::ConstValue& tensor_value,
                                  ONNX_NAMESPACE::TensorProto* tensor_proto) {
  MY_LOG(3)
      << "Converting ORT tensor value to ONNX TensorProto using external data";

  // NOTE: This function stores tensor data as external data references instead
  // of copying the actual data into the TensorProto. This approach:
  // 1. Reduces memory usage by avoiding data duplication
  // 2. Improves performance by eliminating large memory copies
  // 3. Maintains references to the original ORT tensor data
  // 4. Requires careful lifetime management of the source tensors

  try {
    // Get tensor type and shape information
    auto type_info = tensor_value.GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // Set tensor name (if available)
    // Note: ORT ConstValue doesn't have a direct name accessor
    // This would need to be provided from the calling context
    // tensor_proto->set_name("tensor_name");

    // Set element type
    auto element_type = tensor_info.GetElementType();
    tensor_proto->set_data_type(static_cast<int32_t>(element_type));

    // Set dimensions
    auto shape_info = tensor_info.GetShape();
    for (int64_t dim_value : shape_info) {
      tensor_proto->add_dims(dim_value);
    }

    // Get tensor data
    size_t element_count = tensor_info.GetElementCount();
    size_t element_size = 0;

    // Calculate element size based on data type
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      element_size = sizeof(float);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      element_size = sizeof(double);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      element_size = sizeof(int32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      element_size = sizeof(int64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      element_size = sizeof(uint8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      element_size = sizeof(int8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      element_size = sizeof(uint16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      element_size = sizeof(int16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      element_size = sizeof(uint32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      element_size = sizeof(uint64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      element_size = sizeof(bool);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      element_size = sizeof(uint16_t); // Float16 is stored as uint16
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      element_size = sizeof(uint16_t); // BFloat16 is stored as uint16
      break;
    default:
      return ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "Unsupported tensor element type for conversion");
    }

    // Get raw tensor data
    const void* tensor_data = tensor_value.GetTensorRawData();
    size_t data_size = element_count * element_size;

    // Store data as external data instead of copying
    // Generate a unique external data location (could be memory address or
    // file-based)
    std::string external_data_location =
        "#" + std::to_string(reinterpret_cast<uintptr_t>(tensor_data));

    // Set external data info in TensorProto
    auto* external_data = tensor_proto->mutable_external_data();
    auto* location_entry = external_data->Add();
    location_entry->set_key("location");
    location_entry->set_value(external_data_location);

    auto* offset_entry = external_data->Add();
    offset_entry->set_key("offset");
    offset_entry->set_value("0");

    auto* length_entry = external_data->Add();
    length_entry->set_key("length");
    length_entry->set_value(std::to_string(data_size));

    MY_LOG(3) << "TensorProto conversion completed - elements: "
              << element_count << ", data_size: " << data_size << " bytes"
              << ", external_location: " << external_data_location;
    return nullptr;

  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}

void IRConverter::save_model_for_debugging(
    const ONNX_NAMESPACE::ModelProto* model_proto) const {
  // Save converted model to file if debug level > 4 and output file is
  // specified
  if (ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER) > 4) {
    std::string output_file =
        ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE);
    if (!output_file.empty()) {
      MY_LOG(5) << "Saving converted ONNX model to: " << output_file;
      try {
        std::ofstream ofs(output_file, std::ios::binary);
        if (ofs.is_open()) {
          if (model_proto->SerializeToOstream(&ofs)) {
            MY_LOG(5) << "Successfully saved ONNX model to: " << output_file;
          } else {
            MY_LOG(1) << "Failed to serialize ONNX model to: " << output_file;
          }
        } else {
          MY_LOG(1) << "Failed to open output file: " << output_file;
        }
      } catch (const std::exception& e) {
        MY_LOG(1) << "Exception while saving ONNX model: " << e.what();
      }
      // also save it as a text format
      std::string output_file_txt = output_file + ".txt";
      MY_LOG(5) << "Saving converted ONNX model to text format: "
                << output_file_txt;
      try {
        std::ofstream ofs_txt(output_file_txt, std::ios::out);
        if (ofs_txt.is_open()) {
          ofs_txt << model_proto->DebugString();
        } else {
          MY_LOG(1) << "Failed to open output file for text format: "
                    << output_file_txt;
        }
      } catch (const std::exception& e) {
        MY_LOG(1) << "Exception while saving ONNX model to text format: "
                  << e.what();
      }
    } else {
      MY_LOG(5)
          << "Debug level > 4 but no output file specified. Set "
             "MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE to save the model.";
    }
  }
}

} // namespace morphizen
