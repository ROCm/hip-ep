/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define MORPHIZEN_USER 1
#include "./ir-converter-imp.hpp"
#include "./ort-graph-wrapper.hpp"
#include <fstream>
#include <glog/logging.h>
#include <morphizen-utils/morphizen-utils.hpp>
#include <morphizen/graph.hpp>
#include <morphizen/morphizen-ort-api-ext.hpp>
#include <morphizen/node_attr.hpp>
#include <unordered_set>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER, "0")
DEF_ENV_PARAM_2(MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE,
                "MorphiZen-EP-IR-Converter.onnx", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER) >= n)
namespace morphizen {

// Constructor definition
IRConverterImp::IRConverterImp(const ApiPtrs& api_ptrs, const OrtGraph& graph,
                               const IRConverterConfig& config)
    : ApiPtrs(api_ptrs), graph_(*this, graph), config_(config) {}

ModelUniquePtr IRConverterImp::to_onnx_model(const ApiPtrs& api_ptrs,
                                             const OrtGraph& graph,
                                             const IRConverterConfig& config) {
  // Forward call to instance method
  const ORTCHAR_T* api_model_path = nullptr;
  api_ptrs.throw_if_error(
      api_ptrs.ort_api.Graph_GetModelPath(&graph, &api_model_path));
  auto model_path = std::filesystem::path(api_model_path);
  MY_LOG(1) << "Converting ORT graph to ONNX model at: " << model_path;
  auto opset_imports = std::vector<std::pair<std::string, int64_t>>();
  auto graph_wrapper = OrtGraphWrapper(api_ptrs, graph);
  for (const auto& [domain, version] : graph_wrapper.guess_opset()) {
    opset_imports.emplace_back(domain, (int64_t)version);
    MY_LOG(3) << "Add opset: " << domain << " with version: " << version;
  }
  // Use MORPHIZEN_ORT_API for model creation
  auto model = ModelUniquePtr(
      MORPHIZEN_ORT_API(create_empty_model)(model_path, opset_imports),
      [](onnxruntime::Model* model) {
        MORPHIZEN_ORT_API(model_delete)(model);
      });
  IRConverterImp converter(api_ptrs, graph, config);
  converter.throw_if_error(converter.convert_to_model(*model.get()));
  return model;
}

OrtStatus* IRConverterImp::convert_to_model(morphizen::Model& model) const {
  // This is a stub implementation. Replace with actual conversion logic.
  MY_LOG(1) << "graph name =" << graph_.name();
  auto& main_graph = MORPHIZEN_ORT_API(model_main_graph)(model);

  throw_if_error(convert_metadata(main_graph, model));
  throw_if_error(convert_graph(main_graph));

  // Save converted model to file for debugging if enabled
  save_model_for_debugging(model);

  return nullptr;
}
OrtStatus* IRConverterImp::convert_metadata(morphizen::Graph& /*graph*/,
                                            morphizen::Model& model) const {
  Ort::AllocatorWithDefaultOptions allocator;
  auto metadata = graph_.get_model_metadata();
  auto customized_keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
  for (auto& key : customized_keys) {
    auto value =
        metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
    auto cxx_key = std::string(key.get());
    auto cxx_value = std::string(value.get());
    MORPHIZEN_ORT_API(model_set_meta_data)(model, cxx_key, cxx_value);
  }
  return nullptr;
}

void IRConverterImp::save_model_for_debugging(
    const morphizen::Model& model) const {
  // Save converted model to file if debug level > 4 and output file is
  // specified
  if (ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER) > 4) {
    std::string output_file =
        ENV_PARAM(MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE);
    if (!output_file.empty()) {
      MY_LOG(4) << "Saving converted ONNX model to: " << output_file;
      auto& main_graph = MORPHIZEN_ORT_API(model_main_graph)(
          const_cast<morphizen::Model&>(model));
      morphizen_cxx::GraphConstRef(main_graph)
          .save(output_file, output_file + ".dat",
                std::numeric_limits<size_t>::max());
    } else {
      MY_LOG(4)
          << "Debug level > 4 but no output file specified. Set "
             "MORPHIZEN_DEBUG_IR_CONVERTER_OUTPUT_FILE to save the model.";
    }
  }
}

OrtStatus* IRConverterImp::convert_graph(morphizen::Graph& graph) const {
  MY_LOG(2) << "Converting ORT graph '" << graph_.name() << "' to ONNX graph";
  // Set basic graph properties
  morphizen_cxx::GraphRef(graph).set_name(graph_.name());
  // throw_if_error(convert_nodes(graph, graph_proto));
  // - Converting inputs
  throw_if_error(convert_graph_inputs(graph));
  // - Converting initializers
  throw_if_error(convert_graph_initializers(graph));
  // - Converting Nodes
  throw_if_error(convert_graph_nodes(graph));
  // - Converting outputs
  throw_if_error(convert_graph_outputs(graph));
  // - Resolve the graph
  morphizen_cxx::GraphRef(graph).resolve(true);
  MY_LOG(2) << "Graph conversion completed";
  return nullptr;
}

OrtStatus* IRConverterImp::convert_graph_inputs(morphizen::Graph& graph) const {
  MY_LOG(2) << "Converting graph inputs to ONNX format";
  // Get inputs from the ORT graph
  auto inputs = graph_.inputs();
  auto new_inputs = std::vector<morphizen::NodeArg*>();
  new_inputs.reserve(inputs.size());
  for (const OrtValueInfo* input : inputs) {
    // Create ValueInfo wrapper for the input
    auto value_info =
        Ort::ConstValueInfo(input); // Create ONNX ValueInfoProto for the input
    morphizen::NodeArg* node_arg = nullptr;
    throw_if_error(convert_value_info_proto(value_info, graph, &node_arg));
    CHECK(node_arg != nullptr);
    new_inputs.push_back(node_arg);
    MY_LOG(3) << "Added input: " << value_info.GetName();
  }
  morphizen_cxx::GraphRef(graph).set_inputs(new_inputs);
  MY_LOG(2) << "Converted " << inputs.size() << " inputs";
  return nullptr;
}

std::vector<morphizen::NodeArg*>
IRConverterImp::guess_missing_output(std::vector<morphizen::NodeArg*> outputs,
                                     morphizen::Graph& graph) const {

  auto nodes = graph_.nodes();
  // collect all inputs
  std::unordered_set<const OrtValueInfo*> all_inputs;
  for (auto& node : nodes) {
    std::vector<const OrtValueInfo*> node_inputs = {};
    size_t num_of_inputs = 0;
    throw_if_error(ort_api.Node_GetNumInputs(node, &num_of_inputs));
    node_inputs.resize(num_of_inputs);
    throw_if_error(
        ort_api.Node_GetInputs(node, node_inputs.data(), num_of_inputs));
    for (auto input : node_inputs) {
      if (input != nullptr) { // input == nullptr mean optional argument.
        all_inputs.insert(input);
      }
    }
  }

  // guess output: add dangling node outputs that are not consumed by any other
  // node and are not already in the output list. Skip optional intermediates
  // from multi-output ops where at least one sibling result IS consumed.
  for (auto& node : nodes) {
    std::vector<const OrtValueInfo*> node_outputs = {};
    size_t num_of_outputs = 0;
    throw_if_error(ort_api.Node_GetNumOutputs(node, &num_of_outputs));
    node_outputs.resize(num_of_outputs);
    throw_if_error(
        ort_api.Node_GetOutputs(node, node_outputs.data(), num_of_outputs));

    bool any_sibling_consumed = false;
    if (num_of_outputs > 1) {
      for (auto sibling : node_outputs) {
        if (sibling != nullptr && all_inputs.count(sibling) > 0) {
          any_sibling_consumed = true;
          break;
        }
      }
    }

    for (auto output : node_outputs) {
      if (output != nullptr) { // output == nullptr mean optional argument.
        if (all_inputs.count(output) == 0) {
          if (any_sibling_consumed) {
            auto vi = Ort::ConstValueInfo(output);
            MY_LOG(3) << "Skipping optional intermediate output: "
                      << vi.GetName();
            continue;
          }
          // Create ValueInfo wrapper for the output
          auto value_info = Ort::ConstValueInfo(output);
          morphizen::NodeArg* node_arg = nullptr;
          throw_if_error(
              convert_value_info_proto(value_info, graph, &node_arg));
          CHECK(node_arg != nullptr);
          if (std::find(outputs.begin(), outputs.end(), node_arg) ==
              outputs.end()) {
            outputs.push_back(node_arg);
            MY_LOG(3) << "Added output: " << value_info.GetName();
          }
        }
      }
    }
  }
  return outputs;
}

OrtStatus*
IRConverterImp::convert_graph_outputs(morphizen::Graph& graph) const {
  MY_LOG(2) << "Converting graph outputs to ONNX format";
  // Get outputs from the ORT graph
  auto outputs = graph_.outputs();
  auto new_outputs = std::vector<morphizen::NodeArg*>();
  new_outputs.reserve(outputs.size());
  for (const OrtValueInfo* output : outputs) {
    // Create ValueInfo wrapper for the output
    auto value_info = Ort::ConstValueInfo(
        output); // Create ONNX ValueInfoProto for the output
    morphizen::NodeArg* node_arg = nullptr;
    throw_if_error(convert_value_info_proto(value_info, graph, &node_arg));
    CHECK(node_arg != nullptr);
    new_outputs.push_back(node_arg);
    MY_LOG(3) << "Added output: " << value_info.GetName();
  }
  // guess output
  new_outputs = guess_missing_output(new_outputs, graph);

  morphizen_cxx::GraphRef(graph).set_outputs(new_outputs);
  MY_LOG(2) << "Converted " << outputs.size() << " outputs";
  return nullptr;
}

OrtStatus*
IRConverterImp::convert_graph_initializers(morphizen::Graph& graph) const {
  MY_LOG(2) << "Converting graph initializers to ONNX format";
  // Get initializers from the ORT graph
  auto initializers = graph_.initializers();
  for (const OrtValueInfo* initializer : initializers) {
    // Create ValueInfo wrapper for the initializer
    auto value_info =
        Ort::ConstValueInfo(initializer); // Create ONNX ValueInfoProto
    const OrtValue* ort_value = nullptr;
    throw_if_error(
        ort_api.ValueInfo_GetInitializerValue(value_info, &ort_value));
    if (ort_value == nullptr) {
      MY_LOG(2) << "Initializer value is null for: " << value_info.GetName();
      throw_error(std::string("cannot get OrtValue from OrtValueInfo: name=") +
                  value_info.GetName());
    }
    // Get tensor type and shape information
    auto tensor_value = Ort::ConstValue(ort_value);
    auto type_info = tensor_value.GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // Set tensor name (if available)
    // Note: ORT ConstValue doesn't have a direct name accessor
    // This would need to be provided from the calling context
    // tensor_proto->set_name("tensor_name");

    // Set element type
    auto element_type = tensor_info.GetElementType();
    auto shape = tensor_info.GetShape();

    // Get raw tensor data
    const void* tensor_data = tensor_value.GetTensorRawData();
    size_t data_size = tensor_value.GetTensorSizeInBytes();

    // For tensors above the threshold, use the no-copy memory-address
    // mechanism: encode the raw pointer as an int64 in the external data offset
    // field with location sentinel "*/_ORT_MEM_ADDR_/*". The downstream decoder
    // reconstructs the pointer — zero copy. ORT owns the tensor data for the
    // session lifetime.
    morphizen::TensorProto* raw_proto = nullptr;
    if (data_size > config_.external_data_threshold) {
      MY_LOG(2) << "Initializer '" << value_info.GetName()
                << "': no-copy path (size=" << data_size << ")";
      raw_proto = MORPHIZEN_ORT_API_EXT(tensor_proto_new_with_external_data)(
          value_info.GetName(), shape, element_type, "*/_ORT_MEM_ADDR_/*",
          data_size,
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor_data)));
    } else {
      MY_LOG(2) << "Initializer '" << value_info.GetName()
                << "': copy path (size=" << data_size << ")";
      raw_proto = MORPHIZEN_ORT_API_EXT(tensor_proto_new_raw_data)(
          value_info.GetName(), shape, element_type, tensor_data, data_size);
    }

    morphizen_cxx::GraphRef(graph).add_initialized_tensor(*raw_proto);
    MORPHIZEN_ORT_API(tensor_proto_delete)(raw_proto);
    MY_LOG(3) << "Added initializer: " << value_info.GetName();
  }

  MY_LOG(2) << "Converted " << initializers.size() << " initializers";
  return nullptr;
}
OrtStatus*
IRConverterImp::convert_value_info_proto(const Ort::ConstValueInfo& value_info,
                                         morphizen::Graph& graph,
                                         morphizen::NodeArg** node_arg) const {
  MY_LOG(3) << "Converting ORT ValueInfo to ONNX ValueInfoProto";
  // Get type information and convert it
  auto type_info = value_info.TypeInfo();
  auto shape = std::vector<int64_t>();
  auto name = value_info.GetName();
  int element_type = 0; // Placeholder for element type
  throw_if_error(convert_type_proto(type_info, &element_type, &shape));
  auto existing_node_arg_opt =
      morphizen_cxx::GraphConstRef(graph).find_node_arg(name);
  if (existing_node_arg_opt.has_value()) {
    *node_arg =
        const_cast<morphizen::NodeArg*>(existing_node_arg_opt.value().ptr());
  } else {
    *node_arg = &morphizen::node_arg_new(graph, name, &shape, element_type);
  }
  return nullptr;
};
OrtStatus*
IRConverterImp::convert_type_proto(const Ort::ConstTypeInfo& type_info,
                                   int* element_type,
                                   std::vector<int64_t>* shape) const {
  MY_LOG(3) << "Converting ORT TypeInfo to ONNX TypeProto";

  try {
    // Get the ONNX type from the type info
    auto onnx_type = type_info.GetONNXType();

    switch (onnx_type) {
    case ONNX_TYPE_TENSOR: {
      MY_LOG(4) << "Converting tensor type";
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      // Set element type
      *element_type = tensor_info.GetElementType();
      // Set shape information
      *shape = tensor_info.GetShape();
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

static int64_t get_attr_value_int(const OrtApi& ort_api,
                                  const OrtOpAttr* attr) {
  int64_t value = 0;
  size_t out = {};
  Ort::ThrowOnError(ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_INT,
                                       &value, sizeof(value), &out));
  return value;
}
static std::vector<int64_t> get_attr_value_ints(const OrtApi& ort_api,
                                                const OrtOpAttr* attr) {
  int64_t i = {};
  size_t out = {};
  // first call to get the bytes needed
  // 1. A status == nullptr means that ReadOpAttr was successful. A status !=
  // nullptr means failure.
  // 2. The ReadOpAttr function should normally be called twice: once to get the
  // needed buffer size (returns a status != nullptr), and a second time to
  // actually read the ints (returns status == null on success).
  // 3. This code tries a subtle optimization in the first call to ReadOpAttr.
  // It passes in a buffer (&i) of size 1 just in case there is only 1 int. In
  // this case, status == nullptr and we need to return {i}.
  auto status = ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_INTS, &i,
                                   sizeof(i), &out);
  if (status) {
    size_t num_i = out / sizeof(int64_t);
    std::vector<int64_t> ints(num_i, 0);
    Ort::ThrowOnError(ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_INTS,
                                         ints.data(), out, &out));
    return ints;
  } else {
    if (out == 0u) {
      return {};
    }
    return {i};
  }
}
static float get_attr_value_float(const OrtApi& ort_api,
                                  const OrtOpAttr* attr) {
  float value = 0.0f;
  size_t out = {};
  Ort::ThrowOnError(ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_FLOAT,
                                       &value, sizeof(value), &out));
  return value;
}
static std::vector<float> get_attr_value_floats(const OrtApi& ort_api,
                                                const OrtOpAttr* attr) {
  float f = {};
  size_t out = {};
  auto status = ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_FLOATS, &f,
                                   sizeof(f), &out);
  if (status) {
    size_t num_f = out / sizeof(float);
    std::vector<float> floats(num_f, 0.0f);
    Ort::ThrowOnError(ort_api.ReadOpAttr(
        attr, OrtOpAttrType::ORT_OP_ATTR_FLOATS, floats.data(), out, &out));
    return floats;
  } else {
    if (out == 0u) {
      return {};
    }
    return {f};
  }
}
static std::string get_attr_value_string(const OrtApi& ort_api,
                                         const OrtOpAttr* attr) {
  char c = {};
  size_t out = {};
  // first call to get the bytes needed
  auto status =
      ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_STRING, &c, 1, &out);
  if (status) {
    std::string chars(out, '\0');
    Ort::ThrowOnError(ort_api.ReadOpAttr(
        attr, OrtOpAttrType::ORT_OP_ATTR_STRING, chars.data(), out, &out));
    return chars;
  } else {
    return {c};
  }
}
static std::vector<std::string> get_attr_value_strings(const OrtApi& ort_api,
                                                       const OrtOpAttr* attr) {
  char c = {};
  size_t out = {};
  auto status = ort_api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_STRINGS, &c,
                                   sizeof(char), &out);
  if (status) {
    std::vector<char> chars(out, '\0');
    Ort::ThrowOnError(ort_api.ReadOpAttr(
        attr, OrtOpAttrType::ORT_OP_ATTR_STRINGS, chars.data(), out, &out));
    std::vector<std::string> strings;
    char* char_st = chars.data();
    char* char_ed = char_st + out;
    while (char_st < char_ed) {
      strings.emplace_back(char_st);
      while (*char_st != '\0') {
        char_st++;
      }
      char_st++;
    }
    return strings;
  } else {
    if (out == 0u) {
      return {};
    }
    return {std::string{c}};
  }
}

OrtStatus* IRConverterImp::convert_graph_nodes(morphizen::Graph& graph) const {
  MY_LOG(2) << "Converting graph nodes to ONNX format";
  // Get nodes from the ORT graph
  auto nodes = graph_.nodes();

  for (const OrtNode* node : nodes) {
    // Get node information from ORT API
    const char* op_type = nullptr;
    const char* name = nullptr;
    const char* domain = nullptr;
    const char* description = nullptr;
    std::vector<const OrtValueInfo*> inputs = {};
    size_t num_of_inputs = 0;
    throw_if_error(ort_api.Node_GetNumInputs(node, &num_of_inputs));
    inputs.resize(num_of_inputs);
    throw_if_error(ort_api.Node_GetInputs(node, inputs.data(), num_of_inputs));
    std::vector<const OrtValueInfo*> outputs = {};
    size_t num_of_outputs = 0;
    throw_if_error(ort_api.Node_GetNumOutputs(node, &num_of_outputs));
    outputs.resize(num_of_outputs);
    throw_if_error(
        ort_api.Node_GetOutputs(node, outputs.data(), num_of_outputs));
    size_t node_id = 0;
    throw_if_error(ort_api.Node_GetOperatorType(node, &op_type));
    throw_if_error(ort_api.Node_GetName(node, &name));
    // throw_if_error(ort_api.Node_GetDescription(node, &description));
    description = ""; // Default to empty string if not available
    throw_if_error(ort_api.Node_GetDomain(node, &domain));
    throw_if_error(ort_api.Node_GetId(node, &node_id));
    auto node_inputs = std::vector<const morphizen::NodeArg*>();
    node_inputs.reserve(inputs.size());
    for (auto input : inputs) {
      morphizen::NodeArg* node_arg = nullptr;
      if (input != nullptr) { // input == nullptr mean optionsl argument.
        throw_if_error(convert_value_info_proto(Ort::ConstValueInfo(input),
                                                graph, &node_arg));
      }
      node_inputs.push_back(node_arg);
    }
    auto node_outputs = std::vector<const morphizen::NodeArg*>();
    node_outputs.reserve(outputs.size());
    for (auto output : outputs) {
      morphizen::NodeArg* node_arg = nullptr;
      if (output != nullptr) { // output == nullptr mean optionsl argument.
        throw_if_error(convert_value_info_proto(Ort::ConstValueInfo(output),
                                                graph, &node_arg));
      }
      node_outputs.push_back(node_arg);
    }

    morphizen::NodeAttributesBuilder attrs_builder;

    std::vector<const OrtOpAttr*> api_node_attributes;
    size_t num_of_attributes = 0;
    throw_if_error(ort_api.Node_GetNumAttributes(node, &num_of_attributes));
    api_node_attributes.resize(num_of_attributes);
    throw_if_error(ort_api.Node_GetAttributes(node, api_node_attributes.data(),
                                              api_node_attributes.size()));

    for (const OrtOpAttr* attr : api_node_attributes) {
      const char* attr_name = nullptr;
      OrtOpAttrType attr_type = OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;
      throw_if_error(ort_api.OpAttr_GetName(attr, &attr_name));
      throw_if_error(ort_api.OpAttr_GetType(attr, &attr_type));
      MY_LOG(3) << "Processing attribute: " << attr_name
                << " with type: " << static_cast<int>(attr_type);
      switch (attr_type) {
      case OrtOpAttrType::ORT_OP_ATTR_INT: {
        auto value = get_attr_value_int(ort_api, attr);
        MY_LOG(3) << "Attribute " << attr_name << " is of type INT with value "
                  << value;
        attrs_builder.add(attr_name, value);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_INTS: {
        attrs_builder.add(attr_name, get_attr_value_ints(ort_api, attr));
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_FLOAT: {
        auto value = get_attr_value_float(ort_api, attr);
        MY_LOG(3) << "Attribute " << attr_name
                  << " is of type FLOAT with value " << value;
        attrs_builder.add(attr_name, value);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_FLOATS: {
        attrs_builder.add(attr_name, get_attr_value_floats(ort_api, attr));
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRING: {
        auto value = get_attr_value_string(ort_api, attr);
        MY_LOG(3) << "Attribute " << attr_name
                  << " is of type STRING with value " << value;
        attrs_builder.add(attr_name, value);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRINGS: {
        attrs_builder.add(attr_name, get_attr_value_strings(ort_api, attr));
        break;
      }
      default: {
        MY_LOG(2) << "Unsupported attribute type: " << attr_type
                  << " for attribute: " << attr_name;
        return ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Unsupported attribute type encountered during conversion");
      }
      }
    }

    auto node_attributes = attrs_builder.build();
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4996) // deprecated declaration
#else
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    morphizen::graph_add_node(graph, name, op_type, description, node_inputs,
                              node_outputs, std::move(node_attributes), domain);
#ifdef _MSC_VER
#  pragma warning(pop)
#else
#  pragma GCC diagnostic pop
#endif
    MY_LOG(3) << "Added node: " << name << " (op_type: " << op_type << ")";
  }

  MY_LOG(2) << "Converted " << nodes.size() << " nodes";
  return nullptr;
}

} // namespace morphizen
