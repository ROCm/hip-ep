/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/op_invoker.hpp"

#include <filesystem>
#include <string>

#include "morphizen/model.hpp"
#include <glog/logging.h>
#include <morphizen/morphizen_ort_api.h>

namespace morphizen {

Ort::MemoryInfo OpInvoker::CreateDefaultCpuMemInfo() {
  return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::unique_ptr<OpInvoker> OpInvoker::Create(
    const char *op_name, const char *domain, int version,
    const Ort::OpAttr *attr_values, size_t attr_count,
    const ONNXTensorElementDataType *input_type_values, size_t input_count,
    const ONNXTensorElementDataType *output_type_values, size_t output_count) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "morphizen");
  Ort::SessionOptions options{nullptr};
  return OpInvoker::Create(env, options, op_name, domain, version, attr_values,
                           attr_count, input_type_values, input_count,
                           output_type_values, output_count);
}

std::unique_ptr<OpInvoker> OpInvoker::Create(
    const Ort::Env &env, const Ort::SessionOptions &options,
    const char *op_name, const char *domain, int version,
    const Ort::OpAttr *attr_values, size_t attr_count,
    const ONNXTensorElementDataType *input_type_values, size_t input_count,
    const ONNXTensorElementDataType *output_type_values, size_t output_count) {
  return std::make_unique<OpInvoker>(PrivateTag{}, env, options, op_name,
                                     domain, version, attr_values, attr_count,
                                     input_type_values, input_count,
                                     output_type_values, output_count);
}

OpInvoker::OpInvoker(PrivateTag, const Ort::Env &env,
                     const Ort::SessionOptions &options, const char *op_name,
                     const char *domain, int version,
                     const Ort::OpAttr *attr_values, size_t attr_count,
                     const ONNXTensorElementDataType *input_type_values,
                     size_t input_count,
                     const ONNXTensorElementDataType *output_type_values,
                     size_t output_count) {
  // create model
  std::string model_name(op_name);
  std::filesystem::path curr_path = std::filesystem::current_path();
  std::filesystem::path model_path = curr_path / (model_name + ".onnx");

  std::unique_ptr<morphizen_cxx::Model> model =
      morphizen_cxx::Model::create(model_path, {{domain, version}});
  morphizen_cxx::GraphRef graph = model->main_graph();

  // inputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input_args;
  for (size_t i = 0; i < input_count; ++i) {
    std::string name = "Input_" + std::to_string(i);
    ONNX_NAMESPACE::TensorProto_DataType data_type =
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(input_type_values[i]);

    input_args.emplace_back(graph.new_node_arg(name, data_type));
  }

  // outputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output_args;
  for (size_t i = 0; i < output_count; ++i) {
    std::string name = "Output_" + std::to_string(i);
    ONNX_NAMESPACE::TensorProto_DataType data_type =
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
            output_type_values[i]);

    output_args.emplace_back(graph.new_node_arg(name, data_type));
  }

  // attributes
  morphizen::NodeAttributesPtr node_attrs =
      morphizen::NodeAttributesPtr(MORPHIZEN_ORT_API(node_attributes_new)());

  const OrtOpAttr *const *ort_attr_values =
      reinterpret_cast<const OrtOpAttr *const *>(attr_values);
  for (size_t i = 0; i < attr_count; ++i) {
    const ONNX_NAMESPACE::AttributeProto *attr_proto =
        reinterpret_cast<const ONNX_NAMESPACE::AttributeProto *>(
            ort_attr_values[i]);
    morphizen::AttributeProtoPtr attr_proto_ptr =
        morphizen::attr_proto_clone(*attr_proto);
    MORPHIZEN_ORT_API(node_attributes_add)
    (*node_attrs, std::move(*attr_proto_ptr));
  }

  // node
  graph.add_node(model_name + "_op", domain, op_name, "", input_args,
                 output_args, std::move(node_attrs));

  // mark graph inputs/outputs
  std::vector<morphizen_cxx::NodeArgConstRef> inputs;
  for (auto &arg : input_args) {
    inputs.push_back(arg.value());
  }

  std::vector<morphizen_cxx::NodeArgConstRef> outputs;
  for (auto &arg : output_args) {
    outputs.push_back(arg.value());
  }

  graph.set_inputs(inputs);
  graph.set_outputs(outputs);

  // create session
  morphizen::ModelProto *model_proto =
      MORPHIZEN_ORT_API(model_to_proto)(*model);
  morphizen::DllSafe<std::string> model_string =
      MORPHIZEN_ORT_API(model_proto_serialize_as_string)(*model_proto);

  session_ = Ort::Session(env, model_string.get()->c_str(),
                          model_string.get()->size(), options);

  MORPHIZEN_ORT_API(model_proto_delete)(model_proto);
}

OpInvoker::~OpInvoker() {}

void OpInvoker::Invoke(const Ort::Value *input_values, size_t input_count,
                       Ort::Value *output_values, size_t output_count) {
  Ort::RunOptions run_options{nullptr};
  const OrtMemoryInfo *const *mem_info_arr{nullptr};
  Invoke(run_options, input_values, input_count, output_values, output_count,
         mem_info_arr);
}

static bool HasValue(const Ort::Value &value) {
  return static_cast<const OrtValue *>(value) && value.HasValue();
}

void OpInvoker::Invoke(const Ort::RunOptions &run_options,
                       const Ort::Value *input_values, size_t input_count,
                       Ort::Value *output_values, size_t output_count,
                       const OrtMemoryInfo *const *mem_info_arr) {
  std::vector<std::string> input_names = session_.GetInputNames();
  std::vector<std::string> output_names = session_.GetOutputNames();

  CHECK_EQ(input_names.size(), input_count)
      << "input_names.size(): " << input_names.size()
      << " and input_count: " << input_count << " should be equal";
  CHECK_EQ(output_names.size(), output_count)
      << "output_names.size(): " << output_names.size()
      << " and output_count: " << output_count << " should be equal";

  Ort::MemoryInfo default_cpu_mem_info = CreateDefaultCpuMemInfo();

  // Run with IO Binding
  try {
    Ort::IoBinding binder(session_);

    for (size_t i = 0; i < input_count; ++i) {
      binder.BindInput(input_names[i].c_str(), input_values[i]);
    }

    for (size_t i = 0; i < output_count; ++i) {
      if (HasValue(output_values[i])) {
        binder.BindOutput(output_names[i].c_str(), output_values[i]);
      } else {
        binder.BindOutput(output_names[i].c_str(), mem_info_arr
                                                       ? mem_info_arr[i]
                                                       : default_cpu_mem_info);
      }
    }

    session_.Run(run_options, binder);

    std::vector<Ort::Value> bound_output_values = binder.GetOutputValues();

    CHECK_EQ(bound_output_values.size(), output_count)
        << "bound_output_values.size(): " << bound_output_values.size()
        << " and output_count: " << output_count << " should be equal";

    for (size_t i = 0; i < output_count; ++i) {
      if (!HasValue(output_values[i])) {
        output_values[i] = std::move(bound_output_values[i]);
      }
    }
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "Session Run Error: " << e.what() << "\n";
    throw e;
  }
}

} // namespace morphizen
