/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MlirCustomOp.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>

// Protobuf headers
#include "google/protobuf/util/json_util.h"
#include "metadata.pb.h"

// Component headers
#include "InferenceState.h"

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {

// Tensor marshaling state - holds tensors, shapes, and span
struct TensorData {
  std::vector<tensor_t> tensors;
  std::vector<std::vector<int64_t>> shapes; // Storage for shape arrays
  span_t span;
};

static size_t ort_element_size(ONNXTensorElementDataType dtype) {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return 1;
  default:
    return 4;
  }
}

static size_t onnx_elem_type_size(int elem_type) {
  switch (elem_type) {
  case 1:
    return 4; // FLOAT
  case 2:
    return 1; // UINT8
  case 3:
    return 1; // INT8
  case 5:
    return 2; // INT16
  case 6:
    return 4; // INT32
  case 7:
    return 8; // INT64
  case 9:
    return 1; // BOOL
  case 10:
    return 2; // FLOAT16
  case 11:
    return 8; // DOUBLE
  case 12:
    return 4; // UINT32
  case 13:
    return 8; // UINT64
  case 16:
    return 2; // BFLOAT16
  default:
    return 4;
  }
}

// Marshal input tensors from ORT context
TensorData marshal_input_tensors(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();

  MY_LOG(2) << "Marshaling " << num_inputs << " input tensors";

  TensorData data;
  data.tensors.resize(num_inputs);
  data.shapes.resize(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_tensor = ctx.GetInput(i);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    data.shapes[i] = tensor_info.GetShape();

    data.tensors[i].data = const_cast<void *>(input_tensor.GetTensorRawData());
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size =
        ort_element_size(tensor_info.GetElementType());

    MY_LOG(3) << "Input[" << i << "]: rank=" << data.tensors[i].rank
              << " element_size=" << data.tensors[i].element_size;
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();

  return data;
}

// Marshal output tensors from ORT context using metadata outputs
TensorData marshal_output_tensors(
    OrtKernelContext *context,
    const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs) {
  if (outputs.size() == 0) {
    LOG(FATAL) << "No output shapes in metadata";
  }

  Ort::KernelContext ctx(context);
  MY_LOG(2) << "Marshaling " << outputs.size() << " output tensors";

  TensorData data;
  data.tensors.resize(outputs.size());
  data.shapes.resize(outputs.size());

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_meta = outputs[i];
    data.shapes[i].assign(output_meta.shape().begin(),
                          output_meta.shape().end());

    // GetOutput allocates the tensor and returns a reference
    auto output_tensor = ctx.GetOutput(i, data.shapes[i]);

    data.tensors[i].data = output_tensor.GetTensorMutableRawData();
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size = onnx_elem_type_size(output_meta.elem_type());

    MY_LOG(3) << "Output[" << i << "]: rank=" << data.tensors[i].rank
              << " element_size=" << data.tensors[i].element_size;
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();

  return data;
}

namespace {

// Parse metadata from JSON string in MetaDefProto
// Logs FATAL and terminates on failure
mlir_metadata::Metadata parse_metadata_from_metadef(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def) {

  auto metadata_json = context->get_meta_def_param(*meta_def);
  mlir_metadata::Metadata metadata;
  auto status =
      google::protobuf::util::JsonStringToMessage(metadata_json, &metadata);

  if (!status.ok()) {
    LOG(FATAL) << "Failed to parse MLIR metadata: " << status.ToString();
  }

  MY_LOG(1) << "Parsed metadata - Artifact filename: "
            << metadata.artifact_filename();
  return metadata;
}

// Load artifact bytes from EPContext file stream
// Logs FATAL and terminates on failure
std::vector<uint8_t> load_artifact_from_epcontext(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::string &artifact_filename) {

  auto artifact_stream = context->open_file_for_read(artifact_filename);
  if (!artifact_stream) {
    LOG(FATAL) << "Failed to open artifact from EPContext: "
               << artifact_filename;
  }

  std::vector<uint8_t> artifact_bytes;
  artifact_bytes.reserve(1024 * 1024); // Start with 1MB

  const size_t chunk_size = 64 * 1024; // 64KB chunks
  uint8_t buffer[chunk_size];
  size_t total_read = 0;

  while (true) {
    size_t bytes_read = artifact_stream->fread(buffer, chunk_size);
    if (bytes_read == 0) {
      break;
    }
    artifact_bytes.insert(artifact_bytes.end(), buffer, buffer + bytes_read);
    total_read += bytes_read;

    if (bytes_read < chunk_size) {
      break; // EOF reached
    }
  }

  if (artifact_bytes.empty()) {
    LOG(FATAL) << "Failed to read artifact bytes from EPContext";
  }

  MY_LOG(1) << "Loaded artifact: " << total_read << " bytes";
  return artifact_bytes;
}
} // anonymous namespace

MlirCustomOp::MlirCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : morphizen::CustomOpImp(context, meta_def, model) {

  MY_LOG(1) << "MlirCustomOp constructor";
  // Parse metadata from JSON
  metadata_ = parse_metadata_from_metadef(context, meta_def);
  // Create inference state from DLL bytes (uses morphizen::Plugin)
  inference_state_ = customop::InferenceState::create(
      load_artifact_from_epcontext(context, metadata_.artifact_filename()));
}

void MlirCustomOp::Compute(const OrtApi *api, OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::Compute() called";

  auto inputs = marshal_input_tensors(context);
  auto outputs = marshal_output_tensors(context, metadata_.outputs());

  int ret = inference_state_->compute(&inputs.span, &outputs.span);
  if (ret != 0) {
    LOG(ERROR) << "inference_compute() failed with code: " << ret;
    // TODO: Throw ORT exception
  }

  MY_LOG(2) << "Compute completed successfully";
}

} // namespace mlir_compilation
