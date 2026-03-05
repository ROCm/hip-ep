/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipCustomOp.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>
#include <sstream>

DEF_ENV_PARAM(HIP_COMPILER_DEBUG, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(HIP_COMPILER_DEBUG) >= n)

namespace hip_compilation {

namespace {

size_t ort_element_size(ONNXTensorElementDataType dtype) {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return 8;
  default: return 4;
  }
}

size_t onnx_elem_type_size(int elem_type) {
  switch (elem_type) {
  case 1:  return 4; // FLOAT
  case 6:  return 4; // INT32
  case 7:  return 8; // INT64
  case 10: return 2; // FLOAT16
  case 11: return 8; // DOUBLE
  case 16: return 2; // BFLOAT16
  default: return 4;
  }
}

// Marshal ORT input tensors into span_t
struct TensorData {
  std::vector<tensor_t> tensors;
  std::vector<std::vector<int64_t>> shapes;
  span_t span;
};

TensorData marshal_inputs(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  auto n = ctx.GetInputCount();

  TensorData data;
  data.tensors.resize(n);
  data.shapes.resize(n);

  for (size_t i = 0; i < n; ++i) {
    auto t = ctx.GetInput(i);
    auto info = t.GetTensorTypeAndShapeInfo();
    data.shapes[i] = info.GetShape();

    data.tensors[i].data = const_cast<void *>(t.GetTensorRawData());
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size = ort_element_size(info.GetElementType());
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();
  return data;
}

TensorData marshal_outputs(OrtKernelContext *context,
                           const std::vector<HipCustomOp::OutputMeta> &metas) {
  Ort::KernelContext ctx(context);

  TensorData data;
  data.tensors.resize(metas.size());
  data.shapes.resize(metas.size());

  for (size_t i = 0; i < metas.size(); ++i) {
    data.shapes[i] = metas[i].shape;
    auto t = ctx.GetOutput(i, data.shapes[i]);

    data.tensors[i].data = t.GetTensorMutableRawData();
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size = onnx_elem_type_size(metas[i].elem_type);
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();
  return data;
}

// Parse simple JSON-like metadata from MetaDefProto parameter string.
// Format: "artifact_filename=<name>;output:<name>,<rank>,<elem_type>,<dim0>x<dim1>x..."
std::pair<std::string, std::vector<HipCustomOp::OutputMeta>>
parseMetadata(const std::string &param) {
  std::string artifact;
  std::vector<HipCustomOp::OutputMeta> outputs;

  std::istringstream ss(param);
  std::string line;
  while (std::getline(ss, line, ';')) {
    if (line.substr(0, 18) == "artifact_filename=") {
      artifact = line.substr(18);
    } else if (line.substr(0, 7) == "output:") {
      HipCustomOp::OutputMeta om;
      std::istringstream ls(line.substr(7));
      std::string token;

      std::getline(ls, om.name, ',');

      std::getline(ls, token, ',');
      om.rank = std::stoi(token);

      std::getline(ls, token, ',');
      om.elem_type = std::stoi(token);

      std::getline(ls, token);
      std::istringstream ds(token);
      std::string dim;
      while (std::getline(ds, dim, 'x'))
        om.shape.push_back(std::stoll(dim));

      outputs.push_back(om);
    }
  }
  return {artifact, outputs};
}

std::vector<uint8_t>
loadArtifact(const std::shared_ptr<const morphizen::PassContext> &ctx,
             const std::string &filename) {
  auto stream = ctx->open_file_for_read(filename);
  if (!stream)
    LOG(FATAL) << "Failed to open artifact: " << filename;

  std::vector<uint8_t> bytes;
  bytes.reserve(1024 * 1024);
  const size_t chunk = 64 * 1024;
  uint8_t buf[chunk];
  while (true) {
    size_t n = stream->fread(buf, chunk);
    if (n == 0)
      break;
    bytes.insert(bytes.end(), buf, buf + n);
    if (n < chunk)
      break;
  }
  if (bytes.empty())
    LOG(FATAL) << "Empty artifact from EPContext";
  return bytes;
}

} // namespace

HipCustomOp::HipCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : morphizen::CustomOpImp(context, meta_def, model) {

  MY_LOG(1) << "HipCustomOp constructor";

  auto param = context->get_meta_def_param(*meta_def);
  auto [artifact_filename, outputs] = parseMetadata(param);
  output_metas_ = std::move(outputs);

  inference_state_ = customop::HipInferenceState::create(
      loadArtifact(context, artifact_filename));
}

void HipCustomOp::Compute(const OrtApi *api, OrtKernelContext *context) const {
  MY_LOG(2) << "HipCustomOp::Compute()";

  auto inputs = marshal_inputs(context);
  auto outputs = marshal_outputs(context, output_metas_);

  int ret = inference_state_->compute(&inputs.span, &outputs.span);
  if (ret != 0)
    LOG(ERROR) << "inference_compute() failed with code: " << ret;
}

} // namespace hip_compilation
