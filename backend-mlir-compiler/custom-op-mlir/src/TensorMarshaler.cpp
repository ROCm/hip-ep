/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TensorMarshaler.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "custom_op_mlir.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

InputTensors::InputTensors(std::vector<tensor_t> &&tensors,
                           std::vector<std::vector<int64_t>> &&shapes,
                           span_t &&span)
    : tensors_(std::move(tensors)), shapes_(std::move(shapes)),
      span_(std::move(span)) {}

InputTensors InputTensors::marshal(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();

  MY_LOG(2) << "Marshaling " << num_inputs << " input tensors";

  std::vector<tensor_t> tensors(num_inputs);
  std::vector<std::vector<int64_t>> shapes(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_tensor = ctx.GetInput(i);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    shapes[i] = tensor_info.GetShape();

    tensors[i].data = const_cast<void *>(input_tensor.GetTensorRawData());
    tensors[i].shape = shapes[i].data();
    tensors[i].rank = shapes[i].size();

    MY_LOG(3) << "Input[" << i << "]: rank=" << tensors[i].rank;
  }

  span_t span;
  span.data = tensors.data();
  span.count = tensors.size();

  return InputTensors(std::move(tensors), std::move(shapes), std::move(span));
}

span_t *InputTensors::span() { return &span_; }

OutputTensors::OutputTensors(std::vector<tensor_t> &&tensors,
                             std::vector<std::vector<int64_t>> &&shapes,
                             span_t &&span)
    : tensors_(std::move(tensors)), shapes_(std::move(shapes)),
      span_(std::move(span)) {}

OutputTensors
OutputTensors::marshal(OrtKernelContext *context,
                       const std::vector<std::vector<int64_t>> &output_shapes) {
  Ort::KernelContext ctx(context);
  auto num_outputs = ctx.GetOutputCount();

  MY_LOG(2) << "Marshaling " << num_outputs << " output tensors";

  std::vector<tensor_t> tensors(num_outputs);
  std::vector<std::vector<int64_t>> shapes = output_shapes;

  for (size_t i = 0; i < num_outputs; ++i) {
    // GetOutput allocates the tensor and returns a reference
    // We must not store the Ort::Value, just get the data pointer
    auto output_tensor = ctx.GetOutput(i, shapes[i]);

    tensors[i].data = output_tensor.GetTensorMutableRawData();
    tensors[i].shape = shapes[i].data();
    tensors[i].rank = shapes[i].size();

    MY_LOG(3) << "Output[" << i << "]: rank=" << tensors[i].rank;
  }

  span_t span;
  span.data = tensors.data();
  span.count = tensors.size();

  return OutputTensors(std::move(tensors), std::move(shapes), std::move(span));
}

span_t *OutputTensors::span() { return &span_; }

} // namespace customop
} // namespace mlir_compilation
