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
#include "mlir_compilation.pb.h"

// Component headers
#include "ArtifactLoader.h"
#include "InferenceState.h"
#include "LlvmJitLoader.h"
#include "MetadataParser.h"
#include "NativeDllLoader.h"
#include "TensorMarshaler.h"

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {

MlirCustomOp::MlirCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : morphizen::CustomOpImp(context, meta_def, model) {

  MY_LOG(1) << "MlirCustomOp constructor";

  // Parse metadata
  auto proto_opt = customop::MetadataParser::parse(context, meta_def);
  if (!proto_opt) {
    LOG(FATAL) << "Failed to parse MLIR compilation metadata";
    return;
  }
  auto proto = *proto_opt;

  // Load artifact bytes
  auto artifact_opt = customop::ArtifactLoader::load(
      context, proto.artifact_filename(), proto.artifact_size());
  if (!artifact_opt) {
    LOG(FATAL) << "Failed to load artifact from EPContext";
    return;
  }
  auto artifact = *artifact_opt;

  // Load artifact based on format and transfer ownership to InferenceState
  if (proto.artifact_format() == "native") {
    auto dll_opt = customop::DllHandle::load(artifact.bytes);
    if (!dll_opt) {
      LOG(FATAL) << "Failed to load native DLL";
      return;
    }

    // Transfer ownership to InferenceState
    inference_state_ = customop::InferenceState::create(std::move(*dll_opt));
    if (!inference_state_) {
      LOG(FATAL) << "Failed to initialize inference state";
      return;
    }

  } else if (proto.artifact_format() == "llvm_ir") {
    auto jit_opt = customop::JitHandle::compile(artifact.bytes);
    if (!jit_opt) {
      LOG(FATAL) << "Failed to compile LLVM IR";
      return;
    }

    // Transfer ownership to InferenceState
    inference_state_ = customop::InferenceState::create(std::move(*jit_opt));
    if (!inference_state_) {
      LOG(FATAL) << "Failed to initialize inference state";
      return;
    }

  } else {
    LOG(FATAL) << "Unknown artifact format: " << proto.artifact_format();
    return;
  }

  // Extract output shapes from metadata
  for (int i = 0; i < proto.outputs_size(); ++i) {
    const auto &output_meta = proto.outputs(i);
    std::vector<int64_t> shape;
    for (int j = 0; j < output_meta.shape_size(); ++j) {
      shape.push_back(output_meta.shape(j));
    }
    output_shapes_.push_back(shape);
    MY_LOG(1) << "Output " << i << " (" << output_meta.name() << "): "
              << "rank=" << output_meta.rank()
              << ", dtype=" << output_meta.dtype();
  }
}

void MlirCustomOp::Compute(const OrtApi *api, OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::Compute() called";

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  MY_LOG(2) << "num_inputs: " << num_inputs << ", num_outputs: " << num_outputs;

  // Marshal input tensors
  auto inputs = customop::InputTensors::marshal(context);

  // Marshal output tensors using shapes from metadata
  std::vector<std::vector<int64_t>> output_shapes;
  if (!output_shapes_.empty()) {
    // Use metadata shapes (correct approach)
    output_shapes = output_shapes_;
    MY_LOG(2) << "Using output shapes from metadata: " << output_shapes.size()
              << " outputs";
  } else {
    // Fallback: assume pass-through (backwards compatibility)
    MY_LOG(1) << "WARNING: No output shapes in metadata, falling back to "
                 "pass-through assumption";
    output_shapes.resize(num_outputs);
    if (num_inputs > 0) {
      auto input_tensor = ctx.GetInput(0);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      auto input_shape = tensor_info.GetShape();
      for (size_t i = 0; i < num_outputs; ++i) {
        output_shapes[i] = input_shape;
      }
    }
  }
  auto outputs = customop::OutputTensors::marshal(context, output_shapes);

  // Execute inference computation
  int ret = inference_state_->compute(inputs.span(), outputs.span());
  if (ret != 0) {
    LOG(ERROR) << "inference_compute() failed with code: " << ret;
    // TODO: Throw ORT exception
  }

  MY_LOG(2) << "Compute completed successfully";
}

} // namespace mlir_compilation
