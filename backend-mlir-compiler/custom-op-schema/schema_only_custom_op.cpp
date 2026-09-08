/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Schema-only registration for ORT custom ops that only need to let ORT
// load/Resolve a graph containing them (see hipep::SchemaOnlyCustomOp in
// schema_only_custom_op.hpp for why one class covers every op here).
//
// Add a new op by adding one (domain, name[, shape_infer_fn]) entry to
// kSchemaOnlyOps below -- no new file, no CMakeLists.txt edit. Each entry
// must set shape_infer_fn (e.g. PropagatePrimaryInputOutput). If an op needs
// real type/shape validation, give it its own ONNX-to-HIP conversion pattern
// instead of a schema-only entry here.
#include "./schema_only_custom_op.hpp"
#include <morphizen/morphizen.hpp>
#include <morphizen/op_def.hpp>

#include <memory>

namespace {

// Copy input-0 type/shape to output-0. Uses the C API directly on
// OrtShapeInferContext (do not construct Ort::ShapeInferContext here: its
// ctor also calls GetInputTypeShape and double-fetching corrupts ORT state).
OrtStatusPtr ORT_API_CALL PropagatePrimaryInputOutput(
    const OrtCustomOp *, OrtShapeInferContext *ort_ctx) {
  const OrtApi &api = Ort::GetApi();
  size_t input_count = 0;
  if (OrtStatus *status =
          api.ShapeInferContext_GetInputCount(ort_ctx, &input_count)) {
    return status;
  }
  if (input_count == 0) {
    return Ort::Status(
               "schema-only custom op requires at least one input for output "
               "type inference",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  OrtTensorTypeAndShapeInfo *input_info = nullptr;
  if (OrtStatus *status =
          api.ShapeInferContext_GetInputTypeShape(ort_ctx, 0, &input_info)) {
    return status;
  }

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (OrtStatus *status = api.GetTensorElementType(input_info, &elem_type)) {
    return status;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    return Ort::Status(
               "schema-only custom op cannot infer output type from undefined "
               "primary input",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  // ORT retains input_info when SetOutputTypeShape succeeds; do not release.
  return api.ShapeInferContext_SetOutputTypeShape(ort_ctx, 0, input_info);
}

struct SchemaOnlyOpSpec {
  const char *domain;
  const char *name;
  hipep::SchemaOnlyShapeInferFn shape_infer_fn = nullptr;
};

constexpr SchemaOnlyOpSpec kSchemaOnlyOps[] = {
    // Each row owns a distinct SchemaOnlyCustomOp instance with its own
    // OrtCustomOp::InferOutputShapeFn; shape_infer_fn is per op, not global.
    {"com.amd", "QMoE", PropagatePrimaryInputOutput},
    // {"domain", "OpName", MyShapeInferFn},
};

int RegisterSchemaOnlyOps(void *state, add_op_t add_op) {
  struct Register : OpRegister {
    Register(void *s, add_op_t a) : OpRegister(s, a) {}
    int register_ops() override {
      for (const auto &spec : kSchemaOnlyOps) {
        if (spec.shape_infer_fn == nullptr) {
          continue;
        }
        AddOp(spec.domain, std::make_unique<hipep::SchemaOnlyCustomOp>(
                               spec.name, spec.shape_infer_fn));
      }
      return 0;
    }
  } reg(state, add_op);
  return reg.register_ops();
}

static ::morphizen::StaticPluginRegister
    g_register("hipep-custom-op-schema", "morphizen_register_ops",
               (void *)&RegisterSchemaOnlyOps);

} // namespace
