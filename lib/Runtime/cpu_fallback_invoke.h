/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "debug_log.h"
#include "hipdnn_ep_runtime.h"

#include <cstdint>

//===----------------------------------------------------------------------===//
// Generic debug CPU fallback staging (model.dll side).
//
// Fill `HipdnnCpuFbGenericDesc` with GPU device pointers + shapes, then call
// hipdnn_cpu_fallback_try_generic. On success (return 0) the caller returns
// from wrap_* without running the GPU kernel.
//===----------------------------------------------------------------------===//

#define HIPDNN_CPU_FB_MAX_IO 8
#define HIPDNN_CPU_FB_MAX_ATTRS 8

typedef struct HipdnnCpuFbTensorView {
  void *device;
  int64_t rank;
  const int64_t *shape;
  int64_t num_elements;
  int64_t hip_dtype;
} HipdnnCpuFbTensorView;

typedef enum HipdnnCpuFbAttrKind {
  HIPDNN_CPU_FB_ATTR_INT = 0,
  HIPDNN_CPU_FB_ATTR_INTS = 1,
  HIPDNN_CPU_FB_ATTR_FLOAT = 2,
} HipdnnCpuFbAttrKind;

typedef struct HipdnnCpuFbAttrView {
  const char *name;
  HipdnnCpuFbAttrKind kind;
  int64_t i;
  const int64_t *ints;
  int32_t ints_len;
  float f;
} HipdnnCpuFbAttrView;

typedef struct HipdnnCpuFbGenericDesc {
  const char *op_name;
  const char *domain;
  int64_t opset;
  int32_t num_inputs;
  int32_t num_outputs;
  HipdnnCpuFbTensorView inputs[HIPDNN_CPU_FB_MAX_IO];
  HipdnnCpuFbTensorView outputs[HIPDNN_CPU_FB_MAX_IO];
  int32_t num_attrs;
  HipdnnCpuFbAttrView attrs[HIPDNN_CPU_FB_MAX_ATTRS];
} HipdnnCpuFbGenericDesc;

// Host-side mirror passed to the EP after D2H staging.
typedef struct HipdnnCpuFbTensorHost {
  const void *host;
  void *host_mut;
  int64_t rank;
  const int64_t *shape;
  int64_t num_elements;
  int64_t hip_dtype;
} HipdnnCpuFbTensorHost;

typedef struct HipdnnCpuFbGenericHostDesc {
  const char *op_name;
  const char *domain;
  int64_t opset;
  int32_t num_inputs;
  int32_t num_outputs;
  HipdnnCpuFbTensorHost inputs[HIPDNN_CPU_FB_MAX_IO];
  HipdnnCpuFbTensorHost outputs[HIPDNN_CPU_FB_MAX_IO];
  int32_t num_attrs;
  HipdnnCpuFbAttrView attrs[HIPDNN_CPU_FB_MAX_ATTRS];
} HipdnnCpuFbGenericHostDesc;

// Returns 0 if CPU fallback succeeded (caller should return 0 from wrap_*).
// Returns 1 to continue with the GPU path.
// Returns -1 on unrecoverable error.
int hipdnn_cpu_fallback_try_generic(RuntimeState *state, void *stream,
                                    const char *env_token,
                                    const HipdnnCpuFbGenericDesc *desc);

// --- Convenience helpers (return 0 = fallback ok, 1 = use GPU, -1 = error) ---

static inline int hipdnn_cpu_fb_try_unary_1d(RuntimeState *state, void *stream,
                                             const char *op_name, void *input,
                                             void *output, int64_t n,
                                             int64_t dtype) {
  if (n <= 0)
    return 1;
  int64_t shape[1] = {n};
  HipdnnCpuFbGenericDesc fb{};
  fb.op_name = op_name;
  fb.opset = 13;
  fb.num_inputs = 1;
  fb.num_outputs = 1;
  fb.inputs[0] = {input, 1, shape, n, dtype};
  fb.outputs[0] = {output, 1, shape, n, dtype};
  return hipdnn_cpu_fallback_try_generic(state, stream, op_name, &fb);
}

static inline int hipdnn_cpu_fb_try_binary_1d(RuntimeState *state, void *stream,
                                              const char *op_name, void *a,
                                              void *b, void *output, int64_t n,
                                              int64_t in_dtype,
                                              int64_t out_dtype) {
  if (n <= 0)
    return 1;
  int64_t shape[1] = {n};
  HipdnnCpuFbGenericDesc fb{};
  fb.op_name = op_name;
  fb.opset = 13;
  fb.num_inputs = 2;
  fb.num_outputs = 1;
  fb.inputs[0] = {a, 1, shape, n, in_dtype};
  fb.inputs[1] = {b, 1, shape, n, in_dtype};
  fb.outputs[0] = {output, 1, shape, n, out_dtype};
  return hipdnn_cpu_fallback_try_generic(state, stream, op_name, &fb);
}

static inline int hipdnn_cpu_fb_try_reduce(
    RuntimeState *state, void *stream, const char *op_name, void *data,
    void *axes, void *output, int64_t data_rank, const int64_t *data_shape,
    int64_t data_num_elements, int64_t axes_rank, const int64_t *axes_shape,
    int64_t axes_num_elements, int64_t out_rank, const int64_t *out_shape,
    int64_t output_num_elements, int64_t dtype, int64_t keepdims,
    int64_t noop_with_empty_axes) {
  HipdnnCpuFbGenericDesc fb{};
  fb.op_name = op_name;
  fb.opset = 13;
  fb.num_outputs = 1;
  fb.outputs[0] = {output, out_rank, out_shape, output_num_elements, dtype};
  int32_t in = 0;
  fb.inputs[in++] = {data, data_rank, data_shape, data_num_elements, dtype};
  if (axes && axes_num_elements > 0) {
    fb.inputs[in++] = {axes, axes_rank, axes_shape, axes_num_elements,
                       HIPDNN_EP_DATATYPE_INT64};
  }
  fb.num_inputs = in;
  fb.num_attrs = 2;
  fb.attrs[0] = {"keepdims", HIPDNN_CPU_FB_ATTR_INT, keepdims, nullptr, 0, 0.f};
  fb.attrs[1] = {"noop_with_empty_axes", HIPDNN_CPU_FB_ATTR_INT,
               noop_with_empty_axes, nullptr, 0, 0.f};
  return hipdnn_cpu_fallback_try_generic(state, stream, op_name, &fb);
}

static inline int64_t hipdnn_cpu_fb_cast_to_onnx_proto(int64_t hip) {
  switch (hip) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return 1;
  case HIPDNN_EP_DATATYPE_HALF:
    return 10;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return 16;
  case HIPDNN_EP_DATATYPE_INT32:
    return 6;
  case HIPDNN_EP_DATATYPE_INT64:
    return 7;
  case HIPDNN_EP_DATATYPE_INT8:
    return 3;
  case HIPDNN_EP_DATATYPE_UINT8:
    return 2;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return 11;
  default:
    return -1;
  }
}

static inline int hipdnn_cpu_fb_try_cast(RuntimeState *state, void *stream,
                                        void *input, void *output, int64_t n,
                                        int64_t src_dtype, int64_t dst_dtype) {
  if (n <= 0)
    return 1;
  if (src_dtype == dst_dtype)
    return 1;
  int64_t shape[1] = {n};
  HipdnnCpuFbGenericDesc fb{};
  fb.op_name = "Cast";
  fb.opset = 13;
  fb.num_inputs = 1;
  fb.num_outputs = 1;
  fb.inputs[0] = {input, 1, shape, n, src_dtype};
  fb.outputs[0] = {output, 1, shape, n, dst_dtype};
  fb.num_attrs = 1;
  const int64_t to_proto = hipdnn_cpu_fb_cast_to_onnx_proto(dst_dtype);
  if (to_proto < 0)
    return 1;
  fb.attrs[0] = {"to", HIPDNN_CPU_FB_ATTR_INT, to_proto, nullptr, 0, 0.f};
  return hipdnn_cpu_fallback_try_generic(state, stream, "Cast", &fb);
}

static inline int hipdnn_cpu_fb_try_gather(
    RuntimeState *state, void *stream, void *data, void *indices, void *output,
    int64_t axis, int64_t data_rank, const int64_t *data_shape,
    int64_t data_num_elements, int64_t indices_rank,
    const int64_t *indices_shape, int64_t indices_num_elements,
    int64_t output_rank, const int64_t *output_shape,
    int64_t output_num_elements, int64_t data_dtype, int64_t indices_dtype) {
  HipdnnCpuFbGenericDesc fb{};
  fb.op_name = "Gather";
  fb.opset = 13;
  fb.num_inputs = 2;
  fb.num_outputs = 1;
  fb.inputs[0] = {data, data_rank, data_shape, data_num_elements, data_dtype};
  fb.inputs[1] = {indices,     indices_rank,     indices_shape,
                  indices_num_elements, indices_dtype};
  fb.outputs[0] = {output, output_rank, output_shape, output_num_elements,
                   data_dtype};
  fb.num_attrs = 1;
  fb.attrs[0] = {"axis", HIPDNN_CPU_FB_ATTR_INT, axis, nullptr, 0, 0.f};
  return hipdnn_cpu_fallback_try_generic(state, stream, "Gather", &fb);
}

static inline int hipdnn_cpu_fb_try_reduce_flat(
    RuntimeState *state, void *stream, const char *op_name, void *data,
    void *axes, void *output, int64_t data_num_elements,
    int64_t axes_num_elements, int64_t output_num_elements, int64_t dtype,
    int64_t keepdims, int64_t noop_with_empty_axes) {
  int64_t data_shape[1] = {data_num_elements};
  int64_t out_shape[1] = {output_num_elements};
  int64_t axes_shape[1] = {axes_num_elements > 0 ? axes_num_elements : 0};
  return hipdnn_cpu_fb_try_reduce(
      state, stream, op_name, data, axes, output, 1, data_shape,
      data_num_elements, axes_num_elements > 0 ? 1 : 0,
      axes_num_elements > 0 ? axes_shape : nullptr, axes_num_elements, 1,
      out_shape, output_num_elements, dtype, keepdims, noop_with_empty_axes);
}
