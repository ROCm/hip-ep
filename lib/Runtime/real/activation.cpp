/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type, bool &ok) {
  ok = true;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    fprintf(stderr, "[REAL] unsupported data_type %lld for MIOpen\n",
            (long long)data_type);
    ok = false;
    return miopenFloat;
  }
}

static miopenActivationMode_t hipdnn_ep_to_miopen_activation(int64_t mode,
                                                             bool &ok) {
  ok = true;
  switch (mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return miopenActivationLOGISTIC;
  case HIPDNN_EP_ACTIVATION_RELU:
    return miopenActivationRELU;
  case HIPDNN_EP_ACTIVATION_TANH:
    return miopenActivationTANH;
  case HIPDNN_EP_ACTIVATION_SOFTPLUS:
    return miopenActivationSOFTRELU;
  default:
    fprintf(stderr, "[REAL] unsupported activation_mode %lld for MIOpen\n",
            (long long)mode);
    ok = false;
    return miopenActivationLOGISTIC;
  }
}

//===----------------------------------------------------------------------===//
// Descriptor cache: 2 tensor descriptors + 1 activation descriptor created
// once per unique (num_elements, data_type, activation_mode) triple and
// reused for the process lifetime.
//===----------------------------------------------------------------------===//

struct ActivationCacheKey {
  int64_t num_elements, data_type, activation_mode;
  bool operator==(const ActivationCacheKey &o) const {
    return num_elements == o.num_elements && data_type == o.data_type &&
           activation_mode == o.activation_mode;
  }
};

struct ActivationCacheKeyHash {
  size_t operator()(const ActivationCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.num_elements);
    hash_combine_val(h, k.data_type);
    hash_combine_val(h, k.activation_mode);
    return h;
  }
};

/// Cached MIOpen descriptors for a single activation shape.
/// Ownership: descriptors are created in queryOrCreateActivation() and live
/// for the process lifetime (never destroyed individually).
struct ActivationCacheEntry {
  miopenTensorDescriptor_t inDesc, outDesc;
  miopenActivationDescriptor_t actDesc;
};

static std::unordered_map<ActivationCacheKey, ActivationCacheEntry,
                          ActivationCacheKeyHash>
    g_activation_cache;

static const ActivationCacheEntry *
queryOrCreateActivation(const ActivationCacheKey &key) {
  auto it = g_activation_cache.find(key);
  if (it != g_activation_cache.end())
    return &it->second;

  bool type_ok, act_ok;
  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type, type_ok);
  miopenActivationMode_t act =
      hipdnn_ep_to_miopen_activation(key.activation_mode, act_ok);
  if (!type_ok || !act_ok)
    return nullptr;
  int n = static_cast<int>(key.num_elements);

  ActivationCacheEntry e{};
  int result = 0;

  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.inDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.outDesc), cache_fail);
  {
    int dims[] = {1, 1, 1, n};
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.inDesc, dt, miopenTensorNCHW, dims, 4),
                      cache_fail);
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.outDesc, dt, miopenTensorNCHW, dims, 4),
                      cache_fail);
  }
  MIOPEN_CHECK_GOTO(miopenCreateActivationDescriptor(&e.actDesc), cache_fail);
  {
    // MIOpen's TANH mode computes y = alpha * tanh(beta * x); plain ONNX Tanh
    // needs alpha = beta = 1. (LOGISTIC/SOFTRELU ignore alpha/beta, so the
    // 0,0,0 default is only correct for those modes -- with the default,
    // TANH would degenerate to 0 * tanh(0) = 0.)
    //
    // alpha/beta are a pure function of `act` here, and the descriptor cache is
    // keyed by activation_mode (see ActivationCacheKey), so a TANH descriptor
    // can never be reused for a different mode (no alpha/beta aliasing). If a
    // future activation needs alpha/beta that vary independently of the mode,
    // add those params to ActivationCacheKey as well.
    double alpha = 0.0, beta = 0.0;
    if (act == miopenActivationTANH) {
      alpha = 1.0;
      beta = 1.0;
    }
    MIOPEN_CHECK_GOTO(
        miopenSetActivationDescriptor(e.actDesc, act, alpha, beta, 0.0),
        cache_fail);
  }
  goto cache_done;

cache_fail:
  if (e.actDesc)
    miopenDestroyActivationDescriptor(e.actDesc);
  if (e.outDesc)
    miopenDestroyTensorDescriptor(e.outDesc);
  if (e.inDesc)
    miopenDestroyTensorDescriptor(e.inDesc);
  return nullptr;

cache_done:
  auto [ins, _] = g_activation_cache.emplace(key, e);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateActivation: cached 2 tensor + 1 act "
                    "desc for num_elements=%lld data_type=%lld mode=%lld\n",
                    (long long)key.num_elements, (long long)key.data_type,
                    (long long)key.activation_mode);

  return &ins->second;
}

//===----------------------------------------------------------------------===//
// Generic MIOpen Activation Forward
//===----------------------------------------------------------------------===//
//
// Applies activation_mode element-wise using miopenActivationForward.
// Tensor is represented as flat 1D [1, 1, 1, num_elements] to satisfy
// MIOpen's 4D tensor descriptor requirement.
//===----------------------------------------------------------------------===//

int wrap_miopenActivationForward(RuntimeState *state, void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  OP_PROFILE(
      "activation",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_miopenActivationForward: null argument\n");
    return -1;
  }

  const char *act_name = hipdnn_ep_activation_name(activation_mode);
  const char *type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: activation=%s, "
      "num_elements=%lld, data_type=%s(%lld), element_size=%lld bytes, "
      "total_size=%lld bytes\n",
      act_name, (long long)num_elements, type_name, (long long)data_type,
      (long long)elem_size, (long long)(num_elements * elem_size));

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenActivationForward: null MIOpen handle\n");
    return -1;
  }

  ActivationCacheKey key{num_elements, data_type, activation_mode};
  const ActivationCacheEntry *c = queryOrCreateActivation(key);
  if (!c) {
    fprintf(stderr, "[REAL] wrap_miopenActivationForward: descriptor cache "
                    "creation failed\n");
    return -1;
  }

  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t st = miopenActivationForward(
      handle, c->actDesc, &alpha, c->inDesc, input, &beta, c->outDesc, output);
  if (st != miopenStatusSuccess) {
    fprintf(stderr,
            "[REAL] wrap_miopenActivationForward: "
            "miopenActivationForward failed (%d)\n",
            st);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: completed successfully\n");
  return 0;
}

//===----------------------------------------------------------------------===//
// GELU Activation (Custom HIP Kernel)
//===----------------------------------------------------------------------===//
//
// Applies GELU activation using custom HIP kernel (hip_elementwise_gelu).
// Supports two modes (per ONNX Gelu spec):
//   - Exact (erf):  GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
//   - Tanh approx:  GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 *
//   x³)))
// Supports data types: f32, f16, bf16, f64.
// MIOpen does not support GELU activation, so we use a custom kernel.
//===----------------------------------------------------------------------===//

static int hipdnn_ep_to_hip_dtype_elementwise_unary(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

// Standalone Softmax runtime entry point — called from
// `ConvertHipToLLVM`'s `MiopenSoftmaxOpLowering` for `onnx.Softmax` paths
// outside fused attention (vision encoder self-attention, primarily; text
// decoders fuse softmax inside hip.gqa and don't reach this path).
// Dispatches to `hip_softmax_row_2d_inplace` (custom HIP kernel) —
// row-wise softmax over a contiguous row-major [rows, cols] fp16 buffer
// with `cols` = the softmax axis size (ONNX Softmax axis = -1 over the
// last dim of the flattened input).
//
// fp16-only today. The buffer element type is taken on faith from the
// MemRef the lowering hands us; the runtime does not validate the dtype.
// If a future model needs fp32 softmax, plumb an `elem_size` (or dtype
// enum) through the lowering and dispatch in the kernel launcher.
//
// Symbol name is `hip_miopen_softmax` (not `hip_softmax`) to match the
// lowering side's `kMiopenSoftmax` constant. The kernel-level dispatch
// is unrelated to MIOpen.
//
// Reusing `hip_gqa_softmax_inplace` (from gqa_kernel.hip) would not work:
// that kernel is column-wise over a GQA-specific layout AND its launcher
// only spawns `total_head_queries` blocks, not `total_head_queries * cols`.
// A standalone softmax wired to that path produces NaN downstream.
extern "C" int hip_miopen_softmax(void *state, const void *input, void *output,
                                  int64_t rows, int64_t cols) {
  RuntimeState *st = static_cast<RuntimeState *>(state);
  OP_PROFILE(
      "softmax",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld", (long long)rows, (long long)cols);
        return std::string(b);
      },
      st);
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] hip_miopen_softmax: null argument\n");
    return -1;
  }
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr,
            "[REAL] hip_miopen_softmax: invalid dims rows=%lld cols=%lld\n",
            (long long)rows, (long long)cols);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(st);

  // Copy input -> output (kernel is in-place; both are fp16 contiguous
  // buffers from `extractContiguousMemRefPtr`).
  hipError_t err = hipMemcpyAsync(
      output, input, static_cast<size_t>(rows) * cols * sizeof(uint16_t),
      hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] hip_miopen_softmax: hipMemcpyAsync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] hip_miopen_softmax: rows=%lld cols=%lld\n",
                    (long long)rows, (long long)cols);
  return hip_softmax_row_2d_inplace(stream, output, static_cast<int>(rows),
                                    static_cast<int>(cols));
}

int wrap_gelu(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type, int64_t approximate) {
  OP_PROFILE(
      "gelu",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_gelu: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);

  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_gelu: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  const char *mode_name = (approximate == 1) ? "tanh" : "erf";
  RUNTIME_DEBUG_LOG("[REAL] wrap_gelu: num_elements=%lld, data_type=%s(%lld), "
                    "approximate=%s(%lld), element_size=%lld bytes, "
                    "total_size=%lld bytes\n",
                    (long long)num_elements, type_name, (long long)data_type,
                    mode_name, (long long)approximate, (long long)elem_size,
                    (long long)(num_elements * elem_size));

  // Call custom HIP kernel with approximate mode
  int result = hip_elementwise_gelu(stream, input, output, num_elements,
                                    hip_dtype, approximate);

  if (result != 0) {
    fprintf(stderr, "[REAL] wrap_gelu: kernel launch failed (%d)\n", result);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_gelu: completed successfully\n");
  return 0;
}

int wrap_leaky_relu(RuntimeState *state, void *input, void *output,
                    int64_t num_elements, int64_t data_type, double alpha) {
  OP_PROFILE(
      "leaky_relu",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_leaky_relu: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);

  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_leaky_relu: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_leaky_relu: num_elements=%lld, "
                    "data_type=%s(%lld), alpha=%f\n",
                    (long long)num_elements, hipdnn_ep_datatype_name(data_type),
                    (long long)data_type, alpha);

  int result =
      hip_leaky_relu(stream, input, output, num_elements, hip_dtype, alpha);

  if (result != 0) {
    fprintf(stderr, "[REAL] wrap_leaky_relu: kernel launch failed (%d)\n",
            result);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_leaky_relu: completed successfully\n");
  return 0;
}
