/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "dynamic_dispatch.h"
#include "../hipdnn_ep_errors.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <vector>

// Stable C ABI; avoids C++ vtable/template issues across the DLL boundary.
#include <ops/dd_c_api.h>

//===----------------------------------------------------------------------===//
// Hardcoded DynamicDispatch configuration
//===----------------------------------------------------------------------===//
//
// The compiler does not yet carry model selection or QDQ data to the runtime,
// so the xclbin/model names are fixed and the QDQ tensors are zero-filled.
// This exercises the DD call sequence but does not produce correct numerics.
//
// DynamicDispatch's tests/test_c_api.cpp (test_wrap_dd_matmul) issues the same
// dd_* calls with the same configuration; keep the two in sync.
//===----------------------------------------------------------------------===//

namespace {

constexpr size_t kQdqParamSize = 16;
constexpr const char *kDesignParam = "8x4";
constexpr const char *kGemmModelName = "8x4_PSMU_ST1_v3.2";
constexpr const char *kConvModelName = "8x4PSP1";

// combined_gemm only runs exact {Bin, B, M, K, N} shapes. With Bin == 1 and
// M == 1 the [B, 1, N] output is byte-identical to a 2-D [1, B * N] result, so
// a 2-D GEMM [1, K] x [K, B * N] maps onto these kernels.
struct DDGemmShape {
  int bin, b, m, k, n;
};
constexpr DDGemmShape kGemmShapes[] = {
    {1, 3, 1, 1024, 64},
    {1, 6, 1, 1024, 64},
};

const DDGemmShape *find_gemm_shape(int64_t M, int64_t N, int64_t K) {
  for (const DDGemmShape &s : kGemmShapes) {
    if (M == s.m && K == s.k && N == int64_t(s.b) * s.n) {
      return &s;
    }
  }
  return nullptr;
}

dd_tensor_t make_dd_tensor(const void *data,
                           std::initializer_list<size_t> shape,
                           const char *dtype) {
  dd_tensor_t t{};
  t.data = const_cast<void *>(data);
  for (size_t d : shape) {
    t.shape[t.ndim++] = d;
  }
  t.dtype = dtype;
  return t;
}

size_t round_up8(int64_t n) { return (size_t(n) + 7) / 8 * 8; }

bool skip_op_state_cache() {
  const char *env = std::getenv("HIPEP_DD_DEBUG_NO_GET_OP_STATE");
  return env && env[0] == '1';
}

// Flushes each line so the log survives a crash inside DD (an SEH fault there
// is not caught by the C API's C++ exception guard).
void dd_log(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fflush(stderr);
}

// DD reads and writes tensors from the host, but the EP's buffers are HIP
// device allocations, so every DD tensor is staged through host memory with a
// synchronized copy on the runtime stream.
bool copy_on_stream(RuntimeState *state, void *dst, const void *src,
                    size_t bytes, hipMemcpyKind kind, const char *what) {
  hipError_t err = hipMemcpyAsync(dst, src, bytes, kind, state->stream);
  if (err == hipSuccess) {
    err = hipStreamSynchronize(state->stream);
  }
  if (err != hipSuccess) {
    dd_log("[DD] %s: copy failed: %s\n", what, hipGetErrorString(err));
    return false;
  }
  return true;
}

bool to_host(RuntimeState *state, void *dst, const void *src, size_t bytes,
             const char *what) {
  return copy_on_stream(state, dst, src, bytes, hipMemcpyDeviceToHost, what);
}

bool to_device(RuntimeState *state, void *dst, const void *src, size_t bytes,
               const char *what) {
  return copy_on_stream(state, dst, src, bytes, hipMemcpyHostToDevice, what);
}

} // namespace

//===----------------------------------------------------------------------===//
// Op-State Management for DynamicDispatch Operators
//===----------------------------------------------------------------------===//
//
// DD operators have significant setup cost (xclbin load, instruction registry,
// weight formatting), so the handle is created and its weights initialized
// once, then cached in the op's RuntimeState slot.
//
// The host weight copy and QDQ buffers stay alive for the handle's lifetime in
// case DD keeps references to the const-param data. The activation staging
// buffers are reused across inferences.
//
// The slot may already hold another op's state: the op's OpStateOpInterface
// constructor runs at session init regardless of the DD lowering (hip.matmul
// installs a hipBLASLt MatmulState). A DD state is recognized by its deletor,
// and any foreign occupant is destroyed and replaced on first use.
//===----------------------------------------------------------------------===//

struct DDGemmState : OpStateT<DDGemmState> {
  dd_gemm_handle_t handle = nullptr;
  DDGemmShape shape{};
  std::vector<uint8_t> weights;
  std::vector<int64_t> qdq;
  std::vector<int32_t> qdq_params;
  std::vector<uint16_t> in_host;
  std::vector<uint16_t> out_host;

  static void destroy(OpState *p) { delete static_cast<DDGemmState *>(p); }
  DDGemmState() { deletor = &destroy; }
  ~DDGemmState() { dd_combined_gemm_destroy(handle); }
};

struct DDConvState : OpStateT<DDConvState> {
  dd_conv_handle_t handle = nullptr;
  std::vector<uint8_t> weights;
  std::vector<int64_t> qdq;
  std::vector<int32_t> qdq_params;
  std::vector<uint16_t> in_host;
  std::vector<uint16_t> out_host;

  static void destroy(OpState *p) { delete static_cast<DDConvState *>(p); }
  DDConvState() { deletor = &destroy; }
  ~DDConvState() { dd_iconv_destroy(handle); }
};

template <class T> static T *find_dd_state(RuntimeState *state, int32_t slot) {
  auto *p = static_cast<OpState *>(hipdnn_ep_op_state_get(state, slot));
  return p && p->deletor == &T::destroy ? static_cast<T *>(p) : nullptr;
}

template <class T>
static void store_dd_state(RuntimeState *state, int32_t slot,
                           std::unique_ptr<T> value) {
  auto *old = static_cast<OpState *>(hipdnn_ep_op_state_get(state, slot));
  if (old && old->deletor) {
    old->deletor(old);
  }
  hipdnn_ep_op_state_set(state, slot, value.release());
}

//===----------------------------------------------------------------------===//
// wrap_dd_matmul - GEMM/MatMul via DynamicDispatch combined_gemm
//===----------------------------------------------------------------------===//

static std::unique_ptr<DDGemmState> create_gemm_state(RuntimeState *state,
                                                      const DDGemmShape &s,
                                                      const void *weights_dev) {
  dd_combined_gemm_config_t cfg{};
  cfg.a_dtype = "uint16";
  cfg.b_dtype = "uint8";
  cfg.c_dtype = "uint16";
  cfg.design_param = kDesignParam;
  cfg.model_name = kGemmModelName;
  cfg.input_shape[0] = s.bin;
  cfg.input_shape[1] = s.b;
  cfg.input_shape[2] = s.m;
  cfg.input_shape[3] = s.k;
  cfg.input_shape[4] = s.n;

  dd_log("[DD] wrap_dd_matmul: dd_combined_gemm_create model=%s "
         "Bin=%d B=%d M=%d K=%d N=%d\n",
         cfg.model_name, s.bin, s.b, s.m, s.k, s.n);
  auto gemm = DDGemmState::create();
  gemm->shape = s;
  gemm->handle = dd_combined_gemm_create(&cfg);
  if (!gemm->handle) {
    dd_log("[DD] wrap_dd_matmul: dd_combined_gemm_create failed\n");
    return nullptr;
  }

  const size_t B = s.b, K = s.k, N = s.n;
  // Reinterprets the (f16) weight buffer as uint8 [B, K, N]; this reads
  // B*K*N bytes, which is within the K x (B*N) f16 buffer.
  gemm->weights.resize(B * K * N);
  if (!to_host(state, gemm->weights.data(), weights_dev, gemm->weights.size(),
               "wrap_dd_matmul weights")) {
    return nullptr;
  }
  gemm->qdq.assign(B * N, 0);
  gemm->qdq_params.assign(B * kQdqParamSize, 0);
  gemm->in_host.resize(size_t(s.bin) * s.m * K);
  gemm->out_host.resize(B * s.m * N);
  dd_tensor_t consts[3] = {
      make_dd_tensor(gemm->weights.data(), {B, K, N}, "uint8"),
      make_dd_tensor(gemm->qdq.data(), {B, N}, "int64"),
      make_dd_tensor(gemm->qdq_params.data(), {B, kQdqParamSize}, "int32"),
  };
  int rc = dd_combined_gemm_initialize_weights(gemm->handle, consts, 3);
  if (rc != DD_OK) {
    dd_log("[DD] wrap_dd_matmul: dd_combined_gemm_initialize_weights "
           "failed: rc=%d\n",
           rc);
    return nullptr;
  }
  return gemm;
}

int wrap_dd_matmul(RuntimeState *state, int32_t op_state_slot,
                   const void *input_a, const void *input_b, const void *bias,
                   void *output, int64_t M, int64_t N, int64_t K, double alpha,
                   double beta, int64_t transA, int64_t transB,
                   int64_t data_type) {
  if (!state) {
    dd_log("[DD] wrap_dd_matmul: null RuntimeState\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (data_type != HIPDNN_EP_DATATYPE_HALF) {
    dd_log("[DD] wrap_dd_matmul: unsupported data type %lld (combined_gemm "
           "requires float16 activations)\n",
           (long long)data_type);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  if (transA || transB) {
    dd_log("[DD] wrap_dd_matmul: transA/transB not supported\n");
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  const DDGemmShape *shape = find_gemm_shape(M, N, K);
  if (!shape) {
    dd_log("[DD] wrap_dd_matmul: no combined_gemm kernel for M=%lld N=%lld "
           "K=%lld; supported (M, K, N):",
           (long long)M, (long long)N, (long long)K);
    for (const DDGemmShape &s : kGemmShapes) {
      dd_log(" (%d, %d, %d)", s.m, s.k, s.b * s.n);
    }
    dd_log("\n");
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  // TODO: bias, alpha and beta are not supported by combined_gemm.
  (void)bias;
  (void)alpha;
  (void)beta;

  const bool cache = op_state_slot >= 0 && !skip_op_state_cache();
  DDGemmState *gemm =
      cache ? find_dd_state<DDGemmState>(state, op_state_slot) : nullptr;
  std::unique_ptr<DDGemmState> uncached;
  if (!gemm) {
    std::unique_ptr<DDGemmState> created =
        create_gemm_state(state, *shape, input_b);
    if (!created) {
      return -1;
    }
    gemm = created.get();
    if (cache) {
      store_dd_state(state, op_state_slot, std::move(created));
    } else {
      uncached = std::move(created);
    }
  }

  const DDGemmShape &s = gemm->shape;
  const size_t in_bytes = gemm->in_host.size() * sizeof(uint16_t);
  const size_t out_bytes = gemm->out_host.size() * sizeof(uint16_t);
  if (!to_host(state, gemm->in_host.data(), input_a, in_bytes,
               "wrap_dd_matmul input")) {
    return -1;
  }
  dd_tensor_t in =
      make_dd_tensor(gemm->in_host.data(),
                     {size_t(s.bin), size_t(s.m), size_t(s.k)}, "uint16");
  dd_tensor_t out = make_dd_tensor(
      gemm->out_host.data(), {size_t(s.b), size_t(s.m), size_t(s.n)}, "uint16");
  int rc = dd_combined_gemm_execute(gemm->handle, &in, &out);
  dd_log("[DD] wrap_dd_matmul: dd_combined_gemm_execute rc=%d\n", rc);
  if (rc != DD_OK) {
    return -1;
  }
  return to_device(state, output, gemm->out_host.data(), out_bytes,
                   "wrap_dd_matmul output")
             ? 0
             : -1;
}

//===----------------------------------------------------------------------===//
// wrap_dd_conv2d - Convolution via DynamicDispatch iconv
//===----------------------------------------------------------------------===//

int wrap_dd_conv2d(RuntimeState *state, int32_t op_state_slot,
                   const void *input, int64_t n, int64_t c, int64_t h,
                   int64_t w, const void *weights, int64_t k, const void *bias,
                   void *output, int64_t out_h, int64_t out_w, int64_t kernel_h,
                   int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                   int64_t pad_top, int64_t pad_left, int64_t pad_bottom,
                   int64_t pad_right, int64_t dilation_h, int64_t dilation_w,
                   int64_t group, int64_t data_type) {
  if (!state) {
    dd_log("[DD] wrap_dd_conv2d: null RuntimeState\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (data_type != HIPDNN_EP_DATATYPE_HALF) {
    dd_log("[DD] wrap_dd_conv2d: unsupported data type %lld (iconv requires "
           "float16 activations)\n",
           (long long)data_type);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  if (n != 1 || dilation_h != 1 || dilation_w != 1 || group != 1) {
    dd_log("[DD] wrap_dd_conv2d: unsupported n=%lld dilation=(%lld, %lld) "
           "group=%lld\n",
           (long long)n, (long long)dilation_h, (long long)dilation_w,
           (long long)group);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  // iconv reads/writes activations padded to a multiple of 8 channels; the
  // EP's buffers are unpadded, so other channel counts would overrun them.
  if (round_up8(c) != size_t(c) || round_up8(k) != size_t(k)) {
    dd_log("[DD] wrap_dd_conv2d: channels must be multiples of 8 "
           "(c=%lld k=%lld)\n",
           (long long)c, (long long)k);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }
  // TODO: bias and padding are not expressible in dd_iconv_config_t.
  (void)bias;
  (void)pad_top;
  (void)pad_left;
  (void)pad_bottom;
  (void)pad_right;

  const bool cache = op_state_slot >= 0 && !skip_op_state_cache();
  DDConvState *conv =
      cache ? find_dd_state<DDConvState>(state, op_state_slot) : nullptr;
  std::unique_ptr<DDConvState> uncached;
  if (!conv) {
    dd_iconv_config_t cfg{};
    cfg.a_dtype = "uint16";
    cfg.b_dtype = "uint8";
    cfg.c_dtype = "uint16";
    cfg.design_param = kDesignParam;
    cfg.model_name = kConvModelName;
    cfg.input_format = "NCHW";
    cfg.strides[0] = int(stride_h);
    cfg.strides[1] = int(stride_w);
    const int in_shape[4] = {int(n), int(c), int(h), int(w)};
    const int out_shape[4] = {int(n), int(k), int(out_h), int(out_w)};
    std::copy(in_shape, in_shape + 4, cfg.input_shape);
    std::copy(out_shape, out_shape + 4, cfg.output_shape);

    dd_log("[DD] wrap_dd_conv2d: dd_iconv_create model=%s in=[%d,%d,%d,%d] "
           "out=[%d,%d,%d,%d] strides=[%d,%d]\n",
           cfg.model_name, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
           out_shape[0], out_shape[1], out_shape[2], out_shape[3],
           cfg.strides[0], cfg.strides[1]);
    auto created = DDConvState::create();
    created->handle = dd_iconv_create(&cfg);
    if (!created->handle) {
      dd_log("[DD] wrap_dd_conv2d: dd_iconv_create failed\n");
      return -1;
    }

    // Reinterprets the (f16) weight buffer as uint8 [CO, CI, KY, KX].
    created->weights.resize(size_t(k) * c * kernel_h * kernel_w);
    if (!to_host(state, created->weights.data(), weights,
                 created->weights.size(), "wrap_dd_conv2d weights")) {
      return -1;
    }
    created->qdq.assign(size_t(k), 0);
    created->qdq_params.assign(kQdqParamSize, 0);
    created->in_host.resize(size_t(n) * c * h * w);
    created->out_host.resize(size_t(n) * k * out_h * out_w);
    dd_tensor_t consts[3] = {
        make_dd_tensor(
            created->weights.data(),
            {size_t(k), size_t(c), size_t(kernel_h), size_t(kernel_w)},
            "uint8"),
        make_dd_tensor(created->qdq.data(), {size_t(k)}, "int64"),
        make_dd_tensor(created->qdq_params.data(), {kQdqParamSize}, "int32"),
    };
    int rc = dd_iconv_initialize_weights(created->handle, consts, 3);
    if (rc != DD_OK) {
      dd_log("[DD] wrap_dd_conv2d: dd_iconv_initialize_weights failed: "
             "rc=%d\n",
             rc);
      return -1;
    }

    conv = created.get();
    if (cache) {
      store_dd_state(state, op_state_slot, std::move(created));
    } else {
      uncached = std::move(created);
    }
  }

  const size_t in_bytes = conv->in_host.size() * sizeof(uint16_t);
  const size_t out_bytes = conv->out_host.size() * sizeof(uint16_t);
  if (!to_host(state, conv->in_host.data(), input, in_bytes,
               "wrap_dd_conv2d input")) {
    return -1;
  }
  dd_tensor_t in =
      make_dd_tensor(conv->in_host.data(),
                     {size_t(n), size_t(c), size_t(h), size_t(w)}, "uint16");
  dd_tensor_t out = make_dd_tensor(
      conv->out_host.data(),
      {size_t(n), size_t(k), size_t(out_h), size_t(out_w)}, "uint16");
  int rc = dd_iconv_execute(conv->handle, &in, &out);
  dd_log("[DD] wrap_dd_conv2d: dd_iconv_execute rc=%d\n", rc);
  if (rc != DD_OK) {
    return -1;
  }
  return to_device(state, output, conv->out_host.data(), out_bytes,
                   "wrap_dd_conv2d output")
             ? 0
             : -1;
}

//===----------------------------------------------------------------------===//
// XRT Context Accessors
//===----------------------------------------------------------------------===//
//
// The XRT device/context is owned by DynamicDispatch behind the C API.
//===----------------------------------------------------------------------===//

extern "C" {

void *hipdnn_ep_state_get_xrt_device(RuntimeState *state) {
  (void)state;
  return nullptr;
}

void *hipdnn_ep_state_get_xrt_context(RuntimeState *state) {
  (void)state;
  return nullptr;
}

} // extern "C"
