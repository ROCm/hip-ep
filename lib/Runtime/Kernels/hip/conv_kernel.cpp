/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

// =============================================================================
// 2D convolution forward via Composable Kernel.
//
// Uses CK's pre-instantiated DeviceGroupedConvFwdMultipleABD instance factory
// (composable_kernel::device_conv_operations). ONNX/MIOpen tensors are packed
// NCHW; the CK NCHW-compatible layout family is NGCHW / GKCYX / NGKHW. For the
// standard-conv case (group == 1) an NGCHW tensor with G=1 is byte-identical to
// NCHW, GKCYX with G=1 is KCYX (the ONNX weight [K,C,kH,kW]), and NGKHW with
// G=1 is NKHW. So we can hand CK the raw NCHW / KCYX buffers unchanged and only
// describe them with canonical [G,N,C,...] lengths + NGCHW-family strides.
//
// NOTE: The NGCHW instances in CK are XDL (CDNA/MI) kernels. On RDNA GPUs the
// only conv instances are WMMA and are channels-last (NHWGC) only, so on such
// GPUs GetInstances() may return instances that none IsSupportedArgument()
// accepts. In that case hip_conv_forward returns non-zero and the caller
// (wrap_miopenConvolutionForward) falls back to MIOpen. Set HIPDNN_EP_DEBUG=1
// to see the instance/support counts.
//
// Grouped conv (group > 1) stays on MIOpen. Bias is NOT fused here; the runtime
// wrapper adds it afterwards.
// =============================================================================

namespace {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr ck::index_t kNumDimSpatial = 2;
// G, N, C plus the two spatial dims -> rank-5 length/stride arrays.
constexpr ck::index_t kNonSpatial = 3;
constexpr ck::index_t kRank = kNumDimSpatial + kNonSpatial;

using InLayout = ck::tensor_layout::convolution::NGCHW;
using WeiLayout = ck::tensor_layout::convolution::GKCYX;
using OutLayout = ck::tensor_layout::convolution::NGKHW;

template <typename InT, typename WeiT, typename OutT>
using DeviceConvOp =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        kNumDimSpatial, InLayout, WeiLayout, ck::Tuple<>, OutLayout, InT, WeiT,
        ck::Tuple<>, OutT, PassThrough, PassThrough, PassThrough>;

struct ConvKey {
  int dtype;
  int64_t n, c, h, w, k, kh, kw, oh, ow, sh, sw, pt, pl, pb, pr, dh, dw;
  bool operator==(const ConvKey &o) const {
    return dtype == o.dtype && n == o.n && c == o.c && h == o.h && w == o.w &&
           k == o.k && kh == o.kh && kw == o.kw && oh == o.oh && ow == o.ow &&
           sh == o.sh && sw == o.sw && pt == o.pt && pl == o.pl && pb == o.pb &&
           pr == o.pr && dh == o.dh && dw == o.dw;
  }
};

struct ConvKeyHash {
  size_t operator()(const ConvKey &k) const {
    size_t h = std::hash<int>{}(k.dtype);
    auto mix = [&](int64_t v) {
      h ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    mix(k.n); mix(k.c); mix(k.h); mix(k.w); mix(k.k); mix(k.kh); mix(k.kw);
    mix(k.oh); mix(k.ow); mix(k.sh); mix(k.sw); mix(k.pt); mix(k.pl); mix(k.pb);
    mix(k.pr); mix(k.dh); mix(k.dw);
    return h;
  }
};

std::mutex g_mu;
// Caches the index of the first CK instance that supports a given problem so
// repeated inferences of the same layer skip the linear IsSupportedArgument
// scan. Process-lifetime; entries are POD ints.
std::unordered_map<ConvKey, int, ConvKeyHash> g_instance_cache;

template <typename InT, typename WeiT, typename OutT>
int run(hipStream_t stream, const void *input, const void *weights,
        void *output, const ConvKey &key) {
  using Op = DeviceConvOp<InT, WeiT, OutT>;
  const auto &op_ptrs = ck::tensor_operation::device::instance::
      DeviceOperationInstanceFactory<Op>::GetInstances();
  CUSTOM_KERNELS_DEBUG_LOG("[CK conv] dtype=%d NxCxHxW=%lldx%lldx%lldx%lld "
                           "K=%lld kHxkW=%lldx%lld instances=%zu\n",
                           key.dtype, (long long)key.n, (long long)key.c,
                           (long long)key.h, (long long)key.w,
                           (long long)key.k, (long long)key.kh,
                           (long long)key.kw, op_ptrs.size());
  if (op_ptrs.empty())
    return -1;

  // MakeArgumentPointer takes lengths in canonical [G, N, C, {spatial}] order
  // (weight [G, K, C, {spatial}], output [G, N, K, {spatial}]); the LAYOUT type
  // parameter and the STRIDES encode the physical memory order. For NGCHW /
  // GKCYX / NGKHW with G == 1 the physical buffers are plain NCHW / KCYX / NKHW.
  const ck::index_t G = 1;
  const ck::index_t N = (ck::index_t)key.n, C = (ck::index_t)key.c;
  const ck::index_t H = (ck::index_t)key.h, W = (ck::index_t)key.w;
  const ck::index_t K = (ck::index_t)key.k;
  const ck::index_t R = (ck::index_t)key.kh, S = (ck::index_t)key.kw;
  const ck::index_t OH = (ck::index_t)key.oh, OW = (ck::index_t)key.ow;

  // Input NGCHW (physical N,G,C,H,W); canonical order [G,N,C,H,W].
  std::array<ck::index_t, kRank> in_len = {G, N, C, H, W};
  std::array<ck::index_t, kRank> in_str = {C * H * W, G * C * H * W, H * W, W,
                                           1};
  // Weight GKCYX (physical G,K,C,R,S); canonical order [G,K,C,R,S].
  std::array<ck::index_t, kRank> wei_len = {G, K, C, R, S};
  std::array<ck::index_t, kRank> wei_str = {K * C * R * S, C * R * S, R * S, S,
                                            1};
  // Output NGKHW (physical N,G,K,H,W); canonical order [G,N,K,OH,OW].
  std::array<ck::index_t, kRank> out_len = {G, N, K, OH, OW};
  std::array<ck::index_t, kRank> out_str = {K * OH * OW, G * K * OH * OW,
                                            OH * OW, OW, 1};

  std::array<ck::index_t, kNumDimSpatial> conv_str = {(ck::index_t)key.sh,
                                                      (ck::index_t)key.sw};
  std::array<ck::index_t, kNumDimSpatial> conv_dil = {(ck::index_t)key.dh,
                                                      (ck::index_t)key.dw};
  std::array<ck::index_t, kNumDimSpatial> lpad = {(ck::index_t)key.pt,
                                                  (ck::index_t)key.pl};
  std::array<ck::index_t, kNumDimSpatial> rpad = {(ck::index_t)key.pb,
                                                  (ck::index_t)key.pr};

  // GetInstances() yields unique_ptr<Op>; accept by generic ref.
  auto make_arg = [&](auto &op) {
    return op->MakeArgumentPointer(
        input, weights, std::array<const void *, 0>{}, output, in_len, in_str,
        wei_len, wei_str, std::array<std::array<ck::index_t, kRank>, 0>{},
        std::array<std::array<ck::index_t, kRank>, 0>{}, out_len, out_str,
        conv_str, conv_dil, lpad, rpad, PassThrough{}, PassThrough{},
        PassThrough{});
  };

  int idx = -1;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_instance_cache.find(key);
    if (it != g_instance_cache.end() && it->second >= 0 &&
        it->second < (int)op_ptrs.size())
      idx = it->second;
  }

  if (idx < 0) {
    int supported = 0;
    for (int i = 0; i < (int)op_ptrs.size(); ++i) {
      auto arg = make_arg(op_ptrs[i]);
      if (op_ptrs[i]->IsSupportedArgument(arg.get())) {
        ++supported;
        if (idx < 0)
          idx = i;
      }
    }
    if (idx < 0) {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[CK conv] none of %zu instances support this problem\n",
          op_ptrs.size());
      return -1;
    }
    CUSTOM_KERNELS_DEBUG_LOG("[CK conv] %d/%zu instances supported; using #%d\n",
                             supported, op_ptrs.size(), idx);
    std::lock_guard<std::mutex> lock(g_mu);
    g_instance_cache[key] = idx;
  }

  auto &op = op_ptrs[idx];
  auto arg = make_arg(op);
  if (!op->IsSupportedArgument(arg.get())) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_instance_cache.erase(key);
    return -1;
  }

  void *ws = nullptr;
  size_t ws_size = op->GetWorkSpaceSize(arg.get());
  if (ws_size > 0) {
    if (hipMalloc(&ws, ws_size) != hipSuccess)
      return -1;
    op->SetWorkSpacePointer(arg.get(), ws);
  }

  (void)hipGetLastError();
  auto invoker = op->MakeInvokerPointer();
  invoker->Run(arg.get(), StreamConfig{stream, /*time_kernel=*/false});
  hipError_t launch = hipGetLastError();

  if (ws)
    (void)!hipFree(ws);
  return launch == hipSuccess ? 0 : -1;
}

template <typename InT, typename WeiT, typename OutT>
int dispatch(hipStream_t stream, const void *input, const void *weights,
             void *output, const ConvKey &key) {
  return run<InT, WeiT, OutT>(stream, input, weights, output, key);
}

} // namespace

extern "C" int hip_conv_forward(void *stream, const void *input,
                                const void *weights, void *output,
                                int64_t input_n, int64_t input_c,
                                int64_t input_h, int64_t input_w,
                                int64_t weights_k, int64_t output_h,
                                int64_t output_w, int64_t kernel_h,
                                int64_t kernel_w, int64_t stride_h,
                                int64_t stride_w, int64_t pad_top,
                                int64_t pad_left, int64_t pad_bottom,
                                int64_t pad_right, int64_t dilation_h,
                                int64_t dilation_w, int64_t group,
                                int hip_dtype) {
  if (!stream || !input || !weights || !output)
    return -1;
  if (group != 1)
    return -1; // grouped/depthwise stays on MIOpen

  ConvKey key{hip_dtype,  input_n,  input_c,  input_h,  input_w,  weights_k,
              kernel_h,   kernel_w, output_h, output_w, stride_h, stride_w,
              pad_top,    pad_left, pad_bottom, pad_right, dilation_h,
              dilation_w};
  auto s = static_cast<hipStream_t>(stream);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return dispatch<float, float, float>(s, input, weights, output, key);
  case HIP_DTYPE_FLOAT16:
    return dispatch<ck::half_t, ck::half_t, ck::half_t>(s, input, weights,
                                                        output, key);
  case HIP_DTYPE_BFLOAT16:
    return dispatch<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>(s, input, weights,
                                                          output, key);
  default:
    CUSTOM_KERNELS_DEBUG_LOG("[CK conv] unsupported hip_dtype %d\n", hip_dtype);
    return -1;
  }
}
