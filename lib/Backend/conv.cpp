/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// CK conv forward implementation behind the HIPBackendVTable conv slot.
//
// Built by hipcc into the per-gfx backend DLL; statically links against
// device_conv_operations.lib + utility.lib. The lengthy CK link cost
// (2.6 GB archive) happens here ONCE at backend-DLL build time -- not on
// every model.dll compile in the EP pipeline.
//
// Process-global cache keyed by
// (N,C,H,W,K,Ky,Kx,Ho,Wo,strides,pads,dilations,group): caches the chosen
// DeviceOp + BaseInvoker so subsequent calls with the same shape skip the
// factory iteration. NOT keyed on data pointers -- weights are transposed on
// every call (negligible for typical conv shapes; revisit if profiling shows
// otherwise).
//
// Scratch (NCHW<->NHWGC transpose buffers + GKYXC weight materialization
// + CK internal workspace) comes from the model.dll's pool via the
// scratch-provider callback registered by main.cpp's
// `backend_set_scratch_provider` (called by model.dll's state_init).
// No GPU memory is owned by this TU; nothing to free on shutdown beyond
// clearing the cache map.
//
// Public surface from this TU is only the two `backend_*_impl` symbols
// referenced by main.cpp's vtable initializer; everything else is file-
// static. No symbols leave the DLL except via the single `HIPBackendAPI`
// pointer exported from main.cpp.

#include "hipdnn_ep_backend.h"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Internal layout-conversion kernels. Self-contained -- the backend DLL has
// no link dependency on hip_custom_kernels.lib. The G dim of NHWGC is
// contiguous with C_per_g and totals to C, so the kernels collapse to plain
// NCHW <-> NHWC.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void nchw_to_nhwc_kernel(const T *__restrict__ src,
                                    T *__restrict__ dst, int64_t N, int64_t C,
                                    int64_t H, int64_t W) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = N * H * W * C;
  if (tid >= total)
    return;
  int64_t c = tid % C;
  int64_t w = (tid / C) % W;
  int64_t h = (tid / (C * W)) % H;
  int64_t n = tid / (C * W * H);
  dst[tid] = src[((n * C + c) * H + h) * W + w];
}

template <typename T>
__global__ void nhwc_to_nchw_kernel(const T *__restrict__ src,
                                    T *__restrict__ dst, int64_t N, int64_t C,
                                    int64_t H, int64_t W) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = N * C * H * W;
  if (tid >= total)
    return;
  int64_t w = tid % W;
  int64_t h = (tid / W) % H;
  int64_t c = (tid / (W * H)) % C;
  int64_t n = tid / (W * H * C);
  dst[tid] = src[((n * H + h) * W + w) * C + c];
}

// KCYX (ONNX weight) -> GKYXC (CK weight) for fp32 / group=1 path.
// For group > 1 we treat the [K, C/g, Y, X] input as [G, K/g, C/g, Y, X]
// and write [G, K/g, Y, X, C/g] -- same kernel, just different lengths.
template <typename T>
__global__ void
kcyx_to_gkyxc_kernel(const T *__restrict__ src, T *__restrict__ dst, int64_t G,
                     int64_t Kpg, int64_t Cpg, int64_t Y, int64_t X) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = G * Kpg * Y * X * Cpg;
  if (tid >= total)
    return;
  // dst index decomposition (NHWGC-style innermost = Cpg):
  int64_t cp = tid % Cpg;
  int64_t x = (tid / Cpg) % X;
  int64_t y = (tid / (Cpg * X)) % Y;
  int64_t kp = (tid / (Cpg * X * Y)) % Kpg;
  int64_t g = tid / (Cpg * X * Y * Kpg);
  // src layout: [G, Kpg, Cpg, Y, X] = [K_full, Cpg, Y, X] when reshaped
  int64_t src_idx = (((g * Kpg + kp) * Cpg + cp) * Y + y) * X + x;
  dst[tid] = src[src_idx];
}

template <typename T>
__global__ void bias_add_nhwc_kernel(T *__restrict__ data,
                                     const T *__restrict__ bias, int64_t N,
                                     int64_t C) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= N * C)
    return;
  int64_t c = tid % C;
  data[tid] += bias[c];
}

namespace {

template <typename T>
hipError_t launch_nchw_to_nhwc(hipStream_t stream, const T *src, T *dst,
                               int64_t N, int64_t C, int64_t H, int64_t W) {
  int64_t total = N * C * H * W;
  if (total == 0)
    return hipSuccess;
  int block = 256;
  uint32_t grid = static_cast<uint32_t>((total + block - 1) / block);
  hipLaunchKernelGGL(nchw_to_nhwc_kernel<T>, dim3(grid), dim3(block), 0, stream,
                     src, dst, N, C, H, W);
  return hipGetLastError();
}

template <typename T>
hipError_t launch_nhwc_to_nchw(hipStream_t stream, const T *src, T *dst,
                               int64_t N, int64_t C, int64_t H, int64_t W) {
  int64_t total = N * C * H * W;
  if (total == 0)
    return hipSuccess;
  int block = 256;
  uint32_t grid = static_cast<uint32_t>((total + block - 1) / block);
  hipLaunchKernelGGL(nhwc_to_nchw_kernel<T>, dim3(grid), dim3(block), 0, stream,
                     src, dst, N, C, H, W);
  return hipGetLastError();
}

template <typename T>
hipError_t launch_weight_transpose(hipStream_t stream, const T *src, T *dst,
                                   int64_t G, int64_t Kpg, int64_t Cpg,
                                   int64_t Y, int64_t X) {
  int64_t total = G * Kpg * Cpg * Y * X;
  if (total == 0)
    return hipSuccess;
  int block = 256;
  uint32_t grid = static_cast<uint32_t>((total + block - 1) / block);
  hipLaunchKernelGGL(kcyx_to_gkyxc_kernel<T>, dim3(grid), dim3(block), 0,
                     stream, src, dst, G, Kpg, Cpg, Y, X);
  return hipGetLastError();
}

template <typename T>
hipError_t launch_bias_add_nhwc(hipStream_t stream, T *data, const T *bias,
                                int64_t rows, int64_t channels) {
  int64_t total = rows * channels;
  if (total == 0)
    return hipSuccess;
  int block = 256;
  uint32_t grid = static_cast<uint32_t>((total + block - 1) / block);
  hipLaunchKernelGGL(bias_add_nhwc_kernel<T>, dim3(grid), dim3(block), 0,
                     stream, data, bias, rows, channels);
  return hipGetLastError();
}

// ---------------------------------------------------------------------------
// CK type wiring.
// ---------------------------------------------------------------------------

// fp16 type. CK uses ck::half_t; storage element is ushort/uint16_t. We
// receive raw void* so the kernels just work in 16-bit element units.
using F16 = ck::half_t;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
namespace conv_layout = ck::tensor_layout::convolution;
using NHWGC = conv_layout::NHWGC;
using GKYXC = conv_layout::GKYXC;
using NHWGK = conv_layout::NHWGK;
using EmptyTuple = ck::Tuple<>;

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2, NHWGC, GKYXC, EmptyTuple, NHWGK, F16, F16, EmptyTuple, F16, PassThrough,
    PassThrough, PassThrough>;

using Factory =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>;

using BaseInvoker = ck::tensor_operation::device::BaseInvoker;

std::array<ck::index_t, 5> a_lens(int N, int C, int H, int W, int g) {
  return {g, N, C / g, H, W};
}
std::array<ck::index_t, 5> a_strides(int C, int H, int W, int g) {
  int Cpg = C / g;
  return {Cpg, H * W * C, 1, W * C, C};
}
std::array<ck::index_t, 5> b_lens(int K, int C, int Y, int X, int g) {
  return {g, K / g, C / g, Y, X};
}
std::array<ck::index_t, 5> b_strides(int K, int C, int Y, int X, int g) {
  int Kpg = K / g;
  int Cpg = C / g;
  return {Kpg * Y * X * Cpg, Y * X * Cpg, 1, X * Cpg, Cpg};
}
std::array<ck::index_t, 5> e_lens(int N, int K, int Ho, int Wo, int g) {
  return {g, N, K / g, Ho, Wo};
}
std::array<ck::index_t, 5> e_strides(int K, int Ho, int Wo, int g) {
  int Kpg = K / g;
  return {Kpg, Ho * Wo * K, 1, Wo * K, K};
}

// ---------------------------------------------------------------------------
// Process-global state.
// ---------------------------------------------------------------------------

struct ConvShapeKey {
  int32_t N, C, H, W, K, Ky, Kx, Ho, Wo;
  int32_t sh, sw, pt, pl, pb, pr, dh, dw, group;
  bool operator==(const ConvShapeKey &o) const {
    return std::memcmp(this, &o, sizeof(*this)) == 0;
  }
};

struct ConvShapeKeyHash {
  size_t operator()(const ConvShapeKey &k) const noexcept {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&k);
    size_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < sizeof(k); ++i) {
      h ^= p[i];
      h *= 0x100000001b3ULL;
    }
    return h;
  }
};

struct ConvCacheEntry {
  std::unique_ptr<DeviceOp> op;
  std::unique_ptr<BaseInvoker> invoker;
  size_t workspace_bytes = 0;
};

struct ConvState {
  std::mutex mu;
  std::unordered_map<ConvShapeKey, ConvCacheEntry, ConvShapeKeyHash> cache;
};

ConvState &state() {
  static ConvState s;
  return s;
}

// Provided by main.cpp; forwards to the model.dll's pool grow-on-demand
// helper through the registered scratch-provider callback.
extern "C" void *backend_get_scratch(size_t needed_bytes);

// Returns the cache entry for the given shape, building it on miss.
// Caller MUST hold state().mu.
ConvCacheEntry *lookup_or_build_locked(const ConvShapeKey &key) {
  auto &s = state();
  auto it = s.cache.find(key);
  if (it != s.cache.end())
    return &it->second;

  ConvCacheEntry entry;
  auto al = a_lens(key.N, key.C, key.H, key.W, key.group);
  auto as = a_strides(key.C, key.H, key.W, key.group);
  auto bl = b_lens(key.K, key.C, key.Ky, key.Kx, key.group);
  auto bs = b_strides(key.K, key.C, key.Ky, key.Kx, key.group);
  auto el = e_lens(key.N, key.K, key.Ho, key.Wo, key.group);
  auto es = e_strides(key.K, key.Ho, key.Wo, key.group);
  std::array<ck::index_t, 2> conv_strides = {key.sh, key.sw};
  std::array<ck::index_t, 2> conv_dilations = {key.dh, key.dw};
  std::array<ck::index_t, 2> left_pads = {key.pt, key.pl};
  std::array<ck::index_t, 2> right_pads = {key.pb, key.pr};
  PassThrough op_a, op_b, op_cde;

  for (auto &op : Factory::GetInstances()) {
    auto arg = op->MakeArgumentPointer(
        nullptr, nullptr, std::array<const void *, 0>{}, nullptr, al, as, bl,
        bs, std::array<std::array<ck::index_t, 5>, 0>{},
        std::array<std::array<ck::index_t, 5>, 0>{}, el, es, conv_strides,
        conv_dilations, left_pads, right_pads, op_a, op_b, op_cde);
    if (!op->IsSupportedArgument(arg.get()))
      continue;
    entry.op = std::move(op);
    entry.invoker = entry.op->MakeInvokerPointer();
    entry.workspace_bytes = entry.op->GetWorkSpaceSize(arg.get());
    auto inserted = s.cache.emplace(key, std::move(entry));
    return &inserted.first->second;
  }
  return nullptr;
}

} // namespace

void backend_conv_shutdown_impl(void) {
  // Scratch is owned by model.dll's pool, not by this TU -- nothing to
  // hipFree here. Clearing the cache drops the unique_ptr<DeviceOp> +
  // BaseInvoker entries; CK's kernel handles release with them.
  auto &s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  s.cache.clear();
}

int backend_conv_fwd_fp16_nchw_impl(void *stream_void, const void *input, int N,
                                    int C, int H, int W, const void *weights,
                                    int K, int kernel_h, int kernel_w,
                                    const void *bias, void *output, int Ho,
                                    int Wo, int stride_h, int stride_w,
                                    int pad_top, int pad_left, int pad_bottom,
                                    int pad_right, int dilation_h,
                                    int dilation_w, int group) {
  if (!input || !weights || !output)
    return -1;
  if (group <= 0 || C % group != 0 || K % group != 0)
    return -1;

  hipStream_t stream = static_cast<hipStream_t>(stream_void);

  ConvShapeKey key{};
  key.N = N;
  key.C = C;
  key.H = H;
  key.W = W;
  key.K = K;
  key.Ky = kernel_h;
  key.Kx = kernel_w;
  key.Ho = Ho;
  key.Wo = Wo;
  key.sh = stride_h;
  key.sw = stride_w;
  key.pt = pad_top;
  key.pl = pad_left;
  key.pb = pad_bottom;
  key.pr = pad_right;
  key.dh = dilation_h;
  key.dw = dilation_w;
  key.group = group;

  // Scratch layout: | nhwc_in | nhwc_out | gkyxc_w | ck_workspace |, each
  // 64-byte aligned. We hold the lock across the whole call -- the cache
  // map is shared across threads; for v1 we serialize. The CK kernels
  // run async on the supplied stream so this only serializes host
  // bookkeeping, not the GPU work itself.
  constexpr size_t kElem = sizeof(F16);
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  size_t in_bytes = static_cast<size_t>(N) * C * H * W * kElem;
  size_t out_bytes = static_cast<size_t>(N) * K * Ho * Wo * kElem;
  size_t w_bytes =
      static_cast<size_t>(K) * (C / group) * kernel_h * kernel_w * kElem;

  std::lock_guard<std::mutex> lk(state().mu);

  ConvCacheEntry *entry = lookup_or_build_locked(key);
  if (!entry) {
    fprintf(stderr,
            "[ep_backend] no CK instance supports shape "
            "N=%d C=%d H=%d W=%d K=%d Y=%d X=%d g=%d\n",
            N, C, H, W, K, kernel_h, kernel_w, group);
    return -1;
  }

  size_t off_in = 0;
  size_t off_out = align_up(off_in + in_bytes, 64);
  size_t off_w = align_up(off_out + out_bytes, 64);
  size_t off_ws = align_up(off_w + w_bytes, 64);
  size_t total = align_up(off_ws + entry->workspace_bytes, 64);

  // Pull scratch from the model.dll's pool via the registered provider.
  // Failure here means model.dll either didn't register a provider yet
  // (state_init misconfiguration) or the underlying ensure_workspace
  // grow returned an error (out of GPU memory).
  void *scratch = backend_get_scratch(total);
  if (!scratch) {
    fprintf(stderr,
            "[ep_backend] scratch provider unavailable or returned null "
            "for %zu bytes (did model.dll call set_scratch_provider?)\n",
            total);
    (void)stream;
    return -1;
  }

  uint8_t *base = static_cast<uint8_t *>(scratch);
  void *nhwc_in = base + off_in;
  void *nhwc_out = base + off_out;
  void *gkyxc_w = base + off_w;
  void *ck_workspace =
      entry->workspace_bytes ? static_cast<void *>(base + off_ws) : nullptr;

  if (launch_nchw_to_nhwc<F16>(stream, static_cast<const F16 *>(input),
                               static_cast<F16 *>(nhwc_in), N, C, H,
                               W) != hipSuccess)
    return -1;

  // KCYX -> GKYXC. K = G*Kpg, C = G*Cpg.
  if (launch_weight_transpose<F16>(stream, static_cast<const F16 *>(weights),
                                   static_cast<F16 *>(gkyxc_w), group,
                                   K / group, C / group, kernel_h,
                                   kernel_w) != hipSuccess)
    return -1;

  auto al = a_lens(N, C, H, W, group);
  auto as = a_strides(C, H, W, group);
  auto bl = b_lens(K, C, kernel_h, kernel_w, group);
  auto bs = b_strides(K, C, kernel_h, kernel_w, group);
  auto el = e_lens(N, K, Ho, Wo, group);
  auto es = e_strides(K, Ho, Wo, group);
  std::array<ck::index_t, 2> conv_strides = {stride_h, stride_w};
  std::array<ck::index_t, 2> conv_dilations = {dilation_h, dilation_w};
  std::array<ck::index_t, 2> left_pads = {pad_top, pad_left};
  std::array<ck::index_t, 2> right_pads = {pad_bottom, pad_right};
  PassThrough op_a, op_b, op_cde;

  auto arg = entry->op->MakeArgumentPointer(
      nhwc_in, gkyxc_w, std::array<const void *, 0>{}, nhwc_out, al, as, bl, bs,
      std::array<std::array<ck::index_t, 5>, 0>{},
      std::array<std::array<ck::index_t, 5>, 0>{}, el, es, conv_strides,
      conv_dilations, left_pads, right_pads, op_a, op_b, op_cde);
  if (!entry->op->IsSupportedArgument(arg.get())) {
    fprintf(
        stderr,
        "[ep_backend] arg unsupported on real-pointer build (unexpected)\n");
    return -1;
  }
  if (entry->workspace_bytes)
    entry->op->SetWorkSpacePointer(arg.get(), ck_workspace);

  StreamConfig sconf;
  sconf.stream_id_ = stream;
  sconf.time_kernel_ = false;
  entry->invoker->Run(arg.get(), sconf);

  // Optional bias add: NHWC output is [N, Ho, Wo, K], channels are innermost
  // so per-token broadcast across K matches the kernel's [rows, C] contract
  // with rows = N*Ho*Wo, C = K.
  if (bias) {
    int64_t rows = static_cast<int64_t>(N) * Ho * Wo;
    if (launch_bias_add_nhwc<F16>(stream, static_cast<F16 *>(nhwc_out),
                                  static_cast<const F16 *>(bias), rows,
                                  K) != hipSuccess)
      return -1;
  }

  if (launch_nhwc_to_nchw<F16>(stream, static_cast<const F16 *>(nhwc_out),
                               static_cast<F16 *>(output), N, K, Ho,
                               Wo) != hipSuccess)
    return -1;

  return 0;
}
