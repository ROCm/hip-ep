/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Drop-in replacement for the MIOpen-based wrap_skip_simplified_layer_norm
// hot path, backed by a ck_dsl-generated fused (Add + RMSNorm * Gamma)
// kernel for the Qwen3.5 residual-stream shape (M=1, hidden_dim=4096, f32).
//
// Why this exists: the baseline issues 3 MIOpen launches per call
// (OpTensor(ADD) for input+skip, optional OpTensor(ADD) for bias, then
// miopenT5LayerNormForward). Each MIOpen API call carries ~30-50 us of
// CPU descriptor / handle dispatch overhead on top of the actual GPU
// launch. The Qwen3.5 forward fires ~64 of these calls per layer
// (SkipSimplifiedLayerNormalization shows up 64x in text.onnx); on a
// decode wall of ~55 ms/token that compounds to a measurable share.
//
// The ck_dsl kernel collapses the 3 launches into 1 launch that streams
// A and B once, writes the residual back to X, and emits the normed Y
// in a single pass over HBM. Bias is *not* supported here -- Qwen3.5
// passes bias=null on every call (verified in the trace), so the
// baseline path is kept for the bias != null edge case.

#include "../hipdnn_ep_runtime.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

#include "ck_dsl/ck_dsl_add_rmsnorm2d_f16_gfx1151_hsaco.h"
#include "ck_dsl/ck_dsl_add_rmsnorm2d_f32_gfx1151_hsaco.h"
#include "ck_dsl_kernel.h"

// Catch a truncated / mis-regenerated embedded HSACO at compile time.
static_assert(sizeof(kCkDslAddRmsnorm2dF32Gfx1151Hsaco) ==
                  kCkDslAddRmsnorm2dF32Gfx1151Hsaco_size,
              "embedded HSACO length mismatch (regenerate header)");
static_assert(sizeof(kCkDslAddRmsnorm2dF16Gfx1151Hsaco) ==
                  kCkDslAddRmsnorm2dF16Gfx1151Hsaco_size,
              "embedded HSACO length mismatch (regenerate header)");

namespace {

// Per-architecture fused (Add + RMSNorm) kernel for the N=4096 shape, one
// table per I/O dtype. The symbols come from AddRMSNorm2DBF16Spec(
// n_per_block=4096, block_size=256, vec=4, dtype=<dt>, save_residual=True)
// .kernel_name() (note: the instance name keeps its "bf16" prefix for all
// dtypes). To support a discrete GPU, generate its HSACO and add a row keyed
// by its gcnArchName.
const std::unordered_map<std::string, ckdsl::KernelDef> &kernelTableF32() {
  static const std::unordered_map<std::string, ckdsl::KernelDef> table = {
      {"gfx1151",
       {kCkDslAddRmsnorm2dF32Gfx1151Hsaco,
        "ck_dsl_add_rmsnorm2d_bf16_f32_N4096_b256_v4_sr", "f32/N=4096/b=256",
        /*block_size=*/256}},
  };
  return table;
}

const std::unordered_map<std::string, ckdsl::KernelDef> &kernelTableF16() {
  static const std::unordered_map<std::string, ckdsl::KernelDef> table = {
      {"gfx1151",
       {kCkDslAddRmsnorm2dF16Gfx1151Hsaco,
        "ck_dsl_add_rmsnorm2d_bf16_f16_N4096_b256_v4_sr", "f16/N=4096/b=256",
        /*block_size=*/256}},
  };
  return table;
}

// Shape the kernels above were compiled for. Any call that doesn't match
// drops back to the caller's baseline path.
constexpr int kHiddenDim = 4096;

// Decode-shaped cutoff: only route calls with at most this many rows (M)
// through ck_dsl. ck_dsl wins on small M (autoregressive decode) but loses to
// MIOpen on prefill-shaped large M, so keep big batches on the MIOpen baseline.
// Prototype heuristic; tune via a per-M sweep.
constexpr int64_t kMaxCkDslDecodeRows = 1;

// ABI matches AddRMSNorm2DBF16Spec.signature() (natural alignment; the 5
// leading pointers are 8-aligned and M/N/eps pack contiguously, so field
// offsets match the ck_dsl signature without packing):
//   A, B, Gamma, X, Y           : 5 x ptr<f32, global>
//   M, N                        : 2 x i32
//   eps                         : f32
struct KernelArgs {
  const void *A;
  const void *B;
  const void *Gamma;
  void *X;
  void *Y;
  int32_t M;
  int32_t N;
  float eps;
};

} // namespace

// Returns:
//   0           on successful launch.
//   -2 (kRejectFallback) when the shape doesn't match what this HSACO was
//                        compiled for, the device has no matching kernel, or
//                        the HSACO failed to load -- the caller must fall back
//                        to its baseline MIOpen path.
//   hipError_t  on a HIP launch failure.
int ck_dsl_skip_simplified_layer_norm(void *stream, const void *input,
                                      const void *skip, const void *gamma,
                                      void *output, void *residual_sum_out,
                                      int64_t num_rows, int64_t hidden_dim,
                                      float epsilon,
                                      int64_t element_size_bytes) {
  constexpr int kRejectFallback = -2;

  if (hidden_dim != kHiddenDim)
    return kRejectFallback;
  if (element_size_bytes != 4 && element_size_bytes != 2)
    return kRejectFallback;
  if (num_rows <= 0 || !input || !skip || !gamma || !output ||
      !residual_sum_out)
    return kRejectFallback;
  if (num_rows > INT32_MAX)
    return kRejectFallback;
  // Keep prefill-shaped (large M) calls on MIOpen; ck_dsl only for decode.
  if (num_rows > kMaxCkDslDecodeRows)
    return kRejectFallback;

  // Pick the kernel built for the running device + I/O dtype; fall back to
  // MIOpen if there is none (e.g. an arch we haven't generated an HSACO for).
  const auto &table =
      (element_size_bytes == 4) ? kernelTableF32() : kernelTableF16();
  auto it = table.find(ckdsl::deviceArch());
  if (it == table.end())
    return kRejectFallback;

  hipFunction_t function = ckdsl::loadKernel(it->second);
  if (!function)
    return kRejectFallback;

  KernelArgs args;
  args.A = input;
  args.B = skip;
  args.Gamma = gamma;
  args.X = residual_sum_out;
  args.Y = output;
  args.M = static_cast<int32_t>(num_rows);
  args.N = static_cast<int32_t>(hidden_dim);
  args.eps = epsilon;

  size_t args_size = sizeof(args);
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &args_size,
      HIP_LAUNCH_PARAM_END,
  };

  hipError_t err =
      hipModuleLaunchKernel(function,
                            /*gx*/ static_cast<unsigned>(num_rows),
                            /*gy*/ 1,
                            /*gz*/ 1,
                            /*bx*/ it->second.block_size,
                            /*by*/ 1,
                            /*bz*/ 1,
                            /*sharedMemBytes*/ 0,
                            /*stream*/ static_cast<hipStream_t>(stream),
                            /*kernelParams*/ nullptr,
                            /*extra*/ extra);
  if (err != hipSuccess) {
    fprintf(stderr,
            "ck_dsl_skip_simplified_layer_norm: hipModuleLaunchKernel "
            "failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
