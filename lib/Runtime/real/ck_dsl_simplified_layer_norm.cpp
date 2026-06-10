/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Drop-in replacement for miopenT5LayerNormForward, backed by a
// ck_dsl-generated RMSNorm kernel for the Qwen3.5 residual-stream shape:
//
//   - f32 / hidden_dim=4096   (residual stream RMSNorm; ~41 calls per
//     decode token, faster on GPU time than MIOpen's T5LayerNorm)
//
// Any other (dtype, hidden_dim) or non-matching device returns -2 so the
// caller falls back to MIOpen unchanged.

#include "../hipdnn_ep_runtime.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

#include "ck_dsl/ck_dsl_rmsnorm2d_f32_gfx1151_hsaco.h"
#include "ck_dsl_kernel.h"

// Catch a truncated / mis-regenerated embedded HSACO at compile time.
static_assert(sizeof(kCkDslRmsnorm2dF32Gfx1151Hsaco) ==
                  kCkDslRmsnorm2dF32Gfx1151Hsaco_size,
              "embedded HSACO length mismatch (regenerate header)");

namespace {

// Per-architecture RMSNorm kernel for the f32 / N=4096 shape. To support a
// discrete GPU, generate its HSACO and add a row keyed by its gcnArchName --
// the lookup below then picks it automatically on that device.
const std::unordered_map<std::string, ckdsl::KernelDef> &kernelTable() {
  static const std::unordered_map<std::string, ckdsl::KernelDef> table = {
      {"gfx1151",
       {kCkDslRmsnorm2dF32Gfx1151Hsaco,
        "ck_dsl_rmsnorm2d_fwd_f32_N4096_b256_v4", "f32/N=4096/b=256",
        /*block_size=*/256}},
  };
  return table;
}

// ABI matches RMSNorm2DSpec.signature() (natural alignment; the 3 leading
// pointers are 8-aligned and M/N/eps pack contiguously, so field offsets
// match the ck_dsl signature without packing):
//   X, Gamma, Y    : 3 x ptr<f32, global>
//   M, N           : 2 x i32
//   eps            : f32
struct KernelArgs {
  const void *X;
  const void *Gamma;
  void *Y;
  int32_t M;
  int32_t N;
  float eps;
};

int launch(hipFunction_t function, unsigned block_size, hipStream_t stream,
           const void *input, const void *gamma, void *output, int32_t num_rows,
           int32_t hidden_dim, float epsilon) {
  KernelArgs args{};
  args.X = input;
  args.Gamma = gamma;
  args.Y = output;
  args.M = num_rows;
  args.N = hidden_dim;
  args.eps = epsilon;
  size_t args_size = sizeof(args);
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &args_size,
      HIP_LAUNCH_PARAM_END,
  };
  hipError_t err = hipModuleLaunchKernel(function,
                                         /*gx*/ static_cast<unsigned>(num_rows),
                                         /*gy*/ 1, /*gz*/ 1,
                                         /*bx*/ block_size,
                                         /*by*/ 1, /*bz*/ 1,
                                         /*sharedMemBytes*/ 0, stream,
                                         /*kernelParams*/ nullptr, extra);
  if (err != hipSuccess) {
    fprintf(stderr,
            "ck_dsl_simplified_layer_norm: hipModuleLaunchKernel failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

// Returns:
//   0           on successful launch.
//   -2          when no compiled HSACO matches (caller falls back to MIOpen).
//   hipError_t  on a HIP launch failure.
//
// element_size_bytes: 2 = f16, 4 = f32. Matches the convention used by the
// wrap_miopenT5LayerNormForward wrapper (miopenHalf vs miopenFloat).
int ck_dsl_simplified_layer_norm(void *stream, const void *input,
                                 const void *gamma, void *output,
                                 int64_t num_rows, int64_t hidden_dim,
                                 float epsilon, int64_t element_size_bytes) {
  constexpr int kRejectFallback = -2;

  if (num_rows <= 0 || hidden_dim <= 0 || !input || !gamma || !output)
    return kRejectFallback;
  if (num_rows > INT32_MAX || hidden_dim > INT32_MAX)
    return kRejectFallback;
  // Only the f32 / N=4096 shape is compiled.
  if (element_size_bytes != 4 || hidden_dim != 4096)
    return kRejectFallback;

  // Pick the kernel built for the running device; fall back to MIOpen if there
  // is none (e.g. an architecture we haven't generated an HSACO for yet).
  const auto &table = kernelTable();
  auto it = table.find(ckdsl::deviceArch());
  if (it == table.end())
    return kRejectFallback;

  hipFunction_t function = ckdsl::loadKernel(it->second);
  if (!function)
    return kRejectFallback;

  return launch(function, it->second.block_size,
                static_cast<hipStream_t>(stream), input, gamma, output,
                static_cast<int32_t>(num_rows),
                static_cast<int32_t>(hidden_dim), epsilon);
}
