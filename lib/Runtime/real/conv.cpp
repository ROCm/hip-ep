/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// Conv — forward convolution via the custom HIP kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature (matches ConvLowering.cpp):
//   wrap_conv(state, input, weights, bias, output,
//             data_type, spatial_rank,
//             N, Cin, Cout,
//             in0, in1, in2,
//             out0, out1, out2,
//             k0..k2, s0..s2, p0..p2 (pad_begin), dil0..dil2,
//             group)
//
// `bias` is fused into the kernel, so a convolution is a single launch.
//
// The kernel derives everything it needs from its arguments: there is no op
// state, no solution cache and no workspace.
//
// Only pads_begin is passed. Pad positions are never read, so the trailing pad
// affects nothing except how many output positions exist, and that is already
// in out_d*.

static int hipdnn_ep_to_hip_dtype(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

int wrap_conv(RuntimeState *state, const void *input, const void *weights,
              const void *bias, void *output, int64_t data_type,
              int64_t spatial_rank, int64_t N, int64_t Cin, int64_t Cout,
              int64_t in0, int64_t in1, int64_t in2, int64_t out0, int64_t out1,
              int64_t out2, int64_t k0, int64_t k1, int64_t k2, int64_t s0,
              int64_t s1, int64_t s2, int64_t p0, int64_t p1, int64_t p2,
              int64_t dil0, int64_t dil1, int64_t dil2, int64_t group) {
  OP_PROFILE(
      "conv",
      [&] {
        char b[128];
        snprintf(b, sizeof(b),
                 "%lldx%lldx[%lld,%lld,%lld],k=%lldx[%lld,%lld,%lld],g=%lld,%s",
                 (long long)N, (long long)Cin, (long long)in0, (long long)in1,
                 (long long)in2, (long long)Cout, (long long)k0, (long long)k1,
                 (long long)k2, (long long)group,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !weights || !output) {
    fprintf(stderr, "[REAL] wrap_conv: null argument\n");
    return -1;
  }
  if (spatial_rank < 1 || spatial_rank > 3) {
    fprintf(stderr, "[REAL] wrap_conv: unsupported spatial_rank %lld\n",
            (long long)spatial_rank);
    return -1;
  }
  if (group < 1 || Cin % group != 0 || Cout % group != 0) {
    fprintf(stderr,
            "[REAL] wrap_conv: group %lld does not divide Cin %lld / Cout "
            "%lld\n",
            (long long)group, (long long)Cin, (long long)Cout);
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_conv: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_conv: dtype=%s(%lld) spatial_rank=%lld N=%lld Cin=%lld "
      "Cout=%lld group=%lld in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] "
      "k=[%lld,%lld,%lld] s=[%lld,%lld,%lld] pbegin=[%lld,%lld,%lld] "
      "dil=[%lld,%lld,%lld] bias=%s\n",
      hipdnn_ep_datatype_name(data_type), (long long)data_type,
      (long long)spatial_rank, (long long)N, (long long)Cin, (long long)Cout,
      (long long)group, (long long)in0, (long long)in1, (long long)in2,
      (long long)out0, (long long)out1, (long long)out2, (long long)k0,
      (long long)k1, (long long)k2, (long long)s0, (long long)s1, (long long)s2,
      (long long)p0, (long long)p1, (long long)p2, (long long)dil0,
      (long long)dil1, (long long)dil2, bias ? "yes" : "null");

  int rc = hip_conv(stream, input, weights, bias, output, hip_dtype,
                    static_cast<int>(spatial_rank), N, Cin, Cout, in0, in1, in2,
                    out0, out1, out2, k0, k1, k2, s0, s1, s2, p0, p1, p2, dil0,
                    dil1, dil2, group);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_conv: kernel launch failed (%d)\n", rc);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_conv: completed successfully\n");
  return 0;
}
