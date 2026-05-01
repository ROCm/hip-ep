/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"

#include <cmath>
#include <cstdio>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

// ZP_DEBUG_MATMUL: synchronous D2H copy + CPU stats scan over the matmul_nbits
// output tensor. Capped at the first kZpDebugMatmulMaxLogged calls per thread
// so the CI log stays parseable. Used to pin where Inf/NaN first appears in
// the forward pass under the zp=16 toggle (see PR #186 / commit 0f0fade).
static constexpr int64_t kZpDebugMatmulMaxLogged = 600;

static void log_matmul_nbits_stats(RuntimeState *state, const void *output,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t batch_count, int64_t elem_size,
                                   bool has_zp) {
  static thread_local int64_t s_matmul_call_seq = 0;
  int64_t call_seq = s_matmul_call_seq++;
  if (call_seq >= kZpDebugMatmulMaxLogged) {
    return;
  }
  if (!state || !output) {
    return;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    return;
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  size_t total_elems = static_cast<size_t>(M * N * batch_count);
  size_t total_bytes = total_elems * static_cast<size_t>(elem_size);
  if (total_elems == 0 || total_bytes == 0) {
    return;
  }

  std::vector<char> host_buf(total_bytes);
  if (hipMemcpyAsync(host_buf.data(), output, total_bytes,
                     hipMemcpyDeviceToHost, hip_stream) != hipSuccess) {
    return;
  }
  if (hipStreamSynchronize(hip_stream) != hipSuccess) {
    return;
  }

  int64_t nan_count = 0;
  int64_t inf_count = 0;
  float max_abs = 0.0f;
  if (elem_size == 2) {
    const _Float16 *p = reinterpret_cast<const _Float16 *>(host_buf.data());
    for (size_t i = 0; i < total_elems; i++) {
      float v = static_cast<float>(p[i]);
      if (std::isnan(v)) {
        nan_count++;
      } else if (std::isinf(v)) {
        inf_count++;
      } else {
        float a = std::fabs(v);
        if (a > max_abs) {
          max_abs = a;
        }
      }
    }
  } else if (elem_size == 4) {
    const float *p = reinterpret_cast<const float *>(host_buf.data());
    for (size_t i = 0; i < total_elems; i++) {
      float v = p[i];
      if (std::isnan(v)) {
        nan_count++;
      } else if (std::isinf(v)) {
        inf_count++;
      } else {
        float a = std::fabs(v);
        if (a > max_abs) {
          max_abs = a;
        }
      }
    }
  } else {
    return;
  }

  fprintf(stderr,
          "[ZP_DEBUG_MATMUL] seq=%lld M=%lld N=%lld K=%lld zp=%s "
          "max_abs=%.3f nan=%lld inf=%lld total=%lld\n",
          (long long)call_seq, (long long)M, (long long)N, (long long)K,
          has_zp ? "yes" : "null", max_abs, (long long)nan_count,
          (long long)inf_count, (long long)total_elems);
}

int wrap_matmul_nbits(RuntimeState *state, const void *A, const void *B,
                      const void *scales, const void *zero_points,
                      const void *g_idx, const void *bias, void *output,
                      int64_t M, int64_t N, int64_t K, int64_t batch_count,
                      int64_t bits, int64_t block_size, int64_t elem_size,
                      int64_t zp_elem_size) {
  OP_PROFILE(
      "matmul_nbits",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(b);
      },
      state);
  if (!state || !A || !B || !scales || !output) {
    fprintf(stderr, "wrap_matmul_nbits: null argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_matmul_nbits(M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, bits=%lld, block_size=%lld, elem_size=%lld, "
                    "zp_elem_size=%lld, zero_points=%s, g_idx=%s, bias=%s)\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)bits,
                    (long long)block_size, (long long)elem_size,
                    (long long)zp_elem_size, zero_points ? "yes" : "null",
                    g_idx ? "yes" : "null", bias ? "yes" : "null");

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_matmul_nbits: null stream\n");
    return -1;
  }

  if (g_idx) {
    fprintf(stderr, "wrap_matmul_nbits: g_idx not supported\n");
    return -1;
  }

  int result = 0;
  HIP_CHECK(hip_matmul_nbits(stream, A, B, scales, zero_points, bias, output, M,
                             N, K, batch_count, bits, block_size, elem_size,
                             zp_elem_size));

  log_matmul_nbits_stats(state, output, M, N, K, batch_count, elem_size,
                         zero_points != nullptr);

cleanup:
  return result;
}
