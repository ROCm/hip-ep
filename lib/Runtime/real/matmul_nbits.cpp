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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

// ZP_DEBUG_MATMUL: synchronous D2H copy + CPU stats scan over the matmul_nbits
// output tensor. Capped at the first kZpDebugMatmulMaxLogged calls per thread
// so the CI log stays parseable. Used to pin where Inf/NaN first appears in
// the forward pass under the zp=16 toggle (see PR #186 / commit 0f0fade).
//
// Note: this file is compiled to bitcode and JIT-linked at runtime by morphizen.
// The runtime link does NOT include compiler-rt, so we cannot use _Float16 or
// __half conversions (those emit a __extendhfsf2 reference). Instead we read
// fp16 as raw uint16_t and decode by bit manipulation.
static constexpr int64_t kZpDebugMatmulMaxLogged = 600;

static inline bool fp16_bits_is_nan(uint16_t h) {
  return ((h >> 10) & 0x1FU) == 0x1FU && (h & 0x3FFU) != 0U;
}

static inline bool fp16_bits_is_inf(uint16_t h) {
  return ((h >> 10) & 0x1FU) == 0x1FU && (h & 0x3FFU) == 0U;
}

// IEEE 754 binary16 -> binary32. Treats fp16 subnormals as 0 (good enough for
// max-abs tracking; subnormals are below 6e-5 and irrelevant for overflow
// detection). NaN/Inf should be filtered out by the callers above before
// invoking this.
static inline float fp16_bits_to_float(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
  uint32_t exp = static_cast<uint32_t>((h >> 10) & 0x1FU);
  uint32_t mant = static_cast<uint32_t>(h & 0x3FFU);
  uint32_t out;
  if (exp == 0U) {
    out = sign;
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (mant << 13);
  } else {
    out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

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
    const uint16_t *p = reinterpret_cast<const uint16_t *>(host_buf.data());
    for (size_t i = 0; i < total_elems; i++) {
      uint16_t h = p[i];
      if (fp16_bits_is_nan(h)) {
        nan_count++;
      } else if (fp16_bits_is_inf(h)) {
        inf_count++;
      } else {
        float a = std::fabs(fp16_bits_to_float(h));
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
