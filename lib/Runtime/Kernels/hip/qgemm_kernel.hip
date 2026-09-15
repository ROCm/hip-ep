/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A, B, C) = Q(alpha * DQ(A)' @ DQ(B)' + beta * DQ(C))
//
// a = (A - z_a) * s_a
// b = (B - z_b[n]) * s_b[n]
// c = (C - z_c) * s_c
// Y = saturate(round((alpha * a' @ b' + beta * c) / s_y) + z_y)
//
// Neither scale depends on k, so both leave the sum:
//
//   acc[m,n] = sum over k of (A[m,k] - z_a) * (B[k,n] - z_b[n])
//   Y[m,n]   = saturate(round(M_ab * s_b[n] * acc[m,n]
//                             + M_c * (C[m,n] - z_c))
//                       + z_y)
//   M_ab     = alpha * s_a * s_b / s_y    (folded by the caller)
//   M_c      = beta * s_c / s_y           (folded by the caller)
//
// acc is an exact integer, so the only rounding is the multiply by the
// coefficient and the final round -- strictly less error than the unfused path,
// which accumulates K products in float.
//
// A per-output-channel B is what M_ab cannot absorb: s_b and z_b are then
// device arrays indexed by n, and the per-tensor B_scale contributes its 1.0
// identity to M_ab instead. Both cases are one coefficient per column, read
// before the reduction.
//
// REFERENCE IMPLEMENTATION
// -----------------------
// One thread per output element, no tiling and no reuse: every thread re-reads
// a full row of A and a full column of B from global memory, so the same A
// element is fetched N times and the same B element M times. That is the
// arithmetic spelled out literally, which is what makes it the baseline a tuned
// kernel is checked against -- see qmatmul_kernel.hip and qconv_kernel.hip for
// what the staged form of the same reduction looks like.
//
// Accumulator width
// -----------------
// |A - z_a| reaches 65535 for a 16-bit activation and |B - z_b| reaches 255
// even for a 4-bit weight whose zero point sits at the far end of the storage
// range, which bounds one term at 1.7e7 -- past int32 after two terms. The
// accumulator is therefore int64, which nothing here makes expensive.
//
// Narrowing the total to float before applying the coefficient costs 2^-24
// relative error, and the coefficient immediately brings the value back into
// the output's quantized range, so the absolute error stays under 0.01 LSB of
// a 16-bit output.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

// Widen one element of an 8-, 16- or 32-bit quantized buffer.
//
// The dtype is a runtime argument rather than a template parameter because A,
// C and Y vary independently: templating all three would instantiate this
// kernel 80 times to save a branch that a reference implementation is not
// trying to save.
__device__ inline int32_t qgemm_load(const void *__restrict__ buf, int64_t idx,
                                     int dtype) {
  switch (dtype) {
  case HIP_DTYPE_INT8:
    return static_cast<const int8_t *>(buf)[idx];
  case HIP_DTYPE_UINT8:
    return static_cast<const uint8_t *>(buf)[idx];
  case HIP_DTYPE_INT16:
    return static_cast<const int16_t *>(buf)[idx];
  case HIP_DTYPE_UINT16:
    return static_cast<const uint16_t *>(buf)[idx];
  default: // HIP_DTYPE_INT32, the only remaining accepted code
    return static_cast<const int32_t *>(buf)[idx];
  }
}

// Read one LOGICAL element of B or B_zero_points.
//
// `bits` is 4 or 8. At 4, two values share a byte, low nibble first over the
// flattened row-major sequence -- the ONNX INT4/UINT4 convention, matching
// qconv_kernel.hip and qdq_kernel.hip. `lin` counts logical elements, never
// bytes.
//
// `is_signed` decides how a nibble widens, and the two disagree on every value
// above 7, so it must track B's element type rather than being assumed.
__device__ inline int32_t qgemm_read_quant(const uint8_t *__restrict__ buf,
                                           int64_t lin, int bits,
                                           bool is_signed) {
  if (bits == 4) {
    const uint8_t byte = buf[lin >> 1];
    const uint8_t nibble = (lin & 1) ? (byte >> 4) : (byte & 0xF);
    if (is_signed) {
      // Shifting into the high nibble of an int8 and back propagates bit 3.
      return static_cast<int32_t>(static_cast<int8_t>(nibble << 4)) >> 4;
    }
    return static_cast<int32_t>(nibble);
  }
  if (is_signed)
    return static_cast<int32_t>(static_cast<int8_t>(buf[lin]));
  return static_cast<int32_t>(buf[lin]);
}

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only, hence the explicit bounds.
__device__ inline void qgemm_store(void *__restrict__ buf, int64_t idx,
                                   int dtype, float v) {
  switch (dtype) {
  case HIP_DTYPE_INT8:
    static_cast<int8_t *>(buf)[idx] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, v)));
    break;
  case HIP_DTYPE_UINT8:
    static_cast<uint8_t *>(buf)[idx] =
        static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, v)));
    break;
  case HIP_DTYPE_INT16:
    static_cast<int16_t *>(buf)[idx] =
        static_cast<int16_t>(fminf(32767.0f, fmaxf(-32768.0f, v)));
    break;
  default: // HIP_DTYPE_UINT16
    static_cast<uint16_t *>(buf)[idx] =
        static_cast<uint16_t>(fminf(65535.0f, fmaxf(0.0f, v)));
    break;
  }
}

// threadIdx.x indexes n so that consecutive threads write consecutive Y
// elements; n is the contiguous axis of the output.
constexpr int kBlockN = 16;
constexpr int kBlockM = 16;

__global__ __launch_bounds__(kBlockN *kBlockM) void qgemm_naive_kernel(
    const void *__restrict__ A, const uint8_t *__restrict__ B,
    const void *__restrict__ C, const float *__restrict__ B_scales,
    const uint8_t *__restrict__ B_zero_points, void *__restrict__ Y, int64_t M,
    int64_t N, int64_t K, bool trans_a, bool trans_b, int a_dtype, int b_dtype,
    int c_dtype, int y_dtype, int b_bits, bool b_signed, int64_t c_dim0,
    int64_t c_dim1, float M_ab, float M_c, int32_t a_zp, int32_t b_zp,
    int32_t c_zp, int32_t y_zp) {
  (void)b_dtype; // carried as b_signed, which is what the nibble read needs
  const int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockN + threadIdx.x;
  const int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockM + threadIdx.y;
  if (m >= M || n >= N)
    return;

  // Both halves of a per-channel B index by n alone, so they are read once per
  // thread rather than once per k.
  const int32_t bzp = B_zero_points
                          ? qgemm_read_quant(B_zero_points, n, b_bits, b_signed)
                          : b_zp;
  const float coeff = B_scales ? M_ab * B_scales[n] : M_ab;

  int64_t acc = 0;
  for (int64_t k = 0; k < K; ++k) {
    // trans_a stores A as [K, M] and trans_b stores B as [N, K]; only the
    // stride changes, the logical extents do not.
    const int32_t a =
        qgemm_load(A, trans_a ? k * M + m : m * K + k, a_dtype) - a_zp;
    const int32_t b =
        qgemm_read_quant(B, trans_b ? n * K + k : k * N + n, b_bits, b_signed) -
        bzp;
    acc += static_cast<int64_t>(a) * b;
  }

  float scaled = coeff * static_cast<float>(acc);
  if (C) {
    // Unidirectional broadcast to [M, N]: an extent of 1 repeats along its
    // axis, which is the whole of ONNX Gemm's bias rule once the caller has
    // normalized C to rank 2.
    const int64_t cm = (c_dim0 == 1) ? 0 : m;
    const int64_t cn = (c_dim1 == 1) ? 0 : n;
    scaled += M_c * static_cast<float>(
                        qgemm_load(C, cm * c_dim1 + cn, c_dtype) - c_zp);
  }
  qgemm_store(Y, m * N + n, y_dtype, rintf(scaled) + static_cast<float>(y_zp));
}

bool is_quantized_8_or_16(int dtype) {
  return dtype == HIP_DTYPE_INT8 || dtype == HIP_DTYPE_UINT8 ||
         dtype == HIP_DTYPE_INT16 || dtype == HIP_DTYPE_UINT16;
}

} // namespace

extern "C" int hip_qgemm(void *stream, const void *A, const void *B,
                         const void *C, const void *B_scales,
                         const void *B_zero_points, void *Y, int64_t M,
                         int64_t N, int64_t K, int trans_a, int trans_b,
                         int a_dtype, int b_dtype, int c_dtype, int y_dtype,
                         int b_bits, int64_t c_dim0, int64_t c_dim1,
                         float M_ab, float M_c, int64_t a_zp, int64_t b_zp,
                         int64_t c_zp, int64_t y_zp) {
  if (!A || !B || !Y) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0)
    return 0;

  // The kernel indexes a scale and a zero point by the same n; neither has a
  // meaningful stand-in, so one without the other is a caller bug.
  if (static_cast<bool>(B_scales) != static_cast<bool>(B_zero_points)) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: B_scales and B_zero_points "
                    "must be given together\n");
    return -1;
  }
  if (b_bits != 4 && b_bits != 8) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: unsupported b_bits=%d\n",
            b_bits);
    return -1;
  }

  bool b_signed;
  switch (b_dtype) {
  case HIP_DTYPE_INT8:
    b_signed = true;
    break;
  case HIP_DTYPE_UINT8:
    b_signed = false;
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported b_dtype=%d (expected "
            "8-bit storage)\n",
            b_dtype);
    return -1;
  }
  if (!is_quantized_8_or_16(a_dtype) || !is_quantized_8_or_16(y_dtype)) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported a_dtype=%d / y_dtype=%d "
            "(expected 8- or 16-bit quantized)\n",
            a_dtype, y_dtype);
    return -1;
  }
  // int32 is accepted only for C: an ONNX quantizer emits a Gemm bias at
  // s_c = s_a * s_b, which needs the wider accumulator type.
  if (C && !is_quantized_8_or_16(c_dtype) && c_dtype != HIP_DTYPE_INT32) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported c_dtype=%d (expected 8-, "
            "16- or 32-bit quantized)\n",
            c_dtype);
    return -1;
  }
  if (C && (c_dim0 <= 0 || c_dim1 <= 0)) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: invalid C extents %lldx%lld\n",
            (long long)c_dim0, (long long)c_dim1);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qgemm: M=%lld N=%lld K=%lld trans=(%d,%d) "
      "dtypes=(%d,%d,%d,%d) b_bits=%d per_channel=%s M_ab=%g M_c=%g "
      "zp=(%lld,%lld,%lld,%lld) C=%s\n",
      (long long)M, (long long)N, (long long)K, trans_a, trans_b, a_dtype,
      b_dtype, c_dtype, y_dtype, b_bits, B_scales ? "yes" : "no", (double)M_ab,
      (double)M_c, (long long)a_zp, (long long)b_zp, (long long)c_zp,
      (long long)y_zp, C ? "yes" : "null");

  const dim3 block(kBlockN, kBlockM);
  const dim3 grid(static_cast<unsigned>((N + kBlockN - 1) / kBlockN),
                  static_cast<unsigned>((M + kBlockM - 1) / kBlockM));

  (void)hipGetLastError();
  hipLaunchKernelGGL(
      qgemm_naive_kernel, grid, block, 0, static_cast<hipStream_t>(stream), A,
      static_cast<const uint8_t *>(B), C, static_cast<const float *>(B_scales),
      static_cast<const uint8_t *>(B_zero_points), Y, M, N, K, trans_a != 0,
      trans_b != 0, a_dtype, b_dtype, c_dtype, y_dtype, b_bits, b_signed,
      c_dim0, c_dim1, M_ab, M_c, static_cast<int32_t>(a_zp),
      static_cast<int32_t>(b_zp), static_cast<int32_t>(c_zp),
      static_cast<int32_t>(y_zp));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qgemm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
