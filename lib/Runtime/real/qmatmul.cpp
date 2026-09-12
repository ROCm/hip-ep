/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A,B) = Q(DQ(A) @ DQ(B))
//
// a = (A - z_a) * s_a
// b = (B - z_b) * s_b
// Y = saturate(round((a @ b) / s_y) + z_y)
//
// (a @ b)[m,n] = s_a*s_b * sum_k (A[m,k] - z_a) * (B[k,n] - z_b)
//              = s_a*s_b * (acc[m,n] - z_b*rowA[m] - z_a*colB[n] + K*z_a*z_b)
//   acc[m,n] = sum_k A[m,k]*B[k,n]
//   rowA[m]  = sum_k A[m,k]
//   colB[n]  = sum_k B[k,n]
// --->
// let M = s_a * s_b / s_y
// Y = saturate(round(M * (acc - z_b*rowA - z_a*colB + K*z_a*z_b)) + z_y)
//
// acc, rowA and colB are exact integers, so the only rounding is the single
// multiply by M and the final round -- strictly less error than the unfused
// path, which accumulates K products in f32.
//
// A may be 16-bit (A16W8) while B stays 8-bit. The kernel keeps that exact by
// splitting A into two bytes, A = hi*256 + lo, and running one 8-bit dot per
// byte; see lib/Runtime/Kernels/hip/qmatmul_kernel.hip for the derivation.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>

// Maps the quantized types qmatmul accepts; which width each edge may use is
// the caller's rule.
static int hipdnn_to_hip_dtype_qmatmul(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  default:
    return -1;
  }
}

// Largest K that cannot overflow the int32 accumulator for the widest product
// each dtype pair can produce. Unsigned operands are the tighter case: 255*255
// per term against 127*128 for signed.
//
// A 16-bit A contributes 255 as well, because what the accumulator actually
// sees is its low byte -- the wider `hi` plane is bounded by 128 (int16) or 255
// (uint16) and never sets the limit. The GEMV path (M == 1) accumulates in
// int64 and is not bound by this at all, but the limit is applied uniformly so
// that a shape's validity does not depend on its batch size.
static int64_t max_safe_k(int a_dtype, int b_dtype) {
  const int64_t a_mag = (a_dtype == HIP_DTYPE_INT8) ? 128 : 255;
  const int64_t b_mag = (b_dtype == HIP_DTYPE_UINT8) ? 255 : 128;
  return INT32_MAX / (a_mag * b_mag);
}

int wrap_qmatmul(RuntimeState *state, const void *A, const void *B, void *Y,
                 int64_t M, int64_t N, int64_t K, int64_t batch_count,
                 int64_t b_batch_stride, int64_t trans_a, int64_t trans_b,
                 int64_t a_data_type, int64_t b_data_type, int64_t y_data_type,
                 float M_scale, int64_t A_zero_point, int64_t B_zero_point,
                 int64_t Y_zero_point) {
  OP_PROFILE(
      "qmatmul",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld,b=%lld:%s/%s->%s",
                 (long long)M, (long long)N, (long long)K,
                 (long long)batch_count, hipdnn_ep_datatype_name(a_data_type),
                 hipdnn_ep_datatype_name(b_data_type),
                 hipdnn_ep_datatype_name(y_data_type));
        return std::string(b);
      },
      state);

  if (!state || !A || !B || !Y) {
    fprintf(stderr, "wrap_qmatmul: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) {
    fprintf(stderr,
            "wrap_qmatmul: invalid dims M=%lld N=%lld K=%lld batch=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)batch_count);
    return -1;
  }

  const int a_dtype = hipdnn_to_hip_dtype_qmatmul(a_data_type);
  const int b_dtype = hipdnn_to_hip_dtype_qmatmul(b_data_type);
  const int y_dtype = hipdnn_to_hip_dtype_qmatmul(y_data_type);
  // Only A has a 16-bit path: the kernel splits it into two byte planes and
  // runs one 8-bit dot per plane, which leaves a wider B with nothing to use.
  bool isW8 = b_dtype == HIP_DTYPE_INT8 || b_dtype == HIP_DTYPE_UINT8;
  if (a_dtype < 0 || b_dtype < 0 || y_dtype < 0 || !isW8) {
    fprintf(stderr,
            "[REAL] wrap_qmatmul: unsupported data types A=%s(%lld) "
            "B=%s(%lld) Y=%s(%lld); A and Y must be 8- or 16-bit quantized "
            "and B must be 8-bit\n",
            hipdnn_ep_datatype_name(a_data_type), (long long)a_data_type,
            hipdnn_ep_datatype_name(b_data_type), (long long)b_data_type,
            hipdnn_ep_datatype_name(y_data_type), (long long)y_data_type);
    return -1;
  }

  const int64_t k_limit = max_safe_k(a_dtype, b_dtype);
  // Splitting a 16-bit A into two bytes is what keeps it on the 8-bit dot4
  // pipeline, and the price is an int32 accumulator, which is what bounds K.
  // Even the tightest dtype pairing still allows K = 33025, far past the
  // reduction widths real graphs use, so this rarely fires.
  if (K > k_limit) {
    fprintf(stderr,
            "[REAL] wrap_qmatmul: K=%lld exceeds the int32 accumulator limit "
            "%lld for this dtype pair\n",
            (long long)K, (long long)k_limit);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  // Split-K scratch. The kernel owns the decision -- it depends on the
  // device's compute-unit count, not just the extents -- and reports 0 for
  // shapes that already fill the GPU, so the common case reserves nothing. A
  // reservation that fails is not fatal either: hip_qmatmul falls back to the
  // single-pass kernel when it is handed no workspace.
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  const size_t splitk_bytes = hip_qmatmul_workspace_bytes(M, N, K, batch_count);
  if (splitk_bytes > 0 &&
      hipdnn_ep_state_ensure_workspace(state, splitk_bytes) == 0) {
    workspace = hipdnn_ep_state_get_workspace(state);
    workspace_bytes = hipdnn_ep_state_get_workspace_size(state);
  }

  if (hipdnn_ep_debug_enabled()) {
    fprintf(stderr,
            "[REAL] wrap_qmatmul: M=%lld N=%lld K=%lld batch=%lld "
            "b_batch_stride=%lld trans=(%lld,%lld) dtypes=(%s,%s,%s) "
            "M_scale=%g zp=(%lld,%lld,%lld)\n",
            (long long)M, (long long)N, (long long)K, (long long)batch_count,
            (long long)b_batch_stride, (long long)trans_a, (long long)trans_b,
            hipdnn_ep_datatype_name(a_data_type),
            hipdnn_ep_datatype_name(b_data_type),
            hipdnn_ep_datatype_name(y_data_type), (double)M_scale,
            (long long)A_zero_point, (long long)B_zero_point,
            (long long)Y_zero_point);
  }

  return hip_qmatmul(stream, A, B, Y, M, N, K, batch_count, b_batch_stride,
                     trans_a != 0, trans_b != 0, a_dtype, b_dtype, y_dtype,
                     M_scale, A_zero_point, B_zero_point, Y_zero_point,
                     workspace, workspace_bytes);
}
