/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A, B, C) = Q(alpha * DQ(A)' @ DQ(B)' + beta * DQ(C))
//
// Replaces the DQ x2 (or x3) -> Gemm -> Q chain a quantized export produces.
// The derivation and the ABI live in QGemmLowering.cpp and in wrap_qgemm's
// comment in hipdnn_ep_runtime.h; this file only validates the arguments and
// translates the HIPDNN dtype codes into the kernel's.
//
// There is no op state: hip.qgemm carries no OpStateOpInterface, and the
// kernel derives everything from its arguments.
//
// Both coefficients are already folded by lowering, which is what keeps this a
// translation layer. The one factor that could not fold -- a per-output-channel
// B_scales -- stays a device array and is applied inside the kernel.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// Maps the quantized types qgemm accepts. Which width each edge may use is the
// caller's rule, checked below rather than here.
static int hipdnn_to_hip_dtype_qgemm(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  default:
    return -1;
  }
}

static bool is_8_or_16_bit(int hip_dtype) {
  return hip_dtype == HIP_DTYPE_INT8 || hip_dtype == HIP_DTYPE_UINT8 ||
         hip_dtype == HIP_DTYPE_INT16 || hip_dtype == HIP_DTYPE_UINT16;
}

int wrap_qgemm(RuntimeState *state, const void *A, const void *B,
               const void *C, const void *B_scales, const void *B_zero_points,
               void *Y, int64_t M, int64_t N, int64_t K, int64_t trans_a,
               int64_t trans_b, int64_t a_data_type, int64_t b_data_type,
               int64_t c_data_type, int64_t y_data_type, int64_t b_bits,
               int64_t c_dim0, int64_t c_dim1, float M_ab, float M_c,
               int64_t A_zero_point, int64_t B_zero_point, int64_t C_zero_point,
               int64_t Y_zero_point) {
  OP_PROFILE(
      "qgemm",
      [&] {
        char b[128];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld,w%lldb%s:%s/%s->%s",
                 (long long)M, (long long)N, (long long)K, (long long)b_bits,
                 B_scales ? ",pc" : "", hipdnn_ep_datatype_name(a_data_type),
                 hipdnn_ep_datatype_name(b_data_type),
                 hipdnn_ep_datatype_name(y_data_type));
        return std::string(b);
      },
      state);

  if (!state || !A || !B || !Y) {
    fprintf(stderr, "[REAL] wrap_qgemm: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    fprintf(stderr, "[REAL] wrap_qgemm: invalid dims M=%lld N=%lld K=%lld\n",
            (long long)M, (long long)N, (long long)K);
    return -1;
  }
  // The kernel indexes a scale and a zero point by the same n, so neither has
  // a meaningful stand-in; the IR verifier rejects the half-given form too.
  if (static_cast<bool>(B_scales) != static_cast<bool>(B_zero_points)) {
    fprintf(stderr, "[REAL] wrap_qgemm: B_scales and B_zero_points must be "
                    "given together\n");
    return -1;
  }
  // 4 means two values per byte; 8 means full width. Anything else would hand
  // the kernel a stride nothing agreed on.
  if (b_bits != 4 && b_bits != 8) {
    fprintf(stderr, "[REAL] wrap_qgemm: unsupported b_bits %lld\n",
            (long long)b_bits);
    return -1;
  }

  const int a_dtype = hipdnn_to_hip_dtype_qgemm(a_data_type);
  const int b_dtype = hipdnn_to_hip_dtype_qgemm(b_data_type);
  const int y_dtype = hipdnn_to_hip_dtype_qgemm(y_data_type);
  // B is 8-bit STORAGE for both widths: a 4-bit weight holds two nibbles per
  // byte rather than changing element type.
  const bool b_is_8bit =
      b_dtype == HIP_DTYPE_INT8 || b_dtype == HIP_DTYPE_UINT8;
  if (!is_8_or_16_bit(a_dtype) || !is_8_or_16_bit(y_dtype) || !b_is_8bit) {
    fprintf(stderr,
            "[REAL] wrap_qgemm: unsupported data types A=%s(%lld) B=%s(%lld) "
            "Y=%s(%lld); A and Y must be 8- or 16-bit quantized and B must "
            "have 8-bit storage\n",
            hipdnn_ep_datatype_name(a_data_type), (long long)a_data_type,
            hipdnn_ep_datatype_name(b_data_type), (long long)b_data_type,
            hipdnn_ep_datatype_name(y_data_type), (long long)y_data_type);
    return -1;
  }

  // c_data_type is meaningful only alongside a C pointer; lowering reports
  // UNSUPPORTED when the bias is absent, which must not be diagnosed as an
  // unsupported type.
  int c_dtype = HIP_DTYPE_INT32;
  if (C) {
    c_dtype = hipdnn_to_hip_dtype_qgemm(c_data_type);
    // int32 is accepted only here: an ONNX quantizer emits a Gemm bias at
    // C_scale = A_scale * B_scale, which needs the wider type.
    if (!is_8_or_16_bit(c_dtype) && c_dtype != HIP_DTYPE_INT32) {
      fprintf(stderr,
              "[REAL] wrap_qgemm: unsupported C type %s(%lld); expected 8-, "
              "16- or 32-bit quantized\n",
              hipdnn_ep_datatype_name(c_data_type), (long long)c_data_type);
      return -1;
    }
    if (c_dim0 <= 0 || c_dim1 <= 0) {
      fprintf(stderr, "[REAL] wrap_qgemm: invalid C extents %lldx%lld\n",
              (long long)c_dim0, (long long)c_dim1);
      return -1;
    }
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qgemm: M=%lld N=%lld K=%lld trans=(%lld,%lld) "
      "dtypes=(%s,%s,%s,%s) b_bits=%lld per_channel=%s M_ab=%g M_c=%g "
      "zp=(%lld,%lld,%lld,%lld) C=%s[%lldx%lld]\n",
      (long long)M, (long long)N, (long long)K, (long long)trans_a,
      (long long)trans_b, hipdnn_ep_datatype_name(a_data_type),
      hipdnn_ep_datatype_name(b_data_type),
      hipdnn_ep_datatype_name(c_data_type),
      hipdnn_ep_datatype_name(y_data_type), (long long)b_bits,
      B_scales ? "yes" : "no", (double)M_ab, (double)M_c,
      (long long)A_zero_point, (long long)B_zero_point,
      (long long)C_zero_point, (long long)Y_zero_point, C ? "yes" : "null",
      (long long)c_dim0, (long long)c_dim1);

  int rc = hip_qgemm(stream, A, B, C, B_scales, B_zero_points, Y, M, N, K,
                     trans_a != 0, trans_b != 0, a_dtype, b_dtype, c_dtype,
                     y_dtype, static_cast<int>(b_bits), c_dim0, c_dim1, M_ab,
                     M_c, A_zero_point, B_zero_point, C_zero_point,
                     Y_zero_point);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qgemm: kernel launch failed (%d)\n", rc);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qgemm: completed successfully\n");
  return 0;
}
