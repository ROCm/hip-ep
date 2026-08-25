/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

// Map HIPDNN_EP_DATATYPE_* -> hip_dtype_t for custom kernels (e.g. hip_expand
// used to materialise broadcasting before the flat integer kernel, which
// cannot broadcast on its own).  The two enum systems use different
// orderings.  Local copy of the helper in cast.cpp -- runtime files are
// bitcode TUs and cannot share statics without extra plumbing.
static int hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  default:
    return -1;
  }
}


// Per-instance op state for hip.add/mul/min/max. The state itself is empty --
// these ops no longer need any shared per-device data -- but the slot must
// still be constructed because the compiler unconditionally emits a
// generateOpStateInit call for any op implementing OpStateOpInterface (see
// Hip_AddOp/MulOp/MinOp/MaxOp in HipOps.td).
struct OpTensorState : OpStateT<OpTensorState> {};

extern "C" int8_t hipdnn_ep_op_state_construct_optensor(RuntimeState *state,
                                                        int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, OpTensorState::create().release());
  return 0;
}

//===----------------------------------------------------------------------===//
// Generic Element-wise Tensor Operation
//===----------------------------------------------------------------------===//
//
// Dispatches to custom HIP kernels: hip_elementwise_{mul,add,min,max} for
// int16/int32/int64, and the native broadcasting kernel
// hip_elementwise_binary_bcast for float16/float32.  Any other data_type /
// tensor_op combination is unsupported and returns an error.
//
// Each operand's shape is passed as 4D (N, C, H, W). For the integer path,
// non-matching operands are materialised to the output shape via hip_expand
// before the flat kernel runs; the float path broadcasts natively (dimension
// 1 in one operand vs >1 in the other) without materialisation.
// The compiler (HipToLLVM) left-pads shapes with 1 for rank < 4.
// Currently supported data_types: float16, float32, int32, int64, int16
// Currently supported tensor_ops: MUL, ADD, MIN, MAX
//===----------------------------------------------------------------------===//

int wrap_miopenOpTensor(RuntimeState *state, int op_state_slot, void *lhs,
                        void *rhs, void *output, int64_t lhs_n, int64_t lhs_c,
                        int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                        int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                        int64_t out_n, int64_t out_c, int64_t out_h,
                        int64_t out_w, int64_t data_type, int64_t tensor_op) {
  OP_PROFILE(
      "elementwise",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w);
        return std::string(b);
      },
      state);
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr,
            "wrap_miopenOpTensor: null argument (state=%p lhs=%p rhs=%p "
            "output=%p op=%lld dtype=%lld)\n",
            (void *)state, lhs, rhs, output, (long long)tensor_op,
            (long long)data_type);
    return -1;
  }

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  const char *op_name = hipdnn_ep_tensor_op_name(tensor_op);

  // Empty-operand identity-copy path.
  //
  // ONNX broadcast semantics are undefined when one operand has a dim
  // that's 0 and the other has the same dim > 1. In practice, model
  // exporters that emit "empty placeholder + Loop concat accumulator"
  // patterns produce downstream Add/Sub/Mul/Div ops where the empty
  // operand is meant as an additive/multiplicative identity (no
  // contribution). Some exporters' shape inference then sizes the
  // output OUT to MAX(LHS_dim, RHS_dim), which gives a non-empty OUT
  // even when one operand is empty. 0-dim descriptors are not well-defined
  // for the custom elementwise kernels either, so we must handle this here.
  //
  // Treatment: if one operand is empty and the other has the same
  // shape as OUT, copy the non-empty operand into OUT. If both
  // operands are empty OR OUT itself is empty, the call is a no-op.
  const bool out_empty = (out_n == 0 || out_c == 0 || out_h == 0 || out_w == 0);
  const bool lhs_empty = (lhs_n == 0 || lhs_c == 0 || lhs_h == 0 || lhs_w == 0);
  const bool rhs_empty = (rhs_n == 0 || rhs_c == 0 || rhs_h == 0 || rhs_w == 0);
  if (out_empty || lhs_empty || rhs_empty) {
    static int dbg_count = 0;
    if (dbg_count++ < 4) {
      fprintf(stderr,
              "[elementwise-empty] op=%s dtype=%s "
              "lhs=[%lld,%lld,%lld,%lld]%s "
              "rhs=[%lld,%lld,%lld,%lld]%s "
              "out=[%lld,%lld,%lld,%lld]%s\n",
              op_name, type_name, (long long)lhs_n, (long long)lhs_c,
              (long long)lhs_h, (long long)lhs_w, lhs_empty ? "(E)" : "",
              (long long)rhs_n, (long long)rhs_c, (long long)rhs_h,
              (long long)rhs_w, rhs_empty ? "(E)" : "", (long long)out_n,
              (long long)out_c, (long long)out_h, (long long)out_w,
              out_empty ? "(E)" : "");
    }
    if (out_empty)
      return 0;
    // Pick the non-empty operand whose shape matches OUT.
    bool lhs_matches_out = !lhs_empty && lhs_n == out_n && lhs_c == out_c &&
                           lhs_h == out_h && lhs_w == out_w;
    bool rhs_matches_out = !rhs_empty && rhs_n == out_n && rhs_c == out_c &&
                           rhs_h == out_h && rhs_w == out_w;
    if (lhs_matches_out || rhs_matches_out) {
      size_t elem_size = 0;
      switch (data_type) {
      case HIPDNN_EP_DATATYPE_HALF:
        elem_size = 2;
        break;
      case HIPDNN_EP_DATATYPE_FLOAT:
        elem_size = 4;
        break;
      case HIPDNN_EP_DATATYPE_INT32:
        elem_size = 4;
        break;
      case HIPDNN_EP_DATATYPE_INT64:
        elem_size = 8;
        break;
      default:
        fprintf(
            stderr,
            "[elementwise-empty] unsupported dtype %lld for identity-copy\n",
            (long long)data_type);
        return -1;
      }
      size_t bytes =
          static_cast<size_t>(out_n) * out_c * out_h * out_w * elem_size;
      void *src = lhs_matches_out ? lhs : rhs;
      hipStream_t stream =
          static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
      hipError_t err =
          hipMemcpyAsync(output, src, bytes, hipMemcpyDeviceToDevice, stream);
      if (err != hipSuccess) {
        fprintf(stderr, "[elementwise-empty] hipMemcpyAsync failed: %s\n",
                hipGetErrorString(err));
        return -1;
      }
      return 0;
    }
    // Neither operand matches OUT — undefined broadcast. Zero-fill OUT
    // as a safe default (matches CPU treating empty as additive identity
    // and the other operand contributing nothing meaningful).
    fprintf(stderr, "[elementwise-empty] no operand matches OUT shape; "
                    "zero-filling output as a safe default\n");
    hipStream_t stream =
        static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
    size_t elem_size = (data_type == HIPDNN_EP_DATATYPE_HALF) ? 2 : 4;
    size_t bytes =
        static_cast<size_t>(out_n) * out_c * out_h * out_w * elem_size;
    hipError_t err = hipMemsetAsync(output, 0, bytes, stream);
    if (err != hipSuccess)
      return -1;
    return 0;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: op=%s, "
                    "lhs=[%lld,%lld,%lld,%lld], "
                    "rhs=[%lld,%lld,%lld,%lld], "
                    "out=[%lld,%lld,%lld,%lld], "
                    "data_type=%s(%lld)\n",
                    op_name, (long long)lhs_n, (long long)lhs_c,
                    (long long)lhs_h, (long long)lhs_w, (long long)rhs_n,
                    (long long)rhs_c, (long long)rhs_h, (long long)rhs_w,
                    (long long)out_n, (long long)out_c, (long long)out_h,
                    (long long)out_w, type_name, (long long)data_type);

  if (data_type == HIPDNN_EP_DATATYPE_INT64 ||
      data_type == HIPDNN_EP_DATATYPE_INT32 ||
      data_type == HIPDNN_EP_DATATYPE_INT16) {
    if (tensor_op != HIPDNN_EP_TENSOR_OP_MUL &&
        tensor_op != HIPDNN_EP_TENSOR_OP_ADD &&
        tensor_op != HIPDNN_EP_TENSOR_OP_MIN &&
        tensor_op != HIPDNN_EP_TENSOR_OP_MAX) {
      fprintf(stderr,
              "wrap_miopenOpTensor: integer fallback only supports "
              "MUL/ADD/MIN/MAX (got tensor_op=%lld, data_type=%lld)\n",
              (long long)tensor_op, (long long)data_type);
      return -1;
    }
    void *stream = hipdnn_ep_state_get_stream(state);
    int hip_dtype = hipdnn_to_hip_dtype(data_type);
    if (hip_dtype < 0) {
      fprintf(stderr,
              "wrap_miopenOpTensor: integer fallback unsupported dtype %lld\n",
              (long long)data_type);
      return -1;
    }
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const int64_t out_vol = out_n * out_c * out_h * out_w;
    const bool lhs_eq_out_ints =
        (lhs_n == out_n && lhs_c == out_c && lhs_h == out_h && lhs_w == out_w);
    const bool rhs_eq_out_ints =
        (rhs_n == out_n && rhs_c == out_c && rhs_h == out_h && rhs_w == out_w);
    // Materialise broadcasting via hip_expand into the per-state workspace,
    // then call hip_elementwise on same-shape operands. hip_elementwise is a
    // flat kernel and cannot broadcast on its own; vision-encoder shape
    // arithmetic (e.g. multiplying [6,1,1,1] by [1,1,1,1] to drive a Range
    // step) routinely produces broadcasting integer ops, so the integer path
    // must mirror the float-side broadcast handling below.
    void *lhs_use = lhs;
    void *rhs_use = rhs;
    void *ws = nullptr;
    if (!lhs_eq_out_ints || !rhs_eq_out_ints) {
      // We need workspace for any side(s) that need expansion. At most two
      // (lhs to ws_a, rhs to ws_b) -- pack both back-to-back in the workspace.
      const size_t per_side = static_cast<size_t>(out_vol * elem_bytes);
      const size_t needed =
          per_side * static_cast<size_t>((!lhs_eq_out_ints ? 1 : 0) +
                                         (!rhs_eq_out_ints ? 1 : 0));
      if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
        fprintf(stderr,
                "wrap_miopenOpTensor: integer fallback workspace ensure failed "
                "(%zu bytes)\n",
                needed);
        return -1;
      }
      ws = hipdnn_ep_state_get_workspace(state);
      uint8_t *ws_byte = static_cast<uint8_t *>(ws);
      const int64_t in_lhs[4] = {lhs_n, lhs_c, lhs_h, lhs_w};
      const int64_t in_rhs[4] = {rhs_n, rhs_c, rhs_h, rhs_w};
      const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};
      if (!lhs_eq_out_ints) {
        int rc = hip_expand(stream, lhs, ws_byte, in_lhs, 4, out_shape, 4,
                            hip_dtype);
        if (rc != 0) {
          fprintf(stderr,
                  "wrap_miopenOpTensor: integer fallback hip_expand(lhs) "
                  "failed (%d)\n",
                  rc);
          return -1;
        }
        lhs_use = ws_byte;
        ws_byte += per_side;
      }
      if (!rhs_eq_out_ints) {
        int rc = hip_expand(stream, rhs, ws_byte, in_rhs, 4, out_shape, 4,
                            hip_dtype);
        if (rc != 0) {
          fprintf(stderr,
                  "wrap_miopenOpTensor: integer fallback hip_expand(rhs) "
                  "failed (%d)\n",
                  rc);
          return -1;
        }
        rhs_use = ws_byte;
      }
    }
    int rc = -1;
    switch (tensor_op) {
    case HIPDNN_EP_TENSOR_OP_MUL:
      rc = hip_elementwise_mul(stream, lhs_use, rhs_use, output, out_vol,
                               hip_dtype);
      break;
    case HIPDNN_EP_TENSOR_OP_ADD:
      rc = hip_elementwise_add(stream, lhs_use, rhs_use, output, out_vol,
                               hip_dtype);
      break;
    case HIPDNN_EP_TENSOR_OP_MIN:
      rc = hip_elementwise_min(stream, lhs_use, rhs_use, output, out_vol,
                               hip_dtype);
      break;
    case HIPDNN_EP_TENSOR_OP_MAX:
      rc = hip_elementwise_max(stream, lhs_use, rhs_use, output, out_vol,
                               hip_dtype);
      break;
    }
    return rc;
  }

  // Float/half Add/Mul/Min/Max fast path: a single broadcasting HIP kernel.
  // The custom kernel does this work in well under a millisecond with fully
  // coalesced output writes and native per-axis broadcasting (no Expand
  // materialisation). bf16, any op not in {ADD,MUL,MIN,MAX}, or an output
  // volume exceeding the kernel's 32-bit index range fall through to the
  // unsupported-combination error below -- there is no fallback.
  if ((data_type == HIPDNN_EP_DATATYPE_HALF ||
       data_type == HIPDNN_EP_DATATYPE_FLOAT) &&
      (tensor_op == HIPDNN_EP_TENSOR_OP_ADD ||
       tensor_op == HIPDNN_EP_TENSOR_OP_MUL ||
       tensor_op == HIPDNN_EP_TENSOR_OP_MIN ||
       tensor_op == HIPDNN_EP_TENSOR_OP_MAX)) {
    int bcast_op = (tensor_op == HIPDNN_EP_TENSOR_OP_ADD)   ? 0
                   : (tensor_op == HIPDNN_EP_TENSOR_OP_MUL) ? 1
                   : (tensor_op == HIPDNN_EP_TENSOR_OP_MIN) ? 2
                                                            : 3;
    int hip_dtype = hipdnn_to_hip_dtype(data_type);
    void *stream = hipdnn_ep_state_get_stream(state);
    const int64_t lhs_shape4[4] = {lhs_n, lhs_c, lhs_h, lhs_w};
    const int64_t rhs_shape4[4] = {rhs_n, rhs_c, rhs_h, rhs_w};
    const int64_t out_shape4[4] = {out_n, out_c, out_h, out_w};
    int rc = hip_elementwise_binary_bcast(stream, lhs, rhs, output, lhs_shape4,
                                          rhs_shape4, out_shape4, bcast_op,
                                          hip_dtype);
    // rc == -2: output volume exceeds the kernel's 32-bit index range; no
    // fallback exists, so this falls through to the unsupported-combination
    // error below. Any other rc is terminal.
    if (rc != -2)
      return rc;
  }

  fprintf(stderr, "wrap_miopenOpTensor: not support datatype: %s, op: %s\n",
          type_name, op_name);
  return -1;
}


static int sub_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t lhs_n, int64_t lhs_c,
                         int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                         int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                         int64_t out_n, int64_t out_c, int64_t out_h,
                         int64_t out_w, int64_t data_type) {
  OP_PROFILE(
      "sub",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w);
        return std::string(b);
      },
      state);
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr, "wrap_elementwise_sub: null argument\n");
    return -1;
  }

  const int64_t out_vol = out_n * out_c * out_h * out_w;
  if (out_vol <= 0)
    return 0;

  int hip_dtype = sub_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_elementwise_sub: unsupported data_type=%s(%lld)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  const bool lhs_eq_out =
      (lhs_n == out_n && lhs_c == out_c && lhs_h == out_h && lhs_w == out_w);
  const bool rhs_eq_out =
      (rhs_n == out_n && rhs_c == out_c && rhs_h == out_h && rhs_w == out_w);

  void *lhs_use = lhs;
  void *rhs_use = rhs;

  if (!lhs_eq_out || !rhs_eq_out) {
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const size_t per_side = static_cast<size_t>(out_vol * elem_bytes);
    const size_t needed = per_side * static_cast<size_t>((!lhs_eq_out ? 1 : 0) +
                                                         (!rhs_eq_out ? 1 : 0));
    if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
      fprintf(stderr,
              "wrap_elementwise_sub: workspace ensure failed (%zu bytes)\n",
              needed);
      return -1;
    }
    void *ws = hipdnn_ep_state_get_workspace(state);
    uint8_t *ws_byte = static_cast<uint8_t *>(ws);
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};

    if (!lhs_eq_out) {
      const int64_t in_lhs[4] = {lhs_n, lhs_c, lhs_h, lhs_w};
      int rc =
          hip_expand(stream, lhs, ws_byte, in_lhs, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "wrap_elementwise_sub: hip_expand(lhs) failed (%d)\n",
                rc);
        return -1;
      }
      lhs_use = ws_byte;
      ws_byte += per_side;
    }
    if (!rhs_eq_out) {
      const int64_t in_rhs[4] = {rhs_n, rhs_c, rhs_h, rhs_w};
      int rc =
          hip_expand(stream, rhs, ws_byte, in_rhs, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "wrap_elementwise_sub: hip_expand(rhs) failed (%d)\n",
                rc);
        return -1;
      }
      rhs_use = ws_byte;
    }

    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_elementwise_sub: broadcast expand lhs%s rhs%s -> "
        "out=[%lld,%lld,%lld,%lld], dtype=%s\n",
        lhs_eq_out ? "(ok)" : "(expanded)", rhs_eq_out ? "(ok)" : "(expanded)",
        (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w,
        hipdnn_ep_datatype_name(data_type));
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_elementwise_sub: same-shape out=[%lld,%lld,%lld,%lld], "
        "dtype=%s\n",
        (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w,
        hipdnn_ep_datatype_name(data_type));
  }

  return hip_elementwise_sub(stream, lhs_use, rhs_use, output, out_vol,
                             hip_dtype);
}
