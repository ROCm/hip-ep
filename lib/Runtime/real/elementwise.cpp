/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>
#include <utility>

// Map HIPDNN_EP_DATATYPE_* -> hip_dtype_t for custom kernels (e.g. hip_expand
// used as the broadcast-materialise fallback when MIOpen rejects double-side
// broadcast).  The two enum systems use different orderings.  Local copy of
// the helper in cast.cpp -- runtime files are bitcode TUs and cannot share
// statics without extra plumbing.
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
  default:
    return -1;
  }
}

// Explicit mapping from backend-independent HIPDNN_EP_DATATYPE_* enum to
// MIOpen-specific miopenDataType_t. No static_cast -- our enum values are
// independent of any library.
static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type, bool &ok) {
  ok = true;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    fprintf(stderr, "[REAL] unsupported data_type %lld for MIOpen\n",
            (long long)data_type);
    ok = false;
    return miopenFloat;
  }
}

// Explicit mapping from backend-independent HIPDNN_EP_TENSOR_OP_* enum to
// MIOpen-specific miopenTensorOp_t.
static miopenTensorOp_t hipdnn_ep_to_miopen_op(int64_t tensor_op, bool &ok) {
  ok = true;
  switch (tensor_op) {
  case HIPDNN_EP_TENSOR_OP_MUL:
    return miopenTensorOpMul;
  case HIPDNN_EP_TENSOR_OP_ADD:
    return miopenTensorOpAdd;
  case HIPDNN_EP_TENSOR_OP_MIN:
    return miopenTensorOpMin;
  case HIPDNN_EP_TENSOR_OP_MAX:
    return miopenTensorOpMax;
  default:
    fprintf(stderr, "[REAL] unsupported tensor_op %lld for MIOpen\n",
            (long long)tensor_op);
    ok = false;
    return miopenTensorOpMul;
  }
}

//===----------------------------------------------------------------------===//
// MIOpen OpTensor descriptor cache
//===----------------------------------------------------------------------===//
//
// Three MIOpen tensor descriptors (lhs, rhs, output) are created once per
// unique (lhs_shape, rhs_shape, out_shape, data_type) combination and reused
// for the process lifetime.  Avoids repeated miopenCreate/Set/Destroy on
// every element-wise inference call.

struct OpTensorCacheKey {
  int64_t lhs_n, lhs_c, lhs_h, lhs_w; // lhs 4D shape
  int64_t rhs_n, rhs_c, rhs_h, rhs_w; // rhs 4D shape
  int64_t out_n, out_c, out_h, out_w; // output 4D shape
  int64_t data_type;                  // HIPDNN_EP_DATATYPE_* enum value
  bool operator==(const OpTensorCacheKey &o) const {
    return lhs_n == o.lhs_n && lhs_c == o.lhs_c && lhs_h == o.lhs_h &&
           lhs_w == o.lhs_w && rhs_n == o.rhs_n && rhs_c == o.rhs_c &&
           rhs_h == o.rhs_h && rhs_w == o.rhs_w && out_n == o.out_n &&
           out_c == o.out_c && out_h == o.out_h && out_w == o.out_w &&
           data_type == o.data_type;
  }
};

struct OpTensorCacheKeyHash {
  size_t operator()(const OpTensorCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.lhs_n);
    hash_combine_val(h, k.lhs_c);
    hash_combine_val(h, k.lhs_h);
    hash_combine_val(h, k.lhs_w);
    hash_combine_val(h, k.rhs_n);
    hash_combine_val(h, k.rhs_c);
    hash_combine_val(h, k.rhs_h);
    hash_combine_val(h, k.rhs_w);
    hash_combine_val(h, k.out_n);
    hash_combine_val(h, k.out_c);
    hash_combine_val(h, k.out_h);
    hash_combine_val(h, k.out_w);
    hash_combine_val(h, k.data_type);
    return h;
  }
};

/// Cached MIOpen tensor descriptors for a single OpTensor shape.
/// Ownership: descriptors are created in queryOrCreateOpTensor() and live
/// for the process lifetime (never destroyed individually).
struct OpTensorCacheEntry {
  miopenTensorDescriptor_t aDesc, bDesc, cDesc; // lhs, rhs, output
};

static std::unordered_map<OpTensorCacheKey, OpTensorCacheEntry,
                          OpTensorCacheKeyHash>
    g_optensor_cache;

/// Look up or create cached MIOpen tensor descriptors for an OpTensor shape.
/// Returns nullptr on any MIOpen API failure (partially created descriptors
/// are cleaned up before returning).
static const OpTensorCacheEntry *
queryOrCreateOpTensor(const OpTensorCacheKey &key) {
  auto it = g_optensor_cache.find(key);
  if (it != g_optensor_cache.end())
    return &it->second;

  bool type_ok;
  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type, type_ok);
  if (!type_ok)
    return nullptr;
  OpTensorCacheEntry e{};
  int result = 0;

  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.aDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.bDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.cDesc), cache_fail);

  {
    int a_dims[] = {static_cast<int>(key.lhs_n), static_cast<int>(key.lhs_c),
                    static_cast<int>(key.lhs_h), static_cast<int>(key.lhs_w)};
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.aDesc, dt, miopenTensorNCHW, a_dims, 4),
                      cache_fail);
    int b_dims[] = {static_cast<int>(key.rhs_n), static_cast<int>(key.rhs_c),
                    static_cast<int>(key.rhs_h), static_cast<int>(key.rhs_w)};
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.bDesc, dt, miopenTensorNCHW, b_dims, 4),
                      cache_fail);
    int c_dims[] = {static_cast<int>(key.out_n), static_cast<int>(key.out_c),
                    static_cast<int>(key.out_h), static_cast<int>(key.out_w)};
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.cDesc, dt, miopenTensorNCHW, c_dims, 4),
                      cache_fail);
  }
  goto cache_done;

cache_fail:
  if (e.cDesc)
    miopenDestroyTensorDescriptor(e.cDesc);
  if (e.bDesc)
    miopenDestroyTensorDescriptor(e.bDesc);
  if (e.aDesc)
    miopenDestroyTensorDescriptor(e.aDesc);
  return nullptr;

cache_done:
  auto [ins, _] = g_optensor_cache.emplace(key, e);
  return &ins->second;
}

//===----------------------------------------------------------------------===//
// Generic Element-wise Tensor Operation via MIOpen
//===----------------------------------------------------------------------===//
//
// Uses miopenOpTensor to compute:
//   output = alpha1 * op(lhs, alpha2 * rhs) + beta * output
// With alpha1=1, alpha2=1, beta=0 this gives: output = op(lhs, rhs)
//
// Each operand's shape is passed as 4D (N, C, H, W) to allow MIOpen-native
// broadcasting.  When a dimension is 1 in one operand but >1 in the other,
// MIOpen broadcasts automatically (e.g. bias addition).
// The compiler (HipToLLVM) left-pads shapes with 1 for rank < 4.
//===----------------------------------------------------------------------===//

int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs, void *output,
                        int64_t lhs_n, int64_t lhs_c, int64_t lhs_h,
                        int64_t lhs_w, int64_t rhs_n, int64_t rhs_c,
                        int64_t rhs_h, int64_t rhs_w, int64_t out_n,
                        int64_t out_c, int64_t out_h, int64_t out_w,
                        int64_t data_type, int64_t tensor_op) {
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
  // even when one operand is empty. The MIOpen tensor-op API rejects
  // 0-dim descriptors, so we must handle this here.
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
        fprintf(stderr,
                "[elementwise-empty] unsupported dtype %lld for identity-copy\n",
                (long long)data_type);
        return -1;
      }
      size_t bytes = static_cast<size_t>(out_n) * out_c * out_h * out_w *
                     elem_size;
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
    fprintf(stderr,
            "[elementwise-empty] no operand matches OUT shape; "
            "zero-filling output as a safe default\n");
    hipStream_t stream =
        static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
    size_t elem_size = (data_type == HIPDNN_EP_DATATYPE_HALF) ? 2 : 4;
    size_t bytes = static_cast<size_t>(out_n) * out_c * out_h * out_w *
                   elem_size;
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

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenOpTensor: null MIOpen handle\n");
    return -1;
  }

  bool op_ok;
  miopenTensorOp_t miopen_op = hipdnn_ep_to_miopen_op(tensor_op, op_ok);
  if (!op_ok) {
    fprintf(stderr, "wrap_miopenOpTensor: unsupported tensor_op %lld\n",
            (long long)tensor_op);
    return -1;
  }

  // Integer dtypes (i32/i64/ui8) aren't supported by miopenOpTensor. Vision
  // encoders run small i64 shape arithmetic via these ops (e.g. multiplying
  // two i64 scalars to compute a downstream Reshape dim); attention chains
  // run i32/i64 Min/Max for the seqlens_k = Min(total_seq_len, max_seq_len)
  // clamp. Route those to a custom HIP kernel that handles same-shape
  // mul/add/min/max. Broadcasting is materialised below by hip_expand into
  // a per-state workspace before the flat kernel runs.
  if (data_type == HIPDNN_EP_DATATYPE_INT64 ||
      data_type == HIPDNN_EP_DATATYPE_INT32) {
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

  // MIOpen's miopenOpTensor requires A.shape == C.shape; only B may broadcast
  // (dim==1) into A. Caller-provided lhs/rhs ordering is dictated by the
  // original ONNX graph and is not normalized by the lowering pass, so when
  // the broadcast-source operand happens to be in lhs position MIOpen rejects
  // with "A and C Tensors do not match". All ops routed here are commutative
  // (MUL/ADD/MIN/MAX -- see hipdnn_ep_to_miopen_op), so we can safely swap
  // lhs<->rhs to put the output-shaped tensor on the A side.
  const bool lhs_eq_out =
      (lhs_n == out_n && lhs_c == out_c && lhs_h == out_h && lhs_w == out_w);
  const bool rhs_eq_out =
      (rhs_n == out_n && rhs_c == out_c && rhs_h == out_h && rhs_w == out_w);
  if (!lhs_eq_out && rhs_eq_out) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_miopenOpTensor: swapping lhs<->rhs to satisfy MIOpen "
        "A==C constraint (was lhs=[%lld,%lld,%lld,%lld] rhs=[%lld,%lld,%lld,"
        "%lld])\n",
        (long long)lhs_n, (long long)lhs_c, (long long)lhs_h, (long long)lhs_w,
        (long long)rhs_n, (long long)rhs_c, (long long)rhs_h, (long long)rhs_w);
    std::swap(lhs, rhs);
    std::swap(lhs_n, rhs_n);
    std::swap(lhs_c, rhs_c);
    std::swap(lhs_h, rhs_h);
    std::swap(lhs_w, rhs_w);
  }

  // True 2-D broadcast: neither lhs nor rhs equals out. The canonical case
  // is `lhs=[1,1,H,1] * rhs=[1,1,1,W] -> out=[1,1,H,W]` from vision-encoder
  // mrope tables and similar position-encoding patterns. MIOpen accepts only
  // single-side broadcast (B may broadcast into A; A must equal C). Expand
  // the smaller-volume side into the runtime workspace, then call MIOpen
  // with the expanded side as A. Workspace is single-use within this call
  // (the next stream op runs after the MIOpen call returns, but the kernel
  // is queued on the same stream and reads the workspace before any later
  // user grows it).
  if (!lhs_eq_out && !rhs_eq_out) {
    const int64_t lhs_vol = lhs_n * lhs_c * lhs_h * lhs_w;
    const int64_t rhs_vol = rhs_n * rhs_c * rhs_h * rhs_w;
    const bool expand_lhs = (lhs_vol <= rhs_vol);
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const int64_t out_vol = out_n * out_c * out_h * out_w;
    const size_t needed = static_cast<size_t>(out_vol * elem_bytes);
    if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
      fprintf(stderr,
              "wrap_miopenOpTensor: failed to ensure workspace %zu bytes for "
              "broadcast expand\n",
              needed);
      return -1;
    }
    void *ws = hipdnn_ep_state_get_workspace(state);
    void *stream = hipdnn_ep_state_get_stream(state);

    const int64_t in_shape[4] = {
        expand_lhs ? lhs_n : rhs_n, expand_lhs ? lhs_c : rhs_c,
        expand_lhs ? lhs_h : rhs_h, expand_lhs ? lhs_w : rhs_w};
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};
    int hip_dtype = hipdnn_to_hip_dtype(data_type);
    if (hip_dtype < 0) {
      fprintf(stderr,
              "wrap_miopenOpTensor: unsupported data_type %lld for expand\n",
              (long long)data_type);
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_miopenOpTensor: 2-D broadcast detected, expanding %s "
        "side [%lld,%lld,%lld,%lld] to [%lld,%lld,%lld,%lld] into workspace\n",
        expand_lhs ? "lhs" : "rhs", (long long)in_shape[0],
        (long long)in_shape[1], (long long)in_shape[2], (long long)in_shape[3],
        (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w);
    int rc = hip_expand(stream, expand_lhs ? lhs : rhs, ws, in_shape, 4,
                        out_shape, 4, hip_dtype);
    if (rc != 0) {
      fprintf(stderr, "wrap_miopenOpTensor: hip_expand failed (%d)\n", rc);
      return -1;
    }

    if (expand_lhs) {
      lhs = ws;
      lhs_n = out_n;
      lhs_c = out_c;
      lhs_h = out_h;
      lhs_w = out_w;
    } else {
      // After expanding rhs to out shape, swap so the expanded side lands on
      // A (lhs); the original lhs becomes the broadcasting B side, which
      // MIOpen handles natively for its 1-axes.
      void *expanded = ws;
      void *orig_lhs = lhs;
      const int64_t orig_lhs_n = lhs_n, orig_lhs_c = lhs_c, orig_lhs_h = lhs_h,
                    orig_lhs_w = lhs_w;
      lhs = expanded;
      lhs_n = out_n;
      lhs_c = out_c;
      lhs_h = out_h;
      lhs_w = out_w;
      rhs = orig_lhs;
      rhs_n = orig_lhs_n;
      rhs_c = orig_lhs_c;
      rhs_h = orig_lhs_h;
      rhs_w = orig_lhs_w;
    }
  }

  OpTensorCacheKey key{lhs_n, lhs_c, lhs_h, lhs_w, rhs_n, rhs_c,    rhs_h,
                       rhs_w, out_n, out_c, out_h, out_w, data_type};
  const OpTensorCacheEntry *c = queryOrCreateOpTensor(key);
  if (!c) {
    fprintf(stderr, "wrap_miopenOpTensor: descriptor cache creation failed\n");
    return -1;
  }

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;

  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: calling miopenOpTensor"
                    "(op=%s, alpha1=%.1f, alpha2=%.1f, beta=%.1f)\n",
                    op_name, alpha1, alpha2, beta);

  miopenStatus_t st =
      miopenOpTensor(handle, miopen_op, &alpha1, c->aDesc, lhs, &alpha2,
                     c->bDesc, rhs, &beta, c->cDesc, output);
  if (st != miopenStatusSuccess) {
    fprintf(stderr, "wrap_miopenOpTensor: miopenOpTensor failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: completed successfully\n");
  return 0;
}

//===----------------------------------------------------------------------===//
// Element-wise Subtraction via Custom HIP Kernel
//===----------------------------------------------------------------------===//
//
// For types unsupported by MIOpen (e.g. int64), dispatches to
// hip_elementwise_sub from the custom kernels library. The caller passes
// element_size_bytes; we map it to the corresponding hip_dtype_t.
//===----------------------------------------------------------------------===//

int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t num_elements,
                         int64_t element_size_bytes) {
  OP_PROFILE(
      "sub",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr, "wrap_elementwise_sub: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  // element_size_bytes is the only dtype hint plumbed from the lowering.
  // 2 = fp16 (vision/text activations), 4 = fp32 / i32 (treat as fp32 since
  // i32 sub from MIOpen would have been routed elsewhere), 8 = i64 (shape
  // arithmetic). The kernel performs subtraction; fp16 vs i32 with the same
  // 4-byte layout would need disambiguation, but in practice the elementwise
  // sub call site only emits fp arithmetic at 4 bytes (Range / norm chains).
  int hip_dtype;
  switch (element_size_bytes) {
  case 2:
    hip_dtype = HIP_DTYPE_FLOAT16;
    break;
  case 4:
    hip_dtype = HIP_DTYPE_FLOAT32;
    break;
  case 8:
    hip_dtype = HIP_DTYPE_INT64;
    break;
  default:
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_elementwise_sub: unsupported element_size=%lld\n",
        (long long)element_size_bytes);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_elementwise_sub: num_elements=%lld, "
      "element_size=%lld, dtype=%d -> calling hip_elementwise_sub\n",
      (long long)num_elements, (long long)element_size_bytes, hip_dtype);

  return hip_elementwise_sub(stream, lhs, rhs, output, num_elements, hip_dtype);
}
