/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_WHERE_MAX_RANK 8

// =============================================================================
// CONDITION BUFFER LAYOUT ASSUMPTION (1 byte / element)
// =============================================================================
// The `condition` buffer is read here as `const bool *`, i.e. one byte per
// element. This relies on an *implicit* contract with the upstream MLIR
// pipeline: the ONNX `tensor(bool)` reaches us either as `memref<...xi1, 1>`
// (hand-written IR / lit tests) or as a 1-byte integer memref
// `memref<...xi8, 1>` / `memref<...xui8, 1>` (the ORT/morphizen ONNX -> MLIR
// frontend mirrors the on-disk `TensorProto` BOOL layout, which is 1 byte
// per element). Under the default LLVM/MLIR memref-to-LLVM lowering all
// three encodings produce a 1-byte-per-element device buffer that matches
// the C/C++ `bool` ABI, so we can safely reinterpret the device pointer as
// `const bool *` regardless of which encoding the IR carried.
//
// IMPORTANT: this is *not* enforced by a verifier or by a type cast in the
// IR -- the lowering pipeline never converts the condition element type
// explicitly. If a future LLVM/MLIR version switches to bit-packed `i1`
// memrefs (multiple bools per byte) without our pipeline being updated,
// this kernel will silently read garbage offsets without any compile-time
// error. Any change to the condition -> bytes lowering must be mirrored
// here (and in the matching host-side launcher in
// `lib/Conversion/HipToLLVM/WhereLowering.cpp`).
// =============================================================================

// Fixed-size shape/stride buffers passed as kernel arguments (by value).
// Capacity is HIP_WHERE_MAX_RANK; entries past `rank` are unused.
struct WhereDimsND {
  int64_t v[HIP_WHERE_MAX_RANK];
};

// N-D broadcast kernel.
// Decomposes the linear output index into coordinates along `out_shape.v[0..rank)`
// using row-major order, then reconstructs a linear index into each operand
// using its precomputed broadcast stride (stride == 0 along axes where the
// operand's padded dim is 1, i.e. broadcast).
template <typename T>
__global__ void where_nd_kernel(const bool *__restrict__ condition,
                                const T *__restrict__ x,
                                const T *__restrict__ y,
                                T *__restrict__ output,
                                int64_t total_elements, int rank,
                                WhereDimsND out_shape, WhereDimsND cond_stride,
                                WhereDimsND x_stride, WhereDimsND y_stride) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  int64_t cond_idx = 0, x_idx = 0, y_idx = 0;
  int64_t tmp = idx;
  for (int k = rank - 1; k >= 0; --k) {
    int64_t dim = out_shape.v[k];
    int64_t coord = tmp % dim;
    tmp /= dim;
    cond_idx += coord * cond_stride.v[k];
    x_idx += coord * x_stride.v[k];
    y_idx += coord * y_stride.v[k];
  }

  output[idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
}

// Left-pad `src_shape` (length `src_rank`) to `out_rank` with 1s into
// `padded_shape`, then compute row-major strides over the padded shape and
// validate NumPy-style broadcast compatibility against `out_shape`.
//
// For each axis k, the padded operand dim must satisfy one of:
//   1. dim == 1            -> broadcast along this axis (stride = 0)
//   2. dim == out_shape[k] -> exact match (stride = running row-major stride)
//   3. otherwise           -> not broadcast-compatible, return false
//
// This mirrors the per-axis rule used by ORT's CUDA EP `ComputeOutputShape`
// in onnxruntime/core/providers/cuda/tensor/where.cc, where each input dim
// is rejected unless it equals the output dim or 1. We additionally enforce
// the rule here at runtime as defense-in-depth: even if the upstream ONNX
// shape inference and MLIR conversion produced a consistent `out_shape`,
// any mismatch surfaces a clear error instead of an out-of-bounds access.
static bool make_broadcast_layout(const int64_t *src_shape, int src_rank,
                                  int out_rank, const int64_t *out_shape,
                                  WhereDimsND *padded_shape,
                                  WhereDimsND *stride, const char *name) {
  int pad = out_rank - src_rank;
  for (int i = 0; i < pad; ++i)
    padded_shape->v[i] = 1;
  for (int i = 0; i < src_rank; ++i)
    padded_shape->v[pad + i] = src_shape[i];

  int64_t running = 1;
  for (int k = out_rank - 1; k >= 0; --k) {
    int64_t dim = padded_shape->v[k];
    if (dim == 1) {
      // NumPy rule (1): dim of 1 broadcasts to any out dim, including
      // out_shape[k] != 1. Stride 0 makes every output index along this
      // axis read the same source element.
      stride->v[k] = 0;
    } else if (dim == out_shape[k]) {
      // NumPy rule (2): non-1 dim must match the out dim exactly; use the
      // accumulated row-major stride.
      stride->v[k] = running;
    } else {
      // NumPy rule (3): dim is neither 1 nor equal to out_shape[k] -> the
      // operand cannot broadcast to the output along this axis. Reject
      // explicitly rather than silently producing wrong indices.
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_where: %s dim %d = %lld is "
              "not broadcast-compatible with out dim %lld\n",
              name, k, (long long)dim, (long long)out_shape[k]);
      return false;
    }
    running *= dim;
  }
  return true;
}

template <typename T>
static int launch_where(hipStream_t stream, const void *condition,
                        const void *x, const void *y, void *output,
                        int64_t total, int out_rank,
                        const WhereDimsND &out_shape,
                        const WhereDimsND &cond_stride,
                        const WhereDimsND &x_stride,
                        const WhereDimsND &y_stride) {
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(where_nd_kernel<T>, dim3(grid_size), dim3(kBlockSize), 0,
                     stream, static_cast<const bool *>(condition),
                     static_cast<const T *>(x), static_cast<const T *>(y),
                     static_cast<T *>(output), total, out_rank, out_shape,
                     cond_stride, x_stride, y_stride);
  return 0;
}

extern "C" int hip_elementwise_where(
    void *stream,
    const void *condition,
    const void *x,
    const void *y,
    void *output,
    const int64_t *cond_shape, int64_t cond_rank,
    const int64_t *x_shape,    int64_t x_rank,
    const int64_t *y_shape,    int64_t y_rank,
    const int64_t *out_shape,  int64_t out_rank,
    int hip_dtype) {
  // Rank validation. Note that `rank == 0` (a scalar tensor) is intentionally
  // accepted for every operand, matching the ONNX Where spec which permits
  // rank-0 tensors under NumPy-style multidirectional broadcasting (a scalar
  // broadcasts to any shape). Scalars are handled correctly by the rest of
  // this function through natural loop degeneracy:
  //   * When `out_rank == 0`, the element-count loop below leaves `total` at
  //     its initial value of 1, which is the correct element count for a
  //     rank-0 output tensor.
  //   * `make_broadcast_layout` with `src_rank == 0` skips both the padding
  //     and stride loops and returns trivially (all-zero strides), which
  //     mirrors the "broadcast scalar" semantics.
  //   * `where_nd_kernel` launches a single thread whose coordinate
  //     decomposition loop also doesn't execute, so all operand indices stay
  //     at 0 and the kernel produces `output[0] = condition[0] ? x[0] : y[0]`.
  // The shape pointers may legitimately be null when the corresponding rank
  // is 0; the upstream wrapper (`wrap_where`) already guarantees this.
  if (out_rank < 0 || out_rank > HIP_WHERE_MAX_RANK || cond_rank < 0 ||
      cond_rank > out_rank || x_rank < 0 || x_rank > out_rank || y_rank < 0 ||
      y_rank > out_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where: invalid ranks "
            "(max=%d): cond=%lld, x=%lld, y=%lld, out=%lld\n",
            HIP_WHERE_MAX_RANK, (long long)cond_rank, (long long)x_rank,
            (long long)y_rank, (long long)out_rank);
    return -1;
  }

  // Initial value of 1 is also the correct element count for the scalar
  // (out_rank == 0) case, where the loop body does not execute.
  int64_t total = 1;
  for (int i = 0; i < out_rank; ++i)
    total *= out_shape[i];
  if (total <= 0)
    return 0;

  WhereDimsND out_shape_nd = {};
  WhereDimsND cond_shape_nd = {}, x_shape_nd = {}, y_shape_nd = {};
  WhereDimsND cond_stride = {}, x_stride = {}, y_stride = {};
  for (int i = 0; i < out_rank; ++i)
    out_shape_nd.v[i] = out_shape[i];

  if (!make_broadcast_layout(cond_shape, (int)cond_rank, (int)out_rank,
                             out_shape, &cond_shape_nd, &cond_stride,
                             "condition"))
    return -1;
  if (!make_broadcast_layout(x_shape, (int)x_rank, (int)out_rank, out_shape,
                             &x_shape_nd, &x_stride, "x"))
    return -1;
  if (!make_broadcast_layout(y_shape, (int)y_rank, (int)out_rank, out_shape,
                             &y_shape_nd, &y_stride, "y"))
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_elementwise_where: dtype=%d, out_rank=%d, "
      "total=%lld\n",
      hip_dtype, (int)out_rank, (long long)total);

  int rc = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    rc = launch_where<float>(hip_stream, condition, x, y, output, total,
                             (int)out_rank, out_shape_nd, cond_stride, x_stride,
                             y_stride);
    break;
  case HIP_DTYPE_FLOAT16:
    rc = launch_where<__half>(hip_stream, condition, x, y, output, total,
                              (int)out_rank, out_shape_nd, cond_stride,
                              x_stride, y_stride);
    break;
  case HIP_DTYPE_BFLOAT16:
    rc = launch_where<__hip_bfloat16>(hip_stream, condition, x, y, output,
                                      total, (int)out_rank, out_shape_nd,
                                      cond_stride, x_stride, y_stride);
    break;
  case HIP_DTYPE_INT32:
    rc = launch_where<int32_t>(hip_stream, condition, x, y, output, total,
                               (int)out_rank, out_shape_nd, cond_stride,
                               x_stride, y_stride);
    break;
  case HIP_DTYPE_INT64:
    rc = launch_where<int64_t>(hip_stream, condition, x, y, output, total,
                               (int)out_rank, out_shape_nd, cond_stride,
                               x_stride, y_stride);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
