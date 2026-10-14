/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>

namespace {

constexpr int64_t kMaximumMemRefRank = 12;

struct RankedMemRefDescriptor {
  void *allocatedPtr;
  void *alignedPtr;
  int64_t offset;
  int64_t sizesAndStrides[];
};

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool validateReachableSpan(int64_t offset, const int64_t *sizes,
                           const int64_t *strides, int64_t rank,
                           int64_t &maximumOffset) {
  if (offset < 0) {
    return false;
  }
  maximumOffset = offset;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (sizes[dimension] <= 1) {
      continue;
    }
    if (strides[dimension] <= 0) {
      return false;
    }
    int64_t contribution = 0;
    if (!checkedMultiply(sizes[dimension] - 1, strides[dimension],
                         contribution) ||
        !checkedAdd(maximumOffset, contribution, maximumOffset)) {
      return false;
    }
  }
  return true;
}

bool hasNonOverlappingElements(const int64_t *sizes, const int64_t *strides,
                               int64_t rank) {
  std::array<int64_t, kMaximumMemRefRank> activeDimensions{};
  int64_t activeCount = 0;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (sizes[dimension] > 1) {
      activeDimensions[static_cast<size_t>(activeCount++)] = dimension;
    }
  }
  std::sort(
      activeDimensions.begin(), activeDimensions.begin() + activeCount,
      [&](int64_t lhs, int64_t rhs) { return strides[lhs] < strides[rhs]; });

  // This is a conservative non-overlap proof: after sorting active dimensions
  // by stride, each new step must start beyond the complete address span of
  // all lower-stride dimensions. It accepts dense permutations and padded
  // layouts while rejecting descriptors that could race destination writes.
  int64_t lowerSpan = 1;
  for (int64_t index = 0; index < activeCount; ++index) {
    const int64_t dimension = activeDimensions[static_cast<size_t>(index)];
    if (strides[dimension] < lowerSpan) {
      return false;
    }
    int64_t contribution = 0;
    if (!checkedMultiply(sizes[dimension] - 1, strides[dimension],
                         contribution) ||
        !checkedAdd(lowerSpan, contribution, lowerSpan)) {
      return false;
    }
  }
  return true;
}

int copyMemRefOnStream(hipStream_t stream, int64_t elemSize,
                       const HipdnnEpUnrankedMemRef *src,
                       const HipdnnEpUnrankedMemRef *dst,
                       const char *entryPoint) {
  if (!src || !dst || !src->descriptor || !dst->descriptor) {
    fprintf(stderr, "%s: null descriptor\n", entryPoint);
    return -1;
  }
  if (src->rank != dst->rank || src->rank < 0 ||
      src->rank > kMaximumMemRefRank) {
    fprintf(stderr, "%s: incompatible or unsupported rank\n", entryPoint);
    return -1;
  }
  if (elemSize <= 0) {
    fprintf(stderr, "%s: element size must be positive\n", entryPoint);
    return -1;
  }

  auto *source = static_cast<const RankedMemRefDescriptor *>(src->descriptor);
  auto *destination =
      static_cast<const RankedMemRefDescriptor *>(dst->descriptor);
  const int64_t rank = src->rank;
  const int64_t *srcSizes = source->sizesAndStrides;
  const int64_t *srcStrides = source->sizesAndStrides + rank;
  const int64_t *dstSizes = destination->sizesAndStrides;
  const int64_t *dstStrides = destination->sizesAndStrides + rank;

  bool empty = false;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (srcSizes[dimension] < 0 || dstSizes[dimension] < 0 ||
        srcSizes[dimension] != dstSizes[dimension]) {
      fprintf(stderr, "%s: source and destination shapes differ\n", entryPoint);
      return -1;
    }
    empty |= srcSizes[dimension] == 0;
  }
  // Empty memrefs touch no storage. Once both runtime shapes are known equal
  // and nonnegative, offsets, strides, pointers, and products are irrelevant.
  // Returning here also avoids overflowing on shapes such as [INT64_MAX,
  // INT64_MAX, 0].
  if (empty) {
    return 0;
  }

  int64_t totalElements = 1;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (!checkedMultiply(totalElements, srcSizes[dimension], totalElements)) {
      fprintf(stderr, "%s: element count overflows int64\n", entryPoint);
      return -1;
    }
  }

  int64_t srcMaximumOffset = 0;
  int64_t dstMaximumOffset = 0;
  if (!validateReachableSpan(source->offset, srcSizes, srcStrides, rank,
                             srcMaximumOffset) ||
      !validateReachableSpan(destination->offset, dstSizes, dstStrides, rank,
                             dstMaximumOffset)) {
    fprintf(stderr, "%s: invalid offset, stride, or address span\n",
            entryPoint);
    return -1;
  }
  if (!hasNonOverlappingElements(dstSizes, dstStrides, rank)) {
    fprintf(stderr, "%s: destination layout has overlapping elements\n",
            entryPoint);
    return -1;
  }
  int64_t totalBytes = 0;
  int64_t srcOffsetBytes = 0;
  int64_t dstOffsetBytes = 0;
  int64_t srcSpanElements = 0;
  int64_t dstSpanElements = 0;
  if (!checkedMultiply(totalElements, elemSize, totalBytes) ||
      !checkedMultiply(source->offset, elemSize, srcOffsetBytes) ||
      !checkedMultiply(destination->offset, elemSize, dstOffsetBytes) ||
      !checkedAdd(srcMaximumOffset, 1, srcSpanElements) ||
      !checkedAdd(dstMaximumOffset, 1, dstSpanElements) ||
      srcSpanElements > std::numeric_limits<int64_t>::max() / elemSize ||
      dstSpanElements > std::numeric_limits<int64_t>::max() / elemSize) {
    fprintf(stderr, "%s: byte range overflows int64\n", entryPoint);
    return -1;
  }
  if (!source->alignedPtr || !destination->alignedPtr) {
    fprintf(stderr, "%s: null data pointer\n", entryPoint);
    return -1;
  }

  const char *srcBase = static_cast<const char *>(source->alignedPtr) +
                        static_cast<size_t>(srcOffsetBytes);
  char *dstBase = static_cast<char *>(destination->alignedPtr) +
                  static_cast<size_t>(dstOffsetBytes);

  if (rank == 0) {
    hipError_t error = hipMemcpyAsync(dstBase, srcBase, elemSize,
                                      hipMemcpyDeviceToDevice, stream);
    if (error != hipSuccess) {
      fprintf(stderr, "%s: rank-0 copy failed: %s\n", entryPoint,
              hipGetErrorString(error));
      return -1;
    }
    return 0;
  }

  // Find the largest suffix that is contiguous on both sides. Strides on a
  // unit dimension are irrelevant because that coordinate can only be zero.
  int64_t rowStart = rank;
  int64_t rowElements = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    if (srcSizes[dimension] == 1) {
      rowStart = dimension;
      continue;
    }
    if (srcStrides[dimension] != rowElements ||
        dstStrides[dimension] != rowElements) {
      break;
    }
    if (!checkedMultiply(rowElements, srcSizes[dimension], rowElements)) {
      fprintf(stderr, "%s: contiguous row size overflows int64\n", entryPoint);
      return -1;
    }
    rowStart = dimension;
  }

  int64_t rowBytes = 0;
  if (!checkedMultiply(rowElements, elemSize, rowBytes)) {
    fprintf(stderr, "%s: contiguous row byte size overflows int64\n",
            entryPoint);
    return -1;
  }
  if (rowStart == 0) {
    hipError_t error =
        hipMemcpyAsync(dstBase, srcBase, static_cast<size_t>(totalBytes),
                       hipMemcpyDeviceToDevice, stream);
    if (error != hipSuccess) {
      fprintf(stderr, "%s: contiguous copy failed: %s\n", entryPoint,
              hipGetErrorString(error));
      return -1;
    }
    return 0;
  }

  const int64_t outerRank = rowStart;
  int64_t outerTotal = 1;
  for (int64_t dimension = 0; dimension < outerRank; ++dimension) {
    if (!checkedMultiply(outerTotal, srcSizes[dimension], outerTotal)) {
      fprintf(stderr, "%s: outer element count overflows int64\n", entryPoint);
      return -1;
    }
  }

  if (outerRank == 1) {
    if (srcStrides[0] < rowElements || dstStrides[0] < rowElements) {
      fprintf(stderr, "%s: pitch is shorter than the copied row\n", entryPoint);
      return -1;
    }
    int64_t srcPitch = 0;
    int64_t dstPitch = 0;
    if (!checkedMultiply(srcStrides[0], elemSize, srcPitch) ||
        !checkedMultiply(dstStrides[0], elemSize, dstPitch)) {
      fprintf(stderr, "%s: byte pitch overflows int64\n", entryPoint);
      return -1;
    }
    hipError_t error = hipMemcpy2DAsync(
        dstBase, static_cast<size_t>(dstPitch), srcBase,
        static_cast<size_t>(srcPitch), static_cast<size_t>(rowBytes),
        static_cast<size_t>(srcSizes[0]), hipMemcpyDeviceToDevice, stream);
    if (error != hipSuccess) {
      fprintf(stderr, "%s: pitched copy failed: %s\n", entryPoint,
              hipGetErrorString(error));
      return -1;
    }
    return 0;
  }

  int result =
      hip_strided_copy(static_cast<void *>(stream), dstBase, srcBase, elemSize,
                       static_cast<int>(outerRank), srcSizes, srcStrides,
                       dstStrides, rowElements, outerTotal);
  if (result == 0) {
    return 0;
  }

  // Preserve correctness for element widths not implemented by the kernel.
  // The validated span arithmetic above guarantees these row offsets cannot
  // overflow signed 64-bit arithmetic.
  int64_t indices[kMaximumMemRefRank] = {};
  for (int64_t row = 0; row < outerTotal; ++row) {
    int64_t srcElementOffset = 0;
    int64_t dstElementOffset = 0;
    for (int64_t dimension = 0; dimension < outerRank; ++dimension) {
      srcElementOffset += indices[dimension] * srcStrides[dimension];
      dstElementOffset += indices[dimension] * dstStrides[dimension];
    }
    hipError_t error = hipMemcpyAsync(
        dstBase + static_cast<size_t>(dstElementOffset * elemSize),
        srcBase + static_cast<size_t>(srcElementOffset * elemSize),
        static_cast<size_t>(rowBytes), hipMemcpyDeviceToDevice, stream);
    if (error != hipSuccess) {
      fprintf(stderr, "%s: strided-row copy failed: %s\n", entryPoint,
              hipGetErrorString(error));
      return -1;
    }
    for (int64_t dimension = outerRank - 1; dimension >= 0; --dimension) {
      if (++indices[dimension] < srcSizes[dimension]) {
        break;
      }
      indices[dimension] = 0;
    }
  }
  return 0;
}

} // namespace

int wrap_hipMemrefCopy(RuntimeState *state, int64_t elem_size,
                       const HipdnnEpUnrankedMemRef *src,
                       const HipdnnEpUnrankedMemRef *dst) {
  OP_PROFILE_CPU("memref_copy", state);
  if (!state) {
    fprintf(stderr, "wrap_hipMemrefCopy: null state\n");
    return -1;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  int result =
      copyMemRefOnStream(stream, elem_size, src, dst, "wrap_hipMemrefCopy");
  if (result != 0) {
    // The interface will stop using this invocation's buffers after observing
    // the error. Drain work queued before this copy before returning so an
    // immediate cleanup cannot race those earlier operations. The generated
    // interface must also drain its shared error-cleanup path to cover work
    // enqueued after this call before the accumulated status is returned.
    hipError_t syncError = hipStreamSynchronize(stream);
    if (syncError != hipSuccess) {
      fprintf(stderr,
              "wrap_hipMemrefCopy: failure-path stream sync failed: %s\n",
              hipGetErrorString(syncError));
    }
  }
  return result;
}

// GPU D2D memcpy via hipMemcpyAsync.
// Follows opaque RuntimeState pattern - extracts stream internally
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes) {
  OP_PROFILE_CPU_SHAPE(
      "memcpy_d2d", [&] { return std::to_string(size_bytes) + "B"; }, state);
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null state\n");
    return -1;
  }
  // An empty tensor is a null pointer with a zero byte count. Copying
  // nothing is a no-op; a null pointer with a positive size is not.
  if (size_bytes == 0)
    return 0;
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null pointer\n");
    return -1;
  }

  // Extract stream from opaque RuntimeState using accessor function
  // (Maintains abstraction barrier - no direct field access)
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err = hipMemcpyAsync(dst_ptr, src_ptr, size_bytes,
                                  hipMemcpyDeviceToDevice, stream);

  if (err != hipSuccess) {
    fprintf(stderr, "wrap_hipMemcpyAsync: copy failed (%zu bytes): %s\n",
            size_bytes, hipGetErrorString(err));
    return -1;
  }

  return 0;
}

int wrap_hipMemcpy2DAsync(RuntimeState *state, void *dst_ptr, size_t dst_pitch,
                          const void *src_ptr, size_t src_pitch, size_t width,
                          size_t height) {
  OP_PROFILE(
      "memcpy_2d",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "w=%zu h=%zu", width, height);
        return std::string(b);
      },
      state);
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null pointer\n");
    return -1;
  }
  if (width == 0 || height == 0) {
    return 0;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipMemcpy2DAsync: dpitch=%zu spitch=%zu "
                    "width=%zu height=%zu (D2D async) begin\n",
                    dst_pitch, src_pitch, width, height);

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err =
      hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height,
                       hipMemcpyDeviceToDevice, stream);

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipMemcpy2DAsync: end -> %d\n", (int)err);

  if (err != hipSuccess) {
    fprintf(stderr,
            "wrap_hipMemcpy2DAsync: copy failed (width=%zu height=%zu): %s\n",
            width, height, hipGetErrorString(err));
    return -1;
  }

  return 0;
}

// Parallel strided D2D copy via a single hip_strided_copy kernel launch,
// expressed in ELEMENT units (not bytes). This is the fast path for a pitched
// copy whose rows are very thin: hipMemcpy2DAsync processes each row as a
// separate tiny transfer on the copy engine, so a copy of `row_elems` elements
// per row over `height` rows degenerates into `height` micro-transfers (e.g.
// the sinusoidal position-embedding sin/cos interleave produces row_elems=1,
// height=40000 -> ~480ms of copy-engine stall). One kernel launch does the
// whole strided copy with one thread per element instead.
//
// Maps the 2D pitched geometry onto hip_strided_copy's rank-1 outer form:
//   outer_sizes = {height}, src/dst outer strides = {pitch_elems},
//   row_elems   = contiguous inner suffix length (stride 1 on both sides).
// On any kernel failure (unsupported element size, launch error) falls back to
// hipMemcpy2DAsync so correctness never depends on the kernel succeeding.
int wrap_strided_copy(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                      int64_t elem_bytes, int64_t height,
                      int64_t src_pitch_elems, int64_t dst_pitch_elems,
                      int64_t row_elems) {
  OP_PROFILE(
      "strided_copy",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "h=%lld row=%lld es=%lld", (long long)height,
                 (long long)row_elems, (long long)elem_bytes);
        return std::string(b);
      },
      state);
  if (!state) {
    fprintf(stderr, "wrap_strided_copy: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_strided_copy: null pointer\n");
    return -1;
  }
  if (height == 0 || row_elems == 0)
    return 0;

  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: elem=%lld height=%lld "
                    "spitch_e=%lld dpitch_e=%lld row_e=%lld begin\n",
                    (long long)elem_bytes, (long long)height,
                    (long long)src_pitch_elems, (long long)dst_pitch_elems,
                    (long long)row_elems);

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  const int64_t outer_sizes[1] = {height};
  const int64_t src_outer_strides[1] = {src_pitch_elems};
  const int64_t dst_outer_strides[1] = {dst_pitch_elems};
  int rc = hip_strided_copy(static_cast<void *>(stream), dst_ptr, src_ptr,
                            elem_bytes, /*outer_rank=*/1, outer_sizes,
                            src_outer_strides, dst_outer_strides, row_elems,
                            /*outer_total=*/height);

  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: end -> %d\n", rc);

  if (rc == 0)
    return 0;

  // Kernel could not handle this copy (e.g. odd element size) -- fall back to
  // the correct (if slower) pitched DMA path.
  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: kernel rc=%d, falling back to "
                    "hipMemcpy2DAsync\n",
                    rc);
  size_t width_bytes = static_cast<size_t>(row_elems) * elem_bytes;
  size_t src_pitch = static_cast<size_t>(src_pitch_elems) * elem_bytes;
  size_t dst_pitch = static_cast<size_t>(dst_pitch_elems) * elem_bytes;
  hipError_t err = hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch,
                                    width_bytes, static_cast<size_t>(height),
                                    hipMemcpyDeviceToDevice, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_strided_copy: fallback 2D copy failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  return 0;
}

// Synchronize the stream, then read back a device-resident i32 scalar (e.g.
// NonZero's non-zero count). The copy is enqueued after the producing kernel
// on the same stream; the synchronize guarantees both have completed before
// the host value is read. Returns 0 on any failure (a zero extent is a safe,
// inert dynamic dim).
int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar) {
  // CPU-only profiling: this call issues a hipStreamSynchronize, so its cost is
  // host-side wait time (GPU drain), invisible to GPU per-op events. Surfacing
  // it as a profiled row attributes the dynamic-shape readback sync overhead
  // (Range / Reshape(-1) / readback_dim) that otherwise lands in "outside
  // scopes".
  OP_PROFILE_CPU("readback_i32", state);
  if (!state || !device_scalar) {
    fprintf(stderr, "hipdnn_ep_readback_i32: null argument\n");
    return 0;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  int32_t host_val = 0;
  // hipMemcpyDefault (not DeviceToHost): the source may be host-accessible
  // memory (host-mapped scratch / UMA pool), where an explicit D2H fails
  // `invalid argument`. Direction is inferred from the pointer via UVA.
  hipError_t err = hipMemcpyAsync(&host_val, device_scalar, sizeof(int32_t),
                                  hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_i32: D2H copy failed: %s\n",
            hipGetErrorString(err));
    return 0;
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_i32: stream sync failed: %s\n",
            hipGetErrorString(err));
    return 0;
  }
  return host_val;
}

// Synchronize the stream, then copy a small device-resident scalar of arbitrary
// byte width (1/2/4/8) back into the caller-provided host buffer. Generalises
// hipdnn_ep_readback_i32 for scalars whose element type is not i32 — e.g. the
// i64 limit/start/delta and f32/f16 operands of a data-dependent `onnx.Range`,
// whose trip count must be computed on the host. The copy is enqueued after the
// producing kernel on the same stream; the synchronize guarantees the producing
// kernel has finished before the host reads. Without this, generated code that
// does a bare `memref.load` of a GPU-written scalar reads stale memory on
// targets where the pool is true device memory (it accidentally works where the
// pool is UMA-mapped host-accessible memory) — yielding a zero trip count and a
// collapsed dynamic dimension.
void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes) {
  // CPU-only profiling: like readback_i32 this issues a hipStreamSynchronize;
  // its cost is host-side GPU-drain time. Profiling it attributes the
  // Range/Expand/Pad/Loop shape-readback sync overhead.
  OP_PROFILE_CPU_SHAPE(
      "readback_scalar", [&] { return std::to_string(num_bytes) + "B"; },
      state);
  if (!state || !host_dst || !device_scalar || num_bytes <= 0) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: invalid argument\n");
    return;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  // hipMemcpyDefault (not DeviceToHost): the source may be host-accessible
  // memory (host-mapped scratch from the host-scalar materialization pass, or a
  // UMA pool), where an explicit D2H fails `invalid argument`. Direction is
  // inferred from the pointer via UVA. Mirrors hipdnn_ep_readback_i32.
  hipError_t err =
      hipMemcpyAsync(host_dst, device_scalar, static_cast<size_t>(num_bytes),
                     hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: D2H copy failed: %s\n",
            hipGetErrorString(err));
    return;
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: stream sync failed: %s\n",
            hipGetErrorString(err));
  }
}

int wrap_copy_d2h(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                  int64_t size_bytes) {
  OP_PROFILE_CPU_SHAPE(
      "copy_d2h", [&] { return std::to_string(size_bytes) + "B"; }, state);
  if (!state || !dst_ptr || !src_ptr || size_bytes < 0) {
    fprintf(stderr, "wrap_copy_d2h: invalid argument\n");
    return -1;
  }
  if (size_bytes == 0) {
    return 0;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  // hipMemcpyDefault, not DeviceToHost: the source can be host-accessible
  // memory such as host-mapped scratch or a UMA pool, which DeviceToHost
  // rejects with `invalid argument`.
  hipError_t err =
      hipMemcpyAsync(dst_ptr, src_ptr, static_cast<size_t>(size_bytes),
                     hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_copy_d2h: D2H copy failed (%lld bytes): %s\n",
            static_cast<long long>(size_bytes), hipGetErrorString(err));
    return -1;
  }
  return 0;
}
