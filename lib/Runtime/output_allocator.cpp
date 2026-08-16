/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Output allocator runtime contract.
//
// Two functions connect the EP and the generated model.dll:
//   * hipdnn_ep_set_output_allocator - the EP calls this (it is EXPORTED from
//     the model.dll) to set the allocator on RuntimeState before
//     inference_compute.
//   * hipdnn_ep_alloc_output - the generated main_graph calls this (it is what
//     hip.alloc_output lowers to). It forwards to the set callback.
//
// Kept in its own .cpp file (not hipdnn_ep_runtime_state.cpp) so the GPU-free
// unit test can build it against the mock runtime types - nothing here touches
// HIP/MIOpen/hipBLASLt. See test/runtime/.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

bool checkedMul(size_t lhs, size_t rhs, size_t &result) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool logicalBytes(const int64_t *shape, int64_t rank, int64_t elem_size,
                  size_t &bytes) {
  if (rank < 0 || elem_size <= 0 || (rank > 0 && !shape))
    return false;
  bytes = static_cast<size_t>(elem_size);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (shape[dim] < 0 ||
        !checkedMul(bytes, static_cast<size_t>(shape[dim]), bytes))
      return false;
  }
  return true;
}

int recordOutputFailure(RuntimeState *state) {
  if (state)
    (void)hipdnn_ep_state_set_error_flag(state);
  return -1;
}

} // namespace

// EP -> model.dll. Exported so the EP can GetProcAddress it (see
// HIPDNN_EP_RT_EXPORT in hipdnn_ep_runtime.h for why the attribute is on both
// the decl and this def, and why it is also listed in export_symbols).
extern "C" HIPDNN_EP_RT_EXPORT void
hipdnn_ep_set_output_allocator(RuntimeState *state,
                               const hipdnn_output_allocator_t *allocator) {
  if (!state)
    return;
  // A null allocator clears the slot ("none set"). The struct layout is a fixed
  // ABI contract (see hipdnn_ep_runtime.h), so a plain copy works on both sides
  // of the model.dll <-> EP boundary.
  if (!allocator) {
    state->output_allocator.self = nullptr;
    state->output_allocator.allocate = nullptr;
    return;
  }
  state->output_allocator = *allocator;
}

// generated main_graph -> runtime. Internal (not exported): the generated code
// calls it directly inside the DLL. Forwards to the EP callback; returns null
// (and logs) when no allocator is set - which only happens if the generated
// code asks for an output buffer but the EP never set a callback.
extern "C" void *hipdnn_ep_alloc_output(RuntimeState *state, int64_t out_idx,
                                        const int64_t *shape, int64_t rank,
                                        int64_t elem_size) {
  size_t bytes = 0;
  if (!state)
    return nullptr;
  if (!logicalBytes(shape, rank, elem_size, bytes)) {
    (void)recordOutputFailure(state);
    return nullptr;
  }
  if (!state->output_allocator.allocate) {
    fprintf(stderr,
            "hipdnn_ep_alloc_output: no output allocator installed "
            "(out_idx=%lld)\n",
            (long long)out_idx);
    (void)recordOutputFailure(state);
    return nullptr;
  }
  void *result = state->output_allocator.allocate(
      state->output_allocator.self, out_idx, shape, rank, elem_size);
  if (!result && bytes != 0)
    (void)recordOutputFailure(state);
  return result;
}

extern "C" int hipdnn_ep_copy_output(RuntimeState *state, void *dst,
                                     const void *src, int64_t rank,
                                     const int64_t *sizes,
                                     const int64_t *src_strides,
                                     int64_t elem_size) {
  size_t totalBytes = 0;
  if (!state || !logicalBytes(sizes, rank, elem_size, totalBytes) ||
      (rank > 0 && !src_strides))
    return recordOutputFailure(state);
  if (totalBytes == 0)
    return 0;
  if (!dst || !src)
    return recordOutputFailure(state);

  size_t contiguousElems = 1;
  int64_t lastIndexedDim = rank - 1;
  if (rank > 0 && src_strides[rank - 1] == 1) {
    contiguousElems = static_cast<size_t>(sizes[rank - 1]);
    lastIndexedDim = rank - 2;
  }
  size_t rowBytes = 0;
  if (!checkedMul(contiguousElems, static_cast<size_t>(elem_size), rowBytes))
    return recordOutputFailure(state);
  if (rowBytes == 0)
    return 0;
  size_t rows = totalBytes / rowBytes;
  auto *dstBytes = static_cast<unsigned char *>(dst);
  auto *srcBytes = static_cast<const unsigned char *>(src);
  for (size_t row = 0; row < rows; ++row) {
    size_t remainder = row;
    size_t srcOffsetElems = 0;
    for (int64_t dim = lastIndexedDim; dim >= 0; --dim) {
      size_t extent = static_cast<size_t>(sizes[dim]);
      size_t coordinate = extent == 0 ? 0 : remainder % extent;
      remainder = extent == 0 ? 0 : remainder / extent;
      if (src_strides[dim] < 0)
        return recordOutputFailure(state);
      size_t contribution = 0;
      if (!checkedMul(coordinate, static_cast<size_t>(src_strides[dim]),
                      contribution) ||
          contribution > std::numeric_limits<size_t>::max() - srcOffsetElems)
        return recordOutputFailure(state);
      srcOffsetElems += contribution;
    }
    size_t srcOffsetBytes = 0;
    size_t dstOffsetBytes = 0;
    if (!checkedMul(srcOffsetElems, static_cast<size_t>(elem_size),
                    srcOffsetBytes) ||
        !checkedMul(row, rowBytes, dstOffsetBytes))
      return recordOutputFailure(state);
    if (wrap_hipMemcpyAsync(state, dstBytes + dstOffsetBytes,
                            srcBytes + srcOffsetBytes, rowBytes) != 0)
      return recordOutputFailure(state);
  }
  return 0;
}
