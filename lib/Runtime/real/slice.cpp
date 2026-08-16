/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "hip/Support/SliceUtils.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <limits>
#include <mutex>
#include <string>

namespace {

int toHipDtype(int64_t dataType) {
  switch (dataType) {
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

} // namespace

int wrap_slice(RuntimeState *state, void *data, void *output,
               const int64_t *data_shape, const int64_t *output_shape,
               const int64_t *starts, const int64_t *steps,
               const int64_t *extents, int64_t rank, int64_t data_type,
               bool params_valid) {
  auto fail = [&]() {
    if (state)
      (void)hipdnn_ep_state_set_error_flag(state);
    return -1;
  };

  int hipDtype = toHipDtype(data_type);
  int64_t elementBytes = hipdnn_ep_datatype_size(data_type);
  if (hipDtype < 0 || elementBytes <= 0) {
    fprintf(stderr, "[REAL] wrap_slice: unsupported data type %s(%lld)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return fail();
  }
  if (rank > std::numeric_limits<int>::max()) {
    fprintf(stderr, "[REAL] wrap_slice: rank exceeds kernel ABI\n");
    return fail();
  }

  hipdnn_ep::slice::PreflightResult checked = hipdnn_ep::slice::preflight(
      state, data, output, data_shape, output_shape, starts, steps, extents,
      rank, elementBytes, params_valid);
  if (checked.status == hipdnn_ep::slice::PreflightStatus::Error) {
    fprintf(stderr,
            "[REAL] wrap_slice: invalid normalized shape/address contract\n");
    return fail();
  }
  if (checked.status == hipdnn_ep::slice::PreflightStatus::EmptyOutput)
    return 0;

  if (!state->slice_metadata_mutex) {
    fprintf(stderr, "[REAL] wrap_slice: metadata mutex is unavailable\n");
    return fail();
  }
  std::lock_guard<std::mutex> metadataLock(*state->slice_metadata_mutex);

  if (static_cast<uint64_t>(rank) >
      std::numeric_limits<size_t>::max() / (4 * sizeof(int64_t))) {
    fprintf(stderr, "[REAL] wrap_slice: metadata byte count overflow\n");
    return fail();
  }
  size_t metadataBytes = static_cast<size_t>(rank) * 4 * sizeof(int64_t);
  if (state->slice_metadata_scratch_size < metadataBytes) {
    // The lock is held from scratch growth through both enqueues. If call A
    // unlocks after enqueueing H2D(A), kernel(A), call B can only enqueue
    // H2D(B) afterward on the same state stream. Stream order therefore makes
    // kernel(A) consume the metadata before H2D(B) overwrites the scratch; no
    // completion wait is needed at unlock. A grow is the exceptional case:
    // drain the stream before freeing storage that an earlier kernel may use.
    if (state->slice_metadata_scratch) {
      hipError_t syncError = hipStreamSynchronize(state->stream);
      if (syncError != hipSuccess) {
        fprintf(stderr,
                "[REAL] wrap_slice: metadata grow stream sync failed: %s\n",
                hipGetErrorString(syncError));
        return fail();
      }
      hipError_t freeError = hipFree(state->slice_metadata_scratch);
      if (freeError != hipSuccess) {
        fprintf(stderr, "[REAL] wrap_slice: metadata scratch free failed: %s\n",
                hipGetErrorString(freeError));
        return fail();
      }
      state->slice_metadata_scratch = nullptr;
      state->slice_metadata_scratch_size = 0;
    }
    hipError_t allocationError =
        hipMalloc(&state->slice_metadata_scratch, metadataBytes);
    if (allocationError != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_slice: metadata scratch allocation failed: %s\n",
              hipGetErrorString(allocationError));
      state->slice_metadata_scratch = nullptr;
      return fail();
    }
    state->slice_metadata_scratch_size = metadataBytes;
  }

  OP_PROFILE(
      "slice",
      [&] {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "r%lld:%s", (long long)rank,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buffer);
      },
      state);

  void *stream = hipdnn_ep_state_get_stream(state);
  void *scratch = state->slice_metadata_scratch;
  int status =
      hip_slice(stream, data, output, data_shape, output_shape, starts, steps,
                static_cast<int>(rank), hipDtype, scratch, metadataBytes);
  if (status != 0)
    return fail();
  RUNTIME_DEBUG_LOG("[REAL] wrap_slice: rank=%lld dtype=%s\n", (long long)rank,
                    hipdnn_ep_datatype_name(data_type));
  return 0;
}
