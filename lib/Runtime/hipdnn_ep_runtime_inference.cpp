/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_errors.h"
#include "hipdnn_ep_runtime.h"

#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#define HIPDNN_STACK_ALLOC(size) _alloca(size)
#else
#include <alloca.h>
#define HIPDNN_STACK_ALLOC(size) alloca(size)
#endif

namespace {

// Packed ranked memref descriptor. Same layout as
// mlir::StridedMemRefType<T, Rank> and as MemRefDescriptor::unpack:
//   {allocated, aligned, offset, sizes[rank], strides[rank]}
// T does not appear in the bytes: LLVM pointers are opaque. Strides are
// row-major from the runtime shape.
//
// Example, rank 2 with shape [2, 4]:
//   ptr, ptr, 0, [2, 4], [4, 1]
size_t packedMemRefBytes(int64_t rank) {
  return 2 * sizeof(void *) +
         (1 + 2 * static_cast<size_t>(rank)) * sizeof(int64_t);
}

void writePackedMemRef(char *storage, void *gpuPtr, const int64_t *shape,
                       int64_t rank) {
  auto **pointers = reinterpret_cast<void **>(storage);
  pointers[0] = gpuPtr;
  pointers[1] = gpuPtr;
  auto *integers = reinterpret_cast<int64_t *>(storage + 2 * sizeof(void *));
  integers[0] = 0;
  for (int64_t dim = 0; dim < rank; ++dim) {
    integers[1 + dim] = shape[dim];
  }
  int64_t stride = 1;
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    integers[1 + rank + dim] = stride;
    stride *= shape[dim];
  }
}

void freePreparedInputs(RuntimeState *state, TensorBuffer *buffers,
                        size_t count) {
  for (size_t index = 0; index < count; ++index) {
    hipdnn_ep_tensor_free_input(state, &buffers[index]);
  }
}

} // namespace

extern "C" int hipdnn_ep_inference_init(RuntimeState **out_state, void *fs,
                                        const void *metadata_blob,
                                        size_t blob_size, const void *config,
                                        HipdnnEpOpStatesInitFn op_states_init) {
  if (!out_state) {
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  int status = HIPDNN_EP_SUCCESS;
  if ((status = hipdnn_ep_state_init_with_fs(out_state, fs, metadata_blob,
                                             blob_size, config)) !=
      HIPDNN_EP_SUCCESS) {
    return status;
  }
  if (op_states_init &&
      (status = op_states_init(*out_state)) != HIPDNN_EP_SUCCESS) {
    goto cleanup;
  }
  return status;

cleanup:
  hipdnn_ep_state_cleanup(*out_state);
  *out_state = nullptr;
  return status;
}

extern "C" int hipdnn_ep_inference_compute(RuntimeState *state, span_t *inputs,
                                           const int64_t *ranks,
                                           size_t input_count,
                                           HipdnnEpMainGraphFn main_graph_fn) {
  if (!state || !main_graph_fn) {
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (input_count != 0 && (!inputs || !ranks)) {
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  hipdnn_ep_runtime_begin_compute(state);

  TensorBuffer *buffers = nullptr;
  void **descriptors = nullptr;
  size_t prepared = 0;
  int status = HIPDNN_EP_SUCCESS;

  if (input_count != 0) {
    buffers = static_cast<TensorBuffer *>(
        HIPDNN_STACK_ALLOC(input_count * sizeof(TensorBuffer)));
    std::memset(buffers, 0, input_count * sizeof(TensorBuffer));

    size_t packedBytes = 0;
    for (size_t index = 0; index < input_count; ++index) {
      if (ranks[index] < 0) {
        status = HIPDNN_EP_ERR_INVALID_DIMENSION;
        goto cleanup;
      }
      packedBytes += packedMemRefBytes(ranks[index]);
    }

    descriptors =
        static_cast<void **>(HIPDNN_STACK_ALLOC(input_count * sizeof(void *)));
    auto *packed = static_cast<char *>(HIPDNN_STACK_ALLOC(packedBytes));

    size_t packedOffset = 0;
    for (size_t index = 0; index < input_count; ++index) {
      if ((status = hipdnn_ep_tensor_prepare_input(
               state, inputs, index, static_cast<size_t>(ranks[index]),
               &buffers[index])) != HIPDNN_EP_SUCCESS) {
        goto cleanup;
      }
      ++prepared;

      void *gpuPtr = hipdnn_ep_tensor_buffer_get_gpu_ptr(&buffers[index]);
      int64_t *shape = hipdnn_ep_tensor_buffer_get_shape_ptr(&buffers[index]);
      if (ranks[index] > 0 && !shape) {
        status = HIPDNN_EP_ERR_NULL_POINTER;
        goto cleanup;
      }
      writePackedMemRef(packed + packedOffset, gpuPtr, shape, ranks[index]);
      descriptors[index] = packed + packedOffset;
      packedOffset += packedMemRefBytes(ranks[index]);
    }
  }

  if ((status = hipdnn_ep_state_reset_error_flag(state)) != HIPDNN_EP_SUCCESS) {
    goto cleanup;
  }
  if ((status = main_graph_fn(state, descriptors)) != HIPDNN_EP_SUCCESS) {
    goto cleanup;
  }
  if ((status = hipdnn_ep_stream_sync(state)) != HIPDNN_EP_SUCCESS) {
    goto cleanup;
  }
  status = hipdnn_ep_state_read_and_clear_error_flag(state);

cleanup:
  freePreparedInputs(state, buffers, prepared);
  return status;
}
