/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "tensor_buffer_internal.h"

#include <new>

extern "C" {

int64_t hipdnn_ep_tensor_buffer_storage_words(void) {
  static_assert(alignof(TensorBuffer) <= alignof(int64_t));
  constexpr size_t wordSize = sizeof(int64_t);
  return static_cast<int64_t>((sizeof(TensorBuffer) + wordSize - 1) / wordSize);
}

void hipdnn_ep_tensor_buffer_construct(void *storage) {
  if (storage)
    ::new (storage) TensorBuffer{};
}

void hipdnn_ep_tensor_free_inputs(RuntimeState *state, TensorBuffer **buffers,
                                  size_t count) {
  if (!buffers && count != 0)
    return;
  for (size_t i = 0; i < count; ++i)
    hipdnn_ep_tensor_free_input(state, buffers[i]);
}

} // extern "C"
