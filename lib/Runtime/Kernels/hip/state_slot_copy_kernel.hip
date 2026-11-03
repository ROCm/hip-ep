/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>

#include <cstdint>

#include "hip_custom_kernels.h"

namespace {

struct StateSlotDesc {
  uint8_t *base;
  uint64_t slot_bytes;
};

static_assert(sizeof(StateSlotDesc) == 16);
static_assert(alignof(StateSlotDesc) == alignof(uint64_t));

__global__ void state_slot_copy_kernel(const StateSlotDesc *descriptors,
                                       int32_t source_slot,
                                       int32_t destination_slot) {
  const StateSlotDesc descriptor = descriptors[blockIdx.x];
  const uint64_t bytes = descriptor.slot_bytes;
  const uint64_t source_offset =
      static_cast<uint64_t>(source_slot) * bytes;
  const uint64_t destination_offset =
      static_cast<uint64_t>(destination_slot) * bytes;
  const uint8_t *source = descriptor.base + source_offset;
  uint8_t *destination = descriptor.base + destination_offset;

  // State tensors and their slots are normally at least 16-byte aligned.
  // Vectorize that common path while retaining a byte-safe fallback for any
  // future odd-sized state tensor.
  if (((reinterpret_cast<uintptr_t>(source) |
        reinterpret_cast<uintptr_t>(destination) | bytes) &
       0xfu) == 0) {
    const auto *source_vectors =
        reinterpret_cast<const uint4 *>(source);
    auto *destination_vectors = reinterpret_cast<uint4 *>(destination);
    const uint64_t vector_count = bytes / sizeof(uint4);
    for (uint64_t i = threadIdx.x; i < vector_count; i += blockDim.x)
      destination_vectors[i] = source_vectors[i];
    return;
  }

  for (uint64_t i = threadIdx.x; i < bytes; i += blockDim.x)
    destination[i] = source[i];
}

} // namespace

extern "C" int hip_state_slot_copy(void *stream,
                                    const void *descriptors_device,
                                    int32_t count, int32_t source_slot,
                                    int32_t destination_slot) {
  if (!descriptors_device || count <= 0 || source_slot < 0 ||
      destination_slot < 0 || source_slot == destination_slot)
    return static_cast<int>(hipErrorInvalidValue);

  constexpr int kBlockSize = 256;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  hipLaunchKernelGGL(
      state_slot_copy_kernel, dim3(static_cast<uint32_t>(count)),
      dim3(kBlockSize), 0, hip_stream,
      static_cast<const StateSlotDesc *>(descriptors_device), source_slot,
      destination_slot);
  return static_cast<int>(hipGetLastError());
}
