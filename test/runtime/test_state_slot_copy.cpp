/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int32_t kDescriptorCount = 96;
constexpr int32_t kSlotCount = 8;
constexpr size_t kGuardBytes = 32;

struct StateSlotDesc {
  void *base;
  uint64_t slot_bytes;
};

static_assert(sizeof(StateSlotDesc) == 16);

bool hip_ok(hipError_t error, const char *operation) {
  if (error == hipSuccess)
    return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return false;
}

uint8_t state_value(int descriptor, int slot, size_t offset) {
  return static_cast<uint8_t>((descriptor * 17 + slot * 29 + offset * 7) &
                              0xff);
}

bool run_promotion(hipStream_t stream,
                   const std::vector<StateSlotDesc> &host_descriptors,
                   StateSlotDesc *device_descriptors, int32_t source_slot,
                   int32_t destination_slot,
                   const std::vector<std::vector<uint8_t>> &before,
                   const std::vector<void *> &allocations) {
  if (!hip_ok(hipMemcpyAsync(device_descriptors, host_descriptors.data(),
                             host_descriptors.size() * sizeof(StateSlotDesc),
                             hipMemcpyHostToDevice, stream),
              "descriptor upload"))
    return false;

  const int launch_status =
      hip_state_slot_copy(stream, device_descriptors, kDescriptorCount,
                          source_slot, destination_slot);
  if (launch_status != static_cast<int>(hipSuccess)) {
    std::fprintf(stderr, "hip_state_slot_copy launch failed: %s\n",
                 hipGetErrorString(static_cast<hipError_t>(launch_status)));
    return false;
  }

  std::vector<std::vector<uint8_t>> after(kDescriptorCount);
  for (int i = 0; i < kDescriptorCount; ++i) {
    after[i].resize(before[i].size());
    if (!hip_ok(hipMemcpyAsync(after[i].data(), allocations[i], after[i].size(),
                               hipMemcpyDeviceToHost, stream),
                "state download"))
      return false;
  }
  // This is intentionally the only synchronization in the write -> promote
  // -> consume sequence, proving stream ordering without a kernel-side sync.
  if (!hip_ok(hipStreamSynchronize(stream), "final stream synchronization"))
    return false;

  for (int descriptor = 0; descriptor < kDescriptorCount; ++descriptor) {
    const size_t slot_bytes =
        static_cast<size_t>(host_descriptors[descriptor].slot_bytes);
    for (size_t byte = 0; byte < after[descriptor].size(); ++byte) {
      uint8_t expected = before[descriptor][byte];
      const size_t destination_begin =
          kGuardBytes + static_cast<size_t>(destination_slot) * slot_bytes;
      if (byte >= destination_begin && byte < destination_begin + slot_bytes) {
        const size_t within_slot = byte - destination_begin;
        expected =
            before[descriptor]
                  [kGuardBytes + static_cast<size_t>(source_slot) * slot_bytes +
                   within_slot];
      }
      if (after[descriptor][byte] != expected) {
        std::fprintf(stderr, "descriptor %d byte %zu: got %u, expected %u\n",
                     descriptor, byte,
                     static_cast<unsigned>(after[descriptor][byte]),
                     static_cast<unsigned>(expected));
        return false;
      }
    }
  }
  return true;
}

} // namespace

int main() {
  hipStream_t stream = nullptr;
  StateSlotDesc *device_descriptors = nullptr;
  std::vector<void *> allocations(kDescriptorCount, nullptr);

  if (!hip_ok(hipStreamCreate(&stream), "hipStreamCreate") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&device_descriptors),
                        kDescriptorCount * sizeof(StateSlotDesc)),
              "descriptor hipMalloc"))
    return 1;

  std::vector<StateSlotDesc> descriptors(kDescriptorCount);
  std::vector<std::vector<uint8_t>> initial(kDescriptorCount);
  bool success = true;
  for (int descriptor = 0; descriptor < kDescriptorCount; ++descriptor) {
    // Exercise both the vectorized and byte-copy paths.
    const size_t slot_bytes = descriptor % 2 == 0
                                  ? static_cast<size_t>(64 + descriptor * 16)
                                  : static_cast<size_t>(65 + descriptor * 13);
    const size_t allocation_bytes = 2 * kGuardBytes + kSlotCount * slot_bytes;
    initial[descriptor].assign(allocation_bytes, 0xa5);
    for (int slot = 0; slot < kSlotCount; ++slot) {
      for (size_t byte = 0; byte < slot_bytes; ++byte)
        initial[descriptor]
               [kGuardBytes + static_cast<size_t>(slot) * slot_bytes + byte] =
                   state_value(descriptor, slot, byte);
    }

    if (!hip_ok(hipMalloc(&allocations[descriptor], allocation_bytes),
                "state hipMalloc") ||
        !hip_ok(hipMemcpyAsync(allocations[descriptor],
                               initial[descriptor].data(), allocation_bytes,
                               hipMemcpyHostToDevice, stream),
                "state upload")) {
      success = false;
      break;
    }
    descriptors[descriptor] = {static_cast<uint8_t *>(allocations[descriptor]) +
                                   kGuardBytes,
                               static_cast<uint64_t>(slot_bytes)};
  }

  if (success)
    success = run_promotion(stream, descriptors, device_descriptors,
                            /*source_slot=*/2, /*destination_slot=*/7, initial,
                            allocations);

  if (success) {
    // Reinitialize and repeat with a different source slot to catch stale
    // descriptor or launch state.
    for (int descriptor = 0; descriptor < kDescriptorCount; ++descriptor)
      success &= hip_ok(hipMemcpyAsync(allocations[descriptor],
                                       initial[descriptor].data(),
                                       initial[descriptor].size(),
                                       hipMemcpyHostToDevice, stream),
                        "state reupload");
    if (success)
      success = run_promotion(stream, descriptors, device_descriptors,
                              /*source_slot=*/4, /*destination_slot=*/7,
                              initial, allocations);
  }

  if (hip_state_slot_copy(stream, nullptr, kDescriptorCount, 1, 7) !=
          static_cast<int>(hipErrorInvalidValue) ||
      hip_state_slot_copy(stream, device_descriptors, 0, 1, 7) !=
          static_cast<int>(hipErrorInvalidValue) ||
      hip_state_slot_copy(stream, device_descriptors, kDescriptorCount, 7, 7) !=
          static_cast<int>(hipErrorInvalidValue)) {
    std::fprintf(stderr, "invalid-argument checks failed\n");
    success = false;
  }

  for (void *allocation : allocations)
    if (allocation)
      (void)hipFree(allocation);
  if (device_descriptors)
    (void)hipFree(device_descriptors);
  if (stream)
    (void)hipStreamDestroy(stream);

  if (!success)
    return 1;
  std::puts("state-slot copy passed for 96 mixed descriptors");
  return 0;
}
