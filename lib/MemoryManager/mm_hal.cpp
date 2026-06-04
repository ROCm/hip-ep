/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_hal.h"

#include <hip/hip_runtime.h>

namespace mm {
namespace detail {
namespace {

Status hip_error_to_status(hipError_t err) {
  if (err == hipSuccess)
    return Status::Ok;
  switch (err) {
  case hipErrorInvalidDevice:
    return Status::ErrInvalidArgument;
  case hipErrorOutOfMemory:
    return Status::ErrOutOfMemory;
  default:
    return Status::ErrHalFailure;
  }
}

class RocmHal final : public Hal {
public:
  Status set_device(int device_id) override {
    return hip_error_to_status(hipSetDevice(device_id));
  }

  Status malloc(void **ptr, std::size_t size) override {
    return hip_error_to_status(hipMalloc(ptr, size));
  }

  Status free(void *ptr) override { return hip_error_to_status(hipFree(ptr)); }

  Status host_mapped_malloc(void **ptr, std::size_t size) override {
    return hip_error_to_status(hipHostMalloc(ptr, size, hipHostMallocMapped));
  }

  Status host_mapped_free(void *ptr) override {
    return hip_error_to_status(hipHostFree(ptr));
  }

  Status memset_async(void *ptr, int value, std::size_t size,
                      stream_t stream) override {
    return hip_error_to_status(
        hipMemsetAsync(ptr, value, size, static_cast<hipStream_t>(stream)));
  }
};

RocmHal g_hal;

} // namespace

Hal *hal_rocm() { return &g_hal; }

} // namespace detail
} // namespace mm
