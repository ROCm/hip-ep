/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Hardware abstraction layer for the memory manager.
 */

#pragma once

#include <cstddef>

#include "mm/mm_types.h"

namespace mm {
namespace detail {

class Hal {
public:
  virtual ~Hal() = default;
  virtual Status set_device(int device_id) = 0;
  virtual Status malloc(void **ptr, std::size_t size) = 0;
  virtual Status free(void *ptr) = 0;
  virtual Status host_mapped_malloc(void **ptr, std::size_t size) = 0;
  virtual Status host_mapped_free(void *ptr) = 0;
  virtual Status memset_async(void *ptr, int value, std::size_t size,
                              stream_t stream) = 0;
};

Hal *hal_rocm();

} // namespace detail
} // namespace mm
