/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_TENSOR_BUFFER_INTERNAL_H
#define HIPDNN_EP_RUNTIME_TENSOR_BUFFER_INTERNAL_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct TensorBuffer {
  void *gpu_ptr = nullptr;
  void *host_ptr = nullptr;
  int64_t *shape_ptr = nullptr;
  size_t rank = 0;
  size_t size_bytes = 0;
  bool is_aliased = false;
};

static_assert(std::is_standard_layout_v<TensorBuffer>);
static_assert(std::is_trivially_destructible_v<TensorBuffer>);

#endif // HIPDNN_EP_RUNTIME_TENSOR_BUFFER_INTERNAL_H
