/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "tensor_buffer_internal.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

int failures = 0;
TensorBuffer *released[3] = {};
size_t releasedCount = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

} // namespace

extern "C" void hipdnn_ep_tensor_free_input(RuntimeState *,
                                            TensorBuffer *buffer) {
  released[releasedCount++] = buffer;
}

int main() {
  const int64_t storageWords = hipdnn_ep_tensor_buffer_storage_words();
  const int64_t expectedWords = static_cast<int64_t>(
      (sizeof(TensorBuffer) + sizeof(int64_t) - 1) / sizeof(int64_t));
  check(storageWords == expectedWords,
        "runtime storage words must cover TensorBuffer");

  auto storage = std::make_unique<int64_t[]>(static_cast<size_t>(storageWords));
  std::memset(storage.get(), 0xa5,
              static_cast<size_t>(storageWords) * sizeof(int64_t));
  hipdnn_ep_tensor_buffer_construct(storage.get());

  auto *buffer = reinterpret_cast<TensorBuffer *>(storage.get());
  check(buffer->gpu_ptr == nullptr, "gpu_ptr must start null");
  check(buffer->host_ptr == nullptr, "host_ptr must start null");
  check(buffer->shape_ptr == nullptr, "shape_ptr must start null");
  check(buffer->rank == 0, "rank must start zero");
  check(buffer->size_bytes == 0, "size_bytes must start zero");
  check(!buffer->is_aliased, "constructed buffer must own no alias");

  TensorBuffer prefix[3] = {};
  TensorBuffer *prefixPtrs[] = {&prefix[0], &prefix[1], &prefix[2]};
  for (size_t count = 0; count <= 3; ++count) {
    releasedCount = 0;
    hipdnn_ep_tensor_free_inputs(nullptr, prefixPtrs, count);
    check(releasedCount == count, "batch cleanup must release exact prefix");
    for (size_t i = 0; i < count; ++i)
      check(released[i] == prefixPtrs[i],
            "batch cleanup must preserve prefix order");
  }

  releasedCount = 0;
  hipdnn_ep_tensor_free_inputs(nullptr, nullptr, 0);
  check(releasedCount == 0, "empty cleanup must not dereference its array");

  return failures == 0 ? 0 : 1;
}
