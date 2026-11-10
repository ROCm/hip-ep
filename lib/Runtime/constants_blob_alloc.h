/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_CONSTANTS_BLOB_ALLOC_H
#define HIPDNN_EP_RUNTIME_CONSTANTS_BLOB_ALLOC_H

#include "runtime_types.h"

#include <cstddef>
#include <cstdio>

namespace hipdnn_ep {

// Allocates the device constants blob. Returns 0 and sets *blob on success.
//
// Only device exhaustion is recoverable: a blob larger than physical VRAM can
// still live in managed (paged) memory. Any other hipMalloc failure (invalid
// device, context error, ...) would fail the managed allocation too, so it is
// reported instead of masked. Constants that are only gathered along axis 0
// avoid this path entirely via `disk_resident`.
inline int alloc_constants_blob(void **blob, size_t total_size) {
  const hipError_t alloc_err = hipMalloc(blob, total_size);
  if (alloc_err == hipSuccess)
    return 0;
  (void)hipGetLastError();
  if (alloc_err != hipErrorOutOfMemory) {
    fprintf(stderr, "hipMalloc failed for constants blob (%zu bytes): %s\n",
            total_size, hipGetErrorString(alloc_err));
    return 1;
  }
  const hipError_t managed_err =
      hipMallocManaged(blob, total_size, hipMemAttachGlobal);
  if (managed_err != hipSuccess) {
    (void)hipGetLastError();
    fprintf(stderr,
            "hipMalloc and hipMallocManaged failed for constants blob "
            "(%zu bytes): %s\n",
            total_size, hipGetErrorString(managed_err));
    return 1;
  }
  fprintf(stderr,
          "hipMalloc ran out of device memory; constants blob uses managed "
          "memory (%zu bytes)\n",
          total_size);
  return 0;
}

} // namespace hipdnn_ep

#endif // HIPDNN_EP_RUNTIME_CONSTANTS_BLOB_ALLOC_H
