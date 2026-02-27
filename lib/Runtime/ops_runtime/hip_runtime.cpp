/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hip_runtime.cpp - Handle lifecycle & device memory
//------------------===//
//
// Provides the extern "C" functions that every MLIR-compiled HIP dialect
// module calls for handle management and device memory allocation:
//
//   hipCreateHandle()      <- hip.create_handle
//   hipDestroyHandle()     <- hip.destroy_handle
//   hip_device_malloc()    <- hip.alloc
//   hip_device_free()      <- hip.free
//
// Link this file alongside any per-op runtime file (e.g. hipblaslt_matmul.cpp).
//
// Compile:
//   cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include
//      hip_runtime.cpp
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>

extern "C" void *hipCreateHandle() {
  fprintf(stderr, "[hip] create_handle\n");
  return nullptr;
}

extern "C" void hipDestroyHandle(void *) {
  fprintf(stderr, "[hip] destroy_handle\n");
}

extern "C" void *hip_device_malloc(int64_t sizeBytes) {
  void *ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, (size_t)sizeBytes);
  if (err != hipSuccess) {
    fprintf(stderr, "[hip] hipMalloc FAILED (%lld bytes): %s\n",
            (long long)sizeBytes, hipGetErrorString(err));
    return nullptr;
  }
  hipMemset(ptr, 0, (size_t)sizeBytes);
  fprintf(stderr, "[hip] alloc %lld bytes -> %p\n", (long long)sizeBytes, ptr);
  return ptr;
}

extern "C" void hip_device_free(void *ptr) {
  fprintf(stderr, "[hip] free %p\n", ptr);
  if (ptr)
    hipFree(ptr);
}
