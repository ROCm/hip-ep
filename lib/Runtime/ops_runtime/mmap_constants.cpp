/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- mmap_constants.cpp - Load externalized constants from sidecar ------===//
//
// Provides the extern "C" functions for loading externalized model constants
// from a sidecar .constants.bin file into GPU-accessible memory:
//
//   hip_load_constants()    -- mmap file, allocate device memory, copy
//   hip_unload_constants()  -- free device memory
//
// Called from compiled MLIR (or host code) to populate the constants buffer
// that --hip-resolve-extern-constants wires into the computation graph.
//
// On APUs with unified memory (gfx1150/1151), hipMalloc allocates in unified
// address space, so no explicit host-to-device copy is needed beyond the
// initial hipMemcpy (the driver handles page migration).
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

extern "C" void *hip_load_constants(void * /*handle*/, const char *bin_path) {
  int fd = open(bin_path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "[hip] hip_load_constants: failed to open '%s': %s\n",
            bin_path, strerror(errno));
    return nullptr;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    fprintf(stderr, "[hip] hip_load_constants: fstat failed: %s\n",
            strerror(errno));
    close(fd);
    return nullptr;
  }
  size_t fileSize = static_cast<size_t>(st.st_size);

  void *mapped =
      mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, /*offset=*/0);
  close(fd);
  if (mapped == MAP_FAILED) {
    fprintf(stderr, "[hip] hip_load_constants: mmap failed: %s\n",
            strerror(errno));
    return nullptr;
  }

  void *devicePtr = nullptr;
  hipError_t err = hipMalloc(&devicePtr, fileSize);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[hip] hip_load_constants: hipMalloc(%zu bytes) failed: %s\n",
            fileSize, hipGetErrorString(err));
    munmap(mapped, fileSize);
    return nullptr;
  }

  err = hipMemcpy(devicePtr, mapped, fileSize, hipMemcpyHostToDevice);
  munmap(mapped, fileSize);
  if (err != hipSuccess) {
    fprintf(stderr, "[hip] hip_load_constants: hipMemcpy failed: %s\n",
            hipGetErrorString(err));
    hipFree(devicePtr);
    return nullptr;
  }

  fprintf(stderr, "[hip] loaded %zu bytes of constants from '%s' -> %p\n",
          fileSize, bin_path, devicePtr);
  return devicePtr;
}

extern "C" void hip_unload_constants(void * /*handle*/, void *constants_ptr) {
  if (constants_ptr) {
    fprintf(stderr, "[hip] unloading constants %p\n", constants_ptr);
    hipFree(constants_ptr);
  }
}
