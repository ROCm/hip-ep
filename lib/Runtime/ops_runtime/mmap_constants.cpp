/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- mmap_constants.cpp - Load externalized constants from sidecar ------===//
//
// Provides the extern "C" functions for loading externalized model constants
// from a sidecar .constants.bin file into GPU-accessible memory:
//
//   hip_load_constants()    -- read file, allocate device memory, copy
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
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime_api.h>

extern "C" void *hip_load_constants(void * /*handle*/, const char *bin_path) {
  FILE *fp = fopen(bin_path, "rb");
  if (!fp) {
    fprintf(stderr, "[hip] hip_load_constants: failed to open '%s': %s\n",
            bin_path, strerror(errno));
    return nullptr;
  }

  if (fseek(fp, 0, SEEK_END) != 0) {
    fprintf(stderr, "[hip] hip_load_constants: fseek failed: %s\n",
            strerror(errno));
    fclose(fp);
    return nullptr;
  }
  long fileSizeLong = ftell(fp);
  if (fileSizeLong < 0) {
    fprintf(stderr, "[hip] hip_load_constants: ftell failed: %s\n",
            strerror(errno));
    fclose(fp);
    return nullptr;
  }
  rewind(fp);
  size_t fileSize = static_cast<size_t>(fileSizeLong);

  void *hostBuf = malloc(fileSize);
  if (!hostBuf) {
    fprintf(stderr,
            "[hip] hip_load_constants: malloc(%zu bytes) failed\n", fileSize);
    fclose(fp);
    return nullptr;
  }

  size_t bytesRead = fread(hostBuf, 1, fileSize, fp);
  fclose(fp);
  if (bytesRead != fileSize) {
    fprintf(stderr,
            "[hip] hip_load_constants: short read (%zu of %zu bytes)\n",
            bytesRead, fileSize);
    free(hostBuf);
    return nullptr;
  }

  void *devicePtr = nullptr;
  hipError_t err = hipMalloc(&devicePtr, fileSize);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[hip] hip_load_constants: hipMalloc(%zu bytes) failed: %s\n",
            fileSize, hipGetErrorString(err));
    free(hostBuf);
    return nullptr;
  }

  err = hipMemcpy(devicePtr, hostBuf, fileSize, hipMemcpyHostToDevice);
  free(hostBuf);
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
