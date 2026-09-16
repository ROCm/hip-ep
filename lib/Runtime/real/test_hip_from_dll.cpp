/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Test functions for model DLL - HIP initialization diagnostics.
 * test_hip_from_dll_manual: step-by-step manual init for diagnostics.
 * test_hip_from_dll: alias that runs the manual init sequence.
 */
#include "error_check_macros.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

extern "C" {

// Manual initialization sequence for HIP diagnostic testing
#ifdef _WIN32
__declspec(dllexport)
#endif
    int test_hip_from_dll_manual() {
  fprintf(stderr,
          "\n[Model DLL Manual] Starting full initialization sequence\n");
  fprintf(stderr, "[Model DLL Manual] Thread ID: %lu\n",
#ifdef _WIN32
          GetCurrentThreadId()
#else
          (unsigned long)pthread_self()
#endif
  );

  // Step 1: HIP device initialization
  fprintf(stderr, "[Model DLL Manual] Step 1: HIP device initialization\n");
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    fprintf(stderr,
            "[Model DLL Manual] ERROR: Failed to get HIP device count\n");
    return 2;
  }
  fprintf(stderr, "[Model DLL Manual] Device count: %d\n", device_count);

  if (hipSetDevice(0) != hipSuccess) {
    fprintf(stderr, "[Model DLL Manual] ERROR: Failed to set HIP device 0\n");
    return 3;
  }

  // Step 2: First device properties check
  fprintf(stderr, "[Model DLL Manual] Step 2: First device properties check\n");
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr,
            "[Model DLL Manual] ERROR: hipGetDeviceProperties failed\n");
    return 4;
  }

  fprintf(stderr, "[Model DLL Manual] Device: %s\n", prop.name);
  fprintf(stderr, "[Model DLL Manual] gcnArchName (BEFORE hipBLASLt): '%s'\n",
          prop.gcnArchName);
  fprintf(stderr, "[Model DLL Manual] gcnArchName length: %zu\n",
          strlen(prop.gcnArchName));

  if (prop.gcnArchName[0] == '\0') {
    fprintf(stderr, "[Model DLL Manual] WARNING: gcnArchName is EMPTY before "
                    "hipBLASLt init\n");
    return 1;
  }

  // Step 3: Stream creation
  fprintf(stderr, "[Model DLL Manual] Step 3: Creating HIP stream\n");
  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) {
    fprintf(stderr, "[Model DLL Manual] ERROR: Failed to create HIP stream\n");
    return 6;
  }

  // Step 4: hipBLASLt initialization (may crash if gcnArchName is empty)
  fprintf(stderr, "[Model DLL Manual] Step 4: hipBLASLt initialization\n");
  hipblasLtHandle_t hipblas_handle = nullptr;
  if (hipblasLtCreate(&hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "[Model DLL Manual] ERROR: Failed to create hipBLASLt handle\n");
    HIP_CLEANUP(hipStreamDestroy(stream));
    return 9;
  }

  // Step 5: Re-check gcnArchName AFTER hipBLASLt initialization
  fprintf(stderr, "[Model DLL Manual] Step 5: Re-checking device properties "
                  "AFTER hipBLASLt\n");
  hipDeviceProp_t prop_after;
  if (hipGetDeviceProperties(&prop_after, 0) == hipSuccess) {
    fprintf(stderr, "[Model DLL Manual] gcnArchName (AFTER hipBLASLt): '%s'\n",
            prop_after.gcnArchName);
    fprintf(stderr, "[Model DLL Manual] gcnArchName length: %zu\n",
            strlen(prop_after.gcnArchName));

    if (prop_after.gcnArchName[0] == '\0') {
      fprintf(stderr, "[Model DLL Manual] ERROR: gcnArchName became EMPTY "
                      "after hipBLASLt!\n");
      hipblasLtDestroy(hipblas_handle);
      HIP_CLEANUP(hipStreamDestroy(stream));
      return 1;
    }
  }

  // Success - cleanup
  fprintf(
      stderr,
      "[Model DLL Manual] All initialization steps completed successfully!\n");
  hipblasLtDestroy(hipblas_handle);
  (void)hipStreamDestroy(stream);

  return 0; // Success
}

// Alias: delegates to the manual sequence above
#ifdef _WIN32
__declspec(dllexport)
#endif
    int test_hip_from_dll() {
  return test_hip_from_dll_manual();
}

} // extern "C"
