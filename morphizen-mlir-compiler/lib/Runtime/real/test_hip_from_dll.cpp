/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Test functions for model DLL - compare manual init vs calling
 * hipdnn_ep_state_init()
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <pthread.h>
#endif

extern "C" {

// VERSION 1: Manual initialization (mimics hipdnn_ep_state_init step-by-step)
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
  fprintf(stderr, "[Model DLL Manual] gcnArchName (BEFORE MIOpen): '%s'\n",
          prop.gcnArchName);
  fprintf(stderr, "[Model DLL Manual] gcnArchName length: %zu\n",
          strlen(prop.gcnArchName));

  if (prop.gcnArchName[0] == '\0') {
    fprintf(stderr, "[Model DLL Manual] WARNING: gcnArchName is EMPTY before "
                    "MIOpen init\n");
    return 1;
  }

  // Step 3: Stream creation
  fprintf(stderr, "[Model DLL Manual] Step 3: Creating HIP stream\n");
  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) {
    fprintf(stderr, "[Model DLL Manual] ERROR: Failed to create HIP stream\n");
    return 6;
  }

  // Step 4: MIOpen initialization
  fprintf(stderr, "[Model DLL Manual] Step 4: MIOpen initialization\n");
  miopenHandle_t miopen_handle = nullptr;
  if (miopenCreate(&miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr,
            "[Model DLL Manual] ERROR: Failed to create MIOpen handle\n");
    hipStreamDestroy(stream);
    return 7;
  }

  if (miopenSetStream(miopen_handle, stream) != miopenStatusSuccess) {
    fprintf(stderr, "[Model DLL Manual] ERROR: Failed to set MIOpen stream\n");
    miopenDestroy(miopen_handle);
    hipStreamDestroy(stream);
    return 8;
  }

  // Step 5: Re-check gcnArchName AFTER MIOpen initialization (CRITICAL)
  fprintf(stderr, "[Model DLL Manual] Step 5: Re-checking device properties "
                  "AFTER MIOpen\n");
  hipDeviceProp_t prop_after_miopen;
  if (hipGetDeviceProperties(&prop_after_miopen, 0) == hipSuccess) {
    fprintf(stderr, "[Model DLL Manual] gcnArchName (AFTER MIOpen): '%s'\n",
            prop_after_miopen.gcnArchName);
    fprintf(stderr, "[Model DLL Manual] gcnArchName length: %zu\n",
            strlen(prop_after_miopen.gcnArchName));

    if (prop_after_miopen.gcnArchName[0] == '\0') {
      fprintf(
          stderr,
          "[Model DLL Manual] ERROR: gcnArchName became EMPTY after MIOpen!\n");
      miopenDestroy(miopen_handle);
      hipStreamDestroy(stream);
      return 1;
    }
  }

  // Step 6: hipBLASLt initialization (may crash if gcnArchName is empty)
  fprintf(stderr, "[Model DLL Manual] Step 6: hipBLASLt initialization\n");
  hipblasLtHandle_t hipblas_handle = nullptr;
  if (hipblasLtCreate(&hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "[Model DLL Manual] ERROR: Failed to create hipBLASLt handle\n");
    miopenDestroy(miopen_handle);
    hipStreamDestroy(stream);
    return 9;
  }

  // Success - cleanup
  fprintf(
      stderr,
      "[Model DLL Manual] All initialization steps completed successfully!\n");
  hipblasLtDestroy(hipblas_handle);
  miopenDestroy(miopen_handle);
  hipStreamDestroy(stream);

  return 0; // Success
}

// VERSION 2: Call hipdnn_ep_state_init() directly (production code path)
#ifdef _WIN32
__declspec(dllexport)
#endif
    int test_hip_from_dll() {
  fprintf(stderr,
          "\n[Model DLL StateInit] Calling hipdnn_ep_state_init() directly\n");
  fprintf(stderr, "[Model DLL StateInit] Thread ID: %lu\n",
#ifdef _WIN32
          GetCurrentThreadId()
#else
          (unsigned long)pthread_self()
#endif
  );

  // Call the actual initialization function used by production code
  RuntimeState* state = nullptr;
  int ret = hipdnn_ep_state_init(&state, nullptr);

  if (ret != 0) {
    fprintf(stderr,
            "[Model DLL StateInit] ERROR: hipdnn_ep_state_init failed with "
            "code: %d\n",
            ret);
    return ret;
  }

  fprintf(stderr, "[Model DLL StateInit] hipdnn_ep_state_init() succeeded!\n");

  // Verify device properties after initialization
  fprintf(stderr, "[Model DLL StateInit] Verifying gcnArchName after full "
                  "initialization\n");
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
    fprintf(stderr, "[Model DLL StateInit] Device: %s\n", prop.name);
    fprintf(stderr, "[Model DLL StateInit] gcnArchName: '%s'\n",
            prop.gcnArchName);
    fprintf(stderr, "[Model DLL StateInit] gcnArchName length: %zu\n",
            strlen(prop.gcnArchName));

    if (prop.gcnArchName[0] == '\0') {
      fprintf(stderr, "[Model DLL StateInit] ERROR: gcnArchName is EMPTY after "
                      "initialization!\n");
      hipdnn_ep_state_cleanup(state);
      return 1;
    }
  }

  // Cleanup
  fprintf(stderr, "[Model DLL StateInit] Cleaning up\n");
  hipdnn_ep_state_cleanup(state);

  fprintf(stderr, "[Model DLL StateInit] Test completed successfully!\n");
  return 0; // Success
}

} // extern "C"
