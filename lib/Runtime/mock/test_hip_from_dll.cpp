/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Simple test function for model DLL - just call hipGetDeviceProperties
 */
#include "runtime_types.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

extern "C" {

// Export test function for DLL
#ifdef _WIN32
__declspec(dllexport)
#endif
    int test_hip_from_dll() {
  fprintf(stderr, "\n[Model DLL test_hip_from_dll] Called\n");
  fprintf(stderr, "[Model DLL] Thread ID: %lu\n",
#ifdef _WIN32
          GetCurrentThreadId()
#else
          (unsigned long)pthread_self()
#endif
  );

  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  fprintf(stderr, "[Model DLL] hipGetDeviceProperties: err=%d\n", err);

  if (err != hipSuccess) {
    fprintf(stderr, "[Model DLL] ERROR: Failed to get device properties\n");
    return 2;
  }

  fprintf(stderr, "[Model DLL] Device name: '%s'\n", prop.name);
  fprintf(stderr, "[Model DLL] gcnArchName: '%s'\n", prop.gcnArchName);
  fprintf(stderr, "[Model DLL] gcnArchName length: %zu\n",
          strlen(prop.gcnArchName));

  return (prop.gcnArchName[0] == '\0') ? 1 : 0;
}

} // extern "C"
