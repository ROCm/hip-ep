/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hal_factory.cpp - HalAllocator factory implementation --------------===//
//
// Detects hipDeviceAttributeIntegrated at session creation time and returns
// the correct HalAllocator backend.
//
// Under HIPDNN_EP_MM_MOCK_HAL (unit tests), always returns ApuHalAllocator
// (both backends use malloc under the flag; APU is the simpler one).
//
//===----------------------------------------------------------------------===//

#include "hal.h"

#ifndef HIPDNN_EP_MM_MOCK_HAL
#include "runtime_types.h"

// hipDeviceGetAttribute is available in both real HIP and the mock (via
// mock_types.h which declares the function forward); however, the mock's
// hipDeviceProp_t already carries an `integrated` field. We use
// hipGetDeviceProperties for simplicity since the mock provides it too.
#else
// In mock mode we only need the struct definition; mock_types.h provides it.
#include "mock_types.h"

static inline int hipGetDeviceProperties(hipDeviceProp_t *prop,
                                         int /*device*/) {
  prop->integrated = 1; // mock always reports integrated (APU)
  return 0;             // hipSuccess == 0
}
#define hipSuccess 0
#endif

HalAllocator *hal_create_for_device(int device_id) {
  hipDeviceProp_t prop{};
  int rc = hipGetDeviceProperties(&prop, device_id);
  bool integrated = (rc == hipSuccess) && (prop.integrated != 0);

  if (integrated) {
    return new ApuHalAllocator();
  }
  return new DiscreteHalAllocator();
}
