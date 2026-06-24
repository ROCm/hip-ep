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

HalAllocator *hal_create_for_device(int device_id) {
  hipDeviceProp_t prop{};
  // hipGetDeviceProperties returns 0 on success; prop.integrated is 1 for APU.
  int rc = hipGetDeviceProperties(&prop, device_id);
  bool integrated = (rc == hipSuccess) && (prop.integrated != 0);
  if (integrated)
    return new ApuHalAllocator();
  return new DiscreteHalAllocator();
}

#else
// Mock / unit-test path: no HIP SDK available. Always return ApuHalAllocator
// because both backends use malloc under HIPDNN_EP_MM_MOCK_HAL, and APU is
// the simpler of the two for tests. Avoids redefining hipGetDeviceProperties
// which mock_types.h already declares as extern "C".
HalAllocator *hal_create_for_device(int /*device_id*/) {
  return new ApuHalAllocator();
}
#endif
