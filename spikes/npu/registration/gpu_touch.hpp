/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// C-ABI wrapper around a trivial HIP kernel, kept in its own hipcc-compiled
// translation unit (gpu_touch.hip) so that the host code linking
// DynamicDispatch/XRT/ryzen_mm (npu_probe.cpp) never needs to be compiled by
// hipcc's clang-cl frontend. Those SDKs are consumed elsewhere in this
// project (and in the hybrid-llm reference tree) as plain MSVC static
// libraries; T0.2, not this spike, is where clang-cl/MSVC coexistence in one
// binary is proven out. Keeping the device kernel in a separate object file
// sidesteps that question entirely for T0.1.

#include <cstddef>
#include <cstdint>

extern "C" {

// Adds `bits_delta` to the raw bf16 bit pattern of every element of
// `host_ptr` (a buffer obtained from hipHostMalloc(..., hipHostMallocMapped),
// so it is dereferenceable from the CPU and importable by HIP without a
// copy). Resolves the device-side view via hipHostGetDevicePointer,
// launches the kernel, and synchronizes before returning -- the caller can
// read the result straight back through `host_ptr` on the CPU with no
// explicit copy.
//
// Returns false on any HIP failure (device pointer resolution, launch,
// or sync); the launch status always reflects hipGetLastError() taken
// after the launch, never an unconditional success.
bool GpuBumpBf16InPlace(void* host_ptr, size_t n_elems, uint16_t bits_delta);

}  // extern "C"
