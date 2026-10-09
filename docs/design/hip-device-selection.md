<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP Device Selection

**Document Type:** Design
**Status:** Current
**Related:** [morphizen-ep-integration.md](morphizen-ep-integration.md), [output-allocator-design.md](output-allocator-design.md)

---

## Table of Contents

- [Overview](#overview)
- [Two numberings](#two-numberings)
- [Allocator device selection](#allocator-device-selection)
- [HIP last-error hygiene at EP boundaries](#hip-last-error-hygiene-at-ep-boundaries)
- [Device-filtering diagnostics](#device-filtering-diagnostics)
- [Limitations](#limitations)

---

## Overview

The EP runs on the AMD GPU that ORT selected, but ORT and HIP number GPUs
differently. This document describes how the EP's ORT allocator picks the HIP
device, how the EP keeps a failed HIP call made outside the model from being
reported as a kernel failure, and what the EP reports when HIP cannot see the
GPU.

The implementation lives in:

| Area | Path |
|---|---|
| Selection rules (GPU-free) | `morphizen/ort-bridge/src/morphizen-hip-device-select.{hpp,cpp}` |
| ORT allocator | `morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.{hpp,cpp}` |
| Device enumeration | `morphizen/ort-bridge/src/morphizen-ep-factory.cpp` |
| Inference entry and output callback | `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` |
| Runtime device init | `lib/Runtime/hipdnn_ep_runtime_state.cpp` |
| Unit test | `test/runtime/test_hip_device_select.cpp` (`HipDeviceSelectUnitTest`) |

## Two numberings

- **ORT / DXGI.** On Windows, ORT's device discovery tags every GPU with
  `DxgiAdapterNumber`, its `IDXGIFactory1::EnumAdapters1` index, and `LUID`.
  DXGI enumerates every adapter, NVIDIA included, and returns the adapter with
  the primary desktop output as index 0.
- **HIP.** The HIP runtime enumerates AMD devices only, after applying its
  device-filtering environment variables.

On a machine with one AMD GPU, HIP always calls it ordinal 0, but its DXGI
adapter number changes with the display configuration: it is 1 whenever
another GPU drives the primary display, for example an external monitor on a
port wired to a discrete NVIDIA GPU, or a MUX switch in discrete-only mode.

## Allocator device selection

The factory registers two memory infos, `MorphiZen` and
`MorphiZen host accessible`, whose device id is a HIP ordinal. A parent EP can
also forward its own memory infos to this EP's `CreateAllocator`: the AMD GPU
umbrella (amdgpu-ep) registers `default` and `pinned` with the DXGI adapter
number as device id and never calls this EP's `GetSupportedDevices`.

`SelectAllocatorHipDevice(requested, own_memory_info, hip_device_count)`
chooses the ordinal the allocator makes current:

| Memory info | HIP devices | Selected |
|---|---|---|
| Own | `requested` in range | `requested` |
| Own | `requested` out of range | current device |
| Parent EP | 1 | 0 |
| Parent EP | more than 1 | current device |
| Any | 0, or `requested` unreadable | current device |

"Current device" means the allocator leaves the calling thread's HIP device
unchanged. A parent EP's id is never used as an ordinal, because a DXGI
adapter number does not identify a HIP device.

The selection is made on the first allocation, not in the constructor. ORT
constructs allocators while it enumerates EP devices, and `amdhip64` is
delay-loaded so the HIP runtime stays out of that window (and out of the
library unload that follows it). For the same reason the factory and
`GetSupportedDevices` make no HIP calls, and an allocator that never allocated
makes none on destruction.

## HIP last-error hygiene at EP boundaries

HIP keeps a failed call's error in a per-thread last-error slot until
`hipGetLastError()` reads it. Custom-kernel launchers check that slot after a
launch, so a failure left there by unrelated code is reported as the next
kernel's launch failure. Graph outputs are allocated mid-inference
(`hip.alloc_output` calls the EP's output callback, which calls
`KernelContext::GetOutput`, which calls whichever allocator owns the output's
memory info) on the thread that launches the kernels.

The EP therefore reads and discards the slot:

- before calling `inference_compute`, since host code that ran earlier on the
  thread (ORT, another EP, the application) may have left an error; and
- in the output callback after `GetOutput` returns, since the buffer is valid
  at that point whatever the allocator's HIP calls reported.

The ORT allocator also reads the slot after each HIP call of its own that
fails (`hipSetDevice`, `hipHostMalloc`, `hipHostFree`, `hipGetDeviceCount`).

## Device-filtering diagnostics

The HIP runtime hides devices according to `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL`, and honors
`CUDA_VISIBLE_DEVICES` as an alias while `HIP_VISIBLE_DEVICES` is unset. The
alias matters on machines with an NVIDIA GPU, where `CUDA_VISIBLE_DEVICES` is
often set for CUDA applications.

- `GetSupportedDevices` warns once when any of these is set. It reads the
  environment only, so it cannot tell whether the AMD GPU is actually hidden.
- The allocator logs an error, naming the set variables, when the HIP runtime
  reports no devices on the first allocation.
- Runtime state initialization prints the `hipGetDeviceCount` error and the
  set variables when it finds no device.

## Limitations

- **Several AMD GPUs.** ORT requires every `OrtEpDevice` of a factory to share
  one memory info, and runtime state initialization always selects HIP
  device 0, so the EP effectively runs on HIP device 0. Mapping a parent EP's
  DXGI adapter number to an ordinal would need the adapter LUID (ORT's `LUID`
  metadata against `hipDeviceProp_t::luid`); the robust fix is for the parent
  EP to translate the number before forwarding it.
- **Kernel launchers.** Not every custom-kernel launcher clears the last-error
  slot before its launch, as the repository rules require. The boundary reads
  above cover errors left by code outside the model; errors from the model's
  own earlier HIP calls are only covered where the launcher clears first.
