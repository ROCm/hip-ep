<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# iGPU Constants Memory Strategy

**Date:** 2026-03-03
**Document Type:** Tech Note
**Status:** Draft
**Related:** [CONSTANT-HANDLING-DESIGN.md](../design/CONSTANT-HANDLING-DESIGN.md), [MEMORY-MANAGEMENT.md](../design/MEMORY-MANAGEMENT.md)

---

## Overview

On discrete GPUs (dGPU), `hipMalloc` draws from a driver-managed VRAM quota.
On integrated GPUs (iGPU), VRAM and system RAM share the same physical DRAM, but
`hipMalloc` still draws from a driver-managed GPU memory quota — not the full
system RAM. For models with large constants blobs, `hipMalloc` can fail with
`hipErrorOutOfMemory` even when physical memory is available.

`hipHostMalloc` (pinned host memory) avoids this on iGPU:
- Draws from system RAM, not the GPU quota.
- GPU accesses the same physical DRAM directly — no `hipMemcpy` needed.
- Must be freed with `hipHostFree`, not `hipFree`.

---

## Design

### Allocator Selection

`RuntimeState` initialization detects the GPU type via `hipDeviceProp_t.integrated`
(already fetched before stream creation — no extra HIP call). Constants are then
allocated with the appropriate allocator:

| GPU type | Allocator | Upload | Free |
|----------|-----------|--------|------|
| dGPU (`integrated == 0`) | `hipMalloc` | `hipMemcpy` (H2D) | `hipFree` |
| iGPU (`integrated == 1`) | `hipHostMalloc` | `memcpy` into pinned buffer | `hipHostFree` |

On iGPU with mmap available, the data is copied directly from the mmap'd file
into pinned memory. On iGPU without mmap, `fread` writes directly into the
pinned buffer — no staging allocation is needed.

The memory pool (`pool_base`) used for intermediate buffers is always allocated
with `hipMalloc`. Intermediate buffers are GPU-local and small relative to
constants, so the quota constraint does not apply.

### `RuntimeState` field

```c
bool constants_blob_is_host; // true = hipHostMalloc, false = hipMalloc
```

Initialized to `false`. Set to `true` only when `hipHostMalloc` is used.
Read in cleanup to select `hipHostFree` vs `hipFree`.

### Cleanup

```c
if (state->constants_blob_is_host)
    hipHostFree(state->gpu_constants_blob);
else
    hipFree(state->gpu_constants_blob);
```

---

## Related Documents

- [CONSTANT-HANDLING-DESIGN.md](../design/CONSTANT-HANDLING-DESIGN.md) — How constants are discovered, written, and uploaded at runtime
- [MEMORY-MANAGEMENT.md](../design/MEMORY-MANAGEMENT.md) — Overall GPU memory allocation strategy and pool lifecycle
