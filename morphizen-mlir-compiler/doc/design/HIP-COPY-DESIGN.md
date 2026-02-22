<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GPU Memory Copy Design

**Date:** 2026-02-17
**Document Type:** Design
**Status:** Draft
**Related:** [HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md), [01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md)

---

## Overview

Standard MLIR `memref.copy` lowers to CPU `memcpy` intrinsic, causing access violations when operating on GPU memory (address space 1). HIP dialect needs GPU-aware copy operation that lowers to `hipMemcpyAsync`.

---

## Design

### Operation Definition

```mlir
hip.copy(%ctx, %src, %dst) : (!hip.context, memref<..., 1>, memref<..., 1>)
```

**Arguments:**
- `ctx`: Runtime context (provides GPU stream)
- `src`: Source GPU buffer
- `dst`: Destination GPU buffer

**Properties:**
- No return value (destination-passing)
- Both operands must have address space 1 (GPU memory)
- Copy is asynchronous on GPU stream

### Lowering Strategy

```
hip.copy → llvm.call @hipdnn_ep_memcpy_gpu_to_gpu → hipMemcpyAsync(DeviceToDevice)
```

Runtime wrapper extracts:
- Stream from RuntimeState
- Buffer pointers from memref descriptors
- Size from memref dimensions

### Copy Elimination

Two-level optimization:

**Level 1: Canonicalization (generic)**
- Self-copy: `hip.copy(%ctx, %buf, %buf)` → remove
- Single-use: `hip.operation(..., %temp); hip.copy(%ctx, %temp, %out)` → redirect operation to write to `%out`

**Level 2: OnnxToHip optimization (proactive)**
- Avoid generating `hip.copy` when operation can write directly to output argument
- See [01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md) ReturnOpConversion

Canonicalization handles cases where OnnxToHip optimization fails (multi-use buffers, unsupported operations).

### Interface Integration

HIP operations implement `DestinationStyleOpInterface`:

```cpp
OpOperandVector Hip_ConvOp::getDpsInits() {
  return {&getOperation()->getOpOperand(4)};  // output operand
}
```

Enables generic canonicalization pattern:

```cpp
// Works for ANY operation implementing DestinationStyleOpInterface
for (Operation *user : src.getUsers()) {
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
    for (OpOperand *init : dpsOp.getDpsInits()) {
      // Check if user writes to source buffer
    }
  }
}
```

No hard-coded operation checks. New operations automatically supported by implementing interface.

---

## Transformation Flow

```
ONNX dialect:
  func @main(%input) -> %result

OnnxToHip (ReturnOpConversion):
  %temp = hip.alloc(...)
  hip.conv(..., %temp)
  hip.copy(%ctx, %temp, %output)  // Fallback when optimization fails

Canonicalizer (EliminateCopyAfterDPSWrite):
  hip.conv(..., %output)          // Redirected, copy eliminated
  // %temp allocation becomes dead → removed by BufferDeallocation

HipToLLVM:
  llvm.call @hipdnn_ep_memcpy_gpu_to_gpu(...)  // If copy not eliminated

Runtime:
  hipMemcpyAsync(dst, src, size, DeviceToDevice, stream)
```

---

## Comparison with memref.copy

| Aspect | memref.copy | hip.copy |
|--------|-------------|----------|
| Target | CPU memory | GPU memory (address space 1) |
| Lowering | `llvm.memcpy` intrinsic → CPU `memcpy` | Runtime call → `hipMemcpyAsync` |
| Context | Not needed | Requires `!hip.context` for stream |
| Synchronization | Synchronous | Asynchronous on GPU stream |
| Optimization | No dialect-specific patterns | DestinationStyleOpInterface + canonicalization |

---

## Related Documents

- [HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) - HIP dialect operations
- [01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md) - Copy generation in ReturnOpConversion
- [03-Canonicalization.md](mlir/passes/03-Canonicalization.md) - Copy elimination patterns
- [05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) - Lowering to runtime calls
- [BUFFER-LIFETIME-DESIGN.md](BUFFER-LIFETIME-DESIGN.md) - Buffer ownership and deallocation
