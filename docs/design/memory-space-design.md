<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Memory Space Design

**Date:** 2026-06-16
**Document Type:** Design
**Status:** Draft
**Related:** [hip-shape-inference.md](hip-shape-inference.md), [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md)

---

## Table of Contents

- [Problem](#problem)
- [Solution Overview](#solution-overview)
- [Design](#design)
  - [Memory Space Attribute](#memory-space-attribute)
  - [Type Constraints](#type-constraints)
  - [Transfer Operations](#transfer-operations)
  - [Pass Pipeline](#pass-pipeline)
- [Design Validation Examples](#design-validation-examples)
- [Related Documents](#related-documents)

---

## Problem

After bufferization, memref types do not track memory location.

**Example:**
```mlir
%padded = hip.pad(%ctx) ins(%input, %pads) outs(%output)
          -> memref<?x?xf16>
```

**Question:** Is `%padded` in GPU memory (hipMalloc) or host memory?

**Answer:** The type `memref<?x?xf16>` doesn't tell you.

### Consequence 1: Silent SEGFAULTs

```mlir
// GPU kernel output pooled in device memory
%pool = hip.get_pool(%ctx, %size) : memref<?xi8>
%output = memref.view %pool[...] : memref<?xi8> to memref<?x?xf16>
%result = hip.pad(%ctx) ins(...) outs(%output)

// Later: direct host access
%value = memref.load %result[%c0, %c0] : memref<?x?xf16>  // SEGFAULT!
```

On true device memory (gfx1151), host load of device memory crashes.

On UMA architectures, this accidentally works, masking the bug.

### Consequence 2: Manual Synchronization

Converters must manually insert D2H transfers:

```cpp
// In RangeConversion.cpp - manual readback
Value limit = readbackScalarToHost(builder, loc, ctx, limitTensor);
```

Problems:
- Nothing enforces this - easy to forget
- Direct `tensor.extract` compiles but crashes at runtime
- Cannot optimize sync placement - compiler doesn't know what's device vs host

### Consequence 3: Cannot Minimize Synchronization

Without knowing which memrefs are device vs host, must sync after every possibly-device operation just to be safe, or risk SEGFAULT.

Cannot limit sync barriers to the actual device→host crossings only.

---

## Solution Overview

Use memory space attributes on memref types to enforce host/device memory boundaries at compile time.

### Three-Step Solution: Correct by Construction

**1. Enforce Memory Space on All Operations**

Revise all hip operations to require explicit memory space attributes. Every memref type must specify `#hip.mem<device>`, `#hip.mem<host>`, `#hip.mem<pinned>`, or `#hip.mem<managed>`.

```tablegen
// GPU kernel operations - require device memory
def Hip_PadOp : Hip_Op<"pad"> {
  let arguments = (ins
    Hip_TensorOrDeviceMemRef:$input,   // Only accepts device memory
    Hip_TensorOrDeviceMemRef:$output
  );
}

// Pool operation - returns device memory
def Hip_GetPoolOp : Hip_Op<"get_pool"> {
  let results = (outs DeviceMemRef:$pool);  // Always device memory
}
```

**2. OnnxToHip Converters Insert Intentional Transfers**

When an ONNX operation's behavior requires crossing the host/device boundary, converters insert explicit transfers. Primary source: shape inference needing to inspect device data.

```mlir
// ONNX NonZero: GPU computes condition, CPU must count true elements
// Converter inserts intentional D2H transfer for shape calculation:
%cond_device = hip.greater(%ctx) ins(%a, %b)
               -> memref<?x?xi1, #hip.mem<device>>

// Shape inference needs CPU access - insert explicit transfer
%cond_host = memref.alloc() : memref<?x?xi1, #hip.mem<pinned>>
hip.memcpy_d2h_async %ctx, %cond_host, %cond_device
hip.stream_sync %ctx

// Now safe - count on host, allocate output with correct shape
%count = /* count true elements in cond_host */
%output = memref.alloc(%count) : memref<?x2xi64, #hip.mem<device>>
```

Operations requiring shape inference transfers:

| Operation | Why CPU Access Needed |
|-----------|-----------------------|
| NonZero   | Count true elements to allocate output |
| TopK      | Read k value if dynamic |
| Reshape   | Read shape tensor for dynamic reshape |
| Expand    | Read shape tensor |
| Tile      | Read repeats if dynamic |
| Range     | Read start/limit/delta for output size |
| Compress  | Count condition elements |
| Unique    | Count unique elements |

**3. Type System Prevents Violations at Compile Time**

With explicit memory spaces, MLIR type system catches invalid code before it's generated:

```mlir
// ❌ WRONG - Compile-time error, GPU kernel with host memory
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.pad(%ctx) outs(%host)
// Error: operand type 'memref<i64, #hip.mem<host>>' doesn't match
//        constraint 'Hip_TensorOrDeviceMemRef' (requires device memory)

// ✅ RIGHT - Converter ensures correct memory space from start
%device = memref.alloc() : memref<i64, #hip.mem<device>>
hip.pad(%ctx) outs(%device)  // Type-correct, no fix-up needed
```

**Key principle:** No illegal MLIR code generated. Converters and bufferization produce type-correct IR from the start. No fix-up passes needed to correct type violations.

### Implementation Pipeline

```
Slot 1:  OnnxToHip Conversion
           ↓ Converters insert explicit transfers when the ONNX op requires it
           → For shape inference: D2H transfers for dynamic shape calculation
Slot 2:  One-Shot-Bufferize
           ↓ Operations specify memory space via getBufferType()
           → Creates memref.alloc with #hip.mem<device> or #hip.mem<host>
Slot 6a: PromoteStridedHipOperands
           ↓ Preserve memory space when creating contiguous copies
Slot 6b: ShapeInference (CRITICAL)
           ↓ Detect operations requiring dynamic shape calculation from device data
           → Insert hip.memcpy_d2h_async + hip.stream_sync + pinned host buffer
           → Examples: NonZero (count), TopK (k value), Reshape (shape tensor)
Slot 6c: EliminateStreamSyncBarriers
           ↓ Analyze MemoryEffectsOpInterface (similar to MLIR's EliminateBarriers)
           → Combine multiple syncs into the fewest barriers needed
Slot 6+: PoolAllocs
           → Pool device memory allocations into hip.get_pool
```

### Key Properties

**Correct by construction:**
- Type system enforces memory space constraints
- Invalid accesses caught at compile time, not runtime SEGV
- No pattern matching - TableGen constraints verify automatically
- No fix-up passes - converters and bufferization generate type-correct IR from the start

**Performance:**
- Shape inference inserts transfers only when the operation truly needs them (e.g., dynamic shape calculation)
- Synchronization minimized via barrier elimination
- Explicit async operations allow overlap optimization

**Systematic Review Required:**

The codebase has ~70 HIP operations. Each needs review to ensure:
- `getBufferType()` returns correct memory space for results
- OnnxToHip converters respect operation type constraints
- Converters insert intentional transfers when the ONNX op requires them (primarily shape inference)
- Type system prevents accidental violations

This is correct-by-construction design, not detect-and-fix.

**Note on LLVM Lowering:**

`ConvertHipToLLVMPass` registers a type-attribute conversion (`addTypeAttributeConversion`) that maps each `#hip.mem<...>` space to a distinct LLVM address space, numbered by the `MemorySpaceKind` enum (**host = 0, device = 1, pinned = 2, managed = 3**).

This follows the **AMDGPU/MLIR convention**: host is the generic/flat AS 0, device is the global AS 1. The address spaces are a compile-time type-system label only, not the AMDGPU backend's hardware spaces — the JIT target is the host, which flattens every space back to one flat pointer space (see the last bullet), so the casts below are no-ops at runtime and the generated model code passes device pointers to the runtime as plain opaque values.

- The runtime C ABI is flat AS-0 `!llvm.ptr`. Each per-op lowering resolves the operand space via `getMemRefAddressSpace()` and addrspace-casts the device (AS 1) pointer down to AS 0 at the call boundary (see `extractMemRefDataPtr` / `MemoryLowering.cpp`), so runtime declarations stay AS-0.
- The host target collapses every space back to one flat space, so CPU access to host/pinned/managed buffers (e.g. `tensor.extract` after a D2H `hip.memcpy_d2h_async` + `hip.stream_sync`) stays valid.

Without this hook, the stock `MemRefToLLVM` conversion — which only understands integer memory spaces — rejects any memref carrying the non-integer `MemorySpaceAttr` and conversion fails.

---

## Design

This section explains the four building blocks that implement the 3-step solution overview.

### Memory Space Attribute

The foundation is a custom MLIR attribute that annotates memref types with explicit memory space:

**Syntax:**
```mlir
memref<i64, #hip.mem<host>>      // Pageable host (malloc/stack)
memref<i64, #hip.mem<device>>    // GPU VRAM (hipMalloc)
memref<i64, #hip.mem<pinned>>    // Pinned host (hipHostMalloc)
memref<i64, #hip.mem<managed>>   // Unified memory (hipMallocManaged)
```

Four memory spaces mapping to HIP allocation methods:
- `#hip.mem<host>`: Pageable host memory (malloc/stack), CPU-only access
- `#hip.mem<device>`: GPU VRAM (hipMalloc), device-only access, highest bandwidth
- `#hip.mem<pinned>`: Pinned host memory (hipHostMalloc), two usage patterns:
  1. Transfer mode: ~3x faster hipMemcpyAsync bandwidth
  2. Zero-copy mode: direct GPU access without copy (no page faults)
- `#hip.mem<managed>`: Unified memory (hipMallocManaged), automatic migration with page faults

**Rationale for four spaces:**

Cannot merge pinned and managed due to different performance characteristics:
- hipHostMalloc zero-copy: No page faults, deterministic latency
- hipMallocManaged: Page faults on access, migration overhead
- Evidence: "Pinned system memory performs consistently better" for write-once-read-once (Performance Evaluation)

Maps to Solution Step 1: Enforce memory space on all operations - operations declare memory space requirements via type constraints.

### Type Constraints

TableGen type constraints verify memory spaces automatically at IR construction time:

**Pure memref constraints:**
```tablegen
HostMemRef      // Accepts: memref<..., #hip.mem<host>>
DeviceMemRef    // Accepts: memref<..., #hip.mem<device>>
PinnedMemRef    // Accepts: memref<..., #hip.mem<pinned>>
ManagedMemRef   // Accepts: memref<..., #hip.mem<managed>>
```

**Composite constraints for operations:**
```tablegen
Hip_TensorOrDeviceMemRef   // Accepts: tensor<...> OR memref<..., #hip.mem<device>>
Hip_TensorOrPinnedMemRef   // Accepts: tensor<...> OR memref<..., #hip.mem<pinned>>
Hip_TensorOrManagedMemRef  // Accepts: tensor<...> OR memref<..., #hip.mem<managed>>
```

**Why composite:** Operations work on tensors before bufferization and memrefs after bufferization. Constraints must accept both.

**Example operations with different memory space requirements:**
```tablegen
// Data arguments - must be device memory for GPU kernel access
def Hip_MatmulOp : Hip_Op<"matmul"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrDeviceMemRef:$a,      // Device - GPU kernel reads
    Hip_TensorOrDeviceMemRef:$b,      // Device - GPU kernel reads
    Hip_TensorOrDeviceMemRef:$c       // Device - GPU kernel writes
  );
}

// Transfer operations - explicit memory space boundaries
def Hip_MemcpyD2HAsyncOp : Hip_Op<"memcpy_d2h_async"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    PinnedMemRef:$dst,    // Must be pinned (for fast transfers)
    DeviceMemRef:$src     // Must be device
  );
}

// Prefetch operation for managed memory
def Hip_MemPrefetchAsyncOp : Hip_Op<"mem_prefetch_async"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    ManagedMemRef:$ptr,   // Must be managed
    I64:$count,
    I32:$device_id
  );
}
```

**Memory space selection guide:**

| Use Case | Memory Space | Rationale |
|----------|--------------|-----------|
| Large data tensors (activations, weights in VRAM) | `#hip.mem<device>` | >40x faster GPU access vs pinned zero-copy |
| Model weights (>VRAM, oversubscription) | `#hip.mem<managed>` | Only option vs OOM, auto-migration |
| Dynamic shape outputs (NonZero, TopK, Where) | `#hip.mem<pinned>` | GPU writes once, CPU reads once, no page faults |
| Streaming pipeline inputs | `#hip.mem<pinned>` | Fast async transfers, overlap copy+compute |
| Kernel arguments (small scalars/arrays) | Pass by value | No allocation, fastest |
| CPU-only data | `#hip.mem<host>` | Pageable, no GPU access |

**Memory space assignment principle:**

- **Data arguments** (large tensors that GPU kernels read/write): `Hip_TensorOrDeviceMemRef`
  - Must be device memory for GPU kernel access
  - Examples: input images, weight matrices, output tensors
- **Shape arguments** (scalars/1D tensors for dimension calculation): Depends on size and GPU kernel usage
  - Large or variable-length arrays: Use device memory (`Hip_TensorOrDeviceMemRef`)
    - Examples: shape tensor for dynamic reshaping, large index arrays
  - Small fixed/bounded arrays used by GPU kernel: Pass as kernel arguments by value
    - Kernel arguments go via constant memory or registers (fastest)
    - No memory allocation, no pointer dereference, no hipMemcpy overhead
    - Examples: pads (≤16 ints), axes (≤8 ints), repeats (≤8 ints)
    - Constraint: Total kernel arguments typically limited to 4KB
  - Shape arguments only for CPU-side computation: Use host memory (`Hip_TensorOrHostMemRef`)
    - For bufferization/allocation only, not passed to GPU kernels
    - Avoids D2H synchronization overhead
- **Small scalar constants**: Should be passed as kernel argument values, not pointers
  - Inefficient to allocate device memory and dereference pointers for single scalars
  - Examples: start, limit, delta in `hip.range` (currently device pointers - design flaw)
  - Pass directly as i64, f32, etc. in kernel signature

**Current implementation issues - systematic review needed:**

The codebase has ~70 HIP operations, each with a runtime wrapper function. Many operations have inconsistent or suboptimal memory space handling for shape arguments:

Realized example of the by-value principle — `hip.pad`'s `constant_value`:
- Passed **by value**: a plain `f32`/`i32`/... SSA operand, not a device pointer.
- The converter resolves it **fold-first, then-transfer** — a compile-time-constant fill (the common case, e.g. `0.0`) folds to an `arith.constant` with zero device traffic; a runtime fill rides the same `hip.transfer ... to host` as `pads`/`axes` (detailed in the pad-pilot section below) and is read with `tensor.extract`.
- The lowering stages the scalar in a host stack slot; the runtime reads it with a plain `memcpy` — no device allocation, no implicit D2H.

This is the pattern the remaining flaws below should follow.

Examples of remaining design flaws:
- `hip.range` - start, limit, delta are device pointers for 3 scalar values
  - Problem: Allocates device memory + pointer dereference for 3 integers
  - Better: Pass as i64 scalar arguments directly
- `hip.tile` - repeats is device pointer but runtime ignores it entirely (confusing API)
- `hip.expand` - shape is device pointer but runtime ignores it entirely (confusing API)

Better design:
- Small arrays for GPU kernels (pads, axes, repeats): Pass as kernel argument values (via constant memory)
  - Kernel signature: `__global__ void padKernel(..., PadConfig config)` where `struct PadConfig { int pads[8]; int rank; }`
  - Faster than hipMemcpy, no allocation overhead, no pointer dereference
  - Constraint: Fits within 4KB kernel argument limit (easily satisfied for these cases)
- Small scalars (start, limit, delta): Pass as scalar values (i64, f32, etc.)
- Large/variable arrays: Device memory (`Hip_TensorOrDeviceMemRef`) only if too large for kernel arguments
- Data tensors: Device memory (`Hip_TensorOrDeviceMemRef`)

**Scope of work required:**

Each of the ~70 HIP operations needs individual review to determine:
- Which arguments are data (device memory) vs. shape (kernel arguments or host memory)
- Whether small arrays/scalars should be passed as kernel argument values instead of pointers
- Whether type constraints match runtime wrapper expectations
- Whether runtime wrappers have unnecessary D2H copies
- Kernel argument size budget (total < 4KB for all arguments combined)

This is significant refactoring work across:
- Operation definitions (TableGen constraints)
- Runtime wrapper signatures
- HipToLLVM lowering code
- OnnxToHip converters (bufferization logic)

This design document establishes the principle. The systematic operation-by-operation review and fixes are future work.

Maps to Solution Step 1 & 3: Type system should enforce memory space constraints matching the role of each argument. Current implementation has inconsistencies requiring systematic review.

### Transfer Operations

Explicit operations for moving data across host/device boundary:

**Device to host:**
```tablegen
def Hip_MemcpyD2HAsyncOp : Hip_Op<"memcpy_d2h_async"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    HostMemRef:$dst,        // Must be host (impl: host OR pinned — same hipMemcpyKind)
    DeviceMemRef:$src       // Must be device
  );
}
```

**Host to device:**
```tablegen
def Hip_MemcpyH2DAsyncOp : Hip_Op<"memcpy_h2d_async"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    DeviceMemRef:$dst,      // Must be device
    HostMemRef:$src         // Must be host (impl: host OR pinned — same hipMemcpyKind)
  );
}
```

**Synchronization:**
```tablegen
def Hip_StreamSyncOp : Hip_Op<"stream_sync"> {
  let arguments = (ins Hip_ContextType:$ctx);
}
```

Type constraints enforce correct transfer direction automatically. Cannot accidentally copy device→device or host→host using these operations.

Maps to Solution Step 2: OnnxToHip converters and shape inference use these operations to insert intentional transfers when an ONNX op requires crossing the host/device boundary.

#### Implementation status: `hip.transfer` as a parallel mechanism (pad pilot)

The transfer machinery above is now **partially implemented as a parallel mechanism** alongside the two pre-existing host-access mechanisms. All three coexist; the new one does not replace either:

| Mechanism | Phase | Used by | Status |
|-----------|-------|---------|--------|
| `hip.readback_scalar` / `hip.readback_dim` | tensor → bufferized D2H + sync | Range bounds, Loop trip count, dynamic Reshape/Expand **output-shape** arithmetic, FixLoopAccumulatorOffset (memref-phase) | unchanged; every op keeps using it. **Pad no longer uses it** — it reads pad amounts from its `hip.transfer` host copy instead |
| `get_host_scratch` (pinned) + `MaterializeHostScalars` | memref-phase host-mapped scratch | `tensor.from_elements` host-scalar staging (e.g. GQA `seqlens_k`) | unchanged; 100% untouched |
| **`hip.transfer` (+ `hip.memcpy_{h2d,d2h}_async` / `hip.stream_sync`)** | **tensor-phase explicit transfer**, bufferizes to memref-phase async memcpy + sync | **`hip.pad` only** (its `pads`/`axes` buffers) | **new; pilot** |

**`hip.transfer`** is the tensor-phase, value-preserving boundary-crossing op (`hip.transfer %ctx, %src to <host> -> ...`). Its `BufferizableOpInterface` model allocates the destination in the target space, emits the matching async memcpy, and replaces the result. The trailing `hip.stream_sync` is emitted **only for D2H** (the host reads the destination next, so it must wait); **H2D omits it**, because the device destination is consumed by later GPU work on the same stream, which already orders it after the copy. This is the tensor-phase counterpart the original design lacked — converters emit it before bufferization instead of hand-writing memref-phase memcpy/sync.

**Pad pilot (the one wired op).** `PadConversion` wraps `pads` (and `axes`, when present) in `hip.transfer ... to host`; `wrap_pad` then reads them with a plain `memcpy` instead of an internal `hipMemcpyAsync` D2H + `hipStreamSynchronize`. Two consequences:
- **Sync count is unchanged** — the D2H + sync moved out of the runtime and became explicit in the IR, the prerequisite for a future `EliminateStreamSyncBarriers` pass to coalesce crossings.
- **No `hip.readback_scalar`** — the output-shape arithmetic (`data_dim + pads_begin + pads_end`) reads the amounts from that *same* host copy via `tensor.extract`, so the single transfer of `pads` serves both the kernel operand and the dynamic-dim computation. This is the first op fully off readback for its host-scalar reads; the other readback users (Range/Reshape/Expand/Loop, and memref-phase FixLoopAccumulatorOffset) are unchanged.

**Deferred (still future work):** migrating readback / host-scratch / other ops onto `hip.transfer`; the H2D direction (`hip.memcpy_h2d_async` is lowered for symmetry but unused by the pad pilot); a pinned-backed transfer destination; the `ShapeInference`-insertion and `EliminateStreamSyncBarriers` passes; and flipping the unspecified-memory-space acceptance to strict.

### Pass Pipeline

Three passes coordinate to enforce memory spaces and optimize synchronization:

**1. Bufferization (Slot 2):**
- Each operation implements `BufferizableOpInterface::getBufferType()`
- Explicitly declares `#hip.mem<device>`, `#hip.mem<host>`, `#hip.mem<pinned>`, or `#hip.mem<managed>` for each result
- Converts tensors to memrefs with correct memory space
- Output is type-correct by construction - no violations to fix later

**2. ShapeInference (Slot 6b) - Intentional Transfer Insertion:**
- Detects operations requiring dynamic shape calculation from device data
- Examples: NonZero (count true elements), TopK (read k), Reshape (read shape tensor)
- For each shape-dependent operation:
  - Allocate pinned host buffer
  - Insert `hip.memcpy_d2h_async` + `hip.stream_sync`
  - Perform shape calculation on host copy
  - Use calculated shape for subsequent allocations
- This is not fixing violations - these transfers are genuinely required by the ONNX operations

**3. EliminateStreamSyncBarriers (Slot 6c):**
- Custom pass (similar to MLIR's EliminateBarriers, but for `hip.stream_sync`)
- Analyzes `MemoryEffectsOpInterface` to find conflicting memory effects
- Removes redundant `hip.stream_sync` operations
- Combines multiple syncs into the fewest barriers needed

Maps to Solution Step 3: Type system guarantees no access violations at compile time. No fix-up passes needed - code is correct from the start.

**Pipeline flow:**
```
Slot 1:  OnnxToHip Conversion       → converters respect type constraints
Slot 2:  One-Shot-Bufferize         → tensor to memref with explicit space
Slot 6a: PromoteStridedHipOperands  → preserve space in contiguous copies
Slot 6b: ShapeInference             → insert transfers for dynamic shape calculation
Slot 6c: EliminateStreamSyncBarriers → minimize synchronization overhead
Slot 6+: PoolAllocs                 → pool device memory allocations
```

---

## Design Validation Examples

This section contrasts wrong (type violations) vs right (correct-by-construction) approaches.

### Example 1: GPU Kernel with Host Memory

❌ **WRONG** - Type violation (should not compile):
```mlir
// Bufferization incorrectly returns host memory for GPU operation result
%host = hip.pad(%ctx) ins(%input, %pads) outs(%output)
        -> memref<?x?xf16, #hip.mem<host>>  // WRONG - GPU writes to host?
// Type mismatch: GPU kernel result cannot be host memory
// Error: operation 'hip.pad' result type incompatible with device-memory requirement
```

✅ **RIGHT** - Correct from start:
```mlir
// getBufferType() correctly declares device memory
%device = hip.pad(%ctx) ins(%input, %pads) outs(%output)
          -> memref<?x?xf16, #hip.mem<device>>  // Correct - GPU writes to device
// Type-correct, no fix-up needed
```

### Example 2: Host Load from Device Memory

❌ **WRONG** - Runtime SEGV (should not compile):
```mlir
%device = hip.greater(%ctx) ins(%a, %b)
          -> memref<?x?xi1, #hip.mem<device>>
%val = memref.load %device[%c0, %c0]  // CPU reads device memory → SEGV!
// This code should not exist - type constraint violation
```

✅ **RIGHT** - Shape inference inserts transfer:
```mlir
%device = hip.greater(%ctx) ins(%a, %b)
          -> memref<?x?xi1, #hip.mem<device>>
// Shape inference detects CPU needs to count elements
// Inserts intentional transfer:
%host = memref.alloc() : memref<?x?xi1, #hip.mem<pinned>>
hip.memcpy_d2h_async %ctx, %host, %device
hip.stream_sync %ctx
%val = memref.load %host[%c0, %c0]  // Safe - reading from host memory
```

### Example 3: Dynamic Shape Calculation (NonZero)

❌ **WRONG** - Cannot count without transfer:
```mlir
%cond = hip.greater(%ctx) ins(%a, %b)
        -> memref<?x?xi1, #hip.mem<device>>
// How to count true elements for NonZero output allocation?
// Cannot read %cond from CPU without transfer!
%count = ???  // Stuck - need CPU access to device data
```

✅ **RIGHT** - Converter inserts transfer for shape calculation:
```mlir
%cond = hip.greater(%ctx) ins(%a, %b)
        -> memref<?x?xi1, #hip.mem<device>>
// Shape inference inserts transfer for counting
%cond_host = memref.alloc() : memref<?x?xi1, #hip.mem<pinned>>
hip.memcpy_d2h_async %ctx, %cond_host, %cond
hip.stream_sync %ctx
// Now CPU can count to allocate output
%count = /* count true elements in cond_host */
%indices = memref.alloc(%count, %c2) : memref<?x2xi64, #hip.mem<device>>
// GPU writes output indices
hip.nonzero(%ctx) ins(%cond) outs(%indices)
```

### Example 4: Small Scalars Passed by Value

❌ **WRONG** - Device pointer for 3 integers:
```mlir
// Current hip.range implementation - device pointers for scalars
%start_dev = memref.alloc() : memref<i64, #hip.mem<device>>
%limit_dev = memref.alloc() : memref<i64, #hip.mem<device>>
%delta_dev = memref.alloc() : memref<i64, #hip.mem<device>>
hip.range(%ctx) ins(%start_dev, %limit_dev, %delta_dev)
// Problem: 3 device allocations + pointer dereferences for 3 integers
```

✅ **RIGHT** - Pass scalars by value:
```tablegen
// Better design - kernel arguments by value
def Hip_RangeOp : Hip_Op<"range"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    I64:$start,    // Scalar value, not pointer
    I64:$limit,    // Scalar value, not pointer
    I64:$delta     // Scalar value, not pointer
  );
}
```
```mlir
hip.range(%ctx) ins(%start, %limit, %delta)  // No allocation, fastest
```

### Key Takeaways

- **No type violations generated** - `getBufferType()` returns correct memory space from start
- **Intentional transfers only** - shape inference inserts D2H transfers when the ONNX op requires them
- **Type system enforces correctness** - cannot compile invalid host/device access patterns
- **70+ operations need review** - ensure each operation's `getBufferType()` and converters are correct

---

## Related Documents

- [hip-shape-inference.md](hip-shape-inference.md) - Dynamic shape inference for HIP operations
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) - Memory pooling algorithm and graph coloring
- [AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html) - LLVM address space mapping for AMDGPU
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) - Standard GPU dialect memory space patterns
