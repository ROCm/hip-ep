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
- [Attribute Definition](#attribute-definition)
  - [Memory Space Attribute](#memory-space-attribute)
  - [Default Memory Space](#default-memory-space)
  - [LLVM Address Space Mapping](#llvm-address-space-mapping)
- [Pipeline Enforcement](#pipeline-enforcement)
  - [Bufferization (Slot 2)](#bufferization-slot-2)
  - [Strided Operand Promotion (Slot 6a)](#strided-operand-promotion-slot-6a)
  - [Pool Allocation (Slot 6+)](#pool-allocation-slot-6)
  - [Converters](#converters)
- [Memory Space Tracking](#memory-space-tracking)
  - [Custom Type Constraints](#custom-type-constraints)
  - [Using Type Constraints in Operations](#using-type-constraints-in-operations)
  - [Verification Examples](#verification-examples)
- [Memory Transfer Operations](#memory-transfer-operations)
- [Synchronization Optimization](#synchronization-optimization)
  - [Current State](#current-state)
  - [Sync Elimination Pass](#sync-elimination-pass)
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

### Consequence 2: Manual Synchronization That Cannot Be Minimized

Converters must remember to insert the device-to-host copy by hand:

```cpp
// In RangeConversion.cpp - manual readback
Value limit = readbackScalarToHost(builder, loc, ctx, limitTensor);
```

- Nothing enforces it. A direct `tensor.extract` compiles but reads stale or invalid data at runtime.
- Because the compiler does not know which buffers are device and which are host, it cannot place the synchronization only where it is needed. It must either synchronize conservatively after every potentially-device op, or risk a crash.

---

## Solution Overview

Use memory space attributes on memref types to track device vs host placement.

**Core idea:**
```mlir
// The type records where the buffer lives.
%device = hip.get_pool(%ctx, %size) : memref<?xi8, #hip.mem<device>>

// A small scalar that the host must read/write is placed in host memory:
%shape = memref.alloc() : memref<i64, #hip.mem<host>>

memref.store %value, %shape[%c0]      // fine: %shape is host memory
// memref.store %value, %device[%c0]  // would be wrong: %device is GPU memory,
//                                     // not host-writable on a real-device GPU
```

**How it solves the problem:**
- The type carries placement, so it no longer has to be guessed from the surrounding ops.
- HIP ops constrain their operands to the space they need, so an operand in the wrong space is caught at compile time.
- A pass can insert device-to-host copies only where the space actually changes.
- Synchronization can be limited to those crossings instead of after every op.

**Key components:**
1. A memory space attribute (`#hip.mem<device|host>`).
2. Device is the default space; host is requested explicitly.
3. HIP ops constrain operands and results to the space they require.
4. Lowering dispatches on the space.

---

## Attribute Definition

### Memory Space Attribute

The kind is an enum attribute, so `#hip.mem<device>` / `#hip.mem<host>` parse directly:

```tablegen
def HIP_Device : I32EnumAttrCase<"Device", 0, "device">;
def HIP_Host   : I32EnumAttrCase<"Host",   1, "host">;

def HIP_MemorySpaceKind
    : I32EnumAttr<"MemorySpaceKind", "HIP memory space kind",
                  [HIP_Device, HIP_Host]> {
  let cppNamespace = "::mlir::hip";
}

def HIP_MemorySpaceAttr : AttrDef<HIP_Dialect, "MemorySpace"> {
  let mnemonic = "mem";
  let summary = "HIP memory space attribute";
  let description = [{
    Where a HIP buffer lives:
    - `device`: GPU memory (hip.get_pool -> hipMalloc)
    - `host`:   host-mapped memory (hip.get_host_scratch -> hipHostMalloc)

    Syntax: `#hip.mem<device>` or `#hip.mem<host>`.
  }];
  let parameters = (ins EnumParameter<HIP_MemorySpaceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}
```

The `I32EnumAttr` generates the matching `MemorySpaceKind` C++ enum (`Device`, `Host`)
in the `mlir::hip` namespace.

### Default Memory Space

The default space is **device**. Bufferization gives every buffer it creates the
device space (see [Bufferization](#bufferization-slot-2)), because the GPU pool is the
common case. Host memory is the exception and is asked for explicitly:

```mlir
%d = memref.alloc() : memref<i64, #hip.mem<device>>  // device (the default)
%h = memref.alloc() : memref<i64, #hip.mem<host>>    // host (explicit)
```

Device-by-default keeps the design simple and does not fight the upstream MLIR passes
that create plain `memref<...>` temporaries with no space.

### LLVM Address Space Mapping

Maps to AMDGPU LLVM address spaces following MLIR GPU dialect conventions.

**AMDGPU Address Spaces (Reference):**

| Address Space | Name | HIP Memory Space | Usage |
|---|---|---|---|
| 0 | Generic/Flat | `#hip.mem<host>` | Host-accessible memory |
| 1 | Global | `#hip.mem<device>` | Device global memory (GPU-only) |
| 3 | Local | — | Workgroup shared memory (LDS) |
| 4 | Constant | — | Read-only constant data |
| 5 | Private | — | Per-thread scratch |

**Currently supported:** AS 0 (host) and AS 1 (device) only.

**Type Converter Setup:**

```cpp
// In LLVMTypeConverter setup
converter.addMemRefAttrConversion(
  [](Attribute attr, MemRefType type) -> std::optional<unsigned> {
    if (auto space = dyn_cast<MemorySpaceAttr>(attr)) {
      switch (space.getKind()) {
        case MemorySpaceKind::Device:
          return 1u;  // AMDGPU Global (device memory)
        case MemorySpaceKind::Host:
          return 0u;  // AMDGPU Generic/Flat (host-accessible)
      }
    }
    return std::nullopt;
  });
```

**Device memory (`#hip.mem<device>`):**
- Maps to AS 1 (AMDGPU Global)
- GPU kernels access device memory in AS 1
- Used by `hipMalloc` allocations

**Host memory (`#hip.mem<host>`):**
- Maps to AS 0 (AMDGPU Generic/Flat)
- Host-accessible via `hipHostMalloc`
- Uses flat addressing instructions
- Accessible from both CPU and GPU

---

## Pipeline Enforcement

Every pass that creates or transforms memrefs must generate valid memory spaces.

### Bufferization (Slot 2)

Two complementary mechanisms:

**1. Default space (covers most ops).** Set `defaultMemorySpaceFn = device` on the
`OneShotBufferize` options. This applies to every leaf (allocs, `tensor.empty`,
function boundaries) and, because the default `getBufferType` makes a DPS result
follow its tied `outs` operand, keeps result/operand buffer types equal with no
per-op code:

```cpp
opts.defaultMemorySpaceFn = [](TensorType) -> std::optional<Attribute> {
  return MemorySpaceAttr::get(/*ctx*/, MemorySpaceKind::Device);
};
```

**2. Per-op `getBufferType` (only ops that deviate from the default).** Some ops must
place a result in a *specific* space regardless of the default — e.g. an op that
always produces host-staged output. Override `getBufferType` on that op's
bufferization model:

```cpp
FailureOr<bufferization::BufferLikeType>
getBufferType(Operation *op, Value value,
              const bufferization::BufferizationOptions &options,
              const bufferization::BufferizationState &state,
              SmallVectorImpl<Value> &invocationStack) const {
  auto t = cast<RankedTensorType>(value.getType());
  auto space = MemorySpaceAttr::get(op->getContext(), /*this op's space*/);
  return cast<bufferization::BufferLikeType>(MemRefType::get(
      t.getShape(), t.getElementType(), /*layout=*/{}, space));
}
```

Use the override sparingly — only where the op's space differs from `device`. A DPS
op that overrides its result space must ensure its tied `outs` operand resolves to the
same space, or bufferization will see mismatched buffer types.

### Strided Operand Promotion (Slot 6a)

Preserve memory space when creating contiguous copies:

```cpp
// PromoteStridedHipOperands.cpp
auto deviceSpace = srcType.getMemorySpace();  // Inherit from source
auto contiguousType = MemRefType::get(srcType.getShape(),
                                      srcType.getElementType(),
                                      AffineMap(),
                                      deviceSpace);
auto allocOp = memref::AllocOp::create(builder, loc, contiguousType, dynSizes);
```

Temporary allocations inherit source memory space.

### Pool Allocation (Slot 6+)

Mark pool buffer with device space:

```cpp
// PoolAllocs.cpp
auto deviceSpace = MemorySpaceAttr::get(context, MemorySpaceKind::Device);
auto poolType = MemRefType::get({ShapedType::kDynamic},
                                builder.getIntegerType(8),
                                AffineMap(),
                                deviceSpace);
```

Pool is always device memory. All `memref.view` operations inherit device space.

### Converters

Explicitly allocate host memory when needed:

```cpp
// RangeConversion.cpp - scalar needs host access
auto hostSpace = MemorySpaceAttr::get(context, MemorySpaceKind::Host);
auto scalarType = MemRefType::get({}, builder.getI64Type(),
                                  AffineMap(), hostSpace);
auto hostAlloc = memref::AllocOp::create(builder, loc, scalarType, {});
```

No heuristics. The space is explicit at the allocation site.

---
## Memory Space Tracking

Operations enforce memory space requirements using TableGen type constraints.

### Custom Type Constraints

HIP ops are defined once but carry **tensor** types before bufferization (Slot 2)
and **memref** types after it — this is exactly why operands today use
`Hip_TensorOrMemRef` (`AnyTypeOf<[AnyRankedTensor, AnyMemRef]>`). The memory-space
attribute only exists on the memref form, so the space constraints must follow the
same dual shape: accept any ranked tensor (pre-bufferization, no space yet) **or** a
memref carrying the required space (post-bufferization).

Define the memref-with-space predicates, then wrap them so a ranked tensor also
satisfies the constraint:

```tablegen
// In HipBase.td

// memref whose memory space is #hip.mem<device>
def Hip_DeviceMemRef : Type<
  CPred<"::llvm::isa<::mlir::MemRefType>($_self) && "
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace() && "
        "::llvm::isa<::mlir::hip::MemorySpaceAttr>("
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()) && "
        "::llvm::cast<::mlir::hip::MemorySpaceAttr>("
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getKind() == "
        "::mlir::hip::MemorySpaceKind::Device">,
  "device memref (#hip.mem<device>)"
>;

// memref whose memory space is #hip.mem<host>
def Hip_HostMemRef : Type<
  CPred<"::llvm::isa<::mlir::MemRefType>($_self) && "
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace() && "
        "::llvm::isa<::mlir::hip::MemorySpaceAttr>("
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()) && "
        "::llvm::cast<::mlir::hip::MemorySpaceAttr>("
        "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getKind() == "
        "::mlir::hip::MemorySpaceKind::Host">,
  "host memref (#hip.mem<host>)"
>;

// Pre-bufferization tensor OR post-bufferization device/host memref.
// Mirrors Hip_TensorOrMemRef so the same op definition verifies in both phases.
def Hip_TensorOrDeviceMemref : AnyTypeOf<[AnyRankedTensor, Hip_DeviceMemRef],
                             "ranked tensor or device memref (#hip.mem<device>)">;

def Hip_TensorOrHostMemRef : AnyTypeOf<[AnyRankedTensor, Hip_HostMemRef],
                           "ranked tensor or host memref (#hip.mem<host>)">;

def AnyHipMemRef : AnyTypeOf<[Hip_TensorOrDeviceMemref, Hip_TensorOrHostMemRef], "HIP tensor or memref">;
```

Use the `Hip_TensorOr*` forms on op operands (they accept a tensor before
bufferization and a space-tagged memref after). Use the plain `Hip_DeviceMemRef` /
`Hip_HostMemRef` forms where the value is always a memref (for example the result of
`hip.get_pool`). The space is only checked once the type is a memref, so converters
build ops before bufferization exactly as they do with `Hip_TensorOrMemRef` today.

### Using Type Constraints in Operations

**Device-only operations** (GPU kernels):

```tablegen
def Hip_PadOp : Hip_Op<"pad"> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrDeviceMemref:$input,   // device
    AnyHipMemRef:$pads,
    Hip_TensorOrDeviceMemref:$output   // device
  );
}
```

A GPU op constrains its operands to device memory.

**Transfer operations** (cross-boundary): one operand is host and the other is device.
The `hip.memcpy_*` ops are written this way — see [The Copy Ops](#the-copy-ops). Their
lowering reads the operand spaces to pick the right runtime copy.

**Pool operations:**

```tablegen
def Hip_GetPoolOp : Hip_Op<"get_pool"> {
  let arguments = (ins Hip_ContextType:$ctx, Index:$size);
  let results = (outs Hip_DeviceMemRef:$pool);  // always device memory
}
```

`hip.get_pool` is created after bufferization, so its result is always a memref —
hence `Hip_DeviceMemRef`, not the tensor-or-memref form. Its lowering calls `hipMalloc`.

### Verification Examples

**Correct usage:**
```mlir
// Device memory for a GPU op
%device = hip.get_pool(%ctx, %size) : memref<?xi8, #hip.mem<device>>
hip.pad(%ctx) ins(%input) outs(%device)  // ok: device space matches

// Host memory for a scalar
%host = memref.alloc() : memref<i64, #hip.mem<host>>
memref.store %value, %host[]             // ok: host store into host memory
```

**Compile-time errors:**
```mlir
// Wrong space on a HIP op result (get_pool must be device)
%bad = hip.get_pool(%ctx, %size) : memref<?xi8, #hip.mem<host>>
// error: result type 'memref<?xi8, #hip.mem<host>>' doesn't match constraint 'Hip_DeviceMemRef'

// Wrong space on a HIP op operand
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.pad(%ctx) outs(%host)
// error: operand type 'memref<i64, #hip.mem<host>>' doesn't match constraint 'Hip_TensorOrDeviceMemref'
```

These errors come from the operand/result type constraints, checked automatically by
TableGen-generated code — no hand-written `verify()` method, and the message names the
expected constraint. (Builtin `memref.load` / `memref.store` are not space-checked this
way; a host access of a device buffer is handled by the transfer pass below, not by a
verifier.)

---

## Memory Transfer Operations

The scalar-only `hip.readback_scalar` is replaced by directional copy ops that work for
any size.

### Why Not `hip.readback_scalar`

```mlir
%value = hip.readback_scalar(%ctx, %device_scalar) : tensor<i64> -> i64
```

- It is a special-case op for one task, and it only handles scalars.
- It bundles the copy and the synchronization together, so the synchronization cannot
  be moved or removed later.

### The Copy Ops

One op per direction, each in a blocking and a non-blocking form. The non-blocking
(`*_async`) form does the copy but does not wait; a separate `hip.stream_sync` orders
it against the host read. All four carry `MemoryEffectsOpInterface` (used by the
[Sync Elimination Pass](#sync-elimination-pass)). Operand order is `ctx, dst, src`.

```tablegen
def Hip_MemcpyD2HOp : Hip_Op<"memcpy_d2h", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy device memory to host and block until it completes";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrHostMemRef:$dst,    // host destination
    Hip_TensorOrDeviceMemref:$src   // device source
  );
}

def Hip_MemcpyH2DOp : Hip_Op<"memcpy_h2d", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy host memory to device and block until it completes";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrDeviceMemref:$dst,  // device destination
    Hip_TensorOrHostMemRef:$src     // host source
  );
}

def Hip_MemcpyD2HAsyncOp : Hip_Op<"memcpy_d2h_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy device memory to host without blocking";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrHostMemRef:$dst,    // host destination
    Hip_TensorOrDeviceMemref:$src   // device source
  );
}

def Hip_MemcpyH2DAsyncOp : Hip_Op<"memcpy_h2d_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy host memory to device without blocking";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrDeviceMemref:$dst,  // device destination
    Hip_TensorOrHostMemRef:$src     // host source
  );
}
```

The operand type constraints fix the direction, so a wrong-direction copy is rejected
at compile time. The `*_async` ops lower to `hipMemcpyAsync` with no wait; all transfers
run on the single HIP stream, so stream order is implicit and no completion token is
needed.

### Usage Pattern

**Blocking copy** — `memcpy_d2h` waits, so the value is ready right after:

```mlir
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.memcpy_d2h %ctx, %host, %device          // copies and blocks
%value = memref.load %host[] : memref<i64, #hip.mem<host>>
```

**Non-blocking copy** — `memcpy_d2h_async` returns immediately, so a `hip.stream_sync`
is needed before the host reads:

```mlir
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.memcpy_d2h_async %ctx, %host, %device     // copies, does not wait
hip.stream_sync %ctx                          // wait before the host read
%value = memref.load %host[] : memref<i64, #hip.mem<host>>
```

### Automatic Insertion Pass

**Pass:** `MaterializeDeviceLoadsPass`

**Pipeline position:** After bufferization (Slot 2), before pool allocation (Slot 6+)

Specifically: Slot 6c, after `MaterializeHostScalarsPass`.

```
Slot 2:  One-Shot-Bufferize        → creates memref.alloc, memref.load
Slot 6a: PromoteStridedHipOperands → may create more memref.load
Slot 6b: MaterializeHostScalars    → redirects host-fed scalar allocs out of the device pool
Slot 6c: MaterializeDeviceLoads    → turns host loads of device memory into copy + load
Slot 6+: PoolAllocs                → pools all allocations
```

**Transformation:**

Before:
```mlir
%device = memref<i64, #hip.mem<device>>
%value = memref.load %device[] : memref<i64, #hip.mem<device>>
// crashes: the host cannot read device memory on a real-device GPU
```

After:
```mlir
%device = memref<i64, #hip.mem<device>>

// the pass inserts:
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.memcpy_d2h_async %ctx, %host, %device
hip.stream_sync %ctx

%value = memref.load %host[] : memref<i64, #hip.mem<host>>
// safe: reads from host memory
```

**Algorithm:**
1. Walk all `memref.load` operations.
2. Read the loaded memref's memory space attribute; act only when it is `#hip.mem<device>` (host memrefs are left alone).
3. Allocate a host buffer with the same element type.
4. Insert `hip.memcpy_d2h_async` before the load.
5. Insert `hip.stream_sync` before the load (one per host read site).
6. Point the load at the host buffer.

This one pass inserts both the copy and the synchronization, so no separate sync pass
is needed. Converters just emit a plain `memref.load`; the pass does the rest. Any
synchronization it adds that turns out to be unnecessary is removed by the
[Sync Elimination Pass](#sync-elimination-pass).

**Why this position:**
- After bufferization: the `memref.load` ops exist.
- After strided promotion: all loads have been created.
- Before pool allocation: the new host buffers are pooled correctly.

---

## Synchronization Optimization

### Current State

Every `hip.memcpy_d2h` synchronizes the stream immediately:

```mlir
// Range needs start/limit/delta on host
hip.memcpy_d2h %ctx, %host_start, %device_start  // hipMemcpyAsync + hipStreamSynchronize
hip.memcpy_d2h %ctx, %host_limit, %device_limit  // hipMemcpyAsync + hipStreamSynchronize
hip.memcpy_d2h %ctx, %host_delta, %device_delta  // hipMemcpyAsync + hipStreamSynchronize
```

Current codebase: 72 occurrences of `hipStreamSynchronize` across 24 files.

Runtime implementation (`lib/Runtime/real/memory.cpp:117-138`):
```cpp
void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes) {
  hipMemcpyAsync(host_dst, device_scalar, num_bytes,
                 hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);  // Blocks host thread
}
```

### Sync Elimination Pass

`MaterializeDeviceLoads` already pairs each `hip.memcpy_d2h_async` with a
`hip.stream_sync` before the host read, so the only remaining job is removing the
synchronizations that turn out to be unnecessary.

**Pipeline position:** after MaterializeDeviceLoads (Slot 6c), before PoolAllocs (Slot 6+).

```
Slot 6c: MaterializeDeviceLoads     → inserts hip.memcpy_d2h_async + hip.stream_sync
Slot 6d: -eliminate-gpu-barriers    → reuse MLIR's existing pass
Slot 6+: PoolAllocs                 → pools all allocations
```

Reuse MLIR's existing [`-eliminate-gpu-barriers`](https://mlir.llvm.org/doxygen/EliminateBarriers_8cpp_source.html)
pass. It works on any op that implements `MemoryEffectsOpInterface`, not just
`gpu.barrier`, so no custom pass is needed. For each barrier it compares the memory
effects before and after, and removes the barrier when there is no conflict.

### Memory Effects

The elimination pass needs memory effects on the sync op and the async copy ops.

**hip.stream_sync** (the barrier) acts as a full barrier — it reads and writes all
memory:

```tablegen
def Hip_StreamSyncOp : Hip_Op<"stream_sync", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Block until all work on the HIP stream completes";
  let arguments = (ins Hip_ContextType:$ctx);
}
```

```cpp
void StreamSyncOp::getEffects(EffectsVector &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),  DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), DefaultResource::get());
}
```

**The copy ops** already carry the trait in their definitions above. Each reads its
`src` operand and writes its `dst` operand (operand order `ctx, dst, src`, so `dst` is
operand 1 and `src` is operand 2). D2H and H2D have the same shape:

```cpp
// hip.memcpy_d2h and hip.memcpy_d2h_async (H2D is the mirror image):
void MemcpyD2HOp::getEffects(EffectsVector &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(2),   // src (device)
                       DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       &getOperation()->getOpOperand(1),   // dst (host)
                       DefaultResource::get());
}
```

`memref.load` / `memref.store` already carry Read/Write effects, and DPS compute ops
get theirs from the shared `emitDpsMemoryEffects` helper.

---

## Related Documents

- [hip-shape-inference.md](hip-shape-inference.md) - Dynamic shape inference for HIP operations
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) - Memory pooling algorithm and graph coloring
- [AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html) - LLVM address space mapping for AMDGPU
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) - Standard GPU dialect memory space patterns
- [MLIR Constraints](https://mlir.llvm.org/docs/DefiningDialects/Constraints/) - TableGen CPred constraint documentation
- [MLIR Operation Definition](https://mlir.llvm.org/docs/DefiningDialects/Operations/) - ODS operation definition guide
- [Aliasing guarantees on memrefs from different memory spaces](https://discourse.llvm.org/t/aliasing-guarantees-on-memrefs-from-different-memory-spaces/61154) - LLVM Discourse discussion on memory space semantics
- [Defining constraint on ops with CPred](https://discourse.llvm.org/t/defining-constraint-on-ops-with-cpred/4217) - LLVM Discourse example of custom CPred constraints

