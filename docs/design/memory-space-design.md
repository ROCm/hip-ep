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
  - [Sync Coalescing](#sync-coalescing)
  - [Sync Elimination Pass](#sync-elimination-pass)
  - [Memory Effects](#memory-effects)
  - [Migration / Removal Ordering](#migration--removal-ordering)
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
    - `device`: GPU memory (`hip.get_pool`, backed by hipMalloc)
    - `host`:   host-mapped memory (`hip.get_host_scratch`, backed by hipHostMalloc)

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

The generated `inference_compute` / `main_graph` is **host (CPU) code, not a GPU
kernel**. It calls `hipMalloc`, MIOpen / hipBLASLt, and the runtime `wrap_*` /
`hipdnn_ep_*` helpers (all `void *` C ABI); the memref "pointers" are device
addresses held as plain host values. So the memory space is a **compile-time
placement tag** the passes read to decide where to insert D2H copies — not a
hardware address space in the generated code.

**Both spaces lower to LLVM address space 0.** The type converter maps
`#hip.mem<device>` and `#hip.mem<host>` to 0, so no `addrspacecast` is added and
device pointers reach the `void *` runtime ABI unchanged. This matches the
current lowering: `getMemRefAddressSpace` returns 0 for every memref, and
`AllocOpLowering` only inserts an `addrspacecast` for a non-zero space
(`lib/Conversion/HipToLLVM/MemoryLowering.cpp`).

**Type Converter Setup.** Register the mapping with `addTypeAttributeConversion`;
the callback returns the numeric space as an `IntegerAttr`:

```cpp
// Device and host both collapse to AS 0.
converter.addTypeAttributeConversion(
  [](BaseMemRefType memref, hip::MemorySpaceAttr space) -> IntegerAttr {
    return IntegerAttr::get(IntegerType::get(memref.getContext(), 64), 0);
  });
```

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

// memref whose memory space is #hip.mem<host>. Identical to Hip_DeviceMemRef
// above, except the final getKind() check compares against MemorySpaceKind::Host.
def Hip_HostMemRef : Type<CPred<"/* same as Hip_DeviceMemRef, Kind == Host */">,
  "host memref (#hip.mem<host>)">;

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

**Transfer operation** (cross-boundary): `hip.memcpy_d2h_async` has a host destination
and a device source — see [The Copy Op (D2H only)](#the-copy-op-d2h-only). It is the only
transfer op; the other direction (H2D) is a `memref.memory_space_cast`, not a copy.

**Pool operations:**

```tablegen
def Hip_GetPoolOp : Hip_Op<"get_pool"> {
  let arguments = (ins Hip_ContextType:$ctx, Index:$size);
  let results = (outs Hip_DeviceMemRef:$pool);  // always device memory
}
```

`hip.get_pool` is created after bufferization, so its result is always a memref —
hence `Hip_DeviceMemRef`, not the tensor-or-memref form. Its lowering allocates the
device pool via `hipdnn_ep_get_pool_base` (which is `hipMalloc`-backed, grow-on-demand).

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

The two directions are not the same. Reading a device value back to the host (D2H) needs
a copy and a wait. Handing host data to the GPU (H2D) needs neither on a shared-memory
(UMA) GPU. So there is one copy op (for D2H) plus `hip.stream_sync`, and H2D is a free
cast. This replaces the old `hip.readback_scalar` / `hip.readback_dim`.

### Why Not `hip.readback_scalar` / `hip.readback_dim`

```mlir
%value = hip.readback_scalar(%ctx, %device_scalar) : tensor<i64> -> i64
```

`hip.readback_scalar` and its `index`-typed sibling `hip.readback_dim` share two problems:

- They only handle a single scalar.
- They do the copy and the wait in one op, so the wait cannot be moved or removed later.

### The Copy Op (D2H only)

D2H is the one direction that needs a copy op. `hip.memcpy_d2h_async` copies a device
buffer to a host buffer and returns right away; a separate `hip.stream_sync` makes the
host wait for it. Both carry `MemoryEffectsOpInterface` so the
[Sync Elimination Pass](#sync-elimination-pass) can reason about them. Operand order is
`ctx, dst, src`.

```tablegen
def Hip_MemcpyD2HAsyncOp : Hip_Op<"memcpy_d2h_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy device memory to host without blocking";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrHostMemRef:$dst,    // host destination
    Hip_TensorOrDeviceMemref:$src   // device source
  );
}
```

The operand types fix the direction, so a wrong-direction copy is rejected at compile
time. The op lowers to `hipMemcpyAsync` with no wait; all work runs on one HIP stream,
so stream order is implicit and no completion token is needed.

### H2D Needs No Copy

Handing host data to the GPU goes the other way, and on a shared-memory (UMA) GPU it is
free. Host scratch (`hip.get_host_scratch`) is already GPU-readable, so the GPU reads it
in place — the "copy" is just a `memref.memory_space_cast` that changes the type, not the
bytes. No wait is needed: the kernel that reads the buffer runs on the same stream as
whatever filled it, so the stream already orders them. A real H2D copy only comes back on
a discrete-VRAM GPU, and even then the host does not wait.

### Usage Pattern

**Read a device value on the host (D2H)** — copy, wait, then load:

```mlir
%host = memref.alloc() : memref<i64, #hip.mem<host>>
hip.memcpy_d2h_async %ctx, %host, %device     // copies, does not wait
hip.stream_sync %ctx                          // wait before the host read
%value = memref.load %host[] : memref<i64, #hip.mem<host>>
```

**Hand host data to a kernel (H2D)** — just a cast, no copy and no wait:

```mlir
%host = memref.alloc() : memref<i64, #hip.mem<host>>
memref.store %value, %host[]                  // host fills it
%device = memref.memory_space_cast %host
        : memref<i64, #hip.mem<host>> to memref<i64, #hip.mem<device>>
hip.some_kernel(%ctx) ins(%device)            // GPU reads it in place
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

Today each host read of a device scalar goes through `hip.readback_scalar` (or
`hip.readback_dim`), which copies and waits in one step. Several reads mean several waits:

```mlir
// Range needs start/limit/delta on host
%start = hip.readback_scalar(%ctx, %device_start) : ... -> i64  // hipMemcpyAsync + hipStreamSynchronize
%limit = hip.readback_scalar(%ctx, %device_limit) : ... -> i64  // hipMemcpyAsync + hipStreamSynchronize
%delta = hip.readback_scalar(%ctx, %device_delta) : ... -> i64  // hipMemcpyAsync + hipStreamSynchronize
```

`hipStreamSynchronize` appears in many runtime call sites. The separate
`wrap_hipMemcpyD2H` runtime helper exists but is dead — no caller; the live D2H path
is `hipdnn_ep_readback_*`.

Runtime implementation (`lib/Runtime/real/memory.cpp`, `hipdnn_ep_readback_scalar`):
```cpp
void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes) {
  // hipMemcpyDefault (not DeviceToHost): the source may be host-accessible
  // (host-mapped scratch or a UMA pool), where an explicit D2H fails.
  hipMemcpyAsync(host_dst, device_scalar, num_bytes,
                 hipMemcpyDefault, stream);
  hipStreamSynchronize(stream);  // Blocks host thread
}
```

### Sync Coalescing

The goal is many copies but one wait. The `Range` reads above are three copies and three
waits; grouped, they become three copies and one wait:

```mlir
hip.memcpy_d2h_async %ctx, %host_start, %device_start   // copy, no wait
hip.memcpy_d2h_async %ctx, %host_limit, %device_limit   // copy, no wait
hip.memcpy_d2h_async %ctx, %host_delta, %device_delta   // copy, no wait
hip.stream_sync %ctx                                    // one wait for all three
```

Reaching this is an active step, not automatic. In this design `MaterializeDeviceLoads`
does it at insertion time: it emits the async copies grouped, then one shared
`hip.stream_sync` (keeping copy and sync as separate ops). The
[Sync Elimination Pass](#sync-elimination-pass) then drops any waits left over. A general
pass that reorders copies from any source to widen this is future work; because the copy
and sync stay separate ops, it can build on the same IR without changes here.

### Sync Elimination Pass

`MaterializeDeviceLoads` already pairs each `hip.memcpy_d2h_async` with a
`hip.stream_sync` before the host read, so the only remaining job is removing the
synchronizations that turn out to be unnecessary.

**Pass:** `EliminateStreamSyncsPass`

**Pipeline position:** after MaterializeDeviceLoads (Slot 6c), before PoolAllocs (Slot 6+).

```
Slot 6c: MaterializeDeviceLoads     → inserts hip.memcpy_d2h_async + hip.stream_sync
Slot 6d: EliminateStreamSyncs       → custom pass (see below)
Slot 6+: PoolAllocs                 → pools all allocations
```

This needs a custom pass, `-hip-eliminate-stream-syncs`. MLIR's
[`-eliminate-gpu-barriers`](https://mlir.llvm.org/doxygen/EliminateBarriers_8cpp_source.html)
**cannot** be reused: it matches only `gpu::BarrierOp` (its pattern is
`OpRewritePattern<gpu::BarrierOp>`), so it never sees `hip.stream_sync`. It uses
`MemoryEffectsOpInterface` only to analyze the ops *around* a barrier — not to decide
what counts as a barrier — and offers no trait or interface to opt a custom sync op in.

The custom pass follows the same idea — drop a sync when no memory access around it
actually needs the ordering it enforces (for example, an adjacent sync already covers it):

1. Match each `hip.stream_sync`.
2. Via `MemoryEffectsOpInterface`, collect the memory effects of the ops before and
   after it, stopping at the neighboring syncs.
3. Erase the sync if no before/after pair conflicts — i.e. no two accesses to aliasing
   memory where at least one is a write.

### Memory Effects

The elimination pass needs memory effects on the sync op and the async copy op.

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

**The copy op** already carries the trait in its definition above. It reads its `src`
operand and writes its `dst` operand (operand order `ctx, dst, src`, so `dst` is operand
1 and `src` is operand 2):

```cpp
void MemcpyD2HAsyncOp::getEffects(EffectsVector &effects) {
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

### Migration / Removal Ordering

Retire `hip.readback_scalar` / `hip.readback_dim` in this order, so correctness never
depends on guessed memory placement:

1. Split D2H readback into `hip.memcpy_d2h_async` + `hip.stream_sync`.
2. Add sync coalescing and the [Sync Elimination Pass](#sync-elimination-pass).
3. Remove the fused readback ops last, once the memory-space attributes and op memory
   effects are solid.

---

## Related Documents

- [hip-shape-inference.md](hip-shape-inference.md) - Dynamic shape inference for HIP operations
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) - Memory pooling algorithm and graph coloring
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) - Standard GPU dialect memory space patterns
- [MLIR Constraints](https://mlir.llvm.org/docs/DefiningDialects/Constraints/) - TableGen CPred constraint documentation
- [MLIR Operation Definition](https://mlir.llvm.org/docs/DefiningDialects/Operations/) - ODS operation definition guide
- [Aliasing guarantees on memrefs from different memory spaces](https://discourse.llvm.org/t/aliasing-guarantees-on-memrefs-from-different-memory-spaces/61154) - LLVM Discourse discussion on memory space semantics
- [Defining constraint on ops with CPred](https://discourse.llvm.org/t/defining-constraint-on-ops-with-cpred/4217) - LLVM Discourse example of custom CPred constraints
