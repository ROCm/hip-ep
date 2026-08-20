<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Loop carrier descriptor-return ABI

## Contract

Each carrier contract is the compatible join of `v_init`, the source Loop
result, the body current argument, and the body yield. Rank and element type
must agree. Conflicting static extents are rejected; a dimension remains static
only when every participant is ranked and agrees on the same static value.
Otherwise it is dynamic. The seed is cast to this joined type and later shape
inference never narrows the contract from the seed.

Bufferization preserves one ranked memref result per carrier and appends a
compiler-visible `!hip.loop_frame` ownership token. A zero-trip loop returns
the borrowed `v_init` descriptor unchanged and a null frame token. A loop that
executes its body returns the final successfully published descriptor and its
owning frame; `v_init` is never mutated, freed, or adopted.

## Outlined body ABI

An outlined body has the source-level signature:

```text
(context, iter, cond_in, current_carriers..., captures..., loop_frame)
    -> (i32 status, [cond_out,] next_carriers...)
```

Carrier results remain function results and lower using MLIR's standard ranked
memref descriptor return convention. They are not promoted to out-parameters.
The generated trampoline unpacks the current descriptor set, calls the body,
and validates all returned descriptors in a separate next set. The runtime
publishes the complete set only when body status, allocation status, condition
copy, and every carrier publication succeed. The trampoline preserves the first
nonzero current-pointer or publication status; a current-pointer failure
returns before body execution, and a publication failure prevents the runtime
from swapping the next descriptor set.

Before a body return, any carrier descriptor not equal to the current borrowed
descriptor and not rooted in that body's own carrier bank is materialized into
an exact `hip.loop_alloc` plus checked copy. This includes nested-loop results
owned by child frames and arbitrary captured/body-produced descriptors, so a
parent trampoline only ever observes pass-through or its own bank.

`!hip.loop_frame` is an opaque per-invocation handle. `hip.loop_alloc` uses it
to obtain a carrier's writable next bank from
`hipdnn_ep_loop_frame_alloc(frame, carrier_index, exact_shape, rank,
element_size)`.

## Ownership and allocation

Each invocation checks out two independent device-bank blocks per carrier as
needed. Checkout is exclusive until frame destruction, so current remains
read-only, next remains writable, nested/concurrent frames cannot share a
pointer, and a checked-out block never relocates. The generic two-bank
ping-pong contract is unchanged. Exact logical sizes and contiguous strides
live in the returned memref descriptor and are independent of block capacity.
Checked multiplication rejects negative extents and byte-size overflow. A
zero-element descriptor has null data and zero logical bytes; publishing it
advances the logical bank without checking out, freeing, or discarding a
retained block. A later nonzero iteration resumes normal bank reuse or growth.
Null remains an allocation failure for nonzero bytes.

The RuntimeState owns a mutex-protected best-fit cache of independent blocks.
Checkout takes the smallest cached capacity at least as large as the request.
A miss allocates one block. Frame destruction and bank growth return blocks to
the cache only after successful stream synchronization; a synchronization
failure quarantines the owning frame and its blocks until a later successful
graph sync. Uncertain in-flight memory is never offered to another frame.
The cache keeps the largest useful blocks up to the observed concurrent block
high-water; undersized blocks displaced by synchronized growth are released,
while retained blocks live across inferences and are freed exactly once during
normal RuntimeState teardown. This cache is separate from relocatable compiler
pools and deliberately is not one contiguous arena.

`hip.loop_alloc` deliberately has no Allocate memory effect. Carrier banks are
therefore excluded from PoolAllocs and ownership-based deallocation. Ordinary
body `memref.alloc` operations remain poolable.

`hip-finalize-loop-frames` inserts `hip.loop_frame_destroy` after the last
carrier descriptor use and after any graph-output copy. An escaping nested
carrier transfers its child frame to the enclosing parent frame; explicit child
destruction unlinks it. Normal invocations therefore retain no per-run frame.
Their drained bank blocks remain in the RuntimeState cache. If synchronization
fails during destruction or growth, only that frame is quarantined for a later
successful synchronization or state cleanup.

Lifetime analysis follows conditional zero-trip/pass-through aliases
transitively through later loop `v_init` edges and view chains. Therefore
`b = loop(a)` keeps `a`'s frame alive through every possible consumer of `b`.

## Graph outputs and errors

A loop result returned directly or through a compatible cast/reshape/subview
chain is copied once from the exact final descriptor into an identity-layout
`hip.alloc_output` buffer. `hip.copy_output` supports strided sources and
checks pointers, sizes, strides, arithmetic, and copy status. Zero-byte null
pointers are valid. Frame storage is never adopted as an output.

`hip-prepare-loop-body-failures` runs after PoolAllocs and inserts a status
branch immediately after every `hip.loop_alloc`. Allocation failure returns
status plus safe current descriptors before any later body kernel or copy.
Driver failure exposes only borrowed initial descriptors and a null frame
token. Every failure sets the existing generated-graph runtime error flag.

## Concurrency and nesting

Iteration storage, condition storage, events, descriptor sets, and checked-out
carrier banks are frame-local. Nested loops and simultaneous driver
invocations on one runtime state therefore do not share loop slots or bank
pointers. The
available-block cache and failed-synchronization quarantine list are shared and
mutex-protected.

## Diagnostics and retention estimate

`HIPDNN_EP_LOOP_BANK_TRACE=1` enables low-overhead cache event lines with
best-fit hits/misses, successful allocation/free counts, active/cached/peak
bytes, quarantine bytes, and synchronization/quarantine reason. The variable
is read once when loop state is initialized; no cache diagnostics are emitted
by default.

For 27 sequential equivalent Loop frames whose two final bank capacities are
`A` and `B`, the cache retains `A + B` bytes and avoids up to 52 repeated
allocations after the first frame. Relative to allocating/finally freeing two
blocks per frame, the estimated cumulative allocation traffic recovered is
`26 * (A + B)` bytes. This is an allocation-volume estimate, not a Windows VRAM
measurement; peak/process VRAM must be reported only after CI reruns it.
