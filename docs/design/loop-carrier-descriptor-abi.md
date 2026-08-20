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

Each invocation owns two high-water device banks per carrier. Current is
read-only to the body; next is writable and may grow. Exact logical sizes and
contiguous strides live in the returned memref descriptor and are independent
of retained bank capacity. Checked multiplication rejects negative extents and
byte-size overflow. A zero-element descriptor has null data and zero logical
bytes; publishing it advances the logical bank without allocating, freeing, or
discarding retained bank capacity. A later nonzero iteration resumes normal
bank reuse or growth. Null remains an allocation failure for nonzero bytes.

`hip.loop_alloc` deliberately has no Allocate memory effect. Carrier banks are
therefore excluded from PoolAllocs and ownership-based deallocation. Ordinary
body `memref.alloc` operations remain poolable.

`hip-finalize-loop-frames` inserts `hip.loop_frame_destroy` after the last
carrier descriptor use and after any graph-output copy. An escaping nested
carrier transfers its child frame to the enclosing parent frame; explicit child
destruction unlinks it. Normal invocations therefore retain no per-run frame.
If synchronization fails during destruction, only that frame is quarantined
for a later successful synchronization or state cleanup.

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

Iteration storage, condition storage, events, descriptor sets, and carrier
banks are frame-local. Nested loops and simultaneous driver invocations on one
runtime state therefore do not share loop slots. Only the failed-destruction
quarantine list is shared and mutex-protected.
