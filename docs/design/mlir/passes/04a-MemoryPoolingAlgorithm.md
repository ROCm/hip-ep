<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Memory Pooling Algorithm

**Date:** 2026-03-02
**Document Type:** Implementation
**Status:** Draft
**Related:** [04-MemoryPooling.md](04-MemoryPooling.md)

---

## Overview

Greedy best-fit strip packing for assigning pool offsets to `memref.alloc` ops. Buffers with non-overlapping lifetimes can share the same pool location.

---

## Data Structures

```cpp
struct AllocInfo {
  memref::AllocOp op;
  uint64_t sizeBytes;
  unsigned defOrder;      // program-order index of defining op
  unsigned lastUseOrder;  // program-order index of last transitive use
  uint64_t poolOffset;    // assigned by strip packing
};
```

Program order is a flat integer index assigned by walking the function with `funcOp.walk(...)`.

---

## Algorithm Steps

### 1. Collect Static Allocs

Walk the function. For each `memref.alloc` with a static shape and supported element type (float or integer):
- Compute `sizeBytes = product(dims) × element_bytes`
- Record `defOrder` from the program-order map
- Compute `lastUseOrder` by transitive use-def traversal (BFS over `Value.getUsers()` including results of user ops)

Dynamic shapes and unsupported element types are skipped; their `memref.alloc` ops remain unchanged.

### 2. Liveness: Transitive Use-Def Traversal

```
lastUseOrder(val):
  last = order[val.getDefiningOp()]
  worklist = {val}
  while worklist not empty:
    v = worklist.pop()
    for user in v.getUsers():
      last = max(last, order[user])
      for result in user.getResults():
        worklist.push(result)
  return last
```

This tracks the liveness end through `memref.view`, `memref.subview`, and similar ops that propagate the buffer value.

### 3. Interval Overlap Check

Two buffers `a` and `b` have overlapping lifetimes if:

```
!(a.lastUseOrder < b.defOrder || b.lastUseOrder < a.defOrder)
```

Overlapping buffers cannot occupy the same pool location.

### 4. Strip Packing (Largest-First)

```
Sort allocs by sizeBytes descending.
hiWater = 0

for each alloc (largest first):
  candidates = {0} ∪ {start, end of each already-placed alloc}
  candidates += hiWater  // fallback: append

  bestOffset = alignUp(hiWater, 256)
  bestGap = ∞

  for each candidate c in sorted order:
    aligned = alignUp(c, 256)
    end = aligned + cur.sizeBytes

    conflict = false
    for each placed alloc j:
      if lifetimes overlap(cur, j):
        if ranges overlap([aligned, end), [j.offset, j.offset+j.size)):
          conflict = true; break

    if not conflict:
      gap = aligned - c        // alignment padding wasted
      if gap < bestGap:
        bestGap = gap
        bestOffset = aligned

  cur.poolOffset = bestOffset
  hiWater = max(hiWater, bestOffset + cur.sizeBytes)
```

Alignment is 256 bytes (`kPoolAlignment`), matching the hipdnn-ep pool allocator.

### 5. Emit IR Replacements

After packing:
1. Emit `hip.get_pool(%ctx) : memref<?xi8, 1>` once at function entry
2. For each alloc in original order:
   - Emit `arith.constant <offset> : index`
   - Emit `memref.view %pool[%offset][] : memref<original-type, 1>`
   - Replace all uses of the `memref.alloc` result with the `memref.view` result
   - Erase the `memref.alloc`
3. Annotate module: `"hipdnn.pool_size" = <final pool size>`

---

## Example

**Input (4 buffers):**

```
buf0: 12.8MB  defOrder=0  lastUseOrder=4
buf1: 12.8MB  defOrder=1  lastUseOrder=5
buf2:  3.2MB  defOrder=5  lastUseOrder=7
buf3:  3.2MB  defOrder=6  lastUseOrder=8
```

**Overlap matrix:**
- buf0 ↔ buf1: overlap (both live 1–4) → cannot share
- buf0 ↔ buf2: no overlap (buf0 ends at 4, buf2 starts at 5) → can share
- buf1 ↔ buf2: overlap (both live at 5) → cannot share
- buf1 ↔ buf3: no overlap → can share
- buf2 ↔ buf3: overlap → cannot share

**Sorted order:** buf0, buf1, buf2, buf3

**Assignment:**
1. buf0 (12.8MB) → offset 0. hiWater = 12,845,056
2. buf1 (12.8MB) → offset 0 conflicts with buf0; try offset 12,845,056 → no conflict. hiWater = 25,690,112
3. buf2 (3.2MB) → offset 0 aligned: no conflict with buf0 (no lifetime overlap), no conflict with buf1 (non-overlapping ranges). Assign 0. hiWater unchanged.
4. buf3 (3.2MB) → offset 0: conflicts with buf2 (overlap in lifetime). Offset 3,211,264: no conflict. Assign 3,211,264. hiWater unchanged.

**Result:** pool size = 25,690,112 bytes (25.6 MB) vs 32.0 MB without pooling = 20% savings.

---

## Complexity

- Sorting: O(B log B)
- Packing loop: O(B² × C) where C = number of candidate offsets ≤ 2B
- Overall: O(B³) — acceptable for B < 1000 buffers

---

## Related Documents

- [04-MemoryPooling.md](04-MemoryPooling.md) - Pass overview and IR transformation
