# CumSum scan test

Checks `hip_cumsum` against a CPU reference across both of its launch
strategies, and reports the per-call time of the path it selects.

`hip_cumsum` picks between a per-thread serial scan (one thread per
`(outer, inner)` slice) and a block-cooperative tiled scan (one block per slice,
scanned through LDS). The choice is made on the axis length, crossing over at
512.

The cooperative path exists because the serial one degenerates on the shape a
decoder actually uses. An attention mask is `[1, context]` scanned on axis 1,
which collapses to `outer = inner = 1`: one slice, one thread, and the whole
context scanned serially by it. That op runs once per generated token and grows
with context.

## Build and run

From this directory:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx1151 \
  test_cumsum_scan.cpp ../../../hip/cumsum_kernel.hip \
  -I../../../include -o test_cumsum_scan.exe
./test_cumsum_scan.exe            # correctness
./test_cumsum_scan.exe --bench    # per-call wall time
```

Substitute your own `--offload-arch`. Exit status is 0 when every check passes,
1 otherwise, and each failure prints the first worst index with the value found
and the value expected.

Both strategies are covered in a single run, because the case list straddles the
selection boundary: the short-axis cases stay on the serial scan and the
long-axis ones take the cooperative path.

## What it checks

Numerics are exact for the integer types and for fp16 here, so the comparison is
bit-exact rather than toleranced. The payload is 0/1 -- what a mask carries, and
small enough that every fp16 partial sum at these axis lengths is integral and
under 2048. Output buffers are poisoned before each call, so a kernel that
writes nothing fails instead of matching a zeroed buffer.

| Case | Why it can break |
|---|---|
| Axis 511 and 512 | Straddles the selection boundary in both directions. The pair also reads as the mechanism's own before/after, since the two do near-identical work and land on opposite paths. |
| Axis 256, 257, 1000, 16241 | Tile edges. Exactly one tile, one past it, and ragged tails -- where the inactive lanes of the final tile must contribute zero to the carry rather than garbage. |
| `exclusive` x `reverse`, all four | On the serial path these live inside one thread's loop. On the cooperative path the exclusive shift and the reversed traversal each have to compose with the carry threaded between tiles. |
| `inner > 1` (2x1000x3) | The cooperative kernel indexes its slice with the same stride arithmetic; a block that assumed contiguity passes every `inner == 1` case. |
| 4096x16 (vision-shaped) | Regression guard on the pre-existing path, which is still the right one for short axes and must keep being selected. |
| 8192x1024 | Many slices *and* a long axis. This is the case that decides whether the selection needs a slice-count term: the cooperative path stays well ahead even here, so it does not. |
