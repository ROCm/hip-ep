<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Convolution split-K

`hip_conv` (`lib/Runtime/Kernels/hip/conv_kernel.hip`) lowers a convolution to
an im2col GEMM with `M = Cout / group`, `N = output positions` and
`K = Cin / group * kernel taps`. Some shapes have a long `K` but few output
positions, for example a 3x3 over 512 channels at 7x7 (`K = 4608`, 49 columns).
Every tile then makes only a handful of blocks, so most of the device idles
while each block walks the whole reduction. Split-K cuts the reduction across
`blockIdx.y` to turn its length into parallelism.

## How it works

- The reduction is cut into `split` slices of `chunk` taps. Each block writes
  its slice's partial sum, in fp32, to a workspace.
- `conv_splitk_reduce_kernel` then adds the slices in index order, adds the bias
  once, and writes the output. The order is fixed, so the result does not depend
  on which block finished first.
- A chunk is a multiple of the tile's `BK`, so a slice never starts in the
  middle of a staged slice of the reduction.

## The workspace

The partial sums need device memory, which `hip_conv` asks for through a
callback (`hip_conv_workspace_fn`, in `hip_custom_kernels.h`). The runtime serves
it from the session's `conv_scratch`: one buffer shared by every convolution on
the session stream, grown on demand and freed with the session. Sharing is safe
because the stream serialises convolutions.

A null callback, or a callback that returns null, is not an error: the kernel
falls back to a plan that needs no workspace.

## Scalar tiles (fp32)

`planSplitK` runs before the tile ladder, for shapes no tile covers on its own.
It applies only when `M >= 32` and `K >= 2 * kSplitMinTaps`.

- It tries the 128x128, 64x64, 32x128 and 32x32 tiles, widest first, skipping
  any tile more than twice as wide as the output.
- For each it asks `sliceSplit` for the split that takes the grid to the target
  of two blocks per CU. The first tile that reaches the target wins, otherwise
  the one that gets closest. A tile whose grid already reaches the target with no
  split ends the search with no plan, so the ladder handles that shape.
- `sliceSplit` caps the count so a slice keeps at least `kSplitMinTaps` taps and
  the partial sums fit in `kSplitMaxWorkspace`.
- The scalar split path runs for fp32 only. bf16 shares the tile kernels but is
  not split: no bf16 convolution can reach the runtime from a model yet, because
  the importer types bf16 as f32, so nothing could test it. A type is allowed by
  adding it to the kernel's split-K element-type list together with a test that
  reaches it.

## WMMA kernels (fp16)

fp16 has its own kernels and its own planner, `planFp16Split`, which uses the
same `sliceSplit` arithmetic. It does not choose a tile: it splits whatever tile
the fp16 ladder chose.

- It applies only when that tile's grid is at most one block per CU, so that the
  grid is properly starved. Above that, a split costs more in workspace traffic
  than the parallelism returns.
- The target is again two blocks per CU, a slice keeps at least `kSplitMinTaps`
  taps, and a chunk is a multiple of the WMMA depth (16).
- The window kernel (stride-1, rank-2, kernel larger than 1x1) splits over input
  channels instead of taps: a chunk is a multiple of 16 channels and a slice
  keeps at least 32.
- The dedicated fp16 1x1 kernel is dispatched before split planning and never
  splits. A 1x1 convolution reaches the WMMA split path only when the 1x1 kernel
  rejects it (for example a reduction that is not a multiple of 8, or stride 2).

## Constants

Both limits were chosen on one part, a 40-CU gfx1151 (an 8060S, which reports 20
WGPs; `deviceUnitCount()` is a WGP count on RDNA and CUs per reported unit is
derived from the wave size). Neither is derived from the hardware.

- `kSplitMinTaps = 256`. A slice shorter than this spends more on its block's
  ramp-up and on the reduce kernel than the extra blocks return. It was swept at
  64, 128 and 256 on eight fp32 convolutions (3x3 at 64, 128, 256 and 512
  channels from 56x56 down to 7x7, a stride-2 case, batch 8, and the 3-channel
  stem), with the block target held at 80. 256 was the best or within noise of
  the best on every shape, and the only one that kept the 512-channel 7x7 shape
  at 0.169 ms against 0.207 ms for 128 and 0.201 ms for 64. Each point was three
  rounds, so differences under a few percent are noise.
- The block target of two per CU was swept at 40, 80, 160 and 320 absolute blocks
  (1, 2, 4 and 8 per CU), with the minimum at 128. 320 over-split and lost
  25-55% on the five single-batch 3x3 shapes, 40 was a little slower than 80,
  and 80 and 160 were within a few percent of each other except on the
  512-channel 7x7 shape, which preferred 160. Two per CU was kept. The sweep
  fixed the target in blocks, so scaling it with the CU count is an
  extrapolation that has not been measured on any other part.
- `kSplitMaxWorkspace = 32 MB` was set a priori and never varied. It bounds the
  session scratch that holds the partial sums for the session's lifetime and the
  traffic of writing and re-reading `split` fp32 copies of the output. It is a
  footprint limit rather than a tuned value.

These sweeps were fp32 only. The fp16 kernels use the same constants and were
only compared with split-K off, and at a few targets on ResNet-50, not swept
shape by shape.

## Not yet done

Everything here was measured on gfx1151 only.

- The constants have to be re-derived per architecture before they are relied on
  elsewhere. Other gfx11.5 parts (gfx1150, gfx1152, gfx1153), gfx12 and CDNA
  differ in CU count, per-CU throughput, cache size and memory bandwidth, which
  is what they trade off against. The minimum slice length should follow per-CU
  throughput and memory latency, and the workspace cap the last-level cache or
  memory budget.
- The split planners have no GPU-free test: they live in the kernel source, and
  the numeric tests (`test/numeric/tests/test_conv.py`, `TestConvSplitK`) check
  results, not which plan ran.
