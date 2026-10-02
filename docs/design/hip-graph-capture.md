<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP graph capture — host control vs. wrap dispatch

**Date:** 2026-09-21
**Document Type:** Design
**Status:** Policy
**Related:** [output-allocator-design.md](output-allocator-design.md), [hip-shape-inference.md](hip-shape-inference.md), [compiler-runtime-contract.md](compiler-runtime-contract.md), [per-op-profiling.md](per-op-profiling.md)

## Purpose

Inference-time GPU–CPU stalls come from **host control values**, not from moving activations. This document records what HIP graphs can capture today and which remaining synchronized host reads are **graph breaks by ABI**, not leftover wrap bugs.

Do **not** remove `hip.alloc_output`'s host `index` extent, `hipdnn_ep_readback_*`, or `hip.readback_dim` / `hip.readback_scalar` in order to make a whole `inference_compute` capturable. Changing that is an ORT output-allocator redesign, not a wrap optimization.

## What a HIP graph cannot capture

A captured graph records kernel launches, D2D `hipMemcpy*Async`, and `hipMemsetAsync` on a stream. It does **not** record:

- `hipStreamSynchronize` / `hipEventSynchronize` used as a host wait
- Host branches on values that were read back from the GPU
- `hipMalloc` (or workspace grow) whose size comes from GPU-computed data
- Kernel grid dimensions or host-copied kernel arguments that change with live GPU scalars

A `wrap_*` is therefore capturable only when it enqueues a **fixed** sequence of kernels / D2D copies / memsets whose grids and host arguments are known from host memref shapes, dialect attributes, or compile-time constants.

Launch count may depend on host-stable values (`num_experts`, `k`, `num_tokens`, `present_seq`). Decode vs. prefill are different capture sequences.

## Graph outputs need a host extent

The generated ABI is `inference_compute(state, inputs)`. Graph outputs are allocated in-graph with `hip.alloc_output`. The EP callback and ORT require a host shape for that request; `hipdnn_ep_readback_i32` / `hipdnn_ep_readback_scalar` (`lib/Runtime/real/memory.cpp`) implement the D2H + stream sync. Lowering is `hip.readback_dim` / `hip.readback_scalar` (`lib/Conversion/HipToLLVM/ReadbackDimLowering.cpp`, `lib/Conversion/OnnxToHip/ReadbackScalar.h`).

A data-dependent extent that is a **graph output** must be a real host SSA `index` *before* `hip.alloc_output`. Keeping that count only on the GPU while the allocator ABI still needs a host `index` is not supported.

| Source | Typical host read | Why it stays |
|---|---|---|
| `onnx.Compress` selected length | Device scan (`hip.nonzero`) then `hip.readback_dim` | ORT output shape must match the kept count, not the padded capacity |
| `onnx.Range` length | Start/limit/delta on host, or readback if they are GPU scalars | Output extent is data-dependent; kernel grid and `hip.alloc_output` need it on the host |
| Dynamic Pad output extents | Host pads/axes when folded; otherwise shape reification + possible readback | Graph-output Pad still sizes `hip.alloc_output` from a host index |
| `hip.loop` trip count `M` | Host `M` for counted loops; cond readback on the dynamic path | Counted `hipdnn_ep_run_counted_loop` skips per-iter cond sync; `hipdnn_ep_run_loop` event-syncs `cond_out` every iter — also a capture break (`lib/Runtime/hipdnn_ep_runtime_loop.cpp`) |

Internal (non-output) tensors should allocate an **upper bound** and pass a device-resident count into the consumer (ScatterND-style indices, QMoE routing slots, GQA `seqlens_k`). That pattern does not apply to ORT graph outputs without an allocator ABI change (over-allocate + slice).

## Policy

1. **Fold first.** Compile-time controls stay on the host (`tensor.dim`, `tensor.from_elements`, dialect attributes). Do not round-trip a host-known scalar through the GPU.
2. **Do not use unsynchronized host loads of device memory.** GPU-computed scalars that the host must observe use `hip.readback_scalar` / `hip.readback_dim`. A bare `tensor.extract` / `memref.load` of device data is a race.
3. **Treat remaining readbacks as graph breaks.** Capture only the fixed-shape subgraph around them, or end the captured region before the readback. Do not delete the readback to “fix” capture.
4. **Data-dependent graph outputs.** Until ORT accepts over-allocate + slice, keep the synchronized host extent. Alternatives (capture break, capture only the static prefix) are runtime/session policy, not converter deletion of `hip.alloc_output`.
5. **End-of-compute sync is outside any captured graph.** Generated `inference_compute` already ends in `hipdnn_ep_stream_sync` so GPU writes are complete on return to ORT. That wait belongs after replay, not inside a captured sequence.
6. **`HIPDNN_EP_PERF=1` extra sync is measurement-only.** Do not enable it when measuring throughput or when capturing graphs. See [per-op-profiling.md](per-op-profiling.md).

## Related wrap work (not this ABI)

Ops whose **control** tensors can stay on device (fused GQA `seqlens_k`, QMoE routing, Slice/Pad clamp tables, GEMM beta scale) should not D2H those controls. That work does not replace output-allocator readbacks.
