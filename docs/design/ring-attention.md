<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Experimental Ring attention

Configure `HIP_EP_GQA_IMPL` as `existing` (default), `ring_q_resident`, or
`ring_kv_outer`. The latter two select different prefill schedules and share a
partitioned Ring decode kernel. For example, pass
`--cmake_extra_defines HIP_EP_GQA_IMPL=ring_q_resident` to `build.py`.

The ONNX GroupQueryAttention interface, HIP dialect, runtime entry signature,
and cache-update helper are unchanged. Selection happens before cache mutation.
The Ring kernels only compute attention over the prepared present cache. There
is no Page attention, distributed transport, new ONNX operator, or runtime
selector in this implementation.

Ring accepts FP16 separate Q/K/V with external RoPE, B=1, 24 query heads,
4 KV heads, head dimension 256, scale 1/16, causal attention, 1..65536 new tokens,
and at most 65664 valid cache tokens (a 64K prompt plus 128 output tokens).
It targets a concrete gfx1151 build and checks the active device. Unquantized caches use BNSD layout; Q/output use BSHD.
Both prefill variants accumulate in FP32. Decode uses up to 198144 bytes of
runtime-owned scratch; prefill uses none. Calls use the existing runtime stream.

Other geometries, architectures, quantized caches, packed inputs, in-op RoPE,
bias, windows, and sinks retain existing GQA routing. Ring builds reject softcap
because neither existing route implements it. The compiler rejects BF16 and
mismatched unquantized cache element types in Ring builds: the unchanged runtime
ABI conveys element width, which cannot distinguish BF16 from FP16. FP32 keeps
its existing route. The `existing` build retains its previous behavior.

Invalidate model caches after changing the selection. Validate with
`HIPDNN_EP_STRICT=1`, CPU fallback disabled, and temporary `HIPDNN_EP_DEBUG=1`;
Ring dispatch reports the selection and lengths. Disable debug for timing.
Run `test_gqa_ring_geometry`, `test_gqa_ring_shared_cache_transitions`, and
`test_gqa_ring_long_context` in the numeric suite against all three builds,
and the Ring-only dtype rejection LIT tests. Direct kernel checks do not
establish ONNX integration or a performance improvement.

The kernel is adapted from the METIS Ring exports, contributed under the MIT
license. WMMA score ownership and V swizzle also derive from hip-ep v8x (MIT).
