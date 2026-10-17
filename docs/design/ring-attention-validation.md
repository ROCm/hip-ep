<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Ring attention validation

## Prefill performance and memory

Median full-model prefill latency; lower is better. Peak memory is measured within the prefill phase.

| Input tokens | Existing s | Ring-Q s | Ring-KV s | Existing peak GiB | Ring-Q peak GiB | Ring-KV peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 9.467 | 9.697 | 9.488 | 17.9956 | 17.9910 | 17.9849 |
| 32,768 | 87.723 | 105.253 | 87.023 | 34.0887 | 34.0780 | 34.0719 |
| 65,536 | 221.179 | 280.256 | 221.246 | 52.4682 | 52.4516 | 52.4456 |

## Decode performance and memory

Median full-model throughput for the 127 decode tokens after first-token selection; higher is better. Peak memory is measured within the decode phase.

| Input tokens | Existing tok/s | Ring-Q tok/s | Ring-KV tok/s | Existing peak GiB | Ring-Q peak GiB | Ring-KV peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 12.238 | 12.631 | 12.630 | 17.9955 | 17.9910 | 17.9849 |
| 32,768 | 9.615 | 10.778 | 11.021 | 34.0881 | 34.0777 | 34.0713 |
| 65,536 | 7.650 | 9.295 | 9.273 | 52.4670 | 52.4514 | 52.4448 |

At 32K, Ring-Q/Ring-KV decode improves by approximately 12%/15% over existing; at 64K, both improve by approximately 21%. Both share the same decode implementation; differences between their measurements reflect separate runs and their surrounding runtime state. Ring-Q prefill is slower, so its overall request time remains worse. Ring-KV prefill is close to baseline and gives a modest overall improvement. Prefill code was not changed by the decode optimization.

Full request latency (prefill, first-token selection and 127 subsequent tokens), median of three requests:

| Input tokens | Existing total s | Ring-Q total s | Ring-KV total s |
| --- | --- | --- | --- |
| 4,096 | 19.866 | 19.742 | 19.544 |
| 32,768 | 100.931 | 117.036 | 98.542 |
| 65,536 | 237.811 | 293.920 | 234.954 |

### Memory decomposition

For each phase, **total = model parameters + KV cache + other memory**. GiB uses 2^30 bytes; displayed values are rounded independently.

| Input tokens | Selection | Parameters GiB | KV GiB | Prefill other GiB | Prefill total GiB | Decode other GiB | Decode total GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | existing | 14.8338 | 0.2578 | 2.9040 | 17.9956 | 2.9039 | 17.9955 |
| 4,096 | ring_q_resident | 14.8338 | 0.2578 | 2.8995 | 17.9910 | 2.8994 | 17.9910 |
| 4,096 | ring_kv_outer | 14.8338 | 0.2578 | 2.8934 | 17.9849 | 2.8933 | 17.9849 |
| 32,768 | existing | 14.8338 | 2.0078 | 17.2471 | 34.0887 | 17.2465 | 34.0881 |
| 32,768 | ring_q_resident | 14.8338 | 2.0078 | 17.2364 | 34.0780 | 17.2361 | 34.0777 |
| 32,768 | ring_kv_outer | 14.8338 | 2.0078 | 17.2304 | 34.0719 | 17.2298 | 34.0713 |
| 65,536 | existing | 14.8338 | 4.0078 | 33.6267 | 52.4682 | 33.6254 | 52.4670 |
| 65,536 | ring_q_resident | 14.8338 | 4.0078 | 33.6100 | 52.4516 | 33.6098 | 52.4514 |
| 65,536 | ring_kv_outer | 14.8338 | 4.0078 | 33.6040 | 52.4456 | 33.6032 | 52.4448 |

- **Parameters:** 15,927,620,288 bytes (14.833752 GiB), the two GPU constants blobs, including the model's stored weight/constant representation.
- **KV:** allocated FP16 K and V capacity for 16 full-attention layers, `65,536 * (input_tokens + 128)` bytes. Past/present aliases count once. The capacity is the same during prefill and decode, and is identical across implementations.
- **Other:** the residual after subtracting parameters and KV from the sampled phase peak. It includes the 48 linear-attention layers' state, activations, allocator pools/reservations, scratch and other allocations. It is not isolated attention-kernel working memory.
- **Peak:** maximum observed per-process amdgpu DRM VRAM+GTT allocation during that phase across the three measured requests; target sampling interval 20 ms. This can miss brief peaks and is not an allocator's exact high-water mark. Decode can retain buffers allocated during prefill. Host RSS is separate and must not be added on this UMA system.
- **Ring kernel scratch:** zero explicit global prefill scratch; at most 387 KiB decode scratch. These figures exclude common Q/K/V/output storage, shared memory and registers. The full-model data do **not** establish a substantial memory reduction over existing GQA.

### Measurement setup

lx01: Ryzen AI MAX+ PRO 395 / Radeon 8060S (`gfx1151`, 20 CUs, 96 GiB GPU-reserved UMA), Ubuntu 24.04.4, TheRock 7.13, ONNX Runtime 1.30 and ONNX Runtime GenAI 0.14, Release builds. Batch 1, unchunked prefill, greedy generation of 128 tokens, identical seeded prompt prefixes, serial GPU execution, debug disabled, HIP-synchronized host-wall timing. One warmup plus three measured requests per configuration in a fresh process, with fresh Generator/KV/recurrent state per request.

The benchmarked kernel/test commit is `6cda37598fcd36e2da8a9992b1ee7ad15aeee52b`, based on upstream `1c9a454141c22bdddfc6071ab8c0c58cb01c36de`. Subsequent publication changes are documentation only. All final Ring cases were rerun. Unchanged 32K/64K existing controls from the same campaign were reused after matching all loaded library hashes. Numerical gates ran before timing. Independent medians need not add exactly to median total latency.

For context, isolated attention-core decode improved from the original Ring implementation from 3.173 to 0.664 ms at 32K (4.78x), and 6.156 to 1.365 ms at 64K (4.51x). Those are **not** speedups over production GQA or whole-model speedups.

## Accuracy checks

- All three builds pass **29 strict ONNX numeric cases**, with CPU fallback disabled. Decode checks every output element. Long prefill checks reference rows over all valid keys, all output finiteness and the full valid caches.
- **68 final direct-kernel checks** pass against an FP32 reference (`atol=rtol=0.005`), covering all 24 heads and all 256 dimensions, history boundaries, input scales 0.125/1/8, nondefault streams, poisoned unused cache capacity, output/scratch redzones, undersized scratch rejection, oldest/final-token dominance and NaN-head isolation.
- Full-model Ring prefill and matching-prefix first-decode logits pass cosine similarity >=0.999 and relative L2 <=0.05 against existing at 4K/32K/64K. The worst first-decode cosine is 0.999828 and worst relative L2 is 0.020067. All 128 warmup tokens match existing for both Ring selections at all three lengths. Each configuration's three measured generations repeat identically.
- Two natural prompts (arithmetic and a one-sentence sky explanation) produce identical answers with all three builds; matching-prefix logit checks also pass. This is bounded functionality evidence, not a broad quality benchmark.
- An earlier FP16-probability decode candidate passed weaker checks but failed 64K full-model logits (cosine 0.941767); it is excluded. The final decoder keeps probabilities and weighted V accumulation in FP32, with a dedicated underflow regression test.

## Test plan and new tests

Added to `test/numeric/tests/test_gqa.py`:

| Tests | Cases | Purpose |
| --- | --- | --- |
| `test_gqa_ring_geometry` | 8 | Static/dynamic Ring geometry, prefill, continuation and decode |
| `test_gqa_ring_shared_cache_transitions` | 1 | Actual GPU OrtValue past/present aliasing across query lengths 17, 1, 33, 1 |
| `test_gqa_ring_long_context` | 7 | 8K/32K/64K prefill, continuation and decode through 65,664 valid tokens |
| `test_gqa_ring_decode_regimes` | 12 | Histories 15/16/17, 31/32/33, 127/128/129, 1023/1024/1025 |
| `test_gqa_ring_decode_regimes_probability_precision` | 1 | Tiny softmax probability times large FP16 V remains representable; prevents FP16 probability underflow |

Added two Ring-build LIT rejection tests for BF16 and mixed unquantized cache dtypes. Existing compiler LIT and the three runtime unit tests (`StaticPlugins`, `OutputAllocator`, `LdsBarrierArchGate`) pass for each selection. `pre-commit run --all-files` and `git diff --check` pass.

Numeric selection used with the repository numeric harness and each installed EP/configuration:

```bash
HIPDNN_EP_STRICT=1 HIPDNN_EP_DEBUG=1 python -m pytest \
  test/numeric/tests/test_gqa.py \
  -k 'test_gqa_ring_geometry or test_gqa_ring_shared_cache_transitions or test_gqa_ring_long_context or test_gqa_ring_decode_regimes' \
  --ep-name MorphiZenExecutionProvider \
  --ep-dll "$EP_LIBRARY" --ep-option "config_file=$EP_CONFIG"
```

The added tests disable CPU fallback. `EP_LIBRARY` and `EP_CONFIG` must point to the same selected installation. The direct HIP stress harnesses and model benchmark receipts are development evidence, not new CI tests in this diff. Nine additional 128K/256K HIP-only guard-extension checks pass; they do not extend the production/ONNX admission limit.

Known existing failures: the earlier broader GQA suite had 8 passes / 14 failures, reproduced in upstream and both original Ring builds. Those unrelated failures are not claimed as fixed; this is not a claim that every repository test passes.

## Models tested and supported

- **Tested end to end:** `amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision `9b37cf56a001720ba7ce98ff80ea651b02e5931b02e5931b`. Published INT4 group-128 weights; FP16 attention activations and KV. Its 64 layers contain 16 full GQA layers and 48 linear-attention layers. Text generation was tested at 4K, 32K and 64K with all three selections.
- **Ring fast-path contract:** `gfx1151`, batch 1, 24 query heads, 4 KV heads, head dimension 256, separate FP16 Q/K/V, causal attention, external RoPE and scale 1/16. Q/output use BSHD and unquantized caches BNSD. Admit 1..65,536 new tokens and at most 65,664 valid cache tokens. Six query heads per KV head are explicitly supported.
- **Fallback:** other geometries/devices and supported packed-input, quantized-cache, in-op RoPE, bias/window/sink cases retain existing routing. Ring builds reject softcap and BF16/mismatched unquantized cache types before dtype erasure; the unchanged ABI cannot distinguish BF16 from FP16 by byte width. FP32 retains its existing route. The default build retains prior behavior.
- No other model family, vision workload, other GPU architecture, graph replay or concurrent-GPU workload was validated. Matching the operator contract establishes eligibility, not proof of full-model compatibility. No Page attention is integrated into ONNX.
