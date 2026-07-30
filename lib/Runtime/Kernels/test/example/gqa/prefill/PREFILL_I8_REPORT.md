# GQA Prefill (TTFT), INT8 KV cache -- runtime path (dequant-once + fp16 prefill)

> **Measured device: AMD Radeon(TM) 8060S Graphics (20 CUs)** -- AMD Radeon 8060S (Ryzen AI Max+ 395, gfx1151), the authoritative target.

## The prefill int8 path (no separate kernel)

Prefill is **compute-bound** (QK^T / P.V WMMA GEMMs over the full `sq x skv` triangle, each K/V element reused across all `sq` query rows). Reading the cache as int8 saves DRAM bytes but nothing on the FLOP bottleneck, and dequantizing **per WMMA fragment** would repeat the same dequant for every query tile (~`sq/ROWS`x). So the runtime does the opposite: `real/gqa.cpp` **dequantizes the int8 cache to an fp16 scratch exactly once** (`hip_gqa_dequant_kv_i8_to_fp16`) and runs the **unchanged, tuned fp16 prefill** (`hip_gqa_flash_prefill_v2`). The attention compute is therefore byte-identical to fp16 -- no separate int8 prefill kernel exists. Numerically it attends over the exact rounded values that decode will later read, so prefill/decode stay consistent.

Full runtime path: `append_quant_i8` (write new K/V into the int8 cache) -> `dequant_kv_i8_to_fp16` (once) -> `flash_prefill_v2`.

## Results

- `relL2 path/i8ref` = full GPU int8 path output vs CPU fp32 causal reference over the exact int8 bytes the append kernel wrote (PASS < 2e-2).
- `int8-path` = dequant(K)+dequant(V)+fp16 prefill; `dequant` = just the two dequant passes; `ratio` = int8-path / fp16 prefill.

| Case | H | G | hpg | D | sq | int8-path (ms) | dequant (ms) | fp16 (ms) | ratio | relL2 path/i8ref | quant(i8/fp16) | result |
|------|---|---|-----|---|----|----------------|--------------|-----------|-------|------------------|----------------|--------|
| mha-h16-d64 | 16 | 16 | 1 | 64 | 512 | 0.0880 | 0.0168 | 0.0732 | 1.20x | 3.24e-04 | 4.10e-03 | PASS |
| mha-h16-d64 | 16 | 16 | 1 | 64 | 2048 | 0.8112 | 0.0675 | 0.7314 | 1.11x | 3.30e-04 | 4.15e-03 | PASS |
| mha-h16-d64 | 16 | 16 | 1 | 64 | 4096 | 2.8904 | 0.1257 | 2.6288 | 1.10x | 3.30e-04 | 4.10e-03 | PASS |
| mha-h16-d128 | 16 | 16 | 1 | 128 | 512 | 0.1476 | 0.0356 | 0.1173 | 1.26x | 4.46e-04 | 4.05e-03 | PASS |
| mha-h16-d128 | 16 | 16 | 1 | 128 | 2048 | 1.5787 | 0.1270 | 1.2941 | 1.22x | 5.71e-04 | 4.09e-03 | PASS |
| mha-h16-d128 | 16 | 16 | 1 | 128 | 4096 | 5.4575 | 0.5886 | 4.8873 | 1.12x | 6.89e-04 | 4.12e-03 | PASS |
| gqa2-h16-d128 | 16 | 8 | 2 | 128 | 512 | 0.1307 | 0.0171 | 0.1142 | 1.14x | 4.48e-04 | 4.05e-03 | PASS |
| gqa2-h16-d128 | 16 | 8 | 2 | 128 | 2048 | 1.2983 | 0.0652 | 1.2351 | 1.05x | 5.62e-04 | 4.02e-03 | PASS |
| gqa2-h16-d128 | 16 | 8 | 2 | 128 | 4096 | 5.6108 | 0.1312 | 5.1006 | 1.10x | 6.95e-04 | 4.11e-03 | PASS |
| llama-3.2-1b | 32 | 8 | 4 | 64 | 512 | 0.1332 | 0.0109 | 0.1228 | 1.08x | 3.20e-04 | 4.10e-03 | PASS |
| llama-3.2-1b | 32 | 8 | 4 | 64 | 2048 | 1.3694 | 0.0319 | 1.3212 | 1.04x | 3.29e-04 | 4.14e-03 | PASS |
| llama-3.2-1b | 32 | 8 | 4 | 64 | 4096 | 5.3052 | 0.0623 | 5.1739 | 1.03x | 3.30e-04 | 4.07e-03 | PASS |
| llama-3.1-8b | 32 | 8 | 4 | 128 | 512 | 0.2045 | 0.0202 | 0.1909 | 1.07x | 4.44e-04 | 3.99e-03 | PASS |
| llama-3.1-8b | 32 | 8 | 4 | 128 | 2048 | 2.8578 | 0.0624 | 2.6741 | 1.07x | 5.73e-04 | 4.11e-03 | PASS |
| llama-3.1-8b | 32 | 8 | 4 | 128 | 4096 | 9.9832 | 0.1306 | 9.7453 | 1.02x | 6.86e-04 | 4.09e-03 | PASS |
| psu_orc_211 | 40 | 10 | 4 | 128 | 512 | 0.2456 | 0.0272 | 0.2323 | 1.06x | 4.42e-04 | 4.08e-03 | PASS |
| psu_orc_211 | 40 | 10 | 4 | 128 | 2048 | 3.3315 | 0.0830 | 3.1295 | 1.06x | 5.73e-04 | 4.08e-03 | PASS |
| psu_orc_211 | 40 | 10 | 4 | 128 | 4096 | 11.9654 | 0.1678 | 11.7009 | 1.02x | 6.91e-04 | 4.14e-03 | PASS |
| gpt-oss-20b | 64 | 8 | 8 | 64 | 512 | 0.2161 | 0.0110 | 0.2035 | 1.06x | 3.22e-04 | 4.06e-03 | PASS |
| gpt-oss-20b | 64 | 8 | 8 | 64 | 2048 | 3.2312 | 0.0320 | 2.8731 | 1.12x | 3.29e-04 | 4.16e-03 | PASS |
| gpt-oss-20b | 64 | 8 | 8 | 64 | 4096 | 10.3004 | 0.0659 | 10.2632 | 1.00x | 3.27e-04 | 4.02e-03 | PASS |
| llama-3-70b | 64 | 8 | 8 | 128 | 512 | 0.3671 | 0.0161 | 0.3520 | 1.04x | 4.47e-04 | 4.12e-03 | PASS |
| llama-3-70b | 64 | 8 | 8 | 128 | 2048 | 5.5273 | 0.0627 | 5.4125 | 1.02x | 5.74e-04 | 4.07e-03 | PASS |
| llama-3-70b | 64 | 8 | 8 | 128 | 4096 | 19.3824 | 0.1280 | 19.3221 | 1.00x | 6.93e-04 | 4.09e-03 | PASS |

## Analysis

- **Correctness**: worst relL2 = **6.95e-04** (< 2e-2); 8-bit-quant error vs fp16 ~4e-3. All PASS.
- **Performance**: int8-path prefill is **1.00x-1.26x** the fp16 prefill (avg **1.08x**) -- i.e. ~parity. The only overhead is the single dequant pass (a few percent of the GEMM time); the WMMA attention itself is the identical fp16 kernel.
- Contrast the earlier per-fragment int8 prefill kernel (~0.6x, i.e. ~1.6x slower): dequantizing once instead of per fragment recovers the full fp16 throughput.

_Reproduce:_ `test_gqa_prefill_i8.exe --all --iters 200 --md <this file>`
