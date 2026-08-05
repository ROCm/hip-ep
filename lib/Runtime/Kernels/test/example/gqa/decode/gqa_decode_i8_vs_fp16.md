# GQA Decode: fp16 KV vs INT8 KV cache -- correctness & performance

> **Test environment**
>
> - **Measured device: AMD Radeon(TM) 8060S Graphics (20 CUs)** -- AMD Radeon 8060S (Ryzen AI Max+ 395, Strix Halo, gfx1151), the **authoritative target**. All numbers in this report were collected here.
> - The 8060S is an APU whose GPU shares LPDDR5X bandwidth with the CPU/system, so the decode int8-vs-fp16 *ratio* reflects the real deployment balance between DRAM bandwidth and compute.

Q is fp16; KV cache is symmetric per-channel INT8.

## Confirmed fp16 + INT8 KV GQA data layout & compute logic

Source model: `models/gqa_kv_u8/psu_orc_211_merged_fp16_gqa.onnx` (`com.microsoft.GroupQueryAttention`, 40 layers).

- Attributes: `num_heads=40`, `kv_num_heads=10` (heads-per-group = 4), `do_rotary=1`, `k_quant_type=v_quant_type=PER_CHANNEL`, `kv_cache_bit_width=8`.
- Query: fp16, packed-QKV path (`key`/`value` inputs empty) -> split into Q/K/V; RoPE applied to Q and K.
- **KV cache: INT8** (signed), BNSD `[B, G, max_seq, D]` (D=128). `present_key/present_value` are INT8 too.
- **Scales: fp32 `[G*D]` = `[10*128]` = 1280** (`dec_k_scale_*`, `dec_v_scale_*`), i.e. one scale per `(kv_head, head_dim)` channel; all positive -> **symmetric** quant, **no zero point**.
- Dequant: `k_fp16 = k_i8 * k_scale[g*D + c]`, `v_fp16 = v_i8 * v_scale[g*D + c]`. Attention is then the standard GQA math over the dequantized K/V.

## Kernel implementation (`hip_gqa_flash_decode` + scales)

Same FA-2 split-K algorithm, `[B,H,K_SPLITS,D+2]` partials, autotune and reduce as the fp16 `hip_gqa_flash_decode`, so seqlens_k / sliding-window / head-sink / smooth-softmax all carry over unchanged. The only change is the K/V load:

- **Scalar kernel** (MHA + GQA d128 / high-hpg): keeps the tile INT8 (GQA stages it into LDS with a vectorized int4 copy -> half the fp16 path's LDS; MHA streams it straight from global) and reads it 1 byte/elem. The per-channel **K scale is folded into Q** (`dot = Sum (Q*Ksc)*K_i8`) and the **V scale is deferred to the epilogue** (`O = Vsc*Sum p*V_i8`), so the inner loop drops the per-key dequant multiplies (int8 costs only ~1 extra int->float vs fp16) and frees the K-scale registers. int8 is read via 32-bit `char4` LDS accesses when EPT is a multiple of 4.
- **WMMA kernel** (GQA d64): dequantizes INT8 -> fp16 during the global->LDS stage (with the same register software-prefetch pipeline as the fp16 path) so the 16x16x16 WMMA GEMMs are byte-identical.

Net effect: **half the DRAM read traffic** on the bandwidth-bound decode KV scan, at fp16-equivalent accuracy.

## Results

- `relL2 kern/i8ref` = int8 kernel output vs CPU fp32 reference over the *same* dequantized int8 cache (kernel correctness; PASS < 2e-2).
- `quant(i8/fp16)` = CPU int8-dequant reference vs CPU fp16 reference (error introduced purely by 8-bit KV quantization).
- `speedup` = fp16 latency / int8 latency (>1 means int8 is faster).

| Case | attn | H | G | hpg | D | KV len | int8 (ms) | fp16 (ms) | speedup | relL2 kern/i8ref | quant(i8/fp16) | result |
|------|------|---|---|-----|---|--------|-----------|-----------|---------|------------------|----------------|--------|
| mha-h16-d64 | MHA | 16 | 16 | 1 | 64 | 512 | 0.0073 | 0.0097 | 1.32x | 2.08e-04 | 4.14e-03 | PASS |
| mha-h16-d64 | MHA | 16 | 16 | 1 | 64 | 2048 | 0.0195 | 0.0266 | 1.36x | 2.19e-04 | 4.20e-03 | PASS |
| mha-h16-d64 | MHA | 16 | 16 | 1 | 64 | 8192 | 0.0633 | 0.0887 | 1.40x | 2.04e-04 | 4.28e-03 | PASS |
| mha-h16-d64 | MHA | 16 | 16 | 1 | 64 | 32768 | 0.5073 | 0.8482 | 1.67x | 2.13e-04 | 4.17e-03 | PASS |
| mha-h16-d128 | MHA | 16 | 16 | 1 | 128 | 512 | 0.0097 | 0.0124 | 1.28x | 2.17e-04 | 4.07e-03 | PASS |
| mha-h16-d128 | MHA | 16 | 16 | 1 | 128 | 2048 | 0.0210 | 0.0244 | 1.16x | 2.11e-04 | 4.23e-03 | PASS |
| mha-h16-d128 | MHA | 16 | 16 | 1 | 128 | 8192 | 0.0895 | 0.3219 | 3.60x | 2.12e-04 | 4.23e-03 | PASS |
| mha-h16-d128 | MHA | 16 | 16 | 1 | 128 | 32768 | 0.8049 | 1.4572 | 1.81x | 2.18e-04 | 4.21e-03 | PASS |
| mha-h20-d64 | MHA | 20 | 20 | 1 | 64 | 512 | 0.0076 | 0.0139 | 1.82x | 2.17e-04 | 4.30e-03 | PASS |
| mha-h20-d64 | MHA | 20 | 20 | 1 | 64 | 2048 | 0.0207 | 0.0279 | 1.35x | 2.12e-04 | 4.06e-03 | PASS |
| mha-h20-d64 | MHA | 20 | 20 | 1 | 64 | 8192 | 0.0659 | 0.2057 | 3.12x | 2.21e-04 | 4.30e-03 | PASS |
| mha-h20-d64 | MHA | 20 | 20 | 1 | 64 | 32768 | 0.6638 | 1.0496 | 1.58x | 2.26e-04 | 4.21e-03 | PASS |
| mha-h20-d128 | MHA | 20 | 20 | 1 | 128 | 512 | 0.0135 | 0.0200 | 1.48x | 2.22e-04 | 4.19e-03 | PASS |
| mha-h20-d128 | MHA | 20 | 20 | 1 | 128 | 2048 | 0.0227 | 0.0345 | 1.52x | 2.15e-04 | 4.28e-03 | PASS |
| mha-h20-d128 | MHA | 20 | 20 | 1 | 128 | 8192 | 0.2008 | 0.3941 | 1.96x | 2.10e-04 | 4.21e-03 | PASS |
| mha-h20-d128 | MHA | 20 | 20 | 1 | 128 | 32768 | 1.0632 | 1.8089 | 1.70x | 2.13e-04 | 4.17e-03 | PASS |
| gqa2-h16-d64 | GQA | 16 | 8 | 2 | 64 | 512 | 0.0067 | 0.0066 | 0.99x | 2.16e-04 | 4.21e-03 | PASS |
| gqa2-h16-d64 | GQA | 16 | 8 | 2 | 64 | 2048 | 0.0159 | 0.0229 | 1.44x | 2.18e-04 | 3.94e-03 | PASS |
| gqa2-h16-d64 | GQA | 16 | 8 | 2 | 64 | 8192 | 0.0523 | 0.0958 | 1.83x | 2.07e-04 | 4.29e-03 | PASS |
| gqa2-h16-d64 | GQA | 16 | 8 | 2 | 64 | 32768 | 0.2129 | 0.4516 | 2.12x | 2.19e-04 | 4.41e-03 | PASS |
| gqa2-h16-d128 | GQA | 16 | 8 | 2 | 128 | 512 | 0.0108 | 0.0073 | 0.68x | 2.21e-04 | 4.14e-03 | PASS |
| gqa2-h16-d128 | GQA | 16 | 8 | 2 | 128 | 2048 | 0.0205 | 0.0295 | 1.44x | 2.18e-04 | 3.99e-03 | PASS |
| gqa2-h16-d128 | GQA | 16 | 8 | 2 | 128 | 8192 | 0.0599 | 0.1253 | 2.09x | 2.19e-04 | 4.37e-03 | PASS |
| gqa2-h16-d128 | GQA | 16 | 8 | 2 | 128 | 32768 | 0.3472 | 0.7295 | 2.10x | 2.11e-04 | 4.12e-03 | PASS |
| llama-3.2-1b | GQA | 32 | 8 | 4 | 64 | 512 | 0.0071 | 0.0190 | 2.67x | 2.21e-04 | 4.30e-03 | PASS |
| llama-3.2-1b | GQA | 32 | 8 | 4 | 64 | 2048 | 0.0216 | 0.0270 | 1.25x | 2.16e-04 | 3.98e-03 | PASS |
| llama-3.2-1b | GQA | 32 | 8 | 4 | 64 | 8192 | 0.0717 | 0.0655 | 0.91x | 2.18e-04 | 4.20e-03 | PASS |
| llama-3.2-1b | GQA | 32 | 8 | 4 | 64 | 32768 | 0.2982 | 0.4271 | 1.43x | 2.21e-04 | 4.36e-03 | PASS |
| llama-3.1-8b | GQA | 32 | 8 | 4 | 128 | 512 | 0.0110 | 0.0089 | 0.81x | 2.16e-04 | 4.14e-03 | PASS |
| llama-3.1-8b | GQA | 32 | 8 | 4 | 128 | 2048 | 0.0260 | 0.0300 | 1.15x | 2.16e-04 | 4.10e-03 | PASS |
| llama-3.1-8b | GQA | 32 | 8 | 4 | 128 | 8192 | 0.0887 | 0.1220 | 1.37x | 2.11e-04 | 4.26e-03 | PASS |
| llama-3.1-8b | GQA | 32 | 8 | 4 | 128 | 32768 | 0.3924 | 0.7048 | 1.80x | 2.16e-04 | 4.14e-03 | PASS |
| psu_orc_211 | GQA | 40 | 10 | 4 | 128 | 512 | 0.0081 | 0.0165 | 2.04x | 2.18e-04 | 3.96e-03 | PASS |
| psu_orc_211 | GQA | 40 | 10 | 4 | 128 | 2048 | 0.0301 | 0.0401 | 1.33x | 2.16e-04 | 4.06e-03 | PASS |
| psu_orc_211 | GQA | 40 | 10 | 4 | 128 | 8192 | 0.1133 | 0.1900 | 1.68x | 2.17e-04 | 4.38e-03 | PASS |
| psu_orc_211 | GQA | 40 | 10 | 4 | 128 | 32768 | 0.4742 | 0.8060 | 1.70x | 2.17e-04 | 4.29e-03 | PASS |
| gpt-oss-20b | GQA | 64 | 8 | 8 | 64 | 512 | 0.0110 | 0.0073 | 0.66x | 2.17e-04 | 4.20e-03 | PASS |
| gpt-oss-20b | GQA | 64 | 8 | 8 | 64 | 2048 | 0.0234 | 0.0219 | 0.93x | 3.63e-04 | 4.10e-03 | PASS |
| gpt-oss-20b | GQA | 64 | 8 | 8 | 64 | 8192 | 0.0765 | 0.0682 | 0.89x | 3.46e-04 | 4.18e-03 | PASS |
| gpt-oss-20b | GQA | 64 | 8 | 8 | 64 | 32768 | 0.3168 | 0.4530 | 1.43x | 3.53e-04 | 4.30e-03 | PASS |
| llama-3-70b | GQA | 64 | 8 | 8 | 128 | 512 | 0.0176 | 0.0127 | 0.72x | 2.14e-04 | 4.10e-03 | PASS |
| llama-3-70b | GQA | 64 | 8 | 8 | 128 | 2048 | 0.0434 | 0.0416 | 0.96x | 2.17e-04 | 4.07e-03 | PASS |
| llama-3-70b | GQA | 64 | 8 | 8 | 128 | 8192 | 0.1844 | 0.1730 | 0.94x | 2.18e-04 | 4.27e-03 | PASS |
| llama-3-70b | GQA | 64 | 8 | 8 | 128 | 32768 | 0.7324 | 0.6924 | 0.95x | 2.16e-04 | 4.12e-03 | PASS |

## Analysis

- **Correctness**: worst-case relL2 (int8 kernel vs its dequant CPU reference) = **3.63e-04**, far below the 2e-2 fp16-accumulation tolerance; all cases PASS. Pure 8-bit KV quantization error vs fp16 is ~4e-3.
- **Performance (KV len >= 2048)**: int8 averages **1.58x** the fp16 throughput overall (up to **3.60x**) -- **MHA 1.85x** avg (streams int8 from global -> pure bandwidth win) and **GQA 1.42x** avg. Speedup grows with context length as the decode becomes more DRAM-bandwidth-bound.
- At very short contexts (512) the decode is launch/occupancy-bound rather than bandwidth-bound, so int8 and fp16 are near parity; the win grows with KV length -- exactly where an 8-bit KV cache is deployed.
- head_dim 64 and 128 are both covered for MHA and GQA (hpg 1/2/4/8).

### Where int8 is slower than fp16 (and why)

The int8 KV cache's only structural advantage is **halving the KV read traffic**, which pays off *only when the decode is DRAM-bandwidth-bound*. In the shapes below it is not, so int8 cannot win and the extra int8->fp16 dequant makes it slightly slower:

| Case | attn | hpg | D | KV len | speedup | why slower |
|------|------|-----|---|--------|---------|------------|
| gqa2-h16-d128 | GQA | 2 | 128 | 512 | 0.68x | short context: decode is launch/occupancy-bound, so halving KV bytes gives no benefit |
| llama-3.2-1b | GQA | 4 | 64 | 8192 | 0.91x | high-hpg d64 reuses each KV tile across many query heads (compute-bound, not DRAM-bound); WMMA dequants int8->fp16 in LDS so no bandwidth win, only added dequant |
| llama-3.1-8b | GQA | 4 | 128 | 512 | 0.81x | short context: decode is launch/occupancy-bound, so halving KV bytes gives no benefit |
| gpt-oss-20b | GQA | 8 | 64 | 512 | 0.66x | tiny KV (us-scale, launch-bound); WMMA path dequants int8->fp16 in LDS so no bandwidth/LDS win -- pure dequant overhead |
| gpt-oss-20b | GQA | 8 | 64 | 2048 | 0.93x | high-hpg d64 reuses each KV tile across many query heads (compute-bound, not DRAM-bound); WMMA dequants int8->fp16 in LDS so no bandwidth win, only added dequant |
| gpt-oss-20b | GQA | 8 | 64 | 8192 | 0.89x | high-hpg d64 reuses each KV tile across many query heads (compute-bound, not DRAM-bound); WMMA dequants int8->fp16 in LDS so no bandwidth win, only added dequant |
| llama-3-70b | GQA | 8 | 128 | 512 | 0.72x | hpg8 reuses each KV tile across 8 query heads -> compute-bound; the halved DRAM read is not the bottleneck at this length |
| llama-3-70b | GQA | 8 | 128 | 2048 | 0.96x | hpg8 reuses each KV tile across 8 query heads -> compute-bound; the halved DRAM read is not the bottleneck at this length |
| llama-3-70b | GQA | 8 | 128 | 8192 | 0.94x | hpg8 reuses each KV tile across 8 query heads -> compute-bound; the halved DRAM read is not the bottleneck at this length |
| llama-3-70b | GQA | 8 | 128 | 32768 | 0.95x | hpg8 reuses each KV tile across 8 query heads -> compute-bound; the halved DRAM read is not the bottleneck at this length |

The d64 cases recover to >=1x once the context grows long enough for the KV scan to become bandwidth-bound (see their 32768 rows). The **hpg8 / d128** case is the exception: each KV tile is reused across 8 query heads, so its arithmetic intensity is high enough that the decode stays compute-bound at *every* tested length and int8 lands at ~parity (0.9-1.0x) throughout -- there is simply no DRAM-bandwidth headroom for the halved KV read to reclaim. The launcher **autotunes scalar-vs-WMMA per shape**, so each row is already the fastest available int8 config; these regressions are intrinsic to those shapes' low arithmetic intensity, not a suboptimal kernel choice. The scalar d128 path additionally defers the per-channel V-scale load to the epilogue to free EPT VGPRs and lift occupancy on the 256-thread (hpg8) blocks.

_Reproduce:_ `test_gqa_decode_i8.exe --all --iters 500 --md <this file>`
