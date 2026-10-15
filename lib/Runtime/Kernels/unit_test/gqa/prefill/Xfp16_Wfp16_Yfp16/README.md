# GQA flash prefill test -- fp16 KV cache

Standalone `hipcc`-only test for the fused FA-2 WMMA prefill kernels
(`sq > 1`; the runtime selects v6x at head_dim 64/96/128 and v8x at 256). No CMake, no EP build.
Entirely self-contained: inputs and the CPU fp32 reference are generated
in-process, no python, no data files, no `unit_test/common/`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... B=1 H=32 G=8 D=128 SQ=512 [PAST=0] [WINDOW=0]
make clean
```

| Target | Meaning |
|---|---|
| `test` | Runs the exe with no single-shape flags, which hits the built-in `cases[]` matrix in `test_gqa_prefill.cpp`'s `main()` (real models + sink/window variants -- see "Shape coverage" below). |
| `test_custom` | One shape from `B=`, `H=`, `G=`, `D=`, `SQ=`, `PAST=`, `WINDOW=` (in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

`test_gqa_prefill.cpp`'s `main()` has an explicit `cases[]` array (75 rows,
each commented with why it exists) -- a hand-picked set of (categorical
situation, typical shape) pairs. Rows 0-28 are
the original real-model/sink/window matrix (typical `sq` of 512/1000/2048,
plus `past` up to 8192 for chunked prefill); rows 29-38 widen the shape
coverage with cheap `sq=128` short-prompt variants across most scenario
types, plus 2 `sq=8192` **pure** long-prompt rows (`past=0`, i.e. one very
long single prompt, as opposed to the existing `past=8192` chunked-prefill
rows which keep `sq` short). The CPU reference (`cpu_reference`, `O(sq^2)`
causal attention) is now multithreaded over `(batch, query-head)` pairs so
this wider range -- especially the 2 `sq=8192` rows -- stays inside the
shared ~30 min tier3 budget. `COVERAGE=1|2|3` (default 3, `make test
COVERAGE=N` or env `HIPDNN_UT_COVERAGE`) picks a fixed index subset of that
array (`kTier1` 26 rows, `kTier2` 68 rows, tier3 all 75) chosen so tier1
still touches every `D`(64/96/128/256)/sink-mode/window-on-off/chunked-prefill
(`past>0`)/must-decline value, window 1, a window wider than the sequence,
batch 2, the sink + smooth + window combination and every prefill kernel
route, plus 2 of the cheap `sq=128` rows; tier2 keeps every tier1 row and
adds all 8 `sq=128` rows (cheap) and all the routing and head_dim 96 rows
but deliberately excludes the 2 expensive
`sq=8192` rows (tier3-only, a budget trade-off). At startup the exe prints
`coverage=N -> running <n>/75 gqa_prefill cases`. No human edits a shape
list -- there is no `shapes.csv` or `gen_data.py` in this leaf.

The original 29 cases (rows 0-28) cover:
- Real model geometries: Qwen3.6-35B-A3B (d=256, on v8x), and gpt-oss-20b
  (d=64), Llama-3.2-1B (d=64) and Llama-3.1-8B (d=128), on v6x.
- Sequence lengths including one (`sq=1000`) deliberately off the 16-row Q
  tile / 16-key KV tile boundary, to exercise partial tiles.
- Attention-sink variants: none, per-head sink tensor, smooth-softmax, and
  both together (the only combination the runtime actually sends).
- A d==128 sink case that must be *rejected* (rc != 0) so the runtime falls
  back to the decomposed path instead of silently dropping the sink.
- Sliding window: window alone, chunked prefill (past > 0, including a
  window deep enough to skip whole KV tiles), a window not aligned to any
  BKV tile (off-by-one edge case), window wider than the sequence (must
  equal full attention), window==1 (degenerate, each query sees itself),
  window combined with the sink, and the full gpt-oss sliding-layer
  configuration (window + sink tensor + smooth together).
- Window at d==128, which v6x implements.

Rows 29-38 (widened coverage) add: short (`sq=128`) prompts across
qwen3.6-d256, gpt_oss-20b, llama-3.1-8b, gpt_oss-sink, gpt_oss-win,
gpt_oss-both, llama-win-d128, and the d128-sink-must-decline case; plus 2
long pure-prefill (`sq=8192`, `past=0`) rows at D=64 (gpt_oss-20b) and D=128
(llama-3.1-8b).

Rows 39-55 walk the prefill kernel routing (the same rule as
`real/gqa.cpp`: v8x at d=256, v6x otherwise, where a KV group that no block
of heads divides splits the block into Q subtiles): d=256 groups of 8, 4 and
6 (Qwen3.6 16/2, Qwen3 16/4, Qwen3.8 24/4), Gemma-3 8/4, a d=256 window that
must be declined, d=128 groups not divisible by 4 (Qwen3-1.7B, Llama-3.2-3B,
Qwen2.5-7B, Llama-2-7B), and d=128 sliding windows (Mistral-7B, plus Qwen2.5
and Qwen3 groups not divisible by 4).

Rows 56-74 add head_dim 96 -- Phi-3-mini (MHA 32/32) with partial tiles, a
chunk and batch 2, a GQA group of 4, Phi-3-mini-4k's 2047-key window (one
that never binds, then binding, unaligned and window-1 cases), and sinks with
and without a window -- plus more block splits: a d=256 group of 5, Gemma-3
and Llama-3.1-8B at batch 2 with a chunk, Llama-2-7B (MHA, d=128), and d=64
groups of 3 and 1 with sinks and a window.

`MODE=lookup` (default): resolves from `hip/autotune/gqa/lut/<arch>.fb` if it
exists for the arch in `OFFLOAD`, else falls back to `autotune` with a
warning. `MODE=lookup` needs `flatc` + its
`include/` (a build tool, not part of this repo):
`FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>` -- see
`unit_test/README.md`.

The prefill kernel never calls the autotune resolver itself (only production
`real/gqa.cpp` does); `MODE=autotune` (the fallback, and the only path when no
`.fb` exists) has the launchers self-tune their configuration per shape,
independent of `MODE`. `MODE=lookup` instead has *this test* call
`hip_gqa_autotune_resolve_prefill()` (the same resolver `real/gqa.cpp` calls in
production) and dispatch the resolved config through
`hip_gqa_flash_prefill_v3_configured()`. Both modes append to
`out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
