# Reliable perf measurement (HIPDNN/MorphiZen EP, gfx1151)

Why this exists: decode TPS on this box has large run-to-run variance (GPU
boost/thermal drift) and **no clock pinning is available** (no rocm-smi/amd-smi
on Windows). Single runs and 2-sample A/Bs mislead — we previously mis-read a
noise fluctuation as a "+5%" win. This harness makes results trustworthy.

## Tool: `perf_measure.py`

Each *sample* is a fresh **process** launch of `model_benchmark` (each reloads +
re-autotunes, so process-level variance is the real noise). Within a process,
`-r/-w` give a steady per-process number.

- **Noise floor:** `python perf_measure.py noise --reps 20`
  Reports mean/median/stddev/CoV/95%CI and the RESOLVABLE-WIN THRESHOLD (~2σ).
- **Paired A/B:** `python perf_measure.py ab --pairs 12 --a-env "KEY=VAL,..."`
  Runs K **interleaved** pairs (A,B,A,B,...); per-pair diffs cancel slow drift.
  Reports the paired delta, its 95% CI, and a SIGNIFICANCE VERDICT.
  A/B configs are env-var dicts, so you toggle ONE code path on the SAME build
  (e.g. `--a-env HIPDNN_EP_MIOPEN_ACT=1` = baseline vs default custom kernel).

## Measured baseline (2026-07-06, p128 g128)

- **Noise floor: CoV ~4-5%.** Single-run resolvable threshold ~5%; a 10-pair
  interleaved A/B resolves ~±1.7% (95% CI).
- **=> optimizations under ~2% are unmeasurable here; under ~5% need the paired
  harness to detect at all.**

## Verdict on the shipped custom kernels (activation + fused skip-RMSNorm)

10-pair interleaved A/B, A = MIOpen baseline, B = custom:
`A=40.61±1.09  B=40.87±1.27  paired delta +0.6% ± 1.7% -> NOT significant.`
Correct + harmless, but no measurable model-level decode gain. (The earlier
"+5%" was a 2-sample noise artifact.)

## Per-op profiler caveat (`HIPDNN_EP_PERF`)

Sum of per-op `gpu(ms)` at steady decode (~52 ms) is ~2× the CLEAN wall
(~25 ms/token): hipEvent-per-op bracketing inflates, worst for tiny ops. So the
"~40% tiny-op tail" was substantially a profiling artifact; the clean tail is
much smaller (decode is ~memory-bound: ~1.5 GB weights/token ÷ ~59 GB/s ≈ 25 ms).
Treat `[PERF]` per-op **shares as rough/relative only**; use the roofline +
paired A/B for decisions. TODO: low-distortion per-op profiler (batched events,
single end-of-inference readback, no per-op sync).

## Rules

1. Establish the noise floor first.
2. Only chase targets projected **above** it (>~5%).
3. Validate with the paired harness — never single/2-sample runs.
