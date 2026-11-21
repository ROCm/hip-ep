<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
## Performance

Primary comparison: **Radeon 8060S (gfx1151), FP16, historical W1 `screen`
measurements**. This is preliminary native operator evidence, not a fresh GPU
measurement of this PR's subsequent CMake/export changes. The original Flash
GQA baseline is `a7621d9b06675108f282510b2e9437683b161909`.

The 22-case matrix uses Page's **no-table default geometry**, resolved with
`HIPDNN_PAGED_ATTENTION_CFG=default` and timed through
`hip_paged_attention_f16_configured_v2`; Page online tuning was not enabled for
these rows. Original dense Flash GQA performs its maintained no-LUT tuning in
untimed preparation. Exact configurations are retained in the
[22-case CSV](data/page-vs-flash-w1-gfx1151.csv).

Complete times include RoPE/KV append plus attention on both sides, and Page's
metadata validation/reduction. Core ratios use identical post-RoPE FP16 inputs.
Ratio = Page / Flash time; above 1 is slower. Times below are microseconds.

| Case | B; Hq/Hkv/D; Q; past | Page complete | Flash complete | Complete ratio | Core ratio |
| --- | --- | --- | --- | --- | --- |
| P01: prefill 256 | 1; 8/2/64; 256; 0 | 15.597 | 11.056 | 1.411x | 1.289x |
| P02: prefill 2K | 1; 8/2/64; 2048; 0 | 181.410 | 153.532 | 1.182x | 1.094x |
| P03: prefill 4K | 1; 8/2/64; 4096; 0 | 603.580 | 529.882 | 1.139x | 1.035x |
| D05: decode 32K | 1; 32/8/128; 1; 32767 | 759.830 | 685.205 | 1.109x | 1.094x |
| D06: decode, 64 Q heads | 1; 64/8/128; 1; 8191 | 224.261 | 210.678 | 1.064x | 1.081x |
| D07: decode D256 | 1; 32/8/256; 1; 8191 | 381.367 | 322.361 | 1.183x | 1.165x |
| D08: batched decode 8K | 4; 32/8/128; [1,1,1,1]; [8191,8191,8191,8191] | 734.382 | 676.464 | 1.086x | 1.060x |
| D10: decode 32K, window 2048 + RoPE | 1; 32/8/128; 1; 32767 | 61.135 | 42.887 | 1.425x | 1.115x |

Across the original 22 cases, unweighted complete-entry geometric means are
**1.151x for 10 prefill cases** and **1.117x for 12 decode cases**; corresponding
common-input core ratios are **1.072x** and **1.058x**. Complete ranges are
1.035–1.411x and 0.921–1.425x. These results retain the slow short-prefill,
windowed-decode and overallocated-cache cases; they do not claim parity with
Flash GQA or a model throughput gain.

All 22 comparisons passed FP32-reference and storage/guard checks. Maximum
Page/CPU relative L2 was `3.02338e-4` (prefill) and `2.11022e-4` (decode); maximum
Page/Flash relative L2 was `1.29638e-4`. Complete-path differences include RoPE
FP16 rounding, not only paging. The separate
[supplemental CSV](data/page-vs-flash-w1-gfx1151-supplemental.csv)
retains ten extension cases and three distinct route probes; these are not
mixed into the 22-case aggregates. Across all 35 rows, maximum Page/Flash
relative L2 was `3.11000e-4` (shared-prefix extension X03).

Method: Windows HIP runtime/driver `71626375`, `gfx11-generic`; shared kernel
objects compiled with `-O2 -ffp-contract=fast`, native driver with `-O3`.
The prefill build rebuilt its driver and linked retained kernel/helper objects.
The owning `ut-adaptive-v3` `screen`/`normal` timer reports the median of five
group means with operand rotation. The 22-case entries accumulated
1.020–75.879 GPU ms each under the historical screen early-stop policy, so
these are **screening results, not formal `confirm` qualification**.
[Timing/build records](data/page-vs-flash-w1-gfx1151-timing.json)
retain actual group budgets, entry names and source/compiler/runtime/binary
hashes. Allocations, uploads, CPU references and tuning are untimed; equal
rotation counts per pair are bounded by 512 MiB per side and do not certify
small cases cold.


## Historical gfx1150/gfx1171 exact-LUT comparison

The table below records a historical FP16 native comparison from 2026-11-20
(R6 implementation epoch), against the original Flash GQA source at
`a7621d9b06675108f282510b2e9437683b161909`. It is evidence for that measured
epoch, not a fresh measurement of a later PR packaging/build revision. The
companion [CSV](data/page-vs-flash-r6.csv) contains all 44 cases and
exact shapes/configurations; [timing records](data/page-vs-flash-r6-timing.json)
preserve the five groups and actual sampling budgets for all 220 entries.

Both sides use the same logical FP16 Q/new-KV/history and sequence metadata.
**Complete** Page includes validation, RoPE, KV append, attention and decode
reduction; complete dense includes the original RoPE, two append launches and
Flash GQA. **Core** comparisons use common post-RoPE FP16 inputs and exclude
those preprocessing stages. Ragged dense decode compacts active requests into
a batched call; ragged dense prefill loops over active requests. These are
operator comparisons, not model end-to-end results.

Ratio = Page time / Flash GQA time; values above 1 mean Page is slower.
Aggregates are unweighted geometric means over the listed cases, not workload
frequency weighted speedups.

| GPU | Cases | Complete ratio: geometric mean (range) | Common-input core ratio: geometric mean (range) |
| --- | --- | --- | --- |
| Radeon 890M, gfx1150 | 10 prefill | 1.107x (1.009–1.454x) | 1.050x (0.996–1.169x) |
| Radeon 890M, gfx1150 | 12 decode | 1.006x (0.950–1.150x) | 0.988x (0.944–1.049x) |
| Radeon 940M, gfx1171 | 10 prefill | 1.283x (1.086–1.883x) | 1.135x (1.051–1.324x) |
| Radeon 940M, gfx1171 | 12 decode | 1.066x (0.971–1.186x) | 1.053x (0.968–1.131x) |

Representative complete-entry times are microseconds, written as
`Page / Flash (ratio)`. Q and past columns list per-request token counts;
Hq/Hkv/D are query heads, KV heads and head dimension.

| Case | B; Hq/Hkv/D; Q; past | gfx1150: Page / Flash (ratio) | gfx1171: Page / Flash (ratio) |
| --- | --- | --- | --- |
| P01: short prefill | 1; 8/2/64; 256; 0 | 19.700 / 13.548 (1.454x) | 28.898 / 20.028 (1.443x) |
| P02: 2K prefill | 1; 8/2/64; 2048; 0 | 396.193 / 388.458 (1.020x) | 738.858 / 624.107 (1.184x) |
| P03: 4K prefill | 1; 8/2/64; 4096; 0 | 1425.493 / 1337.968 (1.065x) | 2701.833 / 2333.846 (1.158x) |
| P06: chunked prefill + RoPE | 1; 32/8/128; 256; 4096 | 1416.682 / 1404.312 (1.009x) | 3565.715 / 3284.221 (1.086x) |
| P07: D256 chunked prefill + RoPE | 1; 16/4/256; 256; 2048 | 900.140 / 860.835 (1.046x) | 2493.504 / 2278.893 (1.094x) |
| P08: D96 MHA, page size 96 | 1; 16/16/96; 256; 0 | 92.424 / 78.458 (1.178x) | 174.903 / 92.869 (1.883x) |
| P10: ragged/windowed prefill + RoPE | 3; 8/2/128; [256,0,512]; [1024,2048,4096] | 243.913 / 212.946 (1.145x) | 598.325 / 458.981 (1.304x) |
| D05: 32K decode | 1; 32/8/128; 1; 32767 | 1435.119 / 1437.055 (0.999x) | 3514.846 / 3285.573 (1.070x) |
| D07: D256 decode | 1; 32/8/256; 1; 8191 | 711.605 / 707.618 (1.006x) | 966.158 / 961.428 (1.005x) |
| D08: batched 8K decode | 4; 32/8/128; [1,1,1,1]; [8191,8191,8191,8191] | 1421.547 / 1493.778 (0.952x) | 3486.527 / 3432.455 (1.016x) |

Page is not uniformly as fast as dense Flash GQA. Short prefill and D96 MHA
have material complete-entry overhead, with the largest observed ratio 1.883x
on gfx1171 P08; ragged/windowed prefill also remains slower. The full CSV
retains these cases and the overallocated-cache decode cases C01/C02.

All 44 comparisons passed their FP32 CPU-reference and storage/guard checks;
no nonfinite output was reported. Maximum Page-vs-CPU relative L2 was
`3.025e-4` for prefill and `2.112e-4` for decode, against unchanged limits of
`2e-3` and `2e-2`. Maximum complete Page-vs-Flash relative L2 was `1.299e-4`;
maximum common-input core difference was `1.298e-4`. Complete-path differences
include the two implementations' RoPE FP16 rounding, so they are not an
isolated measure of error caused by paging.

The measurements used Windows HIP, recorded runtime/driver version
`71626375`, `gfx11-generic` / `gfx11-7-generic` code objects, kernel flags
`-O2 -ffp-contract=fast`, and the owning `ut-adaptive-v3` timer with the named
`confirm` preset and `normal` profile. Reported times are medians of five group
means. This historical timer allowed convergence-based early stopping:
63 of the 220 entries accumulated less than 10,000 valid GPU ms (minimum
105.677 ms). Consequently this dataset does **not** certify the newer strict
10,000-ms-per-entry measurement requirement. Actual budgets, calls, groups and
early-stop flags are preserved instead of claiming a uniform duration.

Untimed preparation includes original dense online tuning; Page uses the
measured exact LUT geometry recorded per row. Allocations, CPU references,
uploads and tuning are excluded from complete-entry timing. Operand rotation
uses equal copy counts per pair, bounded by 512 MiB per side; small cases are
not certified cold. These measurements do not establish an architecture-wide
result, nor a final gfx1151 performance qualification. The separate gfx1151
W1 screen results and later optimization experiments should be identified by
their own epochs/presets when reported.

To run a fresh comparison, use the existing Page UT's `COMPARE=1` build with
`FLASH_SOURCE` pointing to the exact original `gqa_kernel.hip`, then execute
`--compare 1` through the shared named timer preset. To reproduce a specific
historical Page geometry, pin its `splits/bkv/hb/qw/mt/nd` tuple from the CSV;
running the current no-table default is a separate configuration comparison.
