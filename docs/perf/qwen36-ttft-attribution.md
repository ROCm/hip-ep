<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen3.6-35B-A3B prefill: what each optimisation was worth

Eight items came out of the Qwen3.6 prefill investigation. Six changed runtime
code; items 7 (analysis tooling) and 8 (an fp32-gate investigation that
concluded no change was warranted) changed none, so their TTFT is zero by
construction rather than by measurement.

The short version: the merged stack is worth **−394 ms (−3.29%)**, the six items
are **heavily redundant with each other** rather than additive, and they split
into two mechanisms — five of them shrink prefill's slow excursions while one
lowers its floor.

## Operating point

`-Driver vlm -MaxTokens 2 -MaxLength 8192 -ExecutionProvider AMDGPU` against
`Qwen3.6-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml` with the 2K prompt file, which
yields **3985 prompt tokens** (2162 text, 1823 image). 16K was not used: a 16K
VLM prefill exhausts this box's 63.6 GB of shared memory and hard-reboots it.

One artifact set per arm was built from the common merge-base and staged before
any measurement, so no arm was rebuilt mid-campaign. Every arm deploys all three
binaries — `custom_kernels_gfx1151.dll`, `hipgpu.dll`, `hip-compiler.dll` — even
where only one differs, because a file an arm does not overwrite is left in place
and silently joins its result.

`p1` (item 1) is an ancestor of `p6` (item 6), so `p6` vs base measures items 1
and 6 together. Item 6 alone is `p6` vs `p1`.

## Headline: the merged stack

20 interleaved rounds at 4 reps, two blocks of 10 with the second reversed.

| rounds | base | stack | paired delta | 95% CI | |
|---|---|---|---|---|---|
| 1–20, as designed | 12,192 ms | 11,577 ms | −614 ms | [−768, −460] | −5.04% |
| 1–7 | 12,621 ms | 11,597 ms | −1,023 ms | [−1,170, −877] | −8.11% |
| 8–20 | 11,961 ms | 11,567 ms | **−394 ms** | [−451, −336] | **−3.29%** |

**−394 ms is the number to quote.** The stack arm is the same in all three rows
to within 30 ms; the whole spread is the *base* arm, which starts at 12,621 ms
and steps down to 11,961 ms at round 8 and then holds there for 13 rounds. Base
needs roughly 20 minutes of running to reach its steady state, and the optimised
arms reach theirs immediately. Rounds 1–7 therefore measure base's warm-up on
top of the real effect, and the −614 ms full-run figure is inflated by it.

This asymmetric warm-up is unexplained and is the top follow-up: it caps the
precision of every comparison against this baseline. It is not the box shedding
heat — a replication run started after roughly half an hour with no benchmark
running had base at 12,528 ms on its first round and 11,660 ms by its fourth,
warming *into* its fast state rather than out of it.

## Per item

Ten interleaved rounds at 4 reps, all seven arms in one run so every item met
the same conditions, half reversed. Reported as measured.

| item | change | TTFT vs base | 95% CI | |
|---|---|---|---|---|
| 1 | causal conv prefill tiling | −506 ms | [−782, −229] | |
| 2 | GQA prefill LDS staging | −236 ms | [−463, −10] | |
| 3 | linear-attention pass-3 LDS | −595 ms | [−885, −305] | |
| 4 | MoE token bucketing | −313 ms | [−591, −34] | |
| 5 | skip + RMS-norm fusion | −171 ms | [−430, +89] | spans zero |
| 6 | conv channels-last fold | +19 ms | [−94, +131] | spans zero, vs `p1` |
| 7 | analysis tooling | 0 | — | no runtime code |
| 8 | fp32-gate investigation | 0 | — | no runtime code |

**These do not add up, and that is the main finding, not a defect.** They sum to
−1,802 ms against a stack worth −394 ms, 4.6× too much. The individual numbers
are not the error: item 3 alone, at −595 ms, already exceeds the entire stack's
steady-state gain. The items are close to redundant — they all buy back time from
the same place, and once that time is gone the next item has nothing left to
take. The next section shows where that place is.

So this column ranks the items and bounds each one. It cannot be summed, and no
single row should be quoted as that item's standalone contribution to the stack.

Base is also measurably the worst possible reference. Summarising the same CSV
against `p1` instead gives a standard deviation of the paired differences of
106–193 ms, against 317–408 ms for the same comparisons made against base. Any
two optimised arms can be separated 2.5× more precisely than either can be
separated from the baseline, which is why the intervals above are as wide as
they are.

## The mechanism: a fixed floor and a variable tail

Pooling all individual prefill reps per arm rather than the per-round means shows
why. `floor` is the 5th percentile, `tail` the 95th. The two groups are separate
runs, so each is read against its own base.

| arm | floor | median | tail | spread |
|---|---|---|---|---|
| base, per-item run | 11,688 ms | 12,461 ms | 12,938 ms | 1,250 ms |
| p1 | 11,626 ms | 11,834 ms | 12,144 ms | 517 ms |
| p2 | 11,657 ms | 12,210 ms | 12,516 ms | 860 ms |
| p3 | 11,619 ms | 11,723 ms | 12,221 ms | 602 ms |
| p4 | 11,636 ms | 12,156 ms | 12,360 ms | 724 ms |
| p5 | 11,695 ms | 12,276 ms | 12,490 ms | 795 ms |
| p6 | **11,448 ms** | 11,901 ms | 12,412 ms | 965 ms |
| base, stack run | 11,818 ms | 12,016 ms | 12,891 ms | 1,073 ms |
| stack | **11,405 ms** | 11,554 ms | 11,759 ms | 354 ms |

Items 1–5 leave the floor where it was, within 70 ms of base, and cut the tail by
420–795 ms. `p6` is the only arm that moves the floor: 240 ms below base, and
178 ms below `p1`, which is item 6's own share of it. The stack, which contains
item 6, is 413 ms below its own base. That is exactly the
difference between making a kernel faster and removing a kernel launch: the floor
is the host-bound path, and only item 6 shortens it, by deleting 60 of the 100
transpose dispatches. Everything else reduces GPU work that only appears above
the floor.

It also explains the redundancy directly. In the same run, base's reps span
1,073 ms from floor to tail while the stack's span 354 ms, so the stack has very
nearly run out of tail to remove: it now sits close to a floor that none of these
six items can lower much further. Additional kernel optimisation of this graph
will do almost nothing until the launch path itself is addressed.

The replication run, base/`p1`/`p6` over 4 rounds half reversed, is consistent
with the floor reading and inconsistent with the per-item table's verdict on
item 6:

| | mean | vs base | 95% CI |
|---|---|---|---|
| base | 11,993 ms | — | |
| p1 | 11,649 ms | −344 ms | [−712, +24], spans zero |
| p6 | 11,533 ms | −459 ms | [−721, −198] |
| p6 vs p1 | | −116 ms | [−330, +98], spans zero |

Item 6 is worth somewhere around 120–180 ms by both the floor statistic and this
replication, and the +19 ms in the per-item table is the least reliable cell in
that table rather than a refutation.

## Kernel level

Median of three `HIPDNN_EP_PERF=1` passes per arm, warm autotune cache, prefill
inference only. The *noise* column is the base arm's own spread across its three
passes: a delta inside it is not attributable to the change.

| operator | dispatches | base | noise | item 1 | item 6 (vs p1) | stack |
|---|---|---|---|---|---|---|
| `causal_conv` | 30 | 140.1 ms | 8.0 | −103.0 ms | −14.3 ms | −117.4 ms |
| `transpose` | 100 → 40 | 295.5 ms | 49.6 | +10.9 ms | −120.9 ms | −154.0 ms |
| `matmul_nbits` | 391 | 525.6 ms | 24.0 | +45.3 ms | −17.4 ms | −35.8 ms |
| `layernorm` | 111 | 99.5 ms | 30.1 | +3.5 ms | +4.0 ms | +1.8 ms |
| `skip_layernorm` | 80 | 29.2 ms | 1.8 | +0.1 ms | +0.5 ms | +0.2 ms |
| `linear_attention` | 30 | 126.2 ms | 370.2 | — | — | — |
| total, excl. `gqa` | | 1,802.4 ms | 369.5 | −45.2 ms | −128.0 ms | −396.3 ms |

Only `causal_conv` and `transpose` clear their noise floors individually.
`linear_attention` varies by 370 ms between identical passes of the same binary,
which also sets the noise on the total, so item 3's kernel-level effect is not
measurable this way despite item 3 having the largest per-item TTFT number.

Item 6's contribution is the transposes, not the convolution: 60 of the 100
dispatches disappear, worth ~121 ms, and the channels-last kernel is a further
~14 ms faster than the channels-first one it replaces.

The stack's GPU saving (−396 ms) and its TTFT saving (−394 ms) match. Host and
GPU alternate here rather than overlap, so GPU time sits on the critical path in
full — it is simply a small part of it, 1.8 s of a 12 s TTFT.

**Item 5's fused kernel is inert on this graph.** `skip_layernorm` keeps the same
80 dispatches at the same `3985x2048` shape and the same time to within 0.7 ms
against a 1.8 ms noise floor, and `layernorm`'s 111 dispatches and shape
breakdown are unchanged too. Whatever the fusion does, this graph does not reach
it. That is consistent with its TTFT interval spanning zero, and it is the one
item whose implementation is worth re-opening.

## Two traps in the measurement, for whoever repeats this

**The `gqa` row of the per-op profile is wrong and must not be used.** It reports
21,598 ms of GPU time across 10 dispatches inside a prefill whose entire TTFT is
about 12,000 ms, which is impossible. Two further pieces of evidence: item 2's
kernel change moves it by a reproducible +3,327 ms while moving TTFT by −236 ms,
and the sum of every *other* operator is 1,802 ms, which is a credible GPU-busy
figure for this prefill. The fault is isolated to that one row; the rest of the
table is internally consistent. The total row above therefore excludes it.

**The autotune cache is keyed on shape, not on which binary tuned it,** so
without a per-arm `TEMP` a cache written by one arm is silently reused by the
next and the two arms no longer differ in the way you intended. With per-arm
caches the first pass pays the tuning search instead: `matmul_nbits` came out at
543.8 ms on the cache-writing pass against 525.6 and 519.8 ms on the two that
followed, which is why the kernel table is a median of three passes rather than a
single run.

## Reproducing

```powershell
. C:\Users\zyq\hipep-env.ps1
$ab = 'tools\perf-harness\bench\ab_interleaved.ps1'
$common = @{ Manifest = 'arms.json'; Reps = 4; SkipPrime = $true; Driver = 'vlm'
             MaxTokens = 2; MaxLength = 8192; ExecutionProvider = 'AMDGPU' }
& $ab @common -Arms base,stack -Rounds 10 -StartRound 1
& $ab @common -Arms base,stack -Rounds 10 -StartRound 11 -Reverse
python tools\perf-harness\bench\ab_summary.py <out>\ttft_summary.csv --baseline base
```

Discard base's warm-up rounds, and give base at least 20 minutes of running
before treating its level as settled.

Building the arms needs `build_deploy.ps1 -WithCompiler` for any arm that touches
`lib/Conversion`, `lib/Dialect` or `HipOps.td`. Without it `hipgpu.dll` relinks
and the deploy looks complete while compilation still goes through the stale
`hip-compiler.dll`, and the change measures as a clean zero.
