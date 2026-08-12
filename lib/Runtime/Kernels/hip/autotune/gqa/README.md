# GQA autotune

How a GQA launch config is chosen, and where each piece lives.

## Three tiers

A config is resolved by `gqa_autotune_resolve_decode/prefill()` in
`gqa_autotune.cpp`. It tries, in order:

| Tier | Source | What it is | Cost |
|---|---|---|---|
| 1 | `Exact` | Offline LUT, keyed on the measured shape. | one hash lookup |
| 2 | `Bucket` | Same LUT, keyed on power-of-two ceilings of the lengths — the fuzzy match. | one hash lookup |
| 3 | `Heuristic` | A fixed config that is only known to *run*. | none |

Tiers 1 and 2 are both **data in the FlatBuffer**; they differ only in how the
key is built. Nothing is computed on the dispatch path — no timing, no scoring,
no floating point. A config that has to be derived at runtime is a config nobody
reviewed, so deriving one is an offline job (see below) and its output ships as
more rows.

Tier 3 exists only so an unknown shape still runs. It is deliberately dumb; the
way to stop hitting it is to add rows, not to make it smarter.

## Files here

| File | Role | Built by |
|---|---|---|
| `gqa_autotune.fbs` | LUT **schema** — the format, not the data. Lives here, not in `schemas/`, so it sits next to its only reader. | `schemas/CMakeLists.txt` (flatc) |
| `gqa_autotune.h` | Policy API: requests, configs, `GqaTuneSource`. | — |
| `gqa_autotune.cpp` | All three tiers: LUT load/lookup and the heuristic fallback. | `lib/Runtime/CMakeLists.txt` (bitcode) |
| `gqa_cost_model.h` / `.cpp` | **Offline only.** Scores a candidate for a shape; used to generate LUT rows for geometries nobody measured. | `tools/gqa-autotune-lut` |
| `lut/*.json` | The tables, per arch. Reviewable source of truth. | — |
| `lut/*.fb` | What the runtime loads, produced from the JSON by `flatc`. | — |

`gqa_autotune.cpp` is host code compiled into the runtime bitcode: it needs
flatbuffers and the EP `FileSystem`. `gqa_cost_model.cpp` is host C++ too, but it
is **not** linked into the EP or the kernel library — only into the offline tool.
Neither file contains device code; `.hip` is not involved on either side.

The cost model lives here rather than under `tools/` because its tables are
compile-time facts about the kernel instantiations in `../../gqa_kernel.hip`
(register spill, occupancy, waves per block). Editing the kernel invalidates
them, and that is much easier to notice from this directory.

## Using it

```c
hip_gqa_shape_t shape = { .batch = 1, .num_heads = 64, .kv_heads = 8,
                          .head_dim = 64, .q_len = 512, .kv_len = 2112,
                          .window = -1, .cu_count = cus };
double best_score = 0.0;
for (candidate in candidates) {
  hip_gqa_config_t cfg = { ... };
  double score = hip_gqa_config_score(&shape, &cfg);   // higher is better
  if (score > best_score) { best_score = score; best = cfg; }
}
if (best_score == 0.0) { /* model declined -> tier 3 */ }
```

A score of 0 means "not a real instantiation for this shape", which is also the
signal to fall through to the heuristic.

## Filling tiers 1 and 2

`gqa_autotune.fbs` is the schema; the entries live in a FlatBuffer packaged with
EPContext. Both the exact and the bucket ("fuzzy") entries are **data**, not
code: widening a match means adding rows, not writing another matcher.

The shipped tables live in `lut/`, with a reviewable JSON source next to the
binary the runtime loads:

```
lut/gfx1151.json    # source of truth, reviewed in PRs
lut/gfx1151.fb      # what the runtime loads
```

Regenerate the binary after editing the JSON:

```bash
cd lut && flatc --binary --strict-json -o /tmp ../gqa_autotune.fbs gfx1151.json
cp /tmp/gfx1151.bin gfx1151.fb
```

New measurements come from the sweep harness under
`../../../test/example/gqa/autotune`; see `lut/README.md` for what the current
gfx1151 table covers and which buckets are deliberately absent.

### Rows for geometries nobody measured

Measuring every geometry is not practical, and a miss costs a heuristic config.
`tools/gqa-autotune-lut` computes bucket rows from the cost model instead:

```
gqa-lut-rows-from-model --arch gfx1151 --cus 20 \
    --geometry 64:8:64 --geometry 32:8:128 --geometry 16:4:256 \
    --windows 0,128 --out rows.json
```

It emits an `entries` array in the same format as the arch JSON. Review it,
merge it under a distinct `model_key`, and run `flatc`. The rows are marked
`Bucket`, so any measured `Exact` row still wins over them.

These rows are computed, not measured: on the sweep set the model picks the
measured winner 69% of the time and averages +2.5% over it, with `prefill_v8`
exact and `prefill_v7` the weakest (its two live configs are within ~5% of each
other everywhere, so it tends to pick one and stay there). Prefer measuring the
geometries you actually ship.

Note that `max_seq` is part of the decode exact key rather than a wildcard: two
sliding-window shapes with the same effective KV length pick different winners
when their backing cache strides differ.

## Keeping it current

Two inputs, refreshed separately.

**Resource tables** (`blocks_per_cu`, `waves_per_block`, `scratch_bytes`) come
from the compiler and must be regenerated whenever `gqa_kernel.hip` changes
shape:

```
hipcc -c -O3 -std=c++17 -x hip --offload-arch=<arch> -Iinclude -Ihip \
      -Rpass-analysis=kernel-resource-usage hip/gqa_kernel.hip -o /dev/null 2> res.txt
python RdpCapture/ops_analyze/gqa/tools/parse_kernel_resources.py res.txt
```

**Coefficients** are fitted per device from measured sweeps:

```
cd lib/Runtime/Kernels/test/example/gqa/autotune && make && make run
python RdpCapture/ops_analyze/gqa/tools/fit_cost_model.py --emit
```

### What is and is not in the shipping path

The dispatch path is two table lookups and nothing else:

```
gqa.cpp -> gqa_autotune.cpp -> exact key -> bucket key -> heuristic
```

Everything that computes a config runs offline and in-tree: the sweep harness
(`../../../test/example/gqa/autotune`, C++/HIP) measures, `tools/gqa-autotune-lut`
(C++) turns the cost model into rows, `flatc` packs the reviewed JSON.

The only Python left is `RdpCapture/ops_analyze/gqa/tools/`, which **re-fits the
six coefficients** in `gqa_cost_model.cpp` from a sweep and produces the
analysis workbooks. It runs by hand when the kernels change or a new part is
brought up, its output is six numbers that get pasted into `kCoef`, and neither
the build nor the runtime ever invokes it.

## Checking the model against measurement

The sweep harness in `../../../test/example/gqa/autotune/` benchmarks every
candidate for a shape and, in the same run, records what the cost model would
have picked (`model_config` / `model_vs_best_x` / `model_exact` in its
`--best-csv` output). That is how the accuracy figures in `gqa_cost_model.h` are
produced, and how to tell whether a coefficient refit helped.

Online tuning is unchanged and still available: with no LUT packaged, the policy
reports `Online` mode and `gqa.cpp` calls the un-configured kernel entries, which
benchmark as before.
