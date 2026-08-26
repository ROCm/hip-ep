# GQA offline autotune LUTs

`gfx1151.json` is the reviewable source; `gfx1151.fb` is the FlatBuffer the runtime
loads. Rows are generated from measurement, never edited by hand:

```bash
# from RdpCapture/ops_analyze/gqa -- see that directory for the full pipeline
python tools/build_lut.py --store --prune-tolerance 1.02 \
    --selection data/gqa_lut_selection.csv \
    --json ../../../hip-ep/lib/Runtime/Kernels/hip/autotune/gqa/lut/gfx1151.json
```

These are the flags the shipped table was built with, and they reproduce it byte for
byte. `--store` reads every reading ever taken, keyed on the shape rather than on the
grid that asked for it, and takes only those from the current timing harness and the
current version of each kernel; passing result CSVs with `--hipevent` instead is for
looking at a fresh sweep before it has been ingested. **After a kernel change this
command alone is not enough** — see `../README.md`, "Keeping it current".

Check the JSON before packing it. `flatc` validates field names and enum spellings
and nothing else: a duplicate key is not an error (the loader keeps whichever it
parsed last) and neither is a row `rowConsistent()` will refuse, and both are silent
row loss.

```bash
python tools/validate_lut_json.py \
    --fbs ../../../hip-ep/lib/Runtime/Kernels/hip/autotune/gqa/gqa_autotune.fbs \
    --json ../../../hip-ep/lib/Runtime/Kernels/hip/autotune/gqa/lut/gfx1151.json
```

Then, from this directory:

```bash
flatc --binary --strict-json -o /tmp ../gqa_autotune.fbs gfx1151.json
cp /tmp/gfx1151.bin gfx1151.fb
```

Nothing has to be kept in sync by hand any more: the bucket ladder, the window and
parallelism classes and the legal config names are all enums in
`../gqa_autotune.fbs`, and `tools/lut_schema.py` reads them from there and checks
that the ladder is the sequence `seqBucket()` computes. The generator cannot name a
bucket the runtime does not produce.

Compatibility:

- GPU architecture: `gfx1151`
- HIP runtime version: `70151803` (`hipcc 7.1.51803`)
- GQA kernel ABI: `gqa-v1`
- LUT schema version: **5**
- KV-cache format: rows are `kv_dtype = Any`; only fp16 has been measured

Schema 5 is not backward compatible, and the break is the point: a version-4 row
carries exact head counts, an exact batch and a `max_seq`, which this loader has no
field for. An older file is rejected with a message rather than loaded with rows
quietly dropped.

## What the table covers

The fused decode path admits **heads-per-group ∈ {1,2,3,4,5,8,16}** at
**head_dim ∈ {64,128,256}** (`flash_decode_geometry_ok` in `real/gqa.cpp`) — 21
pairs, a closed set — and all 21 are measured. That is the coverage claim worth
making: every geometry that can reach this table has rows keyed on its own pair, so
a model with an unusual q:kv ratio does not fall to the last resort for being
unusual. A ratio outside the set (Qwen2.5-7B's 28:4, heads-per-group 7) never
reaches the fused decode path at all.

| head_dim | heads-per-group measured, as `H:G` |
|---|---|
| 64 | 1 `8:8` `32:32`, 2 `8:4` `16:8`, 3 `6:2` `12:4` `24:8`, 4 `4:1` `16:4` `32:8`, 5 `5:1` `20:4` `40:8`, 8 `8:1` `64:8`, 16 `16:1` `32:2` |
| 128 | 1 `4:4` `16:16` `32:32` `40:40`, 2 `4:2` `16:8` `32:16` `64:32`, 3 `6:2` `12:4` `24:8`, 4 `8:2` `16:4` `32:8` `40:10`, 5 `5:1` `20:4` `40:8`, 8 `8:1` `64:8` `128:16`, 16 `16:1` `32:2` `64:4` |
| 256 | 1 `8:8` `16:16`, 2 `8:4` `16:8` `32:16`, 3 `6:2` `12:4` `24:8`, 4 `8:2` `16:4` `24:6`, 5 `5:1` `20:4` `40:8`, 8 `8:1` `16:2` `24:3` `48:6`, 16 `32:2` `64:4` |

Prefill needs much less of that: its kernels are templated on `head_dim` alone, so
most of its rows are keyed on lengths and answer any geometry. It is not none —
adding MHA and heads-per-group 16 to the grid put 58 v5 and 73 v7 rows on the
heads-per-group tier, where the ten original geometries (all between 2 and 8) had
shown nothing. v8 stays a single row: `ND4_MT1_BKV32` wins all 599 of its measured
shapes across seven geometries.

Each pair is measured at both ends of the **parallelism axis** (`batch*num_heads`),
which is the axis a new model actually moves along. The floor is the pair's smallest
possible geometry — `H = heads-per-group, G = 1`, multi-query attention — and the
ceiling comes from batch 32 on the largest head counts, so a request always has a row
at or below its own parallelism for the probe to walk down to. Scoring the table on
ten geometries at head counts nothing was measured on puts decode within 6.5% of
optimum by total time.

Other axes:

- **Lengths** at every bucket label from 128 to 65536 plus points inside each
  interval (1.03, 1.25, 1.5, 1.75 of each edge). Prefill `seq_q` stops at 4096,
  since `seq_q` is the prefill chunk and a single-shot 64 k prefill is ~13 s per
  layer on this part.
- **Batch** at 1, 2, 3, 4, 8, 16 and 32, keyed both through `batch*num_heads` and,
  where measurement says the factors are not interchangeable, as a batch class of
  its own.
- **Sliding window** for v5 prefill at 128 only. `window_ok` in `real/gqa.cpp` keeps
  windowed decode off the fused path entirely, so rows for it would be unreachable.

## What it costs

See `RdpCapture/ops_analyze/gqa/capture_autotune_lut.md` for the numbers, how they
were measured, and the design alternatives that were measured and rejected. The
reviewable workbook is `RdpCapture/ops_analyze/gqa/GQA_LUT_Capture.xlsx`, where green
marks the config that shipped for each shape.

Two things about reading the per-shape resolution are worth knowing here. Landing on
a coarse tier is not a gap: rows that repeat what a coarser row already says are
pruned, so on a measured geometry a `Fallback` hit means measurement found nothing
finer to say — the whole prefill v8 policy is one row, because `ND4_MT1_BKV32` won
every one of 465 measured v8 shapes. And `Heuristic` is not a tier: it means no
table loaded.

## Packaging

```text
gqa_autotune_lut=/path/to/lib/Runtime/Kernels/hip/autotune/gqa/lut/gfx1151.fb
```

The schema (`../gqa_autotune.fbs`) and the code that reads these tables
(`../gqa_autotune.cpp`) sit one level up; see `../README.md` for the probe order and
what is in each key.
