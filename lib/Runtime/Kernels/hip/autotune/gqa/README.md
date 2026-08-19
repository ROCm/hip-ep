# GQA autotune

How a GQA launch config is chosen, and where each piece lives.

**Changing a GQA kernel? Start at [Keeping it current](#keeping-it-current).** The
rows here are measurements of a specific kernel, and the failure mode is not a stale
table — it is a table rebuilt from readings the old kernel produced, which is silent.
There is a stamp that prevents it and a four-step procedure that uses it.

New to this? `RdpCapture/ops_analyze/gqa/gqa_autotune_guide.html` explains the whole
ruleset from nothing, in one browser page, with a tool for putting a shape in and
seeing which keys it probes.

## Four tiers, all of them rows

A config is resolved by `gqa_autotune_resolve_decode/prefill()` in
`gqa_autotune.cpp`. It builds one key per tier and probes them in order:

| Tier | Key | What it answers |
|---|---|---|
| `Geometry` | head_dim, heads-per-group, a bucket of `batch*num_heads`, optionally a batch class, bucketed lengths | the geometry at its own parallelism — the tier nearly every request lands on |
| `HeadGroup` | head_dim, heads-per-group, bucketed lengths | the geometry, pooled over head counts |
| `Length` | head_dim, bucketed lengths | a heads-per-group with no rows of its own, and most prefill shapes |
| `Fallback` | phase, kv_dtype, head_dim | anything else; ranked on its worst case |
| — `Heuristic` | — | computed, not stored: reached only when no table loaded |

Every tier is **data in the FlatBuffer**; they differ only in how much of the key
is wildcarded, and the tier is itself a field of the key. Nothing is computed on
the dispatch path — no timing, no scoring, no floating point, one hash of one
64-bit word per probe. A config that has to be derived at runtime is a config
nobody reviewed, so deriving one is an offline job (see below) and its output ships
as more rows.

`Heuristic` is not a tier so much as a failure mode: no file, an arch or schema
mismatch, or a table shipped without `Fallback` rows. **A table that loads answers
every shape**, so `Heuristic` in a log means the table did not load, not that a
shape was missed. `gqa_tune_source_name()` is worth logging for exactly this.

What it answers with is a calculation rather than a constant, borrowed from
llama.cpp's `launch_fattn`: the decode split count that fills the machine without
opening another wave, given the compute unit count the driver reports. Against the
fixed `min(8, useful_splits)` it replaces, over every measured decode shape, that
takes the median from 91.0% of optimum to 96.5% and the share within 10% from 52% to
75%. It is deliberately not used above this tier — measured rows beat it, 96% within
10% against 84% — because occupancy is not the only term: each extra split re-reads
the KV and adds a row to the reduce.

Each tier probes twice, once with the request's `kv_dtype` and once with
`GqaTuneKvDtype::Any`. Every shipped row is `Any`: prefill dequantises an Int8
cache to fp16 scratch before it dispatches, so its tuning cannot depend on the
dtype, and decode's Int8 path has not been swept. Measuring one later adds rows
without a schema change.

### What is in the key, and why that and nothing else

The kernels are templated on `(head_dim, heads-per-group)` for decode and on
`head_dim` alone for prefill, and the key follows that rather than the request's
head counts:

- **`heads-per-group`, not `num_heads` and `kv_num_heads`.** Adding the exact head
  counts on top of heads-per-group and the parallelism class splits 1661 decode keys
  into 1885 and changes the median, the p90, the worst case and the share of shapes
  within 5% of optimum by nothing at all. What dropping them buys is *completeness*:
  `flash_decode_geometry_ok` in `real/gqa.cpp` admits heads-per-group in
  {1,2,3,4,5,8,16} at head_dim in {64,128,256}, 21 pairs in all, so a table with rows
  for each pair answers every geometry that can reach it. A key on head counts can
  only ever cover the counts somebody measured — which is why 16:4 and 24:8 used to
  land on the last resort.
- **`batch*num_heads` bucketed, and the batch class as well.** The product is where
  batch and the absolute head count enter together, because what they change is the
  number of independent work items, and it is not a small effect: at batch 4 the
  optimal decode split count drops by about the batch factor, and ignoring batch
  costs a median of 1.10x and up to 1.68x. The previous schema keyed on exact `batch`
  with only batch-1 rows measured, so every batched request reached the last resort.

  The product alone is not enough either. `8:1:64` at batch 32 and `64:8:64` at batch
  4 are the same 256 work items and disagree about WMMA in 7 shapes of 7, so a
  Geometry row may also name a batch class; splitting the product into its two
  factors takes the worst case from 68.7% of optimum to 80.3%. Pruning keeps such a
  row only where the batch changes the answer, which in the shipped table is 362 rows
  of 2788. Beyond those two, exact head counts and exact batch add nothing at all:
  the same measurements group into exactly the same keys either way.

- **A missing parallelism class falls back to a lower one.** The Geometry probe walks
  the axis downward before giving up on the tier, the way a length rounds up to a
  label — both round the request to a key the table has, in the direction that is
  safe. Down is safe because a row measured at less parallelism holds more splits and
  the clamp bounds what that costs, while too few splits leaves the machine idle.
  Measured on held-out geometries, walking beats the pooled row: worst case 66.5% to
  78.7%, share within 10% of optimum 81% to 90%. The grid also measures each pair's
  *floor* (`H = heads-per-group, G = 1`, i.e. multi-query attention), so the walk
  always lands on something measured.
- **Bucketed lengths, four labels to the octave.** `seqBucket()` rounds a length up
  to `{1, 1.25, 1.5, 1.75} x 2^k`, so the row labelled `S12288` answers every
  request in `(10240, 12288]`. The step has been halved twice, each time because a
  wider interval was measured straddling a change in the optimum: `64:8:64` decode
  trades the lead between the WMMA and scalar kernels twice inside `(512, 1024]`,
  and its prefill v5 flips between `MT1` and `MT2` inside `(8192, 12288]`.
- **No `max_seq`.** It is the KV cache capacity, not the current length. Sweeping
  it from `seq_kv` to 128 k at fixed work moves every candidate by the same factor
  and never changes which one wins — at most 1.025x in ranking terms. Keying on it
  would multiply the table by the capacities a deployment might use and match none
  of them. (It does change absolute time — a 64 k+ cache costs about 25% on d=64
  decode — but that is a capacity-planning fact, not a tuning one.)
- **No exact lengths.** There used to be a tier keyed on the request's own
  lengths. It answers nothing a bucket row does not: `effective_skv` grows by one
  token per step, so generating 1000 tokens from an 8 k prompt matches *zero* exact
  keys, and on the boundary shapes where one would match, the bucket row already
  lands on the optimum. It was a third of the table's bytes.
- **`local_window` on prefill v5 only.** `window_ok` in `real/gqa.cpp` admits a
  window to the fused path at head_dim 64 alone, and the fused decode path
  hard-codes `local_window = 0`, so rows for windowed decode would be unreachable.
  `NoWindow` is a concrete value and `Any` is the wildcard; a `Fallback` row needs
  `Any` because it answers both.

A row's wildcards have to agree with its tier, and `rowConsistent()` rejects it
otherwise: wildcarding a field the tier keys on would make the row answer far more
than it was measured on, and setting a field the tier ignores would make it answer
far less than it looks like it does. Both are silent, so both are refused.

### A bucket row serves shapes smaller than its label

The key rounds lengths **up**, so a row must hold the config with the lowest regret
*across its whole interval*, not the config that won at its label.
`RdpCapture/ops_analyze/gqa/tools/build_lut.py` chooses rows that way, from
measurements taken inside the intervals, and pools a row's group over every
geometry it will serve.

### Decode split counts are clamped

A resolved decode config has its `splits` clamped to `ceil(effective_len/16)`, the
splits that have work to do. This is what lets one row cover a whole interval at
short context, where the measured optimum *is* that bound: on `64:8:64` the winner
at 132/160/192/224/256 keys is exactly 9/10/12/14/16 splits. Without the clamp a
row would have to hold the value its shortest length tolerates and would give up
17% at the top of the interval.

### A config is a name, not a bag of knobs

`GqaTuneConfig` names one point of the launch space — `Scalar`, `Wmma`,
`MT2_BKV32`, `NW4_BKV32_MT1`, `ND4_MT1_BKV32` — spelled the way the dispatchers and
the offline sweep spell it. The previous schema carried seven independent ints and
the loader had to check, per phase, which combinations were real. `nd = 4` with
`bkv = 64` has no name, so no table can ask for it.

Decode carries its split count separately, in `splits`, because the split ladder is
not closed: the runtime clamps it per request.

## A row is 12 bytes

`GqaTuneRow` is a FlatBuffers **struct** of twelve `ubyte` fields, so the table is
a length-prefixed array with no vtables and no padding. The same information cost
61 bytes a row in the previous schema, 122 KB for a table that is 33 KB here — and
size is the point: what fits in the file is how many measured shapes the policy can
distinguish. Both halvings of the bucket ladder were affordable because of it.

Lengths, windows and parallelism classes are ubyte-backed **enums**, which buys two
things. The bucket ladder lives in the schema, so the generator cannot name a
bucket the runtime does not compute — the previous scheme could only be caught by
bumping `schema_version`. And the JSON stays readable: a row diffs as
`seq_kv: "S16384"`, not as a magic byte.

## Files here

| File | Role | Built by |
|---|---|---|
| `gqa_autotune.fbs` | LUT **schema** — the format, not the data. Lives here, not in `schemas/`, so it sits next to its only reader. | `schemas/CMakeLists.txt` (flatc) |
| `gqa_autotune.h` | Policy API: requests, configs, `GqaTuneSource`. | — |
| `gqa_autotune.cpp` | Key encoding, the four probes, the split clamp, the loader. | `lib/Runtime/CMakeLists.txt` (bitcode) |
| `lut/*.json` | The tables, per arch. Reviewable source of truth. | `build_lut.py` |
| `lut/*.fb` | What the runtime loads, produced from the JSON by `flatc`. | `flatc` |

`gqa_autotune.cpp` is host code compiled into the runtime bitcode: it needs
flatbuffers and the EP `FileSystem`. It contains no device code; `.hip` is not
involved.

## Using it

```c
hipdnn_ep::GqaDecodeRequest request{
    kv_dtype, batch, num_heads, kv_heads, head_dim,
    effective_skv, kFlashDecodeMaxSplits, /*local_window=*/0};
auto selected = hipdnn_ep::gqa_autotune_resolve_decode(policy, request);
// selected.source is Geometry / HeadGroup / Length / Fallback / Heuristic
```

## Filling the tiers

Every tier is **data**: widening what the table answers means adding rows, never
writing another matcher.

```
lut/gfx1151.json    # source of truth, reviewed in PRs
lut/gfx1151.fb      # what the runtime loads
```

Regenerate the binary after the JSON changes:

```bash
cd lut && flatc --binary --strict-json -o /tmp ../gqa_autotune.fbs gfx1151.json
cp /tmp/gfx1151.bin gfx1151.fb
```

Rows are generated from measurement, not by hand. The pipeline — shape grid, both
timing harnesses, row selection and the quality report — is documented in
`RdpCapture/ops_analyze/gqa/capture_autotune_lut.md`, which also records what the
current table costs and which shapes it is weakest on. Two questions have their own
tools:

- `tools/check_model_coverage.py` replays these probes for a catalog of shipped LLM
  geometries and prints the tier each one lands on. That, not the row count, is the
  answer to "is my model covered?".
- `tools/score_table.py` scores a shipped table against measurements it was not built
  from — held-out geometries, or a CI run. That is the answer to "what does a model
  nobody measured get?", and for this table it is 93.5% of optimum by total time on
  decode and 98.4% on prefill.
- `tools/validate_lut_json.py` checks what `flatc` does not: duplicate keys, rows
  the loader will refuse, and a phase with no `Fallback` row.
- `tools/check_fallback.py` is the gate for after a kernel change: it holds out a
  geometry, rebuilds through the real selection path, scores what the held-out shapes
  get, and refits `kBlocksPerCu` against the current readings. See below.

`lut/README.md` states what the current gfx1151 table covers.

## Keeping it current

**Read this before changing `../../gqa_kernel.hip`.** The table is a claim about
kernels that stop existing the moment one is edited, and the expensive mistake is not
rebuilding it — it is rebuilding it from readings the old kernel produced, which
fails silently and ships a table that is confidently wrong.

Four steps. The first three update the exact half of the policy, the fourth checks
the half that answers shapes nobody measured.

```bash
# 1. Stamp what you changed, in the same commit as the change.
#    kKernelVersions in gqa_autotune_sweep.cpp -- bump the entry for the kernel you
#    touched (not for a rename; only for something that can move which config wins).
#    This is what stops the measurement store reusing that kernel's old readings,
#    and it is per kernel: bumping v7 leaves the decode grid valid.

cd RdpCapture/ops_analyze/gqa

# 2. Measure. The store is asked what is missing first, so this costs what you
#    changed, not what the table contains. Everything from empty is 172 min
#    unattended; one kernel is typically 15-40 min.
powershell -File tools/run_hipevent_sweeps.ps1

# 3. Rebuild, check, pack. All four tiers come out of this one pass.
LUT=../../../hip-ep/lib/Runtime/Kernels/hip/autotune/gqa
python tools/build_lut.py --store --prune-tolerance 1.02 \
    --selection data/gqa_lut_selection.csv --json $LUT/lut/gfx1151.json
python tools/validate_lut_json.py --fbs $LUT/gqa_autotune.fbs \
    --json $LUT/lut/gfx1151.json
(cd $LUT/lut && flatc --binary --strict-json -o /tmp ../gqa_autotune.fbs gfx1151.json \
    && cp /tmp/gfx1151.bin gfx1151.fb)

# 4. Gate the fallback half. Seconds, no GPU. Exits non-zero on a regression.
python tools/check_fallback.py
```

Step 3 needs no separate work for the coarse tiers: `HeadGroup`, `Length` and
`Fallback` rows are selected from the same measurements as `Geometry`, in the same
pass, so they cannot lag behind it. What rebuilding cannot tell you is whether they
still *transfer* — whether a geometry nobody measured still lands near its optimum
through them — and that is a measurement, which is step 4. Step 4 also refits
`kBlocksPerCu` in `gqa_autotune.cpp`: every other layer here is regenerated from
data, that one is a constant fitted once, so it is the one that goes stale in silence.
It is only reached when no table loads at all, so a drift there is not an outage — but
finding out the fuse is the wrong size before you need it is worth the seconds.

What invalidates what:

| Change | Bump | Costs |
|---|---|---|
| A kernel's tiles, inner loop, or candidate set | its `kKernelVersions` entry | remeasure that kernel |
| How the sweep times | `kDecodeHarnessVersion` / `kPrefillHarnessVersion` | remeasure that phase |
| A `GqaTuneConfig` name's meaning | `kGqaKernelAbi` in `gqa_autotune.h` | shipped `.fb` is rejected, not misread |
| A new arch or ROCm version | — | loader checks `gpu_arch` / `rocm_version` |

Two cases need a hand beyond a bump:

- **A new candidate config** needs its name in `GqaTuneConfig`, in
  `prefillV5/V7/V8Candidates` or `flashDecodeCandidateSplits` (what the harnesses
  enumerate), and in the runtime's knob decoder. Then remeasure the kernel: a new
  candidate can win shapes an existing row already answers, and a row is only as good
  as the candidate set it was chosen from.
- **A new `(head_dim, heads-per-group)` pair becomes legal** — extend
  `flash_decode_geometry_ok`, add the pair to `GEOMETRIES` in `gen_lut_grid.py`, and
  measure that grid. Nothing already in the store is invalidated. Until it is
  measured the pair is answered by the `Length` tier.

Readings are never deleted by a bump; they stay in the store labelled with the
version they measured, so `measurement_store.py stats --any-kernel-ver` can show what
a kernel change did to the numbers it changed.

`test/runtime/test_gqa_autotune.cpp` covers the probe order, the quarter-octave
labels, the split clamp, the parallelism tier, the config names and every rule
`rowConsistent()` enforces. It is GPU-free and compiles the same
`gqa_autotune.cpp` that goes into `runtime.bc`.

## Packaging

```text
gqa_autotune_lut=/path/to/lib/Runtime/Kernels/hip/autotune/gqa/lut/gfx1151.fb
```

`HIPDNN_GQA_LUT_FILE` overrides the logical filename at runtime, and
`HIPDNN_GQA_AUTOTUNE_MODE=online` bypasses the table entirely and benchmarks on the
GPU as before. Deciding online is not free: the production tuner issues 12-96
launches for a decode shape and 526-1518 for a prefill one, up to 1769 s on the
worst single shape. A table lookup is zero launches.
