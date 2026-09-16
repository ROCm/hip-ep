# GQA autotune

How a GQA launch config is chosen, and where each piece lives.

**Changing a GQA kernel? Start at [Keeping it current](#keeping-it-current).** The
table is measurements of a specific kernel built with a specific compiler; the
expensive mistake is not a stale table but one rebuilt from readings the old kernel
produced, which fails silently.

## How a config is resolved

A config is resolved by the `hip_gqa_autotune_resolve_decode/prefill()` shims
exported from `custom_kernels_<arch>` (implemented in `gqa_autotune.cpp`). The lookup
is **nearest-neighbour**, matmul_nbits style, and hierarchical — most-impactful axis
first:

1. **Kernel identity — exact match.** `phase` (Decode / PrefillV5 / PrefillV7 /
   PrefillV8), `head_dim` (D64/D128/D256), `kv_dtype` (Fp16/Int8). These are not
   distances: `d64` and `d128` are different kernel instantiations, and prefill's
   variant *is* its head_dim, so borrowing across them is meaningless.
2. **Geometry.** The exact `(num_heads, kv_num_heads)` pair is preferred; if it was
   never measured, the search stays within the same **heads-per-group** ratio (the
   axis the kernels template on), never crossing to a different ratio.
3. **Lengths — nearest measured point in weighted log2 space** over `num_heads`,
   `kv_num_heads`, `batch`, `seq_q`, `seq_kv`:

   ```
   d = sqrt( Σ (w_dim * log2(query_dim / point_dim))^2 )
   ```

   Distance 0 — every dim equal — is an **Exact** hit; otherwise **Nearest**.

If a geometry has no usable point, the per-`(phase, head_dim)` **Fallback** row
answers. If no table loaded at all (arch/schema/ABI mismatch, or an empty embed),
the compiled-in **Heuristic** answers. Both last resorts return a runnable config, so
the op never fails for lack of a table. `GqaTuneSource` reports which of
`Exact / Nearest / Fallback / Heuristic` was used — worth logging.

The tiering lives in the resolver, not the schema, so a later experiment can
re-order or re-tier the axes (or refit the weights) without regenerating the table.
One distance function serves both the exact-geometry search (where the head terms
are zero, so it reduces to lengths + batch) and the same-hpg fuzzy search (where the
head terms pick the closest measured head pair).

### The validator keeps a stale or illegal config from being launched

A candidate config is checked before it is returned. For decode: WMMA is templated
only for some `(head_dim, heads-per-group)` pairs, the split count must be in
`1..64`, and the KV tile height must be a compiled one. A candidate that fails is
rejected and the search moves to the next-nearest point — the same self-correction
matmul_nbits uses — so a table measured against a wider kernel set than the running
one degrades to the nearest *usable* point instead of launching something illegal.

### Decode split counts are clamped

A resolved decode config has its `splits` clamped to `ceil(effective_len/16)`, the
splits that have work to do. This is what lets one measured point serve a whole
neighbourhood of lengths: the stored split count is what the top of the
neighbourhood wants, and the clamp makes the same point right at a shorter length.
`effective_len` is `min(seq_kv, window)` — a sliding window shows up here, as a
smaller scanned length, which is why the table needs no separate windowed rows.

## The table layout

The table is FlatBuffers, one arch per file, and de-duplicates configs the way
matmul_nbits does:

- **`configs[]`** — every distinct launch config once. A config is `kind`
  (`Decode`/`Prefill`) plus its knobs: decode carries `use_wmma` + `bkv`; prefill
  carries the v5/v7/v8 tuple (`m_tiles`, `bkv`, `nw`, `mt`, `nd`). At most 256.
- **`points[]`** — one measured winner per shape: the categorical key, a 1-byte
  `config` index into `configs[]`, a per-point `splits` (decode only), and the
  literal numeric dims `num_heads / kv_num_heads / batch / seq_q / seq_kv`. Thousands
  of points cost one config byte each instead of inlining the knobs.
- **`fallbacks[]`** — one per `(phase, head_dim)`: a `config` index + `splits`, the
  runnable last resort.
- **weights** — `weight_num_heads / _kv_num_heads / _batch / _seq_q / _seq_kv`, the
  per-dim log-space weights. They live in the table so the offline fit and the
  runtime metric cannot drift apart; refitting is a table regeneration.

`Any = 0` on every categorical enum, so a memset field reads as "unclassified" and
is rejected at load rather than silently matched. `pointConsistent()` drops a point
whose key is unclassified, whose dims are non-positive, or whose config kind does not
match its phase; `compatible()` rejects the whole table on a `schema_version`,
`kernel_abi`, or `gpu_arch` mismatch. A rejected row is silent coverage loss, so the
loader counts them and `test-gqa-autotune` asserts the count is zero.

## Where it lives — in the kernel DLL, not runtime.bc

The resolver **and** its table ride inside the per-arch `custom_kernels_<arch>`
shared library, next to `gqa_kernel.hip`:

- `lut/<arch>.fb` is embedded by `lib/Runtime/Kernels/CMakeLists.txt` with pure
  CMake (`file(READ ... HEX)` → a `kGqaLutData[]` C array), so the hip-ep build needs
  no Python and no generated `.cpp` is committed. An arch with no `.fb` gets a size-0
  stub and falls to the heuristic.
- `gqa_autotune.cpp` reads that array as an **in-DLL data symbol**. It is *not*
  compiled into `runtime.bc` any more.
- `real/gqa.cpp` (compiled to bitcode) reaches the resolver through the exported
  `extern "C"` `hip_gqa_autotune_*` shims — the same way it already calls
  `hip_gqa_flash_decode_configured`.

This mirrors matmul_nbits, and it is deliberate: the JIT resolves DLL **function**
symbols reliably, but a **data** symbol does not cross that boundary safely on
Windows. Moving the table into the DLL means the only cross-boundary symbols are
functions. (The old design embedded the table in `runtime.bc`; this replaced it.)

## Files here

| File | Role | Built by |
|---|---|---|
| `gqa_autotune.fbs` | LUT **schema**. Lives here, next to its only reader. | `schemas/CMakeLists.txt` (flatc) → `gqa_autotune_generated.h` |
| `gqa_autotune.h` | POD request/config/result structs, `GqaTuneSource`, and the `extern "C"` `hip_gqa_autotune_*` C-ABI. | — |
| `gqa_autotune.cpp` | The loader + nearest-neighbour resolver + validator + heuristic + the exported shims. | `lib/Runtime/Kernels/CMakeLists.txt` (into `custom_kernels_<arch>`) |
| `lut/<arch>.json` | The table, per arch. Reviewable source of truth (one entry per line). | `scripts/update_lut.py build` |
| `lut/<arch>.fb` | What the DLL embeds, produced from the JSON by `flatc`. | `scripts/update_lut.py compile` |
| `scripts/update_lut.py` | `plan` / `measure` / `build` / `compile` pipeline. | — |
| `../../test/example/gqa/autotune/` | the GPU sweep driver + Makefile. | — |

## Using it

The EP calls the C-ABI (POD structs by pointer):

```c
hipdnn_ep::GqaDecodeRequest request{
    kv_dtype, batch, num_heads, kv_num_heads, head_dim,
    effective_skv, kFlashDecodeMaxSplits, /*local_window=*/0};
hipdnn_ep::GqaDecodeResult selected;
hip_gqa_autotune_resolve_decode(policy, &request, &selected);
// selected.source is Exact / Nearest / Fallback / Heuristic
// selected.distance is the log2 distance to the point that answered (0 on Exact)
```

`hip_gqa_autotune_create()` builds the session policy (mode + CU count) and loads the
embedded table once; `hip_gqa_autotune_destroy()` frees it.

## Updating the LUT

The pipeline is `scripts/update_lut.py`. The full per-step commands, environment
requirements, and acceptance checks are in
[lut/README.md](lut/README.md); the short version:

```bash
SCRIPTS=lib/Runtime/Kernels/hip/autotune/gqa/scripts

python $SCRIPTS/update_lut.py plan      # list shapes to measure -> lut/shapes_gfx1151.json (config=TBD)
python $SCRIPTS/update_lut.py measure --sweep <sweep-exe>   # GPU sweep -> scripts/data/*_best.csv
python $SCRIPTS/update_lut.py build     # *_best.csv -> lut/gfx1151.json (configs[]/points[])
python $SCRIPTS/update_lut.py compile --flatc <flatc>       # -> lut/gfx1151.fb
```

`build` reads `data/*_best.csv` (the sweep's per-shape winners, with decomposed
`cfg_*` columns), de-duplicates configs, emits one point per measured shape (literal
dims, no bucketing), then **prunes saturated points** — within each exact-geometry
group it drops any point a nearest-neighbour lookup can reconstruct exactly (a run
of lengths sharing a winner collapses to its endpoints), verified answer-preserving
by replay. It finally writes the fixed per-`(phase, head_dim)` fallbacks. The
weights default to 1.0 (a `--fit-weights` step is a future addition; prefill's
`seq_kv` weight should end up near zero, since a prefill config is essentially
independent of KV length — which is also why so many prefill points prune away).

**Do not use `--rdpcapture`.** That path delegates to RdpCapture's `build_lut.py`,
which still emits the old tier schema and would produce a table this loader rejects.
The standalone (best-csv) path above is the supported one.

Adding a model geometry: add its `(H, G, d)` to `GEOMETRIES` in
`scripts/update_lut.py` and re-run `plan` → `measure` → `build` → `compile`.

## Keeping it current

**Read this before changing `../../gqa_kernel.hip`.** The table is a claim about
kernels — and about the compiler that built them. Two rules:

- **Sweep with the compiler the model runs under** (currently HIP 7.14 / clang 23).
  A different clang can reorder the winning configs (this cost 20-30x on prefill v7
  once), so a table swept on the wrong toolchain is confidently wrong.
- **Stamp kernel changes.** Bump the kernel's entry in
  `gqa_autotune_sweep.cpp :: kKernelVersions` in the same commit as the kernel change
  (this marks old readings stale), and re-measure. Bump `kGqaKernelAbi` (`gqa-v3`) in
  `gqa_autotune.h` when a config's *meaning* changes, so a stale shipped `.fb` is
  rejected rather than misread.

| Change | Bump | Costs |
|---|---|---|
| A kernel's tiles, inner loop, or candidate set | its `kKernelVersions` entry | remeasure that kernel |
| A config knob's meaning | `kGqaKernelAbi` (+ `kSchemaVersion` on a layout change) | shipped `.fb` is rejected, not misread |
| A new arch / ROCm version | — | loader checks `gpu_arch`; ROCm mismatch only warns |

`test/runtime/test_gqa_autotune.cpp` links the real embedded `.fb` and drives this
resolver GPU-free (arch from `HIPDNN_EP_GQA_ARCH`). It asserts the table loads with
zero rejected rows, that measured shapes resolve from the table (not the heuristic),
that an unmeasured length resolves as nearest, that the split clamp holds, and that
fallbacks are runnable. It compiles the same `gqa_autotune.cpp` that ships in
`custom_kernels_<arch>`.

## Runtime controls

| Control | Effect |
|---|---|
| `HIPDNN_GQA_AUTOTUNE_MODE=lookup\|online` | `lookup` (default) resolves from the table; `online` bypasses it and benchmarks on the GPU (the pre-table path, kept for A/B). Read once per session in `hip_gqa_autotune_create()`. Mirrors matmul_nbits' `HIPDNN_MATMUL_AUTOTUNE_MODE`. |
| `HIPDNN_GQA_LUT_LOG=1` | logs the table load (`loaded N points ...`) and each shape's resolution — `exact`/`nearest` (with the distance), or `fallback`/`heuristic` — so a run's coverage is visible from this one switch. Each distinct line is printed once per process (resolve runs per token, so this dedup keeps a served model from flooding). Mirrors `HIPDNN_MATMUL_LUT_LOG`. Verification only — do not benchmark with it on. |
| `HIPDNN_GQA_AUTOTUNE_LOG=1` | in `gqa_kernel.hip`, logs the config the kernel actually launched (in both `lookup` and `online` modes), so an `online` run can be diffed against the shipped table. Mirrors `HIPDNN_MATMUL_AUTOTUNE_LOG`: setting it alone prints **only** these autotune lines (not the rest of the debug firehose), while `HIPDNN_EP_DEBUG=1` also pulls them in as part of the full `[custom_kernels]` output. |
