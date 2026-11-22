# Kernel unit-test leaf guidelines

These are the rules for every kernel unit-test leaf under
`lib/Runtime/Kernels/unit_test`. A new leaf, and any change to an existing
one, follows this document; a reviewer checks a leaf against it.
[README.md](README.md) covers building, running, the packaged controller,
`out/results.csv` and LUT refresh.

- [Scope](#these-are-unit-tests--comprehensive-shapes-not-a-perf-sweep)
- [Leaf name](#leaf-name--tensor-dtypes-not-a-single-fp16)
- [Required files](#required-files-in-every-leaf--exactly-these-three-nothing-else)
- [Required capabilities](#required-capabilities-in-every-leaf)
- [Shape coverage](#shape-coverage-3-tiers)
- [Required `ut_interface.json` entry](#required-entry-for-every-new-leaf)
- [Adding an op or dtype leaf](#adding-an-op-or-dtype-leaf)

Leaves that predate these rules record what they still lack as `gaps` in
`ut_interface.json`; that list is frozen and only shrinks.

## These are unit tests — comprehensive shapes, not a perf sweep

A kernel unit test's job is to catch correctness regressions across a
**comprehensive** shape surface in well under the ~30 min tier3 budget, not
to characterize performance across the whole *continuous* shape space (that
is what `hip/autotune/<op>/tools/` is for). The coverage model is:

> **all categorical situations, fully** (every mode/dtype/flag the op
> supports) **x a comprehensive set of typical shapes** (multiple M
> including M=1 GEMV/decode and several M>1 prefill points, crossed with
> multiple representative `(K,N)` / `(B,H,G,D,seq)` scenarios), **not the
> full continuous shape space** (no blind `1..4096`-style ladder, and never a
> giant perf shape like `N≈201088`, `M≈4096` at real FFN size).

Concretely:

- **Categorical axes** (group_size, zero-points on/off, TA/TB transpose,
  bias on/off, heads-per-group ratio, head_dim,
  sliding-window/head-sink/smooth-softmax, …) are enumerated in full and run
  at **every** `COVERAGE` tier. Tensor dtypes are not an axis inside a leaf:
  each dtype combo is its own leaf (see "Leaf name" below).
- **Continuous axes** (M/K/N, sequence length) are a comprehensive, explicitly
  listed set of typical values — e.g. matmul_nbits M in
  `{1,16,64,128,512[,1024]}` round-robined across several real `(K,N)` layer
  families (gate/up-proj, down-proj, attn-proj, square); GQA a widened
  typical context-length / prompt-length list (e.g. `{128,512,2048,8192}`)
  — still never the full `1..16384`-style ladder, and never a giant perf
  shape. `COVERAGE=1|2|3` only thins how many of those typical values run
  (tier3 = the full comprehensive set, tier1 = a minimal fast subset); it
  never drops a categorical situation.

A tier3 (default) run of one leaf is on the order of tens to ~150 cases, and
the whole 17-leaf suite at tier3 is budgeted at **up to ~30 min total**
(measured ~20-25 min in practice) — this is intentionally wider than a
"finishes in a minute" smoke test, because the shape axis (M=1 vs M>1,
multiple K/N/seq scenarios) needs real coverage too, not just the categorical
axis. `COVERAGE=1` (tier1) stays a fast, seconds-to-low-minutes path for
quick dev iteration. The CPU references that were plain single-threaded
loops are now multithreaded (`std::thread`, chunked over the independent
output rows/columns/query-heads) specifically so this wider shape set fits
the budget — see each leaf's `test_<op>.cpp` and README for the threading
details.

## Leaf name = tensor dtypes, not a single "fp16"

A leaf folder is **one (X, W, Y) dtype combo**, plus optional suffixes for
storage/quant extras.

```
X<xdtype>_W<wdtype>_Y<ydtype>[_ZP<zpdtype>][_other]
```

| Token | Meaning |
|---|---|
| `X` | activation / input (MatMulNBits A, GQA Q, GatherBlockQuantized indices) |
| `W` | weight / KV (MatMulNBits packed B, GQA K/V cache, GatherBlockQuantized data) |
| `Y` | output (MatMulNBits C, GQA O, GatherBlockQuantized output) |
| `_ZP…` | only if zero-point storage is a distinguishing feature of this leaf (`_ZPu8`, `_ZPfp16`, …) |

Examples:

| Leaf | What it is |
|---|---|
| `matmul_nbits/Xfp16_Wu4_Yfp16` | A fp16, B uint4, C fp16 |
| `matmul_nbits/Xfp16_Wu2_Yfp16` | bits=2 |
| `matmul_nbits/Xfp16_Wu3_Yfp16` | bits=3 |
| `matmul_nbits/Xfp16_Wi8_Yfp16` | unpacked int8 weights |
| `matmul_nbits/Xfp32_Wu4_Yfp32` | A fp32, B uint4, C fp32 (likewise `Wu2`, `Wu3`, `Wi8`) |
| `gqa/decode/Xfp16_Wfp16_Yfp16` | Q/O fp16, KV cache fp16 |
| `gqa/decode/Xfp16_Wi8_Yfp16` | Q/O fp16, KV cache int8 |
| `gqa/prefill/Xfp16_Wfp16_Yfp16` | same for prefill |
| `gqa/prefill/Xfp16_Wi8_Yfp16` | prefill INT8 KV |
| `gqa/softmax/Xfp32_Wbf16_Ybf16` | decomposed-path softmax: fp32 scores, bf16 mask, bf16 probabilities |

If input becomes fp8 later: add `Xfp8_Wfp16_Yfp16`, do **not** rename the
existing fp16 leaf to a bare `fp16/` or `fp8/`. A folder named `fp16` or `i8`
is illegal — it does not say which tensor.

Do **not** use `fp16u4`, `decode/fp16`, `prefill/i8`.

## Required files in every leaf — EXACTLY these three, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_custom` / `clean` / `build` (`build` also copies the exe to `PKG_DIR/PKG_NAME` for the `unit_test/` aggregator) |
| `test_<op>.cpp` | Driver. Entirely self-contained: generates its own inputs (seeded RNG) and computes its own CPU reference **in-process, in C++** — no python, no on-disk data. Shared code comes only from `unit_test/common/` (timer, result binding, ranking). Enumerates the op's categorical situations x typical shapes in C++ (see "Shape coverage" below). Inline empty `resolve()` if needed, guarded by `#ifndef HIPDNN_LUT_LINKED_EXTERNALLY` so `MODE=lookup`'s real resolver (which defines the same symbols) can link. No `autotune_stub.cpp`. |
| `README.md` | Targets, `test_custom` parameters (the same ones as its `ut_interface.json` entry), `MODE`, `COVERAGE`, `TIMING`, supported `OFFLOAD` targets, and `RANK` when the op has a LUT |

Outside its directory, every leaf also needs the registrations in
[Adding an op or dtype leaf](#adding-an-op-or-dtype-leaf), including its
`ut_interface.json` entry.

No `gen_data.py`, no python in a leaf directory, no `shapes.csv`. Shared code
lives only in `unit_test/common/` (adaptive timer, result binding, ranking and
the LUT tools' helpers); each leaf keeps its own coverage-tier resolver and CPU
reference (see [Shape coverage](#shape-coverage-3-tiers) and
[`out/results.csv`](README.md#outresultscsv)).

Forbidden filenames in a leaf: `gen_data.py`, `gen_matmul_nbits*_data.py`,
`gen_model_data*.py`, `bench_*.py`, `verify_*.py`, `zp_perf_compare.md`,
`autotune_stub.cpp`, `test_model/*.json`, `shapes.csv`, any committed `out/` /
`data/`, any home path. If a leaf has a real extra need, justify it in its
README — default is the three files above and nothing more.

`BUILD_DIR=out`.

## Required capabilities in every leaf

A new leaf is complete only when it has all of these. Use the MatMulNBits and
GQA leaves as references.

| Capability | Requirement |
|---|---|
| Correctness | Generate inputs and a CPU reference in-process; compare every case and return non-zero on any failure. |
| Custom shape | `make test_custom <op parameters>` runs exactly the user-given shape through the same correctness check and timer as the suite (for example `SIZE= GS=` for MatMulNBits, `B= H= G= D= SQ= PAST=` for GQA prefill). Provide defaults and document every parameter in the leaf README and the Makefile header. Passing only `ARGS="--coverage 1"` does not count. |
| Coverage tiers | `--coverage 1\|2\|3` / `COVERAGE=1\|2\|3`, as described in [Shape coverage (3 tiers)](#shape-coverage-3-tiers). |
| Timing precision | Time every case with the shared timer in `common/hip_adaptive_timing.h` (`timing::adaptiveRotatedMeanMs` inside a `timing::CaseScope`, and `timing::emitCsvBinding` for the CSV row), so it emits `UT_TIMING` / `UT_TIMING_BINDING` records. Accept `TIMING=screen\|standard\|confirm` (Makefile `TIMING ?= standard`, passed as `HIPDNN_UT_TIMING`) and the optional `ADLX_INCLUDE`. Do not add a private timer or a fixed iteration count. |
| LUT refresh (ops with an offline LUT) | Build a ranking variant with `make build RANK=1` through a UT-only adapter (`common/<op>_rank_adapter.hip`) that compiles the production source unchanged and switches final launches in one process. Support `HIPDNN_UT_RANK=<csv>`, `HIPDNN_UT_RANK_CATALOG=1` and `HIPDNN_UT_RANK_REFERENCE=1`; check every candidate against the CPU reference before timing, with the no-LUT default as candidate zero. `hip/autotune/<op>/scripts/update_lut.py` provides `rank-catalog`, `rank-candidates`, `rank-seed --from-lut`, `rank-measure`, `rank-report`, `rank-build` and `rank-reference-candidates`, and the op's checked-in JSON table is added to the roundtrip in `common/test_ut_lut_from_table.py`. See [Refreshing a LUT with the UT leaves](README.md#refreshing-a-lut-with-the-ut-leaves). |
| Production mode | `MODE=lookup\|autotune` (Makefile `MODE ?= lookup`) sets both the linked resolver and the kernel's own `HIPDNN_<OP>_AUTOTUNE_MODE=lookup\|online`. `lookup` resolves from the table embedded in the production build and falls through to autotune on a miss; `autotune` never reads a table. A missing table is never a failure. See [`MODE=lookup`](README.md#modelookup-default--real-lut-when-one-exists). |
| Config identity | Every row's `config` names the configuration that actually ran, read back from the op's own diagnostics or resolver with its source prefix (`lookup:`, `autotune:`, `fixed:` or `declined`), never inferred from `MODE`. `unlogged` is a gap to fix. See [`out/results.csv`](README.md#outresultscsv). |
| Architectures | gfx1151 is the default validation chip. The leaf builds and runs with the default `OFFLOAD=--offload-arch=gfx11-generic` (gfx115x) and with `OFFLOAD=--offload-arch=gfx1170` or `gfx1171` (gfx11-7-generic); `LUT_ARCH` selects that chip's table. Do not hard-code a chip in the Makefile or driver; a case that supports only some chips checks the device and prints `SKIP` on the others. |
| Results | One `HIPDNN_RESULTS_CSV` row per case when set, and registration in the controller (see [Adding an op or dtype leaf](#adding-an-op-or-dtype-leaf)). |
| Interface record (required) | An entry in `ut_interface.json` with the package executable, capabilities, and every suite, custom and diagnostic flag with its Make variable, defaults and constraints, and `gaps: []`. The `kernel-ut-interface` pre-commit hook checks it in CI. See [Required entry for every new leaf](#required-entry-for-every-new-leaf). |

## Shape coverage (3 tiers)

Every leaf's `test_<op>.cpp` defines, near the top of the file:

- a small, explicit list (or nested loop) of the op's **categorical
  situations** — every group_size / zero-points / transpose / bias /
  head-geometry / window-sink-smooth value the op actually supports for the
  leaf's dtypes. This list is the same size at every tier.
- a **short list of typical shapes** on the continuous axes (M/K/N or
  B/H/G/D/seq) — a handful of values, not a ladder. `COVERAGE` selects a
  subset of *this* list only (tier3 = all of it, tier2 a smaller subset,
  tier1 the minimal one — e.g. "1 decode + 1 prefill" point).

A CLI flag `--coverage N` (also settable via env `HIPDNN_UT_COVERAGE`,
default **3**) picks the tier. At startup the exe prints something like
`coverage=N -> running <n> typical length(s) x <k> categorical case(s) = <n*k>
<op> cases` so the actual case count is always visible. `make test
COVERAGE=N` (any leaf, or the top-level aggregator) builds and runs it.

**Per op family:**

- **gqa decode/prefill (fp16)**: a fixed list of named real-model +
  geometry-sweep cases (17 for decode, 75 hand-authored rows for prefill:
  the original 29, short `sq=128` and long pure-prefill `sq=8192` rows, a
  set that walks every prefill kernel route, and head_dim 96)
  crossed with (decode) or fused with (prefill) a typical context-length /
  prompt-length list (decode: `{128,512,2048,8192}`). Prefill's `O(sq^2)`
  causal-attention CPU reference is multithreaded over `(batch,
  query-head)` pairs.
- **gqa decode/prefill (i8)**: same idea, 13 / 9 named geometries x a typical
  length list (decode: `{128,512,2048,8192}`; prefill:
  `{128,256,512,1024,2048}`). The geometries cover head_dim 64, 96 and 128,
  the range `real/gqa.cpp` accepts for an INT8 KV cache (d256 is rejected
  there). The i8 prefill leaf's two `O(sq^2)` CPU references (i8-cache +
  fp16) are also multithreaded over `(batch, query-head)` pairs.
- **matmul_nbits** (all four bit-widths, fp16 and fp32 activations as
  separate leaves): `group_size {32,64,128} x zero-points {on,off}` (6
  combos, always full) x 12 typical `(M,K,N)` shapes — M in
  `{1,16,64,128,512}` round-robined (not a full M x KN cross) across 4
  representative real-model `(K,N)` layer families (gate/up-proj, attn-proj,
  down-proj, o-proj/square). Every shape is ALSO crossed with the full 6-way
  categorical set, so the shape-list size directly multiplies CPU-reference
  cost by 6 — the reference
  (`genCase()`'s `O(M*K*N)`, no BLAS) is now multithreaded over `N` to keep
  this affordable; see the comment above `genCase()` in any
  `matmul_nbits/*/test_matmul_nbits*.cpp`.
- **qmoe_ragged_mm**: routing `{balanced, empty experts, skewed}` x input
  `{contiguous, FC1 gather}` x bias `{off,on}` x zero points `{off,on}` (24
  combos) on a shape list COVERAGE thins (tier1 = 1 shape, tier2 = 2, tier3 =
  4 plus the GPT-OSS-20B FC1/FC2 dimensions). Every tier also runs the row
  counts on both sides of each GEMV / WMMA16 / WMMA64 boundary, the bucket
  descriptors, the task-policy and launch-grid selectors, and one
  FC1 -> SwiGLU -> FC2 pipeline.
- **gather_block_quantized** (u4, u8 and fp8 weights as separate leaves):
  compact rows `{off,on}` x indices rank `{1,2}`, crossed for u4 / u8 with
  zero points `{off,on}` x signedness `{unsigned,signed}` (16 combos) and for
  fp8 with scale granularity `{per-tensor [1,1], per-row, blockwise}` (12
  combos), on a shape list COVERAGE thins (tier1 = 2 shapes, tier2 = 4,
  tier3 = 6; rows `1024..8192`, row length `64..3840`, block size
  `{16,32,64,128}`, `{1,16,2048}` indices). Every index list includes the first
  and last rows, negative and out-of-range indices. Outputs are compared
  bit-exactly with the CPU reference.
- **cast** (one leaf, `Xany_Yany_bf16`, for every cast with bf16 on one side;
  see its README for why it is not one leaf per pair): bf16 <-> {f32, f16,
  f64, i64, i32, i16, i8, u8} and three unsupported pairs, on sizes COVERAGE
  thins (tier1 = 2 sizes, tier2 = 3, tier3 = 4, up to ~1M elements). Outputs
  are compared bit-exactly with a CPU reference.
- **div** (`Xbf16_Wbf16_Ybf16`): bf16 division over every pair of bf16 special
  values plus seeded random values, and one unsupported dtype, on sizes
  COVERAGE thins (tier1 = 2 sizes, tier2 = 3, tier3 = 4). Outputs are
  compared bit-exactly with an fp32-divide-then-round CPU reference.
- **gqa/softmax** (`Xfp32_Wbf16_Ybf16`): the decomposed-GQA softmax with bf16
  probabilities, unbiased and with full, broadcast one-row and row-offset bf16
  masks (plus fp16/fp16 and fp32/fp32 mask/output regression cases), and
  three rejected calls, on shapes COVERAGE thins (tier1 = 3 shapes, tier2 = 5,
  tier3 = 6). Outputs must be within one unit in the last place (plus a few
  fp32 ulps of compute error) of a double CPU reference.

A new op's test cpp should follow whichever of the above shapes fit it best:
if the kernel's dispatch paths are better covered by a small hand-picked
matrix than a blind cross product, hand-enumerate it (gqa style); if the
categorical axes are truly independent, build the real cross product
(matmul_nbits style). Either way, the array (or the function that builds it)
is the single source of truth for what "typical" means for that op.

## Required entry for every new leaf

A new leaf is not complete without its [`ut_interface.json`](ut_interface.json) entry; it is the
only place an external caller learns how to run the leaf. [External invocation](README.md#external-invocation-ut_interfacejson)
describes the file for callers. The entry has:

- `leaf`, `package_executable` and `source`, matching `kLeaves` and
  `CMakeLists.txt`.
- `capabilities`: `coverage: true`, `timing: "shared"`, `mode` (`lookup`
  for an op with a table, otherwise `label_only`) and `rank` (`true` for an op
  with an offline LUT).
- `suite.args` and `suite.coverage_default`.
- `custom`: `selected_by` and one `params[]` item for every op input of a
  single case, so a caller can reproduce any model shape without reading the
  driver: `name`, `flag` (or `positional` with `format`), `make` (the
  `make test_custom` variable, or `null` with a reason in `description`),
  `type`, `choices` when restricted, `required`, `exe_default`,
  `make_default` and `description`; plus `constraints` in words and a
  `checks` expression for each one a script can test.
- `examples`: at least one custom case by parameter `values`, a real model
  shape that finishes in seconds. Before delivery,
  `python common/ut_interface.py verify --bin <package bin> --leaf <leaf>`
  passes on a GPU.
- `diagnostic_flags` and the leaf's own `env` variables, if any.
- `gaps: []`. Only leaves that predate this standard may list gaps, and only
  those already frozen in `common/test_ut_interface.py`.

The `kernel-ut-interface` pre-commit hook, which CI's pre-commit job runs on
every pull request, and `make -C common test` run
`common/test_ut_interface.py`. It fails when a registered leaf has no entry,
an entry names a flag the driver does not parse or a Make variable or default
the Makefile does not define, a capability disagrees with the source, an
example breaks its own parameters or checks, or a leaf gains a gap. Change the file in the same commit as any leaf flag, Make
variable or registration.

## Adding an op or dtype leaf

Update all of these together:

1. Add the production launcher declaration to `include/hip_custom_kernels.h`,
   implementation to `hip/<op>_kernel.hip`, and that HIP file to
   `lib/Runtime/Kernels/CMakeLists.txt`'s `_kernel_sources`.
2. Add a leaf directory named from all tensor dtypes, containing exactly its
   `Makefile`, self-contained `test_<op>.cpp`, and `README.md`. The leaf must
   provide every capability in
   [Required capabilities in every leaf](#required-capabilities-in-every-leaf).
3. Add the leaf once to the `LEAVES` list in `unit_test/Makefile` for local
   runs, once to `unit_test/CMakeLists.txt` for package install, and
   once to `kernel_ut_main.cpp`'s `kLeaves` for the controller and CSV report.
4. Add the leaf's entry to `ut_interface.json` (required), with every input of
   its custom case, as described in
   [Required entry for every new leaf](#required-entry-for-every-new-leaf).
5. Add the new leaf to `kernel_ut_main.cpp` without declaring whether it has a
   table. The controller always passes the requested `lookup` mode; production
   code must match the actual dtype/bits, shape, and GPU architecture against
   the DLL's embedded table, then fall through to autotune on a miss.
6. If the op supports lookup, add its resolver/schema and checked-in
   `lut/<arch>.fb` data to the production `custom_kernels_<arch>` build. Do
   not embed the FB in a test exe: `lookup` must consume the DLL's embedded
   table, exactly like model execution. A missing compatible table must fall
   through to autotune.
7. Ensure the leaf writes one row per case to `HIPDNN_RESULTS_CSV` when set.
   The controller also writes one `shape=suite` row per leaf, including
   leaves without per-case rows.
