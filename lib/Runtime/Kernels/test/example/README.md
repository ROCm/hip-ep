# Kernel unit tests (`lib/Runtime/Kernels/test/example`)

This folder contains the source for standalone HIP kernel unit tests.

For CI/package testing, the full hip-ep CMake build creates one small,
leaf-specific executable per test and installs it under
`bin/kernel-tests/<arch>/`.  Each executable dynamically links the installed
`custom_kernels_<arch>.dll`, so the GPU kernels and autotune resolver are
compiled and packaged exactly once.  Run an installed test from `bin/` (so the
shared DLL is on the Windows DLL search path), for example:

```
.\kernel-tests\gfx1151\hipdnn-kernel-ut-matmul-nbits-xfp16-wu3-yfp16.exe
```

Use the package entry point to run the complete suite and aggregate every
leaf's result in `bin/out/results.csv`:

```
.\hipdnn-kernel-ut.exe --mode lookup --coverage 3
```

Its two modes match the production kernel behavior:

- `lookup` (default) reads the FB already embedded in
  `custom_kernels_<arch>.dll`; a test executable never carries a second FB.
  If that DLL has no compatible table, it falls through to autotune.
- `autotune` bypasses the table and selects production's `online` mode.  The
  name translation is only at the test command line; `online` remains the
  kernel ABI spelling.

The leaf Makefiles remain available for standalone kernel development.  That
direct-build path intentionally compiles the selected kernel source itself;
it is not the CI/package path.

Repo-relative paths only. `HIP_SDK` and `OFFLOAD` come from the command line.

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

- **Categorical axes** (group_size, zero-points on/off, dtype fp16/fp32,
  TA/TB transpose, bias on/off, heads-per-group ratio, head_dim,
  sliding-window/head-sink/smooth-softmax, INT8-KV, …) are enumerated in
  full and run at **every** `COVERAGE` tier.
- **Continuous axes** (M/K/N, sequence length) are a comprehensive, explicitly
  listed set of typical values — e.g. gemm/matmul_nbits M in
  `{1,16,64,128,512[,1024]}` round-robined across several real `(K,N)` layer
  families (gate/up-proj, down-proj, attn-proj, square); GQA a widened
  typical context-length / prompt-length list (e.g. `{128,512,2048,8192}`)
  — still never the full `1..16384`-style ladder, and never a giant perf
  shape. `COVERAGE=1|2|3` only thins how many of those typical values run
  (tier3 = the full comprehensive set, tier1 = a minimal fast subset); it
  never drops a categorical situation.

A tier3 (default) run of one leaf is on the order of tens to ~150 cases, and
the whole 11-leaf suite at tier3 is budgeted at **up to ~30 min total**
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
| `X` | activation / input (GEMM A, GQA Q) |
| `W` | weight / KV (GEMM B, MatMulNBits packed B, GQA K/V cache) |
| `Y` | output (GEMM C, GQA O) |
| `_ZP…` | only if zero-point storage is a distinguishing feature of this leaf (`_ZPu8`, `_ZPfp16`, …) |

Examples:

| Leaf | What it is |
|---|---|
| `gemm/Xfp16_Wfp16_Yfp16` | dense GEMM fp16 |
| `gemm/Xbf16_Wbf16_Ybf16` | dense GEMM bf16 |
| `gemm/Xfp32_Wfp32_Yfp32` | dense GEMM fp32 |
| `matmul_nbits/Xfp16_Wu4_Yfp16` | A fp16, B uint4, C fp16 |
| `matmul_nbits/Xfp16_Wu2_Yfp16` | bits=2 |
| `matmul_nbits/Xfp16_Wu3_Yfp16` | bits=3 |
| `matmul_nbits/Xfp16_Wi8_Yfp16` | unpacked int8 weights |
| `gqa/decode/Xfp16_Wfp16_Yfp16` | Q/O fp16, KV cache fp16 |
| `gqa/decode/Xfp16_Wi8_Yfp16` | Q/O fp16, KV cache int8 |
| `gqa/prefill/Xfp16_Wfp16_Yfp16` | same for prefill |
| `gqa/prefill/Xfp16_Wi8_Yfp16` | prefill INT8 KV |

If input becomes fp8 later: add `Xfp8_Wfp16_Yfp16`, do **not** rename the
existing fp16 leaf to a bare `fp16/` or `fp8/`. A folder named `fp16` or `i8`
is illegal — it does not say which tensor.

Do **not** use `fp16u4`, `decode/fp16`, `prefill/i8`.

## Layout

```
example/
  README.md
  CMakeLists.txt             # package targets and install rules
  kernel_ut_main.cpp         # hipdnn-kernel-ut.exe controller
  Makefile                   # direct standalone-development aggregator
  gemm/Xfp16_Wfp16_Yfp16/
  gemm/Xbf16_Wbf16_Ybf16/
  gemm/Xfp32_Wfp32_Yfp32/
  matmul_nbits/Xfp16_Wu{2,3,4}_Yfp16/
  matmul_nbits/Xfp16_Wi8_Yfp16/
  gqa/decode/Xfp16_Wfp16_Yfp16/
  gqa/decode/Xfp16_Wi8_Yfp16/
  gqa/prefill/Xfp16_Wfp16_Yfp16/
  gqa/prefill/Xfp16_Wi8_Yfp16/
```

`example/` is UT-only. The offline autotune LUT sweep tools (not UTs) live under
`hip/autotune/<op>/tools/` next to each op's real resolver — e.g.
`hip/autotune/gqa/tools/`, `hip/autotune/matmul_nbits/tools/`,
`hip/autotune/gemm/tools/`.

## Adding an op or dtype leaf

Update all of these together:

1. Add the production launcher declaration to `include/hip_custom_kernels.h`,
   implementation to `hip/<op>_kernel.hip`, and that HIP file to
   `lib/Runtime/Kernels/CMakeLists.txt`'s `_kernel_sources`.
2. Add a leaf directory named from all tensor dtypes, containing exactly its
   `Makefile`, self-contained `test_<op>.cpp`, and `README.md`. The test must
   generate inputs and a CPU reference in-process, return non-zero on failure,
   and support `--coverage 1|2|3`.
3. Add the leaf once to the `LEAVES` list in `example/Makefile` for direct
   development runs, once to `example/CMakeLists.txt` for package install, and
   once to `kernel_ut_main.cpp`'s `kLeaves` for the controller and CSV report.
4. Add the new leaf to `kernel_ut_main.cpp` without declaring whether it has a
   table. The controller always passes the requested `lookup` mode; production
   code must match the actual dtype/bits, shape, and GPU architecture against
   the DLL's embedded table, then fall through to autotune on a miss.
5. If the op supports lookup, add its resolver/schema and checked-in
   `lut/<arch>.fb` data to the production `custom_kernels_<arch>` build. Do
   not embed the FB in a test exe: `lookup` must consume the DLL's embedded
   table, exactly like model execution. A missing compatible table must fall
   through to autotune.
6. Ensure the leaf writes one row per case to `HIPDNN_RESULTS_CSV` when set.
   The controller also writes one `suite/controller` row per leaf, including
   leaves without detailed timing rows.

## Required files in every leaf — EXACTLY these three, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_custom` / `clean` |
| `test_<op>.cpp` | Driver. Entirely self-contained: generates its own inputs (seeded RNG) and computes its own CPU reference **in-process, in C++** — no python, no on-disk data, no `example/common/`. Enumerates the op's categorical situations x typical shapes in C++ (see "Shape coverage" below). Inline empty `resolve()` if needed, guarded by `#ifndef HIPDNN_LUT_LINKED_EXTERNALLY` so `MODE=lookup`'s real resolver (which defines the same symbols) can link. No `autotune_stub.cpp`. |
| `README.md` | Targets + `MODE` + `COVERAGE` |

No `gen_data.py`, no python anywhere under `example/`, no `shapes.csv`, no
`example/common/` (there is nothing left to share — each leaf inlines its own
~30-line coverage-tier resolver and, where it writes `out/results.csv`, its
own tiny CSV writer; see "Shape coverage" and "`out/results.csv`" below).

Forbidden filenames in a leaf: `gen_data.py`, `gen_matmul_nbits*_data.py`,
`gen_model_data*.py`, `bench_*.py`, `verify_*.py`, `zp_perf_compare.md`,
`autotune_stub.cpp`, `test_model/*.json`, `shapes.csv`, any committed `out/` /
`data/`, any home path. If a leaf has a real extra need, justify it in its
README — default is the three files above and nothing more.

`BUILD_DIR=out`.

## Shape coverage (3 tiers)

Every leaf's `test_<op>.cpp` defines, near the top of the file:

- a small, explicit list (or nested loop) of the op's **categorical
  situations** — every group_size / zero-points / dtype / transpose / bias /
  head-geometry / window-sink-smooth / INT8-KV value the op actually
  supports. This list is the same size at every tier.
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

- **gemm**: dtype is fixed by the leaf (`Xfp16_Wfp16_Yfp16`,
  `Xbf16_Wbf16_Ybf16`, `Xfp32_Wfp32_Yfp32`), not a per-row axis, so the three
  leaves together provide the dtype axis at every tier. Each leaf's
  `test_gemm.cpp` hand-enumerates its own identical-shape `cases[]` array — M
  in `{1,16,64,128,512,1024}` round-robined (not a full cross) across 5
  representative `(K,N)` families (square, MLP gate/up-proj, MLP down-proj,
  2 attn-proj sizes), 30 rows total, each labeled with its phase (`GemvNt`,
  `GemvNn`, `Wmma`/`TiledFma-Nt`/`TiledFma-Nn` depending on the leaf's dtype,
  plus `-bias` variants) — **is** the tier-3 set. `kTier1`/`kTier2` are fixed
  index subsets of that array chosen so every TA/TB/bias value still appears
  at tier1 (6 rows) / tier2 (20 rows). The CPU reference (`cpuReference`) is
  multithreaded over M-rows. Every row's `(ta,tb)` pair is unique within its
  `(K,N)` family across the whole grid — needed to route around a discovered
  `hip_gemm` autotune-cache issue where reusing the same `(N,K,ta,tb,dtype)`
  signature at a very different `M` in the same process can return wrong
  results (out of scope to fix here; see the `kernel_ut_coverage_wide`
  handoff `RESULT.md`).
- **gqa decode/prefill (fp16)**: a fixed list of named real-model +
  geometry-sweep cases (13 for decode, 39 hand-authored rows for prefill,
  widened from 29 with short `sq=128` and long pure-prefill `sq=8192` rows)
  crossed with (decode) or fused with (prefill) a typical context-length /
  prompt-length list (decode: `{128,512,2048,8192}`). Prefill's `O(sq^2)`
  causal-attention CPU reference is multithreaded over `(batch,
  query-head)` pairs.
- **gqa decode/prefill (i8)**: same idea, 11 / 8 named geometries x a typical
  length list (decode: `{128,512,2048,8192}`; prefill:
  `{128,256,512,1024,2048}`). The i8 prefill leaf's two `O(sq^2)` CPU
  references (i8-cache + fp16) are also multithreaded over `(batch,
  query-head)` pairs.
- **matmul_nbits** (all four bit-widths): `group_size {32,64,128} x
  zero-points {on,off} x dtype {fp16,fp32}` (12 combos, always full) x 12
  typical `(M,K,N)` shapes — M in `{1,16,64,128,512}` round-robined (not a
  full M x KN cross) across 4 representative real-model `(K,N)` layer
  families (gate/up-proj, attn-proj, down-proj, o-proj/square). Every shape
  is ALSO crossed with the full 12-way categorical set, so the shape-list
  size directly multiplies CPU-reference cost by 12 — the reference
  (`genCase()`'s `O(M*K*N)`, no BLAS) is now multithreaded over `N` to keep
  this affordable; see the comment above `genCase()` in any
  `matmul_nbits/*/test_matmul_nbits*.cpp`.

A new op's test cpp should follow whichever of the above shapes fit it best:
if the kernel's dispatch paths are better covered by a small hand-picked
matrix than a blind cross product, hand-enumerate it (gemm/gqa style); if the
categorical axes are truly independent, build the real cross product
(matmul_nbits style). Either way, the array (or the function that builds it)
is the single source of truth for what "typical" means for that op.

## Uniform make API (standalone development)

`test`, `test_custom`, `clean`.
Standalone: `cd example && make test OFFLOAD=... HIP_SDK=... [COVERAGE=1|2|3]`

### `MODE=lookup` (default) — real LUT when one exists

`MODE ?= lookup` in every leaf Makefile. There are exactly two values, named
after what the kernel itself calls them:

- **`lookup`** (default): resolve the config from
  `hip/autotune/<op>/lut/<arch>.fb`. If this arch has no table, it warns and
  falls back to `autotune` — a missing table is never a hard failure.
- **`autotune`**: always the runtime sweep, never touches a table.

`MODE` also selects the kernel's own source, via
`HIPDNN_{GEMM,MATMUL,GQA}_AUTOTUNE_MODE=lookup|online` in the leaf's run
command. Both halves must agree: the Makefile value alone only decides whether
the real resolver and table are *linked*, so without that wiring a run labelled
`autotune` would still be resolving from a table.

`MODE=lookup` needs `flatc` (a build tool, **not part of this repo** — see
`knowledge/hosts.md`) plus its `include/`, passed on the command line:

```
make test MODE=lookup OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> \
    FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<flatbuffers include dir>
```

`FLATC`/`FLATBUFFERS_INC` are only required when `MODE` actually resolves to
`lookup` (checked at `make direct`/`$(TARGET)` time, not for `clean` etc.).
`flatc` and the flatbuffers headers must be the same release: the generated
header asserts on the version it was produced by, so a mismatched pair fails
the build with "Non-compatible flatbuffers version included".
Under the hood: `flatc --cpp --gen-object-api --scoped-enums` generates
`<op>_autotune_generated.h` into `out/gen/` (unchanged), and the test links
the real `hip/autotune/<op>/<op>_autotune.cpp` resolver instead of the leaf's
inline stub.

**LUT bytes: C23 `#embed`, no generated `.cpp`, no python.** The old
`example/common/embed_lut.py` hex-embed step is gone. Instead, each leaf's
`test_<op>.cpp` has, guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`:

```cpp
// gemm: single-blob ABI.
#include "lut_fb_path.h"
extern "C" const unsigned char kGemmLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kGemmLutData_size = sizeof(kGemmLutData);
```

`matmul_nbits` and `gqa` instead expose a **multi-blob** ABI (one blob per
family-member arch a generic-ISA DLL might embed; each test leaf still only
`#embed`s its own single `.fb`, wrapped as a 1-entry array):

```cpp
// matmul_nbits / gqa: multi-blob ABI.
#include "lut_fb_path.h"
static const unsigned char kLutBlob0[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const unsigned char* const kMatmulNbitsLutBlobs[1]   = { kLutBlob0 };
extern "C" const size_t               kMatmulNbitsLutBlobSizes[1] = { sizeof(kLutBlob0) };
extern "C" const size_t               kMatmulNbitsLutBlobCount    = 1;
// (gqa: same shape, symbols kGqaLutBlobs / kGqaLutBlobSizes / kGqaLutBlobCount)
```

`HIPDNN_LUT_FB` is a `#define` in a one-line Makefile-generated header
(`out/gen/lut_fb_path.h`, written by a plain `echo`, e.g. `#define
HIPDNN_LUT_FB "../../../../hip/autotune/matmul_nbits/lut/gfx1151.fb"`) — a
path macro, not a data file. It is a header rather than a `-D` command-line
define because **hipcc.exe's Windows argument-forwarding mangles embedded `"`
quote characters passed via `-D`** (confirmed: direct `clang.exe` invocation
handles the same quoted `-D` fine; hipcc.exe's wrapper does not). Two other
`#embed` gotchas worth knowing if you touch this:

- `#embed` needs no `-std=c++23` — it works as a Clang extension under the
  existing `-std=c++17` (`-Wno-c23-extensions` silences the pedantic
  warning). Bumping to `-std=c++23` actually **breaks** the build on this
  toolchain (MSVC STL `<cmath>` vs Clang's HIP `<cmath>` `isfinite`
  overload conflict) — do not do it.
- The `extern "C"` array declarations above (the single-blob pair, or the
  multi-blob pointer-array/size-array/count triplet) must **not** be wrapped
  in one shared `extern "C" { ... }` block — that form silently gets
  **internal** linkage from this clang/MSVC-target configuration (no error,
  no warning; the symbol simply never appears in the `.obj`, and the link
  fails with "undefined symbol" in the *other* translation unit that
  references it). Use separate `extern "C" const ... = ...;` statements,
  exactly like the snippets above (this matches what the old
  `embed_lut.py` output already did, for the same reason).

Keep the exact resolver symbol names — `gemm` still uses the single-blob
`kGemmLutData`/`_size` pair. `matmul_nbits` and `gqa` use the multi-blob
`k<Op>LutBlobs[]` / `k<Op>LutBlobSizes[]` / `k<Op>LutBlobCount` triplet
(`kMatmulNbitsLutBlobs`/`kMatmulNbitsLutBlobSizes`/`kMatmulNbitsLutBlobCount`,
`kGqaLutBlobs`/`kGqaLutBlobSizes`/`kGqaLutBlobCount`) — one blob = one array
entry. They are declared `extern "C"` in each op's unmodified
`hip/autotune/<op>/<op>_autotune.cpp`.

For matmul_nbits and gemm the kernel calls `resolve()` itself, so linking the
real resolver is enough. For GQA, `gqa_kernel.hip` does not call the resolver
(only production `real/gqa.cpp` does) — `test_gqa_decode.cpp` /
`test_gqa_prefill.cpp` instead call `hip_gqa_autotune_resolve_decode` /
`_resolve_prefill` themselves (guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`)
and dispatch the resolved config through the production
`hip_gqa_flash_decode_configured` / `_prefill_v3_configured` entries. The GQA
i8 leaves' kernel calls never resolve from the LUT directly (only the fp16
decode/prefill leaves do) — `MODE=lookup` there only needs to satisfy
`gqa_autotune.cpp`'s LUT-data symbols so the link succeeds.

### `out/results.csv`

Every `test` / `test_custom` on a leaf wired for it appends to
`out/results.csv` (truncated once per top-level invocation; `make clean`
removes it with the rest of `out/`). Stable header:

```
op,leaf,arch,mode,shape,config,time_ms,relL2,verdict
```

One row per case.

`time_ms` is the **mean per-launch** time of that case's own timing loop (the
leaf's `kIters` launches between two HIP events), not a total. It is the only
timing in the row: the tuner's own figure is a *peak* (minimum) sample taken
during the sweep, under a different clock state, so it is deliberately not
carried in `config` where it would read as the same measurement.

`config` names the configuration that actually ran, recovered from the op's own
diagnostics (`HIPDNN_{MATMUL,GQA}_AUTOTUNE_MODE` logs, `HIPDNN_MATMUL_LUT_LOG`,
`HIPDNN_EP_DEBUG` for gemm) rather than inferred from `mode`. The prefix says
where the configuration came from, the rest is the configuration itself:

| Prefix | Meaning |
|---|---|
| `lookup:…` | the offline table answered. MatMulNBits also reports `(exact\|nearest\|fallback d=…)`, where `d` is the weighted log2 distance to the measured point |
| `autotune:…` | the table missed, was bypassed, or the op has no lookup wired, and the runtime sweep picked this one |
| `heuristic:…` | gemm's `Wmma` phase only: a table miss in `lookup` mode, which deliberately uses the compiled-in occupancy heuristic instead of sweeping (see `tuneWmma`) |
| `default:…` | gemm's `GemvNt`/`GemvNn`/`TiledFma` phases: a table miss in `lookup` mode, which keeps that phase's compiled-in default config index (each of those tuners' equivalent of `heuristic:` — same "don't sweep on a miss" policy as `Wmma`, just a fixed index instead of an M/N/K-dependent heuristic function) |
| `naive` / `fixed-kernel` | the shape reached a path with no tunable config at all (gemm: `TA=1` or the `NN` layout, both dispatch a single fixed kernel with nothing to choose — see `launchSmallM`; this label does not yet distinguish which fixed path, since neither logs a selection) |
| `declined` | the op rejected the combination by design (a leaf's `expect_reject` case); `time_ms`/`relL2` are 0 because nothing ran |
| `unlogged` | the op emitted no config diagnostic for this case — treat it as a gap to fix, not a result |
| `unclassified:…` | gemm only: the op's diagnostics logged a `-> cfg[…]` line, but its prefix matched none of `lookup`/`heuristic`/`default`/`autotune` — a new source string was added to the kernel without updating the leaf's capture helper (`capture_selected_config` in `test_gemm.cpp`); treat this as a doc/parsing gap to fix, not a result, same as `unlogged` |

A run is either `lookup` or `autotune`, never both, so a leaf has exactly one
`out/results.csv` with one row per case that ran — including declined cases. The
`mode` column records which mode was asked for; an `autotune:` config under
`mode=lookup` is the honest report of a table miss, not a second result set.

Each leaf captures that output per case into a single scratch file under `out/`
and deletes it again, so a completed run leaves only `out/results.csv`. Because
an op logs a selection only on the first encounter with a tune key — which does
not include the activation dtype — a later case sharing that key (e.g. the fp32
twin of an fp16 shape) reuses the earlier selection silently; the leaf remembers
it per key so every row still names a real config.

`hipdnn-kernel-ut.exe` appends one extra row per leaf to the same file, with
`shape=suite`, `config=suite-total` and the leaf process's **total wall time**
in `time_ms` — the one row where that column is not a per-launch mean.

Each leaf that writes this file inlines its own ~30-line `CsvWriter`/`CsvRow`
(no shared header) — see the top of any `test_<op>.cpp`. The two GQA i8 leaves
are the only ones that still emit no per-case rows.
