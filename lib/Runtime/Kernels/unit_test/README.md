# Kernel unit tests (`lib/Runtime/Kernels/unit_test`)

This folder contains the source for standalone HIP kernel unit tests.

There are two ways to build and run them:

- **Standalone development**: `make test` in this directory (see "Uniform make
  API" below). It compiles every leaf, kernel sources included, and runs them
  through the same `hipdnn-kernel-ut.exe` controller the CI runs.
- **Packaging CI**: the hip-ep CMake build, described next. It builds these
  tests by default on Windows with the real runtime.

## Packaging CI: standalone kernel-test executables

The hip-ep CMake build produces these tests as standalone executables for a
packaging or GPU CI to run. `BUILD_HIP_KERNEL_UNIT_TESTS` is **on by default**
(Windows, real runtime), so every hip-ep build, CI included, proves the tests
build; `-DBUILD_HIP_KERNEL_UNIT_TESTS=OFF` skips them.
`.github/workflows/windows-build-real.yml` stages them into the
`gpu-test-package` artifact but does not run them: whether and when to run
them is up to the CI that consumes the package.

Each leaf becomes one small executable per kernel ISA. It compiles only its
host driver and dynamically links `custom_kernels_<arch>.dll`, so the GPU
kernels and the autotune tables are compiled and packaged exactly once, in the
same DLL production uses.

### Enable and build

| Requirement | Value |
|---|---|
| CMake option | `BUILD_HIP_KERNEL_UNIT_TESTS`, default `ON`; `-DBUILD_HIP_KERNEL_UNIT_TESTS=OFF` skips the tests |
| Platform | Windows. On Linux the directory is skipped with a status message. |
| Runtime | Real runtime (`-DBUILD_MOCK_RUNTIME=OFF`, which `build.py` sets when not building the mock). The executables link `custom_kernels_<arch>`. |

A normal `build.py` build needs nothing extra; to skip the tests, add
`--cmake_extra_defines BUILD_HIP_KERNEL_UNIT_TESTS=OFF` (the flag is repeatable):

```
python build.py --config Release --cmake_generator Ninja ^
  --hip_arch "gfx11-generic;gfx1170" ^
  --cmake_prefix_path "<ort-install>;<llvm-install>" ^
  --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON
```

The executables are part of the default build and are installed by the normal
install step. For every entry in `HIP_ARCHITECTURES` the install
prefix gets one `kernel-tests/<arch>/`, named exactly like that entry's
`custom_kernels_<arch>.dll`. With the CI's `gfx11-generic;gfx1170` that is
`kernel-tests/gfx11-generic/` and `kernel-tests/gfx1170/`:

```
<install>/bin/
  hipdnn-kernel-ut.exe                          # suite controller (one)
  custom_kernels_<arch>.dll                     # installed already, one per arch
  kernel-tests/<arch>/
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu2-yfp16.exe
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu3-yfp16.exe
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu4-yfp16.exe
    hipdnn-kernel-ut-matmul-nbits-xfp16-wi8-yfp16.exe
    hipdnn-kernel-ut-matmul-nbits-xfp32-wu2-yfp32.exe
    hipdnn-kernel-ut-matmul-nbits-xfp32-wu3-yfp32.exe
    hipdnn-kernel-ut-matmul-nbits-xfp32-wu4-yfp32.exe
    hipdnn-kernel-ut-matmul-nbits-xfp32-wi8-yfp32.exe
    hipdnn-kernel-ut-gqa-decode-xfp16-wfp16-yfp16.exe
    hipdnn-kernel-ut-gqa-decode-xfp16-wi8-yfp16.exe
    hipdnn-kernel-ut-gqa-prefill-xfp16-wfp16-yfp16.exe
    hipdnn-kernel-ut-gqa-prefill-xfp16-wi8-yfp16.exe
```

### Run (on a machine with a supported AMD GPU)

Copy `<install>/bin` to the GPU machine. At minimum that is
`hipdnn-kernel-ut.exe`, every `custom_kernels_*.dll` and `kernel-tests/`, plus
the HIP runtime the tests were built against (`amdhip64_7.dll` and
`amd_comgr*.dll` from the SDK's `bin`). The install step does not copy the HIP
runtime; the CI's `gpu-test-package/bin` already contains it. Then run the
controller from that `bin` directory:

```
.\hipdnn-kernel-ut.exe --mode lookup --coverage 1
```

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--arch` | `gfxNNNN`, `gfx11-generic`, ... | detected from GPU 0 | Which `kernel-tests/<arch>/` to run. Picked by the same rule the EP uses to load `custom_kernels_<arch>.dll` (`genericTargetFor()` in `backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp`): the ISA's own directory first, then its generic target's. So on the CI package a gfx1151 runs `kernel-tests/gfx11-generic/` and a gfx1170 runs `kernel-tests/gfx1170/`, the same DLLs the EP loads there. `gfx1170` is not a `gfx11-generic` member. |
| `--mode` | `lookup`, `autotune` | `lookup` | `lookup` uses the tables embedded in `custom_kernels_<arch>.dll` and falls through to autotune on a miss, like production. `autotune` bypasses the tables (production's `online` mode). |
| `--coverage` | `1`, `2`, `3` | `3` | Shape tier; see "Shape coverage". `1` suits a per-PR job, `3` a nightly one. |

The controller runs the 13 leaves serially, with `bin` as the working
directory so every leaf finds `custom_kernels_<arch>.dll`. It prints one
`[<op>/<leaf>] PASS|FAIL (<ms> ms)` line per leaf and finally
`Combined report: <path>`.

Each leaf starts with its stderr on `bin/out/<executable>.stderr.log`, which
the controller deletes when the leaf passes and keeps (printing its path) when
it fails. The leaves read their per-case `config` back from that file: the
kernels print their diagnostics from `custom_kernels_<arch>.dll`, whose
statically linked C runtime keeps the stderr the process started with, so a
leaf cannot redirect them itself.

The leaves use the HIP runtime in `bin`, the same copy the controller loads.
Windows searches System32 before a leaf's working directory, and the GPU
driver installs its own `amdhip64_7.dll` and `amd_comgr_*.dll` there, so a
leaf on its own would load the driver's runtime, even with `bin` first on
`PATH`. The controller therefore adds `bin` to the DLL search path with
`SetDllDirectoryW` before launching the leaves, which inherit it. If `bin` has
no HIP runtime, the controller and the leaves fall back to the driver's copy.

| Exit code | Meaning |
|---|---|
| `0` | Every leaf passed. |
| `1` | At least one leaf failed or could not be launched. |
| `2` | Bad argument, no GPU detected, or no packaged tests for that arch. |

The report is `bin/out/results.csv`, truncated at the start of each run.
Header `op,leaf,arch,mode,shape,config,time_ms,relL2,verdict`: one row per case
from the leaves (format in "`out/results.csv`" below), plus one row per leaf
with `shape=suite`, `config=suite-total` and that leaf's `PASS`/`FAIL` in
`verdict`. A failing CI job is diagnosed from this file: filter
`verdict == FAIL`.

To run a single leaf by hand, also start it from `bin`:

```
.\kernel-tests\gfx11-generic\hipdnn-kernel-ut-matmul-nbits-xfp16-wu4-yfp16.exe --coverage 1
```

A leaf started this way does not get the controller's DLL search path, so it
loads the driver's HIP runtime from System32, and its `config` column can read
`unlogged` because it cannot capture the kernels' diagnostics. Run the suite
through the controller to test the package's runtime and record every config.

When `BUILD_TESTING` is on, each leaf is also registered with CTest under the
labels `gpu;kernel-ut;<arch>` (run from the build tree, e.g.
`ctest --test-dir <build> -L kernel-ut`).

### Wiring it into GitHub Actions

The build side is in place in `windows-build-real.yml`: the build produces the
tests (the option is on by default), `hipdnn-kernel-ut.exe` reaches
`staging/bin/` with the other `install/bin/*.exe`, and the staging step copies
`install/bin/kernel-tests/` next to it, failing if the controller or an arch's
executables are missing.

A GPU job that wants the results runs the controller from the downloaded
`gpu-test-package/bin` (suggested; self-hosted GPU runner):

```yaml
- name: Run kernel unit tests (GPU)
  shell: cmd
  working-directory: gpu-test-package/bin
  run: hipdnn-kernel-ut.exe --mode lookup --coverage 1

- name: Upload kernel unit-test report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: kernel-ut-results
    path: gpu-test-package/bin/out/
```

The step fails on any non-zero exit code. The uploaded `results.csv` shows
which op, leaf and shape failed, and a failing leaf's `.stderr.log` sits next
to it.

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
the whole 12-leaf suite at tier3 is budgeted at **up to ~30 min total**
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
| `X` | activation / input (MatMulNBits A, GQA Q) |
| `W` | weight / KV (MatMulNBits packed B, GQA K/V cache) |
| `Y` | output (MatMulNBits C, GQA O) |
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

If input becomes fp8 later: add `Xfp8_Wfp16_Yfp16`, do **not** rename the
existing fp16 leaf to a bare `fp16/` or `fp8/`. A folder named `fp16` or `i8`
is illegal — it does not say which tensor.

Do **not** use `fp16u4`, `decode/fp16`, `prefill/i8`.

## Layout

```
unit_test/
  README.md
  CMakeLists.txt             # package targets and install rules
  kernel_ut_main.cpp         # hipdnn-kernel-ut.exe controller
  Makefile                   # local suite: build every leaf, run hipdnn-kernel-ut.exe
  matmul_nbits/Xfp16_Wu{2,3,4}_Yfp16/
  matmul_nbits/Xfp16_Wi8_Yfp16/
  matmul_nbits/Xfp32_Wu{2,3,4}_Yfp32/
  matmul_nbits/Xfp32_Wi8_Yfp32/
  gqa/decode/Xfp16_Wfp16_Yfp16/
  gqa/decode/Xfp16_Wi8_Yfp16/
  gqa/prefill/Xfp16_Wfp16_Yfp16/
  gqa/prefill/Xfp16_Wi8_Yfp16/
  qmoe_ragged_mm/Xfp16_Wu4_Yfp16/
```

`unit_test/` holds one directory per op and nothing else besides the four
files above. The offline autotune LUT sweep tools (not UTs) live under
`hip/autotune/<op>/tools/` next to each op's real resolver — e.g.
`hip/autotune/gqa/tools/`, `hip/autotune/matmul_nbits/tools/`. Kernel
compile-time guards, which run no kernel and need no GPU (e.g.
`LdsBarrierArchGate`), live in the repository's `test/kernel_guards/` with
the other GPU-free CTest suites.

## Adding an op or dtype leaf

Update all of these together:

1. Add the production launcher declaration to `include/hip_custom_kernels.h`,
   implementation to `hip/<op>_kernel.hip`, and that HIP file to
   `lib/Runtime/Kernels/CMakeLists.txt`'s `_kernel_sources`.
2. Add a leaf directory named from all tensor dtypes, containing exactly its
   `Makefile`, self-contained `test_<op>.cpp`, and `README.md`. The test must
   generate inputs and a CPU reference in-process, return non-zero on failure,
   and support `--coverage 1|2|3`.
3. Add the leaf once to the `LEAVES` list in `unit_test/Makefile` for local
   runs, once to `unit_test/CMakeLists.txt` for package install, and
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
   The controller also writes one `shape=suite` row per leaf, including
   leaves without per-case rows.

## Required files in every leaf — EXACTLY these three, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_custom` / `clean` / `build` (`build` also copies the exe to `PKG_DIR/PKG_NAME` for the `unit_test/` aggregator) |
| `test_<op>.cpp` | Driver. Entirely self-contained: generates its own inputs (seeded RNG) and computes its own CPU reference **in-process, in C++** — no python, no on-disk data, no `unit_test/common/`. Enumerates the op's categorical situations x typical shapes in C++ (see "Shape coverage" below). Inline empty `resolve()` if needed, guarded by `#ifndef HIPDNN_LUT_LINKED_EXTERNALLY` so `MODE=lookup`'s real resolver (which defines the same symbols) can link. No `autotune_stub.cpp`. |
| `README.md` | Targets + `MODE` + `COVERAGE` |

No `gen_data.py`, no python anywhere under `unit_test/`, no `shapes.csv`, no
`unit_test/common/` (there is nothing left to share — each leaf inlines its own
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

A new op's test cpp should follow whichever of the above shapes fit it best:
if the kernel's dispatch paths are better covered by a small hand-picked
matrix than a blind cross product, hand-enumerate it (gqa style); if the
categorical axes are truly independent, build the real cross product
(matmul_nbits style). Either way, the array (or the function that builds it)
is the single source of truth for what "typical" means for that op.

## Uniform make API (standalone development)

From `unit_test/`, the whole suite, run the way the CI runs it:

```
make test  OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> [MODE=lookup|autotune] [COVERAGE=1|2|3]
make build ...   # build only
make clean ...
```

The Makefiles use repo-relative paths only: `HIP_SDK` and `OFFLOAD` come from
the command line. `MODE=lookup` on an arch that has a table (every arch with a
checked-in `lut/<arch>.fb`) also needs `FLATC` and `FLATBUFFERS_INC`; see
"`MODE=lookup`" below.

`make test` runs every leaf's `make build` (add `-j` to build them in
parallel), which compiles the kernel sources into the leaf, and collects the
executables into `out/bin/kernel-tests/<arch>/` under their packaged names. It
builds `out/bin/hipdnn-kernel-ut.exe` from `kernel_ut_main.cpp`, then runs
`hipdnn-kernel-ut.exe --mode <MODE> --coverage <COVERAGE>` from `out/bin`: the
same controller, layout, exit codes and `out/results.csv` (here
`out/bin/out/results.csv`) as the CI package. Only the leaves differ: they
contain the kernels instead of linking `custom_kernels_<arch>.dll`. `out/bin`
has no HIP runtime, so the controller and the leaves use the driver's copy
from System32. For the exact CI binaries, build hip-ep with `build.py` and run
the same command from `<install>/bin`.

In a leaf directory, `test`, `test_custom`, `clean` and `build` work on that
leaf alone; `make test` there runs the leaf directly and writes the leaf's own
`out/results.csv`.

### `MODE=lookup` (default) — real LUT when one exists

`MODE ?= lookup` in every leaf Makefile. There are exactly two values, named
after what the kernel itself calls them:

- **`lookup`** (default): resolve the config from
  `hip/autotune/<op>/lut/<arch>.fb`. If this arch has no table, it warns and
  falls back to `autotune` — a missing table is never a hard failure.
- **`autotune`**: always the runtime sweep, never touches a table.

`MODE` also selects the kernel's own source, via
`HIPDNN_{MATMUL,GQA}_AUTOTUNE_MODE=lookup|online` in the leaf's run
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
`common/embed_lut.py` hex-embed step is gone. Instead, each leaf's
`test_<op>.cpp` has, guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`, the
**multi-blob** LUT ABI (one blob per family-member arch a generic-ISA DLL might
embed; each test leaf still only `#embed`s its own single `.fb`, wrapped as a
1-entry array):

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
HIPDNN_LUT_FB "../../../hip/autotune/matmul_nbits/lut/gfx1151.fb"`) — a
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
- The `extern "C"` array declarations above (the multi-blob
  pointer-array/size-array/count triplet) must **not** be wrapped
  in one shared `extern "C" { ... }` block — that form silently gets
  **internal** linkage from this clang/MSVC-target configuration (no error,
  no warning; the symbol simply never appears in the `.obj`, and the link
  fails with "undefined symbol" in the *other* translation unit that
  references it). Use separate `extern "C" const ... = ...;` statements,
  exactly like the snippets above (this matches what the old
  `embed_lut.py` output already did, for the same reason).

Keep the exact resolver symbol names: `matmul_nbits` and `gqa` use the multi-blob
`k<Op>LutBlobs[]` / `k<Op>LutBlobSizes[]` / `k<Op>LutBlobCount` triplet
(`kMatmulNbitsLutBlobs`/`kMatmulNbitsLutBlobSizes`/`kMatmulNbitsLutBlobCount`,
`kGqaLutBlobs`/`kGqaLutBlobSizes`/`kGqaLutBlobCount`) — one blob = one array
entry. They are declared `extern "C"` in each op's unmodified
`hip/autotune/<op>/<op>_autotune.cpp`.

For matmul_nbits the kernel calls `resolve()` itself, so linking the
real resolver is enough. For GQA, `gqa_kernel.hip` does not call the resolver
(only production `real/gqa.cpp` does) — every GQA leaf instead calls
`hip_gqa_autotune_resolve_decode` / `_resolve_prefill` itself (guarded by
`#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`) and dispatches the resolved config
through the production `hip_gqa_flash_decode_configured` /
`_prefill_v3_configured` entries. The i8 leaves do it the way `real/gqa.cpp`
does for an int8 KV cache: decode resolves with `kv_dtype=INT8`, and prefill
resolves the fp16 prefill that runs on the dequantized cache.

### `out/results.csv`

Every `test` / `test_custom` on a leaf wired for it appends to
`out/results.csv` (truncated once per top-level invocation; `make clean`
removes it with the rest of `out/`). Stable header:

```
op,leaf,arch,mode,shape,config,time_ms,relL2,verdict
```

One row per case.

`shape` lists the case's parameters, enough to tell any two rows of a leaf
apart: MatMulNBits `M1_N5120_K2880_gs32` (`_noz` without zero points), GQA
decode `B1_H64_G8_D64_len2048` and prefill `B1_H64_G8_D64_sq512_past0`, with
`_win<n>`, `_sink` and `_smooth` only when set. qmoe_ragged_mm writes
`R20_E4_N17_K32_bs32_skewed_gather` (rows, experts, N, K, block size, routing,
input layout) with `_bias` and `_zp` when set, and names its selector,
descriptor and pipeline cases (`task_policy_fc1`, `launcher_grid`,
`descriptors_E11_P304_k1`, `pipeline_T20_E5_k2_H32_I16_bs32`).

`time_ms` is the **mean per-launch** time of that case's own timing loop (the
leaf's `kIters` launches between two HIP events), not a total. It is the only
timing in the row: the tuner's own figure is a *peak* (minimum) sample taken
during the sweep, under a different clock state, so it is deliberately not
carried in `config` where it would read as the same measurement.
qmoe_ragged_mm checks correctness only and has no timing loop, so its rows
carry 0.

`config` names the configuration that actually ran, recovered from the op's own
diagnostics (`HIPDNN_{MATMUL,GQA}_AUTOTUNE_LOG`, `HIPDNN_MATMUL_LUT_LOG`) or,
for the fp16 GQA leaves in `lookup` mode, from the resolver they call, rather
than inferred from `mode`. The prefix says
where the configuration came from, the rest is the configuration itself:

| Prefix | Meaning |
|---|---|
| `lookup:…` | the offline table answered. MatMulNBits reports `(exact\|nearest\|fallback d=…)`, where `d` is the weighted log2 distance to the measured point; GQA reports `(exact\|nearest\|fallback\|heuristic)` |
| `autotune:…` | the table missed, was bypassed, or the op has no lookup wired, and the runtime sweep picked this one |
| `fixed:…` | the op has no table and no tuner; a fixed rule picked this, read back through the op's own selectors. qmoe_ragged_mm reports the stage's task policy, queue policy, grid and the task kinds its row groups ran (`fixed:p0-current queue=atomic1 grid=20 tasks=gemv+wmma16`) |
| `declined` | the op rejected the combination by design (a leaf's `expect_reject` case); `time_ms`/`relL2` are 0 because nothing ran |
| `unlogged` | the op emitted no config diagnostic for this case — treat it as a gap to fix, not a result |

A run is either `lookup` or `autotune`, never both, so a leaf has exactly one
`out/results.csv` with one row per case that ran — including declined cases. The
`mode` column records which mode was asked for; an `autotune:` config under
`mode=lookup` is the honest report of a table miss, not a second result set.

Under `hipdnn-kernel-ut.exe` each leaf reads that output back from its stderr
log (see "Run" above); a leaf started any other way redirects its own stderr
to a scratch file under `out/` instead, which captures the kernels only when
they are linked into the leaf (the local make build). Either way a passing run
leaves only `out/results.csv`. Because an op logs a selection only on the
first encounter with a tune key, a later case sharing that key reuses the
earlier selection silently; the leaf remembers it per key so every row still
names a real config.

`hipdnn-kernel-ut.exe` appends one extra row per leaf to the same file. It is
a per-leaf summary, not a case: `shape=suite` and `config=suite-total` are
fixed markers, `time_ms` is the leaf process's **total wall time** (the one row
where that column is not a per-launch mean) and `verdict` is the leaf's
overall `PASS`/`FAIL`.

Each leaf inlines its own ~30-line `CsvWriter`/`CsvRow` (no shared header) —
see the top of any `test_<op>.cpp`.
