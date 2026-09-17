# Kernel unit tests (`lib/Runtime/Kernels/test/example`)

This folder is the **only** entry for standalone HIP kernel unit tests.
Makefile is the only driver. No `.bat` / `setup_*` / `run_*` wrappers.
No full-EP cmake.

Repo-relative paths only. `HIP_SDK` and `OFFLOAD` come from the command line.

## These are unit tests — small and fast, not a perf sweep

A kernel unit test's job is to catch correctness regressions in minutes, not
to characterize performance across the whole shape space (that is what
`hip/autotune/<op>/tools/` is for). The coverage model is:

> **all categorical situations, fully** (every mode/dtype/flag the op
> supports) **x a SMALL set of typical shapes** (a handful of representative
> `(M,K,N)` / `(B,H,G,D,seq)` points — decode + a couple of prefill points),
> **not the full continuous shape space.**

Concretely:

- **Categorical axes** (group_size, zero-points on/off, dtype fp16/fp32,
  TA/TB transpose, bias on/off, heads-per-group ratio, head_dim,
  sliding-window/head-sink/smooth-softmax, INT8-KV, …) are enumerated in
  full and run at **every** `COVERAGE` tier.
- **Continuous axes** (M/K/N, sequence length) are a short, explicitly listed
  set of typical values (e.g. matmul decode `M=1` + two prefill points over
  2-3 real `(K,N)` layer shapes; GQA 2-3 typical context lengths) — never the
  full `1..4096`-style ladder, and never a giant perf shape (`N≈201088`,
  `M≈4096`, …). `COVERAGE=1|2|3` only thins how many of those typical values
  run; it never drops a categorical situation.

A tier3 (default) run of one leaf is on the order of **tens of cases**, not
hundreds, and finishes in well under a minute; the whole 9-leaf suite at
tier3 finishes in a few minutes. This is intentional and by design — a unit
test does not need to, and should not try to, cover the whole shape space.

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
  Makefile                  # CI aggregator
  tests.manifest
  gemm/Xfp16_Wfp16_Yfp16/
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

## Required files in every leaf — EXACTLY these three, nothing else

| File | Role |
|---|---|
| `Makefile` | `test` / `test_custom` / `clean` |
| `test_<op>.cpp` | Driver. Entirely self-contained: generates its own inputs (seeded RNG) and computes its own CPU reference **in-process, in C++** — no python, no on-disk data, no `example/common/`. Enumerates the op's categorical situations x typical shapes in C++ (see "Shape coverage" below). Inline empty `resolve()` if needed, guarded by `#ifndef HIPDNN_LUT_LINKED_EXTERNALLY` so `MODE=lut`'s real resolver (which defines the same symbols) can link. No `autotune_stub.cpp`. |
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

- **gemm**: a hand-enumerated `cases[]` array in `test_gemm.cpp` (M/N/K x
  TA/TB transpose x bias on/off x dtype path fp16/fp32, each row commented
  with the dispatch path it targets, ~16 rows total) **is** the tier-3 set —
  a blind M x N x K sweep would mostly re-test the same WMMA/GEMV/split-K
  path a targeted case already covers. `kTier1`/`kTier2` are fixed index
  subsets of that array chosen so every TA/TB/bias/dtype value still appears
  at tier1.
- **gqa decode/prefill (fp16)**: a fixed list of named real-model +
  geometry-sweep cases (13 for decode, 29 hand-authored rows for prefill)
  crossed with (decode) or fused with (prefill) a small typical
  context-length / prompt-length list.
- **gqa decode/prefill (i8)**: same idea, 11 / 8 named geometries x a small
  typical length list.
- **matmul_nbits** (all four bit-widths): `group_size {32,64,128} x
  zero-points {on,off} x dtype {fp16,fp32}` (12 combos, always full) x 2-4
  typical `(M,K,N)` shapes — decode (`M=1`) at real model FFN/attention-proj
  `(K,N)`, plus 1-2 smaller prefill-representative shapes chosen small enough
  that the in-process CPU reference (`O(M*K*N)`, no BLAS) stays fast. Real
  full-size FFN `(K,N)` at `M=128/512` would make the naive CPU reference
  alone take minutes-to-hours per case — see the comment above `genCase()` in
  any `matmul_nbits/*/test_matmul_nbits*.cpp`.

A new op's test cpp should follow whichever of the above shapes fit it best:
if the kernel's dispatch paths are better covered by a small hand-picked
matrix than a blind cross product, hand-enumerate it (gemm/gqa style); if the
categorical axes are truly independent, build the real cross product
(matmul_nbits style). Either way, the array (or the function that builds it)
is the single source of truth for what "typical" means for that op.

## Uniform make API / CI

`test`, `test_custom`, `clean`.
CI: `cd example && make test OFFLOAD=... HIP_SDK=... [COVERAGE=1|2|3]`

### `MODE=auto` (default) — real LUT when one exists

`MODE ?= auto` in every leaf Makefile:

- **`auto`** (default): `lut` if `hip/autotune/<op>/lut/<arch>.fb` exists for
  the current `OFFLOAD` arch, else `autotune`. Never fails for lack of a
  table — falls back silently.
- **`lut`**: forces `lut`; if the `.fb` is missing for this arch, warns and
  falls back to `autotune` (still does not fail).
- **`autotune`**: always the runtime sweep, never touches a LUT.

`MODE=lut` needs `flatc` (a build tool, **not part of this repo** — see
`knowledge/hosts.md`) plus its `include/`, passed on the command line:

```
make test MODE=auto OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<sdk> \
    FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<flatbuffers include dir>
```

`FLATC`/`FLATBUFFERS_INC` are only required when `MODE` actually resolves to
`lut` (checked at `make direct`/`$(TARGET)` time, not for `clean` etc.).
Under the hood: `flatc --cpp --gen-object-api --scoped-enums` generates
`<op>_autotune_generated.h` into `out/gen/` (unchanged), and the test links
the real `hip/autotune/<op>/<op>_autotune.cpp` resolver instead of the leaf's
inline stub.

**LUT bytes: C23 `#embed`, no generated `.cpp`, no python.** The old
`example/common/embed_lut.py` hex-embed step is gone. Instead, each leaf's
`test_<op>.cpp` has, guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`:

```cpp
#include "lut_fb_path.h"
extern "C" const unsigned char kFooLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kFooLutData_size = sizeof(kFooLutData);
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
- The `extern "C" const unsigned char kFooLutData[] = {...};` declaration
  must **not** be wrapped in a shared `extern "C" { ... }` block with its
  `_size` sibling — that form silently gets **internal** linkage from this
  clang/MSVC-target configuration (no error, no warning; the symbol simply
  never appears in the `.obj`, and the link fails with "undefined symbol" in
  the *other* translation unit that references it). Use two separate
  `extern "C" const ... = ...;` declarations, exactly like the snippet above
  (this matches what the old `embed_lut.py` output already did, for the same
  reason).

Keep the exact resolver symbol names (`kGemmLutData`/`_size`,
`kGqaLutData`/`_size`, `kMatmulNbitsLutData`/`_size`) — they are declared
`extern "C"` in each op's unmodified `hip/autotune/<op>/<op>_autotune.cpp`.

For matmul_nbits and gemm the kernel calls `resolve()` itself, so linking the
real resolver is enough. For GQA, `gqa_kernel.hip` does not call the resolver
(only production `real/gqa.cpp` does) — `test_gqa_decode.cpp` /
`test_gqa_prefill.cpp` instead call `hip_gqa_autotune_resolve_decode` /
`_resolve_prefill` themselves (guarded by `#ifdef HIPDNN_LUT_LINKED_EXTERNALLY`)
and dispatch the resolved config through the production
`hip_gqa_flash_decode_configured` / `_prefill_v3_configured` entries. The GQA
i8 leaves' kernel calls never resolve from the LUT directly (only the fp16
decode/prefill leaves do) — `MODE=lut` there only needs to satisfy
`gqa_autotune.cpp`'s LUT-data symbols so the link succeeds.

### `out/results.csv`

Every `test` / `test_custom` on a leaf wired for it appends to
`out/results.csv` (truncated once per top-level invocation; `make clean`
removes it with the rest of `out/`). Stable header:

```
op,leaf,arch,mode,shape,config,time_ms,relL2,verdict
```

One row per case: `config` is `final` in `MODE=autotune`, or the resolved
LUT config string (and `lut:<source>` for leaves that only log
exact/nearest/fallback) in `MODE=lut`. Each leaf that writes this file
inlines its own ~30-line `CsvWriter`/`CsvRow` (no shared header) — see the
top of any `test_<op>.cpp` that includes one. Not every leaf writes CSV: gemm
never did (its perf/correctness path is inlined in `main()`), and the GQA i8
leaves don't either — both unchanged from before this cleanup.
