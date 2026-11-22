# Kernel unit tests (`lib/Runtime/Kernels/unit_test`)

This folder contains the source for standalone HIP kernel unit tests. A new
or changed leaf follows [GUIDELINES.md](GUIDELINES.md).

MatMulNBits, GQA decode/prefill and ragged W4 MatMul use the shared
[adaptive timing policy](common/README.md). `--timing standard` (default)
ends a measurement at 1,000 calls or one second of GPU time, whichever comes
first, when its five groups agree within 3%, and otherwise at one second and ten
calls; `--timing confirm` (at least ten valid GPU seconds and two calls in
each of five groups; 1% spread diagnostic) is for formal
performance numbers; `--timing screen` (100 calls / 0.1 s) is the short
LUT-ranking mode. Correctness checks are identical in all three. Policy, actual iterations, runtime identity
and dispersion are emitted as `UT_TIMING` JSON records.

The same leaves refresh LUT tables; see
[Refreshing a LUT with the UT leaves](#refreshing-a-lut-with-the-ut-leaves).
Every input an external caller passes, to the controller or to one leaf, is
listed in [`ut_interface.json`](ut_interface.json); see
[External invocation](#external-invocation-ut_interfacejson).

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
  kernel-tests/ut_interface.json                # caller inputs, see "External invocation"
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
    hipdnn-kernel-ut-gather-block-quantized-xi64-wu4-yfp16.exe
    hipdnn-kernel-ut-gather-block-quantized-xi64-wu8-yfp16.exe
    hipdnn-kernel-ut-gather-block-quantized-xi64-wfp8-yfp16.exe
    hipdnn-kernel-ut-cast-xany-yany-bf16.exe
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
| `--coverage` | `1`, `2`, `3` | `3` | Shape tier; see [Shape coverage](GUIDELINES.md#shape-coverage-3-tiers). `1` suits a per-PR job, `3` a nightly one. |
| `--timing` | `screen`, `standard`, `confirm` | `standard` | Timing preset of the shared timer (`HIPDNN_UT_TIMING`); `confirm` for formal numbers. |
| `--leaf` | `<op>/<leaf>`, e.g. `gqa/prefill/Xfp16_Wfp16_Yfp16` | every leaf | Run only this leaf. After `--`, the rest of the command line goes to it instead of `--coverage`; see [External invocation](#external-invocation-ut_interfacejson). |

The controller runs the 17 leaves serially, with `bin` as the working
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

To run a single leaf, name it with `--leaf`; after `--` it receives the
given arguments instead of `--coverage`:

```
.\hipdnn-kernel-ut.exe --leaf matmul_nbits/Xfp16_Wu4_Yfp16 --coverage 1
.\hipdnn-kernel-ut.exe --leaf matmul_nbits/Xfp16_Wu4_Yfp16 -- 1x2880x5120 --group-size 32
```

A leaf executable started directly from `bin` also runs, but it does not get
the controller's DLL search path, so it loads the driver's HIP runtime from
System32, and its `config` column can read `unlogged` because it cannot capture
the kernels' diagnostics.

When `BUILD_TESTING` is on, each leaf is also registered with CTest under the
labels `gpu;kernel-ut;<arch>` (run from the build tree, e.g.
`ctest --test-dir <build> -L kernel-ut`).

### Wiring it into GitHub Actions

The build side is in place in `windows-build-real.yml`: the build produces the
tests (the option is on by default), `hipdnn-kernel-ut.exe` reaches
`staging/bin/` with the other `install/bin/*.exe`, and the staging step copies
`install/bin/kernel-tests/` next to it, failing if the controller or an arch's
executables are missing.

`windows-kernel-ut.yml` (the `kernel_ut` job of `windows-build.yml`) runs
`hipdnn-kernel-ut.exe --mode lookup --coverage 1` from the downloaded
`gpu-test-package/bin` on the self-hosted `StrixHalo` runner:

| Event | Runs when |
|---|---|
| `pull_request`, `merge_group` | The diff touches `lib/Runtime/Kernels/` or `windows-kernel-ut.yml`; otherwise the job is skipped. |
| `push`, `workflow_dispatch` | Always. |

The job fails on any non-zero exit code. It uploads `bin/out/` as the
`kernel-ut-results` artifact (`results.csv` shows which op, leaf and shape
failed; a failing leaf's `.stderr.log` sits next to it), writes a per-leaf
table to the job summary, and on a pull request posts the same table as a PR
comment.

## Leaf guidelines

The rules every leaf follows -- scope, naming, required files and
capabilities, shape coverage, the `ut_interface.json` entry and the checklist
for a new op or dtype -- are in [GUIDELINES.md](GUIDELINES.md). This README
covers building, running and refreshing LUTs.

## Layout

```
unit_test/
  README.md
  GUIDELINES.md              # rules every leaf follows
  ut_interface.json          # inputs of the controller and every leaf (common/ut_interface.py runs them)
  CMakeLists.txt             # package targets and install rules
  kernel_ut_main.cpp         # hipdnn-kernel-ut.exe controller
  Makefile                   # local suite: build every leaf, run hipdnn-kernel-ut.exe
  common/                    # shared timer, ranking and LUT-refresh tools, host tests
  matmul_nbits/Xfp16_Wu{2,3,4}_Yfp16/
  matmul_nbits/Xfp16_Wi8_Yfp16/
  matmul_nbits/Xfp32_Wu{2,3,4}_Yfp32/
  matmul_nbits/Xfp32_Wi8_Yfp32/
  gqa/decode/Xfp16_Wfp16_Yfp16/
  gqa/decode/Xfp16_Wi8_Yfp16/
  gqa/prefill/Xfp16_Wfp16_Yfp16/
  gqa/prefill/Xfp16_Wi8_Yfp16/
  qmoe_ragged_mm/Xfp16_Wu4_Yfp16/
  gather_block_quantized/Xi64_W{u4,u8,fp8}_Yfp16/
  cast/Xany_Yany_bf16/
```

`unit_test/` holds one directory per op and nothing else besides the files
above. LUT refreshes run through these leaves (see below); the older
offline sweep tools under `hip/autotune/<op>/tools/` remain only to reproduce
historical logs. Kernel
compile-time guards, which run no kernel and need no GPU (e.g.
`LdsBarrierArchGate`), live in the repository's `test/kernel_guards/` with
the other GPU-free CTest suites.

## Refreshing a LUT with the UT leaves

LUT tables are refreshed with these leaves rather than a separate sweep
harness, so a table and the unit tests are measured by the same timer. The eight
MatMulNBits leaves and four GQA decode/prefill leaves support it; other ops keep
their previous tools until they are migrated.

A refresh does not choose a timing preset. For each shape the ranking mode:

1. checks every candidate config, always including the no-LUT default, against
   the leaf's CPU reference;
2. measures every candidate once with `screen`;
3. measures the candidates within 5% of the fastest (at most four) plus the
   default, interleaved, at `standard` goals;
4. repeats that round at `confirm` goals only when the decision is close or
   unclear: a winner near the 2% entry line, a winner and runner-up that
   overlap, a nonconverged finalist, or a shape marked `priority=1`;
5. stores the winner only when it is not the default and its median ratio to
   the default is below 0.98; otherwise the default-equivalent entry stays.

Near-tied finalists can swap between runs; that costs little and is expected.
Rules, the `UT_RANK` record and gfx1151 results are in
[common/README.md](common/README.md#native-lut-ranking-ut-rank-v3).

To refresh `<op>` (`matmul_nbits` or `gqa`) from its existing JSON table:

1. From `hip/autotune/<op>/`, generate default + old-table candidates:

   ```sh
   python scripts/update_lut.py rank-seed --from-lut lut/gfx1151.json --out <private>/candidates.csv
   ```

   The command writes the candidate CSV, `.seed.json` identities and a
   `.skipped.json` list with reasons. Only JSON input is supported; FB is
   compiled from JSON. `--sample 20 --seed 111313` selects reproducible points;
   `--leaf`, `--dtype`, `--rank-phase decode|prefill`, and `--keys` filter them.
   Choose `--activation fp16|fp32` for MatMulNBits or
   `--prefill-kv fp16|int8` for GQA prefill at both seed and import/report steps.
   These dimensions are absent from the respective schema keys.
2. Build each selected leaf with
   `make build RANK=1 BUILD_DIR=out_rank HIP_SDK=<sdk>`. On Windows, add
   `ADLX_INCLUDE=<ADLX SDK Include directory>` for GPU clock diagnostics.
3. Run the leaves, inspect the point report, and build a private table:

   ```sh
   python scripts/update_lut.py rank-measure --candidates <private>/candidates.csv --leaf-root <checkout>/lib/Runtime/Kernels/unit_test --out <private>/rank.log
   python scripts/update_lut.py rank-report --rank-log <private>/rank.log --from-lut lut/gfx1151.json --out <private>/report.csv --md <private>/report.md <op options>
   python scripts/update_lut.py rank-build --rank-log <private>/rank.log --base lut/gfx1151.json --out <private>/candidate.json <op options>
   python scripts/update_lut.py compile --out <private>/candidate.json --fb-out <private>/candidate.fb --flatc <flatc>
   ```

   `<op options>` is `--activation fp16|fp32` for MatMulNBits and
   `--prefill-kv fp16|int8` for GQA prefill. `rank-measure` runs one native
   process at a time and accepts `--launcher-json` for the host's execution
   lease wrapper. `rank-report` validates `UT_RANK` and reports the old/default/
   winner ratios, effective selection changes, retests, exclusions and costs.
   `rank-build` preserves weights, fallback/config indices, point order and
   unrelated fields; unchanged selections retain their exact original index.
4. Review the private JSON/FB before a separately authorized installation.

The old table geometry is resolved against the leaf's current production
launch descriptors; its config index is never used as a native ID. The leaf
resolves its actual no-LUT default and deduplicates old=default before measuring.
To include all legal configs, run `rank-catalog --candidates <private>/candidates.csv`
with the same `--leaf-root` and `--out <private>/catalog.log`, then repeat
`rank-seed ... --with-catalog --catalog <private>/catalog.log`. Catalog expansion
is optional; ordinary existing-table refresh compares only default and old.

For shapes absent from the old table, a hand-written CSV with
`leaf,shape,priority,config` and one `default` row per shape remains supported.
Use `rank-catalog` and `rank-candidates --candidates <seed.csv> --catalog <catalog.log>`
to expand it. The host contract tests run through `make -C common test`, or
`python3 common/test_ut_lut_from_table.py`: both existing gfx1151 tables are
reconstructed from synthetic old-config winners with full parsed JSON equality,
including skipped points, weights, fallback/config indices and array order.

Each op's details are in [hip/autotune/matmul_nbits/README.md](../hip/autotune/matmul_nbits/README.md#refresh-through-native-ut-ranking)
and [hip/autotune/gqa/README.md](../hip/autotune/gqa/README.md#refresh-through-native-ut-ranking).

## External invocation (`ut_interface.json`)

[`ut_interface.json`](ut_interface.json) lists every input an external caller
(CI, an agent, a script or a colleague) passes to run these tests, and
[`common/ut_interface.py`](common/ut_interface.py) turns parameter values into
commands and runs them. The package installs both in `bin/kernel-tests/`.

Run one case through the controller, from `bin`, so the leaf gets the
package's HIP runtime, environment, stderr capture and `out/results.csv` rows;
a leaf executable started directly does not (see "Run" above):

```bat
cd <install>\bin
hipdnn-kernel-ut.exe --leaf gqa/prefill/Xfp16_Wfp16_Yfp16 --timing standard -- --b 1 --h 32 --g 8 --d 128 --sq 512 --past 1024 --max-seq 4096
```

The tool (Python 3.8+, standard library only) builds those arguments from
parameter names and checks them before anything runs:

```bat
python kernel-tests\ut_interface.py list
python kernel-tests\ut_interface.py show qmoe/Xfp16_Wu4_Yfp16
python kernel-tests\ut_interface.py command qmoe/Xfp16_Wu4_Yfp16 tokens=1 hidden=2048 inter=768 experts=128 topk=8 block_size=32
python kernel-tests\ut_interface.py run qmoe/Xfp16_Wu4_Yfp16 --bin . tokens=1 hidden=2048 inter=768 experts=128 topk=8 block_size=32
python kernel-tests\ut_interface.py verify --bin .
```

`list` shows every leaf's parameters, `show` one leaf's flags, defaults,
constraints and examples. `command` prints the package command and, when every
given parameter has a Make variable, the `make test_custom` command of the
source tree. `run` executes the case and prints its `results.csv` rows.
`verify` runs every example in the file (default `--timing screen`) and prints
`PASS` or `FAIL` per example; it exits 0 only when all pass. `run` and
`verify` replace `bin/out/results.csv` like any controller run.

The file's fields:

- `tool` and `controller`: the commands above; the controller flags
  (`--leaf`, `--arch`, `--mode`, `--coverage`, `--timing`, `--`), what it
  passes to each leaf, its exit codes and outputs.
- `common`: the Make and environment variables every leaf shares. `COVERAGE`
  (`--coverage`, `HIPDNN_UT_COVERAGE`) selects the shape tier, `TIMING`
  (`HIPDNN_UT_TIMING`) the timing preset, `MODE` with
  `HIPDNN_<OP>_AUTOTUNE_MODE` the config source and `RANK` with
  `HIPDNN_UT_RANK*` LUT ranking; `HIP_SDK`, `OFFLOAD`, `LUT_ARCH`, `FLATC` and
  `FLATBUFFERS_INC` are build inputs.
- `leaves[]`: one entry per registered leaf with its `package_executable`,
  `source`, `capabilities` (`coverage`, `timing` = `shared` or `none`,
  `mode` = `lookup` or `label_only`, `rank`), the `suite` arguments and
  `custom`, the leaf's single-case interface:
  - `selected_by`: what switches the leaf from the suite to one case.
  - `params[]`: every op input of that case. `name` is what the tool takes;
    `flag` is the leaf argument (`positional` with a `format` when it has
    none) and `make` the `make test_custom` variable (`null` when only the
    executable accepts it; `make_args` names a raw `ARGS=` instead), plus
    `type`, `choices`, `required` and `description`. `exe_default` and
    `make_default` can differ: an omitted flag takes `exe_default`, an
    omitted Make variable `make_default`.
  - `constraints` in words, and `checks`: Python expressions over parameter
    names (plus `M`, `K`, `N` of an `MxKxN` shape and `min`) that the tool
    evaluates on the given values, or `exe_default`, before running.
- `examples[]`: named runnable cases, `values` (parameter name to value) for a
  custom case or `coverage` for a suite run; real model shapes that finish in
  seconds.
- `diagnostic_flags` and a leaf's own `env` do not change the op inputs.
- `gaps`: required capabilities a leaf that predates the
  [guidelines](GUIDELINES.md) still lacks (`custom_shape`, `coverage`,
  `shared_timer`, `results_csv`, `make_api`, `arch_default`). `custom: null`
  means the leaf cannot run a caller-chosen shape; its example runs the suite.

Every leaf must have an entry; its required fields are in
[GUIDELINES.md](GUIDELINES.md#required-entry-for-every-new-leaf).

## Uniform make API (standalone development)

From `unit_test/`, the whole suite, run the way the CI runs it:

```
make test  HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<gfx>] [LUT_ARCH=<gfx>] [MODE=lookup|autotune] [COVERAGE=1|2|3] [TIMING=screen|standard|confirm]
make build ...   # build only
make clean ...
```

The Makefiles use repo-relative paths only: `HIP_SDK`, `OFFLOAD` and
`LUT_ARCH` come from the command line. `MODE=lookup` on a chip that has a table
(every chip with a checked-in `lut/<chip>.fb`) also needs `FLATC` and
`FLATBUFFERS_INC`; see "`MODE=lookup`" below.

The build follows the release: compile for the CI target, pick the table by
the real chip.

- `OFFLOAD` defaults to `--offload-arch=gfx11-generic`, the target of the
  `custom_kernels_gfx11-generic.dll` every gfx115x part runs, and the kernels
  build at `-O2` like that DLL, so a test measures the code that ships.
  For gfx1170 or gfx1171 pass `OFFLOAD=--offload-arch=gfx1170` or
  `--offload-arch=gfx1171`. A chip target such as
  `OFFLOAD=--offload-arch=gfx1151` still works and uses that chip's table.
- The LUTs are per chip (`lut/gfx1151.fb`, ...), so for a `gfx*-generic`
  target `LUT_ARCH` defaults to the GPU of the machine that builds, read with
  `$(HIP_SDK)/bin/amdgpu-arch.exe`. Pass `LUT_ARCH=gfxNNNN` when the exe will
  run on another box (e.g. `LUT_ARCH=gfx1150` for a gfx11-generic exe built
  on a gfx1151 and run on a gfx1150). The resolver still
  checks the table's `gpu_arch` against the device at run time and ignores a
  table for another chip, so a wrong `LUT_ARCH` costs the lookup, never
  correctness. `mode_info` prints the table it picked.

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
  `hip/autotune/<op>/lut/<LUT_ARCH>.fb`. If that chip has no table, it warns
  and falls back to `autotune` — a missing table is never a hard failure.
- **`autotune`**: always the runtime sweep, never touches a table.

`MODE` also selects the kernel's own source, via
`HIPDNN_{MATMUL,GQA}_AUTOTUNE_MODE=lookup|online` in the leaf's run
command. Both halves must agree: the Makefile value alone only decides whether
the real resolver and table are *linked*, so without that wiring a run labelled
`autotune` would still be resolving from a table.

`MODE=lookup` needs `flatc` (a build tool, **not part of this repo** — see
`knowledge/hosts.md`) plus its `include/`, passed on the command line:

```
make test MODE=lookup HIP_SDK=<sdk> \
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
gather_block_quantized writes `R1024_C160_bs16_I2x8` (table rows, row length,
block size, indices shape) with `_compact`, `_zp` and `_signed` when set, or
the fp8 scale granularity `_tensor` / `_row` / `_block`; its `config` is always
`fixed:elementwise` and its `time_ms` 0.

`time_ms` is the **mean per-launch** time of that case's timed launches (the
leaf's fixed count of launches between two HIP events), not a total, and it is
measured **cold**. In a model a kernel's weights or KV cache are read once per
step, and the rest of the model streams through the GPU caches before they are
read again, so every launch reads them from DRAM. The timed launches therefore
rotate over copies of that operand -- the MatMulNBits and qmoe_ragged_mm
weights (with their scales and zero points), the GQA K and V caches, of which a
sliding window copies only the rows it reads -- totalling several times the
last-level cache, so each launch reads its copy from DRAM the way consecutive
layers read their own. The activation and the output stay single and warm, as
they are in a model. Untimed launches over the rotation go first, and the timed
launches are held on the GPU until all of them are queued, so the reading is
GPU time rather than the host's submission rate. Flushing the caches between
launches would not reproduce DRAM reads: a read-only flush leaves the reused
lines resident, and a write flush leaves dirty lines for the timed launch to
write back. The copies take a few hundred MB of device memory per case.
`HIPDNN_KERNEL_UT_WARM=1` in the environment of `hipdnn-kernel-ut.exe` (the
leaves inherit it) or of a leaf times one copy back to back on warm caches
instead, for comparison.

`time_ms` is the only timing in the row: the tuner's own figure is a *peak*
(minimum) sample taken during the sweep, under a different clock state, so it
is deliberately not carried in `config` where it would read as the same
measurement. qmoe_ragged_mm times its MatMul cases; its selector, descriptor
and pipeline rows carry 0.

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
to a scratch file beside its executable instead (so it works from any working
directory), which captures the kernels only when they are linked into the leaf
(the local make build). Either way a passing run
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

## QMoE model-shape profiling

`qmoe/Xfp16_Wu4_Yfp16` measures the actual production QMoE wrapper, including
top-k routing, decode/grouped expert work, activation and reduction, for
FP16/packed uint4 fused-SwiGLU softmax-routing signatures. It accepts one
model shape via `make test_custom ARGS="--tokens 1 --hidden 128 --inter 128
--experts 4 --topk 2 --block-size 32 --fc1-bias 0 --fc2-bias 0
--zero-points 0 --normalize 1 --router-weights 0" TIMING=screen`.
The existing `qmoe_ragged_mm` leaf remains a separate primitive test; its
timing does not stand for an entire QMoE model node. See the full-wrapper
leaf README for supported attributes, coverage tiers and timing presets.

PagedAttention FP16 decode/prefill leaves invoke the complete public entry (RoPE, paged append and causal attention). Their semantic custom parameters are registered in `ut_interface.json`; both MODE values label the fixed configuration. See `paged_attention/decode/Xfp16_Wfp16_Yfp16/README.md` and the prefill counterpart.
