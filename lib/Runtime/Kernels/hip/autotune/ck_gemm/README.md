# CK GEMM offline autotune LUT

Ships the measured winner of `ckSelectGemmInstance` for known shapes, so the first call on a shape costs at most three candidates instead of a sweep over every instance in its registry partition.

## Where it sits

```
table proposes up to 3 instances (nearest measured shapes first)
  -> hip_ck_gemm_run accepts exactly one -> it wins
  -> accepts several -> the fastest of those, timed like the sweep
  -> none accepted / no table / online mode -> runtime sweep, as before
```

A proposal is only a starting point: CK still has to accept it for this exact shape. So a stale or distant point costs a rejected launch, never a wrong kernel.

## Key

Exact match on `(ab dtype, d dtype, transA, bias)`, which fixes the registry partition, and on the alignment class of `m`, `n` and `k` (largest of 8 / 4 / 2 / 1 dividing each). Within that group, nearest in log space over `(m, n, k, batch)`, weights stored in the table. If the aligned group has fewer than three distinct answers, the rest of the partition tops up the list. All arguments are `hip_ck_gemm_run`'s column-major ones, i.e. after the call site's operand swap.

Alignment is part of the key because CK tiles are gated on vector widths that must divide these dims, so the winner follows alignment rather than magnitude. Along one f16 -> f32 score-GEMM series (n=6, k=256, m = 1843..2140) odd `m` picks `TNK_E1a`, `m % 4 == 2` mostly `TNK_E2a`, and 4-aligned `m` the full-width `TNK_5` / `TNK_6`: `m = 1844` and `m = 1845` are neighbours in log space but not in tile choice.

The alignment class is computed from each point's own dims at load, so it is not stored in the table.

Points name instances by `hip_ck_gemm_instance_name()` (`"NN_F16:TKN_5"`), not by index: indices shift whenever `kEntries[]` is edited. A name the build no longer has drops its points at load. Bump `kernel_abi` when a tile keeps its name but changes meaning.

## Pruning

`build` keeps a point only if some measured shape needs it: a point is dropped when, without it, every measured shape still resolves to its own measured winner as the first proposal. The check runs per aligned group, never empties one, and mirrors the resolver's ranking (squared log2 distance, ties to the earlier point in key order). An unmeasured shape gets the answer of its nearest kept point.

The resolver ranks in float32 and the script in float64, so a rival point with a different answer must be farther by `TIE_MARGIN` (1e-6 in squared log2 distance) before a removal is accepted; otherwise two near-equidistant neighbours can swap places in the DLL.

`shapes/survey.csv` and the winners CSV keep every shape; only `lut/<arch>.json` / `.fb` are pruned. `--keep-all` skips pruning.

For `lut/gfx1151`: 993 measured shapes, 352 kept, 9,944 bytes (25,328 unpruned).

## Files

| | |
|---|---|
| `ck_gemm_autotune.fbs` | schema |
| `ck_gemm_autotune.cpp` | loader + lookup, exports `hip_ck_gemm_lut_candidates` |
| `shapes/survey.csv` | shape inventory, from `HIPDNN_EP_DEBUG` logs |
| `lut/<arch>.json` | reviewable source of truth |
| `lut/<arch>.fb` | compiled table; the only form embedded into `custom_kernels_<arch>` |
| `scripts/update_lut.py` | extract / build (with pruning) / compile |
| `tools/ck_gemm_autotune_sweep.cpp` | runs `ckSelectGemmInstance` in online mode over a shapes CSV |

## Regenerating

Regenerate after anything that can move the fastest instance: an edit to `kEntries[]`, a Composable Kernel pin bump, or a ROCm SDK / compiler change in `cmake/deps.txt`. Re-extract the shapes only when the model set changes; otherwise reuse `shapes/survey.csv`.

Sweep against the DLL the table ships in, built at the commit the table lands on. A table is embedded into the `custom_kernels_<arch>` whose family lists its arch (`_lut_arch_members` in `lib/Runtime/Kernels/CMakeLists.txt`): gfx1150 / gfx1151 ship in `custom_kernels_gfx11-generic`, gfx1170 / gfx1171 in `custom_kernels_gfx11-7-generic`. Winners measured on a per-arch DLL describe code that does not ship.

Windows, from a configured hip-ep build directory `<build>`; `<rocm>` is `<build>/_therock`:

```powershell
# 0. MSVC environment, needed to link the sweep tool
& "<Visual Studio>\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

# 1. shapes, only when the model set changed: any HIPDNN_EP_DEBUG=1 console logs (CI consoleText works)
python scripts/update_lut.py extract --logs <log>... --out shapes/survey.csv

# 2. the shipping DLL: <build>/bin/custom_kernels_gfx11-generic.dll and its import library
cmake --build <build> --target custom_kernels_gfx11-generic

# 3. sweep on an idle target GPU (about 1.5 minutes for shapes/survey.csv on gfx1151)
<rocm>/bin/clang++ --driver-mode=g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ `
    -I <rocm>/include -I lib/Runtime/Kernels/include -I lib/Runtime/real `
    tools/ck_gemm_autotune_sweep.cpp -o ck_gemm_sweep.exe `
    -L <rocm>/lib -l amdhip64 <build>/lib/Runtime/Kernels/custom_kernels_gfx11-generic.lib
$env:PATH = "<build>\bin;<rocm>\bin;$env:PATH"
ck_gemm_sweep.exe shapes/survey.csv winners.csv

# 4. table; the arch comes from winners.csv
python scripts/update_lut.py build --winners winners.csv --rocm-version 71600 --model-key "<where the shapes came from>"
python scripts/update_lut.py compile --arch gfx1151 --flatc <build>/bin/flatc.exe
```

The sweep writes the device it ran on as the first line of `winners.csv` (`# gpu_arch=gfx1151`). `build` names the table after it and refuses a mismatching `--arch` or winners from different GPUs. Exit code `-1073741515` (`0xC0000135`) from the sweep means a DLL is missing from `PATH`. A `custom_kernels_*.dll` next to an executable wins over `PATH`, so do not leave a copy from an earlier build there.

`--rocm-version` is the SDK version in the `therock_windows` tarball name in `cmake/deps.txt` as `major*10000 + minor*100 + patch` (7.16.0 is `71600`). It is a record only; the loader checks `kernel_abi` and `gpu_arch`. `--model-key` records the CI builds or models the shapes came from.

To check a new table, rebuild the DLL (the `.fb` is a configure dependency, so `cmake --build` picks it up) and run any model with `HIPDNN_CK_GEMM_LUT_LOG=1`. Expect `[ck-gemm-lut] loaded <n> points in <g> groups (0 dropped) for gfx1151`; a nonzero dropped count means instance names the build no longer has.

Two sweeps of the same DLL disagree on about 20% of winners (199 of 993 shapes on gfx1151). Judge a regenerated table by load status and end-to-end numbers, not by per-shape winner diffs against the old one.

## Knobs

`HIPDNN_CK_GEMM_AUTOTUNE_MODE=online` skips the table (A/B against the sweep). `HIPDNN_CK_GEMM_LUT_LOG=1` logs load status and every lookup with its nearest point and distance, to the DLL's stderr.
