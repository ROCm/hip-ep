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

For `lut/gfx1151`: 993 measured shapes, 388 kept, 11,064 bytes (25,584 unpruned).

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

```powershell
# 1. shapes: any HIPDNN_EP_DEBUG=1 console logs (CI consoleText works)
python scripts/update_lut.py extract --logs <log>... --out shapes/survey.csv

# 2. sweep on the target GPU, against the custom_kernels_<arch> it will ship in
clang++ --driver-mode=g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ `
    -I <rocm>/include -I lib/Runtime/Kernels/include -I lib/Runtime/real `
    tools/ck_gemm_autotune_sweep.cpp -o ck_gemm_sweep.exe `
    -L <rocm>/lib -l amdhip64 <build>/lib/Runtime/Kernels/custom_kernels_<arch>.lib
ck_gemm_sweep.exe shapes/survey.csv winners.csv

# 3. table
python scripts/update_lut.py build --winners winners.csv --arch gfx1151 --rocm-version 71600
python scripts/update_lut.py compile --arch gfx1151 --flatc <build>/bin/flatc.exe
```

## Knobs

`HIPDNN_CK_GEMM_AUTOTUNE_MODE=online` skips the table (A/B against the sweep). `HIPDNN_CK_GEMM_LUT_LOG=1` logs load status and every lookup with its nearest point and distance, to the DLL's stderr.
