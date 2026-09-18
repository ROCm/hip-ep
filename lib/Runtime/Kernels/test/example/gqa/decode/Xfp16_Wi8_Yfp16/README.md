# GQA flash decode test -- INT8 KV cache

Standalone `hipcc`-only test for `hip_gqa_flash_decode()` against a symmetric
per-channel INT8 KV cache. No CMake, no EP build. Entirely self-contained:
inputs are generated in-process, no python, no data files, no
`example/common/`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... B=1 H=32 G=8 D=128 MAX_SEQ=4096 TOTAL=2048
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | One shape from `B=`, `H=`, `G=`, `D=`, `MAX_SEQ=`, `TOTAL=` (in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

`test_gqa_decode_i8.cpp`'s `main()` has 11 named geometries -- MHA hpg1
(`mha-h16-d64`, `mha-h16-d128`, `mha-h20-d64`, `mha-h20-d128`), GQA hpg2
(`gqa2-h16-d64`, `gqa2-h16-d128`), GQA hpg4 (`llama-3.2-1b`, `llama-3.1-8b`,
`psu_orc_211`), GQA hpg8 (`gpt-oss-20b`, `llama-3-70b`) -- which run in full
at **every** `COVERAGE` tier. Crossed with that is a typical context-length
list widened to also cover a short/interactive decode length: tier3
(default) = `{128, 512, 2048, 8192}`, tier2 = `{128, 2048, 8192}`, tier1 =
`{2048}` alone. `COVERAGE=1|2|3` (`make test COVERAGE=N` or env
`HIPDNN_UT_COVERAGE`) only picks which of those lengths run. At startup the
exe prints `coverage=N -> running <n> typical length(s) x 11 categorical
case(s) = <n*11> gqa_decode_i8 cases`. No human edits a shape list -- there
is no `shapes.csv` or `gen_data.py` in this leaf.

`MODE=auto` (default): `lut` if `hip/autotune/gqa/lut/<arch>.fb` exists for
the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it (warns + falls
back to `autotune` if the `.fb` is missing). Both build and run fine either
way -- `gqa_kernel.hip` never calls the autotune-LUT resolver for this path
(only production `real/gqa.cpp` does), so `MODE=lut` here only needs to
satisfy `gqa_autotune.cpp`'s (unused) LUT-data symbols so the link succeeds;
the kernel itself always runs its own internal runtime autotune + cache,
independent of `MODE`. `MODE=lut` needs `flatc` + its `include/` (a build
tool, not part of this repo):
`FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>` -- see
`example/README.md`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
