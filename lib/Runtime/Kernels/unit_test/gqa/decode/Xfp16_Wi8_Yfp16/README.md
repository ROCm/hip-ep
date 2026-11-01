# GQA flash decode test -- INT8 KV cache

Standalone `hipcc`-only test for `hip_gqa_flash_decode()` against a symmetric
per-channel INT8 KV cache. No CMake, no EP build. Entirely self-contained:
inputs are generated in-process, no python, no data files, no
`unit_test/common/`.

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

`test_gqa_decode_i8.cpp`'s `main()` has 13 named geometries -- MHA hpg1
(`mha-h16-d64`, `mha-h16-d128`, `mha-h20-d64`, `mha-h20-d128`), GQA hpg2
(`gqa2-h16-d64`, `gqa2-h16-d128`), GQA hpg4 (`llama-3.2-1b`, `llama-3.1-8b`,
`psu_orc_211`), GQA hpg8 (`gpt-oss-20b`, `llama-3-70b`), and head_dim 96 at
MHA and hpg4 (`phi-3-mini-d96`, `gqa4-h32-d96`) -- which run in full
at **every** `COVERAGE` tier. Crossed with that is a typical context-length
list widened to also cover a short/interactive decode length: tier3
(default) = `{128, 512, 2048, 8192}`, tier2 = `{128, 2048, 8192}`, tier1 =
`{2048}` alone. `COVERAGE=1|2|3` (`make test COVERAGE=N` or env
`HIPDNN_UT_COVERAGE`) only picks which of those lengths run. At startup the
exe prints `coverage=N -> running <n> typical length(s) x 13 categorical
case(s) = <n*13> gqa_decode_i8 cases`. No human edits a shape list -- there
is no `shapes.csv` or `gen_data.py` in this leaf.

There is no head_dim 256 geometry: production `real/gqa.cpp` accepts an INT8
KV cache only at head_dim 64, 96 and 128 and rejects d256 before any kernel
launch, so the leaf stays inside that range.

`MODE=lookup` (default): resolves from `hip/autotune/gqa/lut/<arch>.fb` if it
exists for the arch in `OFFLOAD`, else falls back to `autotune` with a
warning. Like production `real/gqa.cpp`, the leaf resolves the int8 decode
config itself (`hip_gqa_autotune_resolve_decode` with `kv_dtype=INT8`) and
launches it through `hip_gqa_flash_decode_configured`; `MODE=autotune` calls
the self-tuning `hip_gqa_flash_decode` instead. `MODE=lookup` needs `flatc` +
its `include/` (a build tool, not part of this repo):
`FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>` -- see
`unit_test/README.md`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
