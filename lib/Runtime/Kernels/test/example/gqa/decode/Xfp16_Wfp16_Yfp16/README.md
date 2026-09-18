# GQA flash decode test -- fp16 KV cache

Standalone `hipcc`-only test for `hip_gqa_flash_decode()` (fp16 KV cache).
No CMake, no EP build. Entirely self-contained: inputs and the CPU fp32
reference are generated in-process, no python, no data files, no
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

`test_gqa_decode.cpp`'s `main()` has 13 named categorical situations (real
models -- gpt-oss-20b full/sliding/smooth, llama-3.1-8b, llama-3.2-1b,
qwen2.5-14b -- plus a geometry sweep over MHA (HpG==1) and GQA (HpG in
{2,8,16}) x head_dim in {64,128,256}), which run in full at **every**
`COVERAGE` tier. Crossed with that is `kLens`, a typical context-length list
widened to also cover a short/interactive decode length: tier3 (default) =
`{128, 512, 2048, 8192}`, tier2 = `{128, 2048, 8192}`, tier1 = `{2048}`
alone. `COVERAGE=1|2|3` (`make test COVERAGE=N` or env `HIPDNN_UT_COVERAGE`)
only picks which of those lengths run; it never drops a categorical
situation. At startup the exe prints `coverage=N -> running <n> typical
length(s) x 13 categorical case(s) = <n*13> gqa_decode cases`. No human
edits a shape list -- there is no `shapes.csv` or `gen_data.py` in this
leaf. Decode's reference cost scales ~linearly with context length (one
query against `eff` keys), so no threading was needed here.

`MODE=auto` (default): `lut` if `hip/autotune/gqa/lut/<arch>.fb` exists for
the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it (warns + falls
back to `autotune` if the `.fb` is missing). `MODE=lut` needs `flatc` + its
`include/` (a build tool, not part of this repo):
`FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>` -- see
`example/README.md`.

`gqa_kernel.hip` never calls the autotune resolver itself (only production
`real/gqa.cpp` does); `MODE=autotune` (the fallback, and the only path when
no `.fb` exists) runs the kernel's own internal runtime autotune + cache,
independent of `MODE`. `MODE=lut` instead has *this test* call
`hip_gqa_autotune_resolve_decode()` (the same resolver `real/gqa.cpp` calls in
production) and dispatch the resolved config through
`hip_gqa_flash_decode_configured()`. Both modes append to `out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
