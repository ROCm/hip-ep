# GQA flash prefill test -- INT8 KV cache

Standalone `hipcc`-only test for the fused FA-2 WMMA prefill kernels against
an INT8 KV cache (dequantized once into an fp16 scratch buffer, then run
through the same fp16 prefill kernels -- exercises the runtime path). No
CMake, no EP build. Entirely self-contained: inputs are generated in-process,
no python, no data files, no `example/common/`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... H=32 G=8 D=128 SQ=512
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | One shape from `H=`, `G=`, `D=`, `SQ=` (B is always 1 here, in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

`test_gqa_prefill_i8.cpp`'s `main()` has 8 named geometries (MHA d64, MHA
d128, GQA2 d128, llama-3.2-1b, llama-3.1-8b, psu_orc_211, gpt-oss-20b,
llama-3-70b), which run in full at **every** `COVERAGE` tier. Crossed with
that is a typical prompt-length list: tier3 (default) =
`{128, 256, 512, 1024, 2048}`, tier2 = `{256, 1024}`, tier1 = `{512}` alone.
`COVERAGE=1|2|3` (`make test COVERAGE=N` or env `HIPDNN_UT_COVERAGE`) only
picks which of those lengths run. At startup the exe prints `coverage=N ->
running <n> typical length(s) x 8 categorical case(s) = <n*8>
gqa_prefill_i8 cases`. No human edits a shape list -- there is no
`shapes.csv` or `gen_data.py` in this leaf.

This leaf runs *every* named shape (including the `H=64` ones) at *every*
length and computes two `O(sq^2)` causal-attention CPU references per case
(i8 + fp16), unlike the fp16 leaf's curated one-case-per-shape list. Both
references (`cpu_reference_i8`, `cpu_reference_fp16`) are now multithreaded
over `(batch, query-head)` pairs (`forEachBH()`), which is what makes adding
`128` (short) and `2048` (long) to the length list affordable without a
full-cross budget blowup.

`MODE=auto` (default): `lut` if `hip/autotune/gqa/lut/<arch>.fb` exists for
the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it (warns + falls
back to `autotune` if the `.fb` is missing). Both build and run fine either
way -- the prefill launchers here never call the autotune-LUT resolver
directly (only production `real/gqa.cpp` does), so `MODE=lut` only needs to
satisfy `gqa_autotune.cpp`'s (unused) LUT-data symbols so the link succeeds;
they always self-tune their launch configuration per shape, independent of
`MODE`. `MODE=lut` needs `flatc` + its `include/` (a build tool, not part of
this repo): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>`
-- see `example/README.md`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
