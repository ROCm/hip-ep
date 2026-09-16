# MatMulNBits bits=2 (uint2) vs bits=4 test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=2, 2-bit packed),
benchmarked back-to-back against bits=4 on the same shape. No CMake, no EP
build.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK%
make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../models/gpt_oss_20b.json]
make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x128x128 GS=128 [NO_ZEROS=1]
make clean
```

| Target | Meaning |
|---|---|
| `test` | Smoke: one small fixed shape (`1x2880x2880`, GS=128). |
| `test_shapes` | Every row in `SHAPES` (default `shapes.csv`). K must be a multiple of 32. |
| `test_model` | Every `M_array` x `KN_pairs` shape in `MODEL_JSON` (shared `../../models/*.json`); shapes with K not %32 are skipped. |
| `test_custom` | One shape from `SIZE=MxKxN`, `GS=`, `NO_ZEROS=1`. |
| `clean` | Removes `out/`, `data/`, `data_model/`. |

`MODE=autotune` (default) links an empty `resolve()` stub so the kernel runs
its own runtime autotune sweep. `MODE=lut`: if
`hip/autotune/matmul_nbits/lut/<arch>.fb` exists for the arch in `OFFLOAD`,
this build prints a one-line notice and falls back to `MODE=autotune` -- the
real FlatBuffers LUT resolver needs a flatc-generated header that only the
CMake build produces, and this test intentionally builds without CMake/flatc.
It never fails because of this.

`gen_data.py` is the single data generator and covers all three modes itself:

```
python gen_data.py SIZE=... --group-size GS --dir data          # test / test_custom
python gen_data.py --model ../../models/gpt_oss_20b.json --group-size GS --out-dir data_model  # test_model
```

(`test_shapes` reuses the single-shape mode once per row of `shapes.csv`, driven by the Makefile.)

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
