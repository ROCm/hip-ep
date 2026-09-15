# GQA flash decode test -- fp16 KV cache

Standalone `hipcc`-only test for `hip_gqa_flash_decode()` (fp16 KV cache).
No CMake, no EP build. `test`/`test_custom` inputs come from `gen_data.py`
(numpy); the built-in `--all`/matrix path keeps in-process random data
since one `data/` directory cannot hold every shape.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK%
make test_shapes OFFLOAD=... HIP_SDK=...
make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../../models/gpt_oss_20b.json]
make test_custom OFFLOAD=... HIP_SDK=... B=1 H=32 G=8 D=128 MAX_SEQ=4096 TOTAL=2048
make clean
```

| Target | Meaning |
|---|---|
| `test` | Smoke: `gendata` writes `data/` for one small fixed shape, then the exe loads it (`--data-dir data`). |
| `test_shapes` | The binary's own built-in `--all` matrix (real models + an MHA/GQA x head_dim geometry sweep); `shapes.csv` documents the representative real-model subset. |
| `test_model` | Every `gqa_decode` entry in `MODEL_JSON` (shared `../../../models/*.json`). |
| `test_custom` | One shape from `B=`, `H=`, `G=`, `D=`, `MAX_SEQ=`, `TOTAL=`; also `gendata`+`--data-dir`. |
| `clean` | Removes `out/` and `data/`. |

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
