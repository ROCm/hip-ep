# GQA flash prefill test -- fp16 KV cache

Standalone `hipcc`-only test for the fused FA-2 WMMA prefill kernels
(`sq > 1`; the runtime selects v5/v7/v8 by head_dim). No CMake, no EP build. `test`/`test_custom` inputs come from `gen_data.py`
(numpy); the built-in `--all`/matrix path keeps in-process random data
since one `data/` directory cannot hold every shape.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK%
make test_shapes OFFLOAD=... HIP_SDK=...
make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../../models/gpt_oss_20b.json]
make test_custom OFFLOAD=... HIP_SDK=... B=1 H=32 G=8 D=128 SQ=512 [PAST=0] [WINDOW=0]
make clean
```

| Target | Meaning |
|---|---|
| `test` | Smoke: `gendata` writes `data/` for one small fixed shape, then the exe loads it (`--data-dir data`). |
| `test_shapes` | The binary's own built-in 30-case matrix (real models + sink/window variants); `shapes.csv` documents the representative subset. |
| `test_model` | Every `gqa_prefill` entry in `MODEL_JSON` (shared `../../../models/*.json`). |
| `test_custom` | One shape from `B=`, `H=`, `G=`, `D=`, `SQ=`, `PAST=`, `WINDOW=`; also `gendata`+`--data-dir`. |
| `clean` | Removes `out/` and `data/`. |

`MODE=auto` (default): `lut` if `hip/autotune/gqa/lut/<arch>.fb` exists for
the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it (warns + falls
back to `autotune` if the `.fb` is missing). `MODE=lut` needs `flatc` + its
`include/` (a build tool, not part of this repo):
`FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>` -- see
`example/README.md`.

The prefill kernel never calls the autotune resolver itself (only production
`real/gqa.cpp` does); `MODE=autotune` (the fallback, and the only path when no
`.fb` exists) has the launchers self-tune their configuration per shape,
independent of `MODE`. `MODE=lut` instead has *this test* call
`hip_gqa_autotune_resolve_prefill()` (the same resolver `real/gqa.cpp` calls in
production) and dispatch the resolved config through
`hip_gqa_flash_prefill_v3_configured()`. Both modes append to
`out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
