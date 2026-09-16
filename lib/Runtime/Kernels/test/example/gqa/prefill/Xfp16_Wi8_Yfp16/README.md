# GQA flash prefill test -- INT8 KV cache

Standalone `hipcc`-only test for the fused FA-2 WMMA prefill kernels against
an INT8 KV cache (dequantized once into an fp16 scratch buffer, then run
through the same fp16 prefill kernels -- exercises the runtime path). No
CMake, no EP build. `test`/`test_custom` inputs come from `gen_data.py`
(numpy); the built-in matrix path keeps in-process random data since one
`data/` directory cannot hold every shape.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK%
make test_shapes OFFLOAD=... HIP_SDK=...
make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../../models/gpt_oss_20b.json]
make test_custom OFFLOAD=... HIP_SDK=... H=32 G=8 D=128 SQ=512
make clean
```

| Target | Meaning |
|---|---|
| `test` | Smoke: `gendata` writes `data/` for one small fixed shape, then the exe loads it (`--data-dir data`). |
| `test_shapes` | The binary's own built-in matrix (8 shapes x 3 seq-lens); `shapes.csv` documents the representative subset. |
| `test_model` | Every `gqa_prefill` entry in `MODEL_JSON` (shared `../../../models/*.json`). |
| `test_custom` | One shape from `H=`, `G=`, `D=`, `SQ=` (B is always 1 here); also `gendata`+`--data-dir`. |
| `clean` | Removes `out/` and `data/`. |

`MODE=autotune` (default): the prefill launchers self-tune their launch
configuration per shape, independent of the `MODE` knob.
`MODE=lut`: if `hip/autotune/gqa/lut/<arch>.fb` exists for the arch in
`OFFLOAD`, this build prints a one-line notice and falls back to
`MODE=autotune` -- the prefill kernel does not call the FlatBuffers LUT
resolver at all today, and wiring that up needs a flatc-generated header this
build intentionally does without. It never fails because of this.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
