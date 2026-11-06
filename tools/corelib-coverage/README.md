<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# corelib-coverage

Lists, for an ONNX model, the ops hip-ep routes to RyzenAI CoreLib (NPU), whether
each can run there, and the `RYZENAICORELIB_PDI` that reaches them. It reads the
graph only: no weights, GPU, NPU, hip-ep or CoreLib are needed, so it runs on any
machine, including for models too large to load. See
[ryzenai-corelib-backend.md](../../docs/design/ryzenai-corelib-backend.md).

```bash
pip install onnx

# Weight-format checks only: which ops hip-ep would send to CoreLib.
python tools/corelib-coverage/corelib_coverage.py models/<name>/model.onnx

# With kernel availability from a DynamicDispatch kernel store.
python tools/corelib-coverage/corelib_coverage.py models/<name>/model.onnx \
    --kernels <DynamicDispatch>/transaction/mds
```

`--kernels` takes any of:

- the store directory (`transaction/mds`); file names are enough, so a sparse or
  LFS-pointer checkout works;
- a text file of kernel names, one per line, as
  `git ls-tree -r --name-only <ref> -- transaction/mds` prints them;
- a DynamicDispatch repository with `--ref <tag>`, for example
  `--ref corelib/kernels/20261029.0`.

Output, per op and shape: the instances (ops with their own weights), and either
the PDIs shipping a kernel with their row counts, or why the op runs on the GPU
(weight format, no kernel at the shape). Then the total and a suggested
`RYZENAICORELIB_PDI`: prefill PDIs ship several rows, decode PDIs one row, each
list ordered by the op instances it reaches.

```text
models/Llama-3.1-8B-Instruct-awq-g128-int4/model.onnx
  op           shape                              inst  NPU kernels (PDI (rows))
  MatMulNBits  K=4096 N=14336 block=128             64  p1 (1..4096), p16 (1), p32 (1)
  SkipRmsNorm  k=4096                               64  p9 (64..4096), p19 (1), p32 (1), p35 (1)
  ...
  total: 290 of 290 op instances with kernels
  RYZENAICORELIB_PDI=1,9;32
```

To keep the setting with a model:

- `--write-pdi` writes `<model>.onnx.corelib_pdi.txt` beside each model, one line
  `RYZENAICORELIB_PDI=<prefill>;<decode>`, which a run script can read;
- `--json <file>` writes every model's table, PDIs and row counts included.

The suggestion comes from kernel names alone. A run with
`RYZENAICORELIB_REPORT=1` confirms what actually ran on the NPU; a PDI list naming
several PDIs per phase makes the NPU switch between them, whose cost is model
dependent.
