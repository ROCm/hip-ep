<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Whole Models on the NPU through RyzenAI CoreLib

When an onnxruntime-genai decoder is a model RyzenAI CoreLib (`ryzenai-corelib`)
knows as a whole, hip-ep can hand it the entire graph instead of compiling it.
CoreLib's model API (`include/ryzenai/corelib/model.h`) reads the ONNX file's
initializers itself, keeps the KV cache, and runs every layer, attention
included, on the AIE4 NPU. Any other model compiles for the GPU as before.

## Enabling

| Setting | Where | Effect |
|---|---|---|
| `use_ryzenai_corelib_model` = `1`/`true` | EP provider option | Try the model-level path |
| `HIPEP_USE_RYZENAI_CORELIB_MODEL` = `1`/`true` | Environment, read at compile time | Same, when the provider option is unset |
| `RYZENAICORELIB_ROOT` | Environment, build and run time | CoreLib install tree (or repository root; `install/` is searched). Build: locates `include/ryzenai/corelib/base.h` (CoreLib 0.66 or later). Run: loads `bin/ryzenai_corelib.dll` (`lib/libryzenai_corelib.so` on Linux), else the default library search path |
| `RYZENAICORELIB_REPORT` = `1` | Environment, run time | At session end, print `[corelib] model report: ...` with the prefills, decodes and rows run on the NPU |

In an onnxruntime-genai `genai_config.json` the option goes in the EP's
`provider_options`, for example `{"AMDGPU": {"profile": "hip",
"use_ryzenai_corelib_model": "true"}}`.

## Hardware

CoreLib's models run on an AIE4 (AIEM) NPU. `ryzenai_corelib_npu_usable()` is
false on AIE2/AIE2P parts (Phoenix, Strix, Krackan) and on an AIE4 part whose
XRT is older than 2.26.0; hip-ep then compiles the model for the GPU.

## Building and deploying

`RYZENAICORELIB_ROOT` names a CoreLib 0.66 or later install tree at build
time; only its headers are used. The configure step prints
`-- RyzenAI CoreLib <version>: <include dir>`; without CoreLib (or in mock
builds) it builds a stub that refuses every model, and warns when
`RYZENAICORELIB_ROOT` is set but holds no usable CoreLib.

At run time the EP loads `ryzenai_corelib.dll` itself on first use, so it still
loads on machines without CoreLib. CoreLib is below 1.0, so the EP refuses a
library whose major or minor version differs from the headers it was built
against and logs `[corelib] <dll> is version ..., but hip-ep was built against
...`. Each CoreLib model library (`ryzenai_corelib_model_<name>.dll`) must sit
beside `ryzenai_corelib.dll`; a model's kernels are linked into its library, so
no kernel store is needed. XRT normally comes from the NPU driver.

## How a model is matched

The level-1 pass (`backend-mlir-compiler/level-1-pass/src/pass_main.cpp`) takes
the path before building any MLIR when:

- the option is on;
- the graph is an onnxruntime-genai decoder: inputs `input_ids`,
  `attention_mask` and `past_key_values.*`; outputs `logits` and `present.*`;
  `GroupQueryAttention` and `MatMulNBits` nodes;
- a CoreLib model accepts the model file: the pass tries each architecture in
  `kCorelibModels` (now `qwen3`) with `ryzenai_corelib_create_model_from_path`,
  which reads the file's initializers and refuses what it does not recognize.

On success the whole graph is fused under the `CORELIBMODEL` domain, with the
parameter `{arch, path, max_sequence_length}`, and no MLIR is compiled. The
session created while matching is handed to the custom op through a registry
(`lib/Runtime/corelib_model_registry.cpp`) keyed by the fused node; a model
loaded another way (an EPContext reload) creates its own. On any refusal (no
CoreLib, no usable AIE4 NPU, CoreLib rejecting the file, an in-memory model with
no file path) the pass logs one `[corelib] model: ...; compiling it for the GPU`
line and compiles the graph as usual.

## Running

`CorelibModelCustomOp` (`backend-mlir-compiler/custom-op-mlir/src`) runs each
call:

- `input_ids` (batch 1) at position `sum(attention_mask) - rows`, the position
  CoreLib's session seeks to, so a new prompt starts over without a reset. A
  mask holding fewer positions than rows (`onnxruntime_perf_test`'s generated
  zeros) runs from position 0, warned once.
- `logits` [1, rows, vocab]: the last row from CoreLib, the others zeros, as
  generation reads only the last.
- `present.*` left unwritten, since CoreLib keeps the cache; with a shared
  past/present buffer it takes the past's shape, else it spans the positions
  so far.

So the contract is onnxruntime-genai generation, not full ONNX semantics: a
caller reading every row's logits or the KV tensors gets zeros. The session's
capacity is `search.max_length` from the `genai_config.json` beside the model,
else CoreLib's own. The runtime side is `lib/Runtime/corelib_model.h`
(`real/ryzenaicorelib_model.cpp`, `mock/ryzenaicorelib_model_stub.cpp`).

## Supported models

| CoreLib model | Export it reads |
|---|---|
| `qwen3` | onnxruntime-genai `model.onnx` of Qwen3 0.6B, 1.7B, 4B or 8B: MatMulNBits block 128 throughout, `lm_head` included; FP16 embedding and norms |

CoreLib's `docs/models.md` lists what each of its models reads. Others that take
ONNX (`phi4mini`, `gemma4`, `qwen36`) can be added to `kCorelibModels` once
their input names match this custom op's contract.

## Measured

AIE4 machine (gfx1170 GPU), CoreLib 0.66, Qwen3-4B-Instruct-2507's
onnxruntime-genai export, onnxruntime-genai's `model_benchmark` (128-token
prompt, 128 tokens generated, greedy):

| | CoreLib model (NPU) | GPU (hip-ep) |
|---|---|---|
| Prompt processing | 1515 tokens/s | 1020 tokens/s |
| Token generation | 29.6 tokens/s (stddev 0.5 ms) | 38.6 tokens/s (stddev 1.4 ms) |
| Session creation (`onnxruntime_perf_test`) | 1.0 s | 5.9 s |

Both generated the same text.

## Limitations

- Batch size 1; only the last row of `logits` is computed; `present.*` is not
  written.
- One CoreLib model so far (`qwen3`), and only exports at block size 128.
- AIE4 only; elsewhere the model compiles for the GPU.
