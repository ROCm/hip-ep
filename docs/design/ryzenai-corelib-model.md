<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Whole Models on the NPU through RyzenAI CoreLib

When an onnxruntime-genai decoder is a model RyzenAI CoreLib (`ryzenai-corelib`)
knows as a whole, hip-ep can hand it the entire graph instead of compiling it.
CoreLib's model API (`include/ryzenai/corelib/model.h`) reads the ONNX file's
initializers itself, keeps the KV cache, and runs every layer, attention
included, on the NPU. Any other model compiles for the GPU as before.

hip-ep names no model: it offers the file to each model CoreLib ships and
passes whichever graph inputs the accepting model's `forward` asks for. A
CoreLib release with new models, or with existing ones reading more formats,
therefore needs no hip-ep change (within the version rule under "Building and
deploying").

## Enabling

| Setting | Where | Effect |
|---|---|---|
| `use_ryzenai_corelib_model` = `1`/`true` | EP provider option | Try the model-level path |
| `HIPEP_USE_RYZENAI_CORELIB_MODEL` = `1`/`true` | Environment, read at compile time | Same, when the provider option is unset |
| `RYZENAICORELIB_ROOT` | Environment, build and run time | CoreLib install tree (or repository root; `install/` is searched). Build: locates `include/ryzenai/corelib/base.h` (CoreLib 0.90.1 or later). Run: loads `bin/ryzenai_corelib.dll` (`lib/libryzenai_corelib.so` on Linux), else the default library search path |
| `RYZENAICORELIB_REPORT` = `1` | Environment, run time | At session end, print `[corelib] model report: ...` with the prefills, decodes and rows run on the NPU |
| `RYZENAICORELIB_MODELS` | Environment, compile time | Comma-separated CoreLib model names to try instead of the model libraries found beside CoreLib, for a CoreLib whose models are built in |

In an onnxruntime-genai `genai_config.json` the option goes in the EP's
`provider_options`, for example `{"AMDGPU": {"profile": "hip",
"use_ryzenai_corelib_model": "true"}}`.

## Hardware

hip-ep uses CoreLib's models where `ryzenai_corelib_npu_usable()` is true. With
CoreLib 0.66 and 0.90.1 that is an AIE4 (AIEM) NPU with XRT 2.26.0 or later;
it is false on AIE2/AIE2P parts (Phoenix, Strix, Krackan), where hip-ep
compiles the model for the GPU. A CoreLib that runs its models on more NPUs
needs no hip-ep change.

## Building and deploying

`RYZENAICORELIB_ROOT` names a CoreLib 0.90.1 or later install tree at build
time; only its headers are used. The configure step prints
`-- RyzenAI CoreLib <version>: <include dir>`; without CoreLib (or in mock
builds) it builds a stub that refuses every model, and warns when
`RYZENAICORELIB_ROOT` is set but holds no usable CoreLib.

At run time the EP loads `ryzenai_corelib.dll` itself on first use, so it still
loads on machines without CoreLib. It follows CoreLib's own compatibility rule
(`base.h`): below 1.0 the API may change in any release, so the EP takes only
the minor version it was built against; from 1.0 the C ABI only grows within a
major version, so it takes that minor or any later one. Otherwise it logs
`[corelib] <dll> is version ..., but hip-ep was built against ...` and compiles
for the GPU. Until CoreLib 1.0, a new CoreLib therefore needs hip-ep rebuilt
against its headers, but no source change.

Each CoreLib model library (`ryzenai_corelib_model_<name>.dll`) must sit beside
`ryzenai_corelib.dll`; those libraries are the models hip-ep tries. A model's
kernels are linked into its library, so no kernel store is needed. XRT
normally comes from the NPU driver.

### Matching the CoreLib version on the target machine

Because `hipgpu.dll` and `ryzenai_corelib.dll` must be built against the same
CoreLib headers (see version compatibility note in the reviewer notes), find
the CoreLib version already deployed on the target machine before building:

```bat
ryzenai_corelib.dll --version        rem if the DLL exposes a CLI
```

or inspect the DLL directly:

```bat
powershell -command "(Get-Item ryzenai_corelib.dll).VersionInfo.ProductVersion"
```

Then check out the matching tag in the `ryzenai-corelib` repo, set
`RYZENAICORELIB_ROOT` to its install tree, and build.

### Rebuilding hipgpu.dll after a CoreLib version change

A CoreLib version bump requires only `hipgpu.dll` to be rebuilt — not a full
hip-ep build. The `hipdnn-ep-corelib` static library embeds the version at
compile time; touching its source forces the recompile:

```bat
rem 1. Save the previous hipgpu.dll under a versioned name
copy "install-corelib-gfx1170\bin\hipgpu.dll" ^
     "install-corelib-gfx1170\bin\hipgpu-corelib-0.90.1.dll"

rem 2. Force recompile of ryzenaicorelib_model.cpp against the new headers
copy /b hip-ep\lib\Runtime\real\ryzenaicorelib_model.cpp +,,

rem 3. Find the hipgpu.dll CMake target name (changes with git commit hash)
cmake --build build\hip-ep-corelib-gfx1170 --config Release ^
      --target help 2>&1 | findstr MORPHIZEN_EP_TARGET

rem 4. Rebuild only hipgpu.dll
cmake --build build\hip-ep-corelib-gfx1170 --config Release --parallel 8 ^
      --target morphizen-core-dynamic-<hash from step 3>

rem 5. Install
cmake --install build\hip-ep-corelib-gfx1170 --config Release
```

If `RYZENAICORELIB_ROOT` itself changed (the new version is in a different
directory), reconfigure before step 4:

```bat
cmake -S hip-ep -B build\hip-ep-corelib-gfx1170 ^
      -DRYZENAICORELIB_ROOT=<new-path>
```

## How a model is matched

The level-1 pass (`backend-mlir-compiler/level-1-pass/src/pass_main.cpp`) takes
the path before building any MLIR when:

- the option is on;
- the graph is an onnxruntime-genai decoder: inputs `attention_mask` and
  `past_key_values.*`; outputs `logits`, with a static vocabulary width, and
  `present.*`;
- a CoreLib model takes the model file and fits the graph. The pass tries each
  model library beside CoreLib: first the one named like `model.type` in the
  `genai_config.json` beside the model (e.g. `qwen3`), then the rest in name
  order, with
  `ryzenai_corelib_create_model_from_path`, which reads the file's
  initializers and refuses what it does not recognize. A model that takes the
  file must also fit: its vocabulary is the width of the graph's `logits`
  (CoreLib's test model reads any file, so this is what turns it away), and
  every input its `forward` requires (`ryzenai_corelib_model_get_method`) is a
  graph input, but for `position`, which the op derives. The first that fits
  is used. `model.type` is only an ordering hint: every CoreLib 0.66 and
  0.90.1 model refused the Qwen3 and Llama exports it was not exported for,
  cleanly and with a reason, so the hint changes no outcome; it keeps the
  matching model ahead of the others, which saves loading attempts.

On success the whole graph is fused under the `CORELIBMODEL` domain, with the
parameter `{arch, path, max_sequence_length}`, and no MLIR is compiled. The
session created while matching is handed to the custom op through a registry
(`lib/Runtime/corelib_model_registry.cpp`) keyed by the fused node; a model
loaded another way (an EPContext reload) creates its own, of the recorded
model. On any refusal (no CoreLib, no usable NPU, no model taking the file, an
in-memory model with no file path) the pass logs one
`[corelib] model: ...; compiling it for the GPU` line, with each model's
reason, and compiles the graph as usual.

## Running

`CorelibModelCustomOp` (`backend-mlir-compiler/custom-op-mlir/src`) runs each
call:

- the graph inputs the model's `forward` takes, by name: CoreLib's names are
  onnxruntime-genai's (`input_ids`, or `inputs_embeds` for an export whose
  embedding runs separately, `per_layer_inputs`, ...). Batch 1; each is passed
  without its batch dimension, its sequence dimension being the rows run, and
  converted where the model does not take its type (int64 to int32, fp16 to
  fp32).
- `position` = `sum(attention_mask) - rows`, the position CoreLib's session
  seeks to, so a new prompt starts over without a reset. A mask holding fewer
  positions than rows (`onnxruntime_perf_test`'s generated zeros) runs from
  position 0, warned once.
- `logits` [1, rows, vocab]: the last row from CoreLib, the others zeros, as
  generation reads only the last.
- `present.*` left unwritten, since CoreLib keeps the cache; with a shared
  past/present buffer it takes the past's shape, else it spans the positions
  so far.

The EP's default device memory is device-only VRAM, so the inputs and
`attention_mask` are read back, and the logits written, with `hipMemcpy` when
they are in GPU memory; CoreLib itself takes host buffers.

So the contract is onnxruntime-genai generation, not full ONNX semantics: a
caller reading every row's logits or the KV tensors gets zeros. The session's
capacity is `search.max_length` from that `genai_config.json`,
else CoreLib's own. When the model holds fewer positions than `max_length`
asks for (configs often ask for the base model's whole context), the
session holds the model's own instead, with a `[corelib] model: holding the
model's own context ...` line; a generation past it fails on that forward. The runtime side is `lib/Runtime/corelib_model.h`
(`real/ryzenaicorelib_model.cpp`, `mock/ryzenaicorelib_model_stub.cpp`).

### onnxruntime-genai and device-only memory

The EP's default allocator returns device-only `hipMalloc` memory. An
onnxruntime-genai build whose AMDGPU device treats that memory as host memory
(older builds `memset` the KV cache directly when creating a generator) faults
with an access violation in `OgaCreateGenerator`. The GPU-only path can get
past it only because the KV cache then happens to land in memory the CPU can
reach; with the CoreLib model the EP has allocated nothing else, so GenAI's
first buffer is unmapped GPU address space. Use the onnxruntime-genai that
[quick_start.md](../quick_start.md) builds (upstream v0.19.0; any release from
v0.16.0 has the AMDGPU device that copies through ONNX Runtime's
`CopyTensors`), or set `MORPHIZEN_DEFAULT_HIPMALLOC=0`, which makes the default
allocator return host-mapped `hipHostMalloc` memory (slower for GPU-only runs).

With that onnxruntime-genai the `AMDGPU` provider loads the AMD GPU umbrella EP
(`amdgpu-ep.dll` with `"profile": "hip"`), which loads hip-ep. If the umbrella
does not pass `use_ryzenai_corelib_model` through to hip-ep, set
`HIPEP_USE_RYZENAI_CORELIB_MODEL=1` instead.

## Supported models

Whatever CoreLib's models read from ONNX; its `docs/models.md` lists each
one's files. With CoreLib 0.66 those were `qwen3` (onnxruntime-genai
`model.onnx` of Qwen3 0.6B to 8B, MatMulNBits block 128 throughout),
`phi4mini` (`model.onnx`), and `gemma4` and `qwen36`, whose exports run the
embedding as a separate model, so the decoder hip-ep sees takes
`inputs_embeds`. CoreLib 0.90.1 ships a larger set of models; consult its
`docs/models.md` for the current list. Models that read GGUF only do not apply
regardless of version.

| Model | Run through hip-ep |
|---|---|
| Qwen3-4B-Instruct-2507 (`qwen3`) | yes, measured below |
| Phi-4-mini-instruct (`phi4mini`, `amd/phi-4-mini-instruct-oga-dml`) | yes, measured below; CoreLib holds 4096 positions from ONNX |
| Gemma 4, Qwen3.6 | not yet tried |

## Measured

AIE4 machine (gfx1170 GPU), CoreLib 0.90.1 (originally measured with 0.66; results were the same); ONNX Runtime 1.31, onnxruntime-genai 0.19 and the AMD
GPU umbrella EP as in [quick_start.md](../quick_start.md); `model_benchmark -l
128 -g 128 -r 3 -w 1` (128-token prompt, 128 tokens generated, 3 iterations
after 1 warmup):

| | Qwen3-4B, CoreLib model (NPU) | Qwen3-4B, GPU (hip-ep) | Phi-4-mini, CoreLib model (NPU) | Phi-4-mini, GPU (hip-ep) |
|---|---|---|---|---|
| Prompt processing | 1815 tokens/s | 327 tokens/s | 1460 tokens/s | 460 tokens/s |
| Token generation | 34.7 tokens/s | 40.4 tokens/s | 33.4 tokens/s | 44.3 tokens/s |
| Model creation | 1.1 s | 8.7 s | 1.0 s | 8.5 s |
| Dedicated GPU memory | 280 MB | 3.4 GB | 291 MB | 3.7 GB |

Each model's two columns were measured back to back. The GPU's Qwen3 prompt
figure varied widely within the run (stddev 32 ms on a 390 ms prompt; another
day measured 914 tokens/s), the NPU's held within 0.2 ms.

Qwen3 is Qwen3-4B-Instruct-2507's onnxruntime-genai export; Phi-4-mini is
`amd/phi-4-mini-instruct-oga-dml` (revision `e751fb68`, the export CoreLib's
`docs/phi4mini.md` names).

For each model the CoreLib session's report counted 639 forwards on the NPU: 4 prefills
(warmup and iterations) and 635 single-row calls, including the ones
`model_benchmark` makes to generate its prompt.

## Limitations

- Batch size 1; only the last row of `logits` is computed; `present.*` is not
  written.
- Only models CoreLib reads from ONNX, with the inputs and outputs named as
  onnxruntime-genai names them; methods other than `forward` (images through
  `encode_image`, drafting) are not used.
- Only where CoreLib's models run (AIE4 with CoreLib 0.90.1 or later);
  elsewhere the model compiles for the GPU.

## Quick reference: CoreLib API surface used by hip-ep

All symbols are loaded at runtime by name from `ryzenai_corelib.dll`
(`lib/Runtime/real/ryzenaicorelib_model.cpp`); none are linked at build time.

### Functions

| Function | Signature | Purpose |
|---|---|---|
| `ryzenai_corelib_get_version` | `(uint32_t* major, uint32_t* minor, uint32_t* patch) → void` | Read DLL version; checked before any other call |
| `ryzenai_corelib_cleanup` | `() → void` | Release all CoreLib resources; called when the last session is destroyed |
| `ryzenai_corelib_has_feature` | `(ryzenai_corelib_feature) → bool` | Check capability; hip-ep tests `ryzenai_corelib_feature_npu_aiem` |
| `ryzenai_corelib_get_last_error_message` | `() → const char*` | Human-readable error from the last failed call |
| `ryzenai_corelib_object_release` | `(ryzenai_corelib_object_ptr) → void` | Release a model or session handle |
| `ryzenai_corelib_create_model_from_path` | `(const char* name, const char* path, ryzenai_corelib_model_ptr* out) → ryzenai_corelib_status` | Load a named model library and offer it the ONNX file |
| `ryzenai_corelib_model_create_session` | `(ryzenai_corelib_model_ptr, const ryzenai_corelib_model_session_options*, ryzenai_corelib_model_session_ptr* out) → ryzenai_corelib_status` | Create an inference session |
| `ryzenai_corelib_model_get_metadata` | `(ryzenai_corelib_model_session_ptr, ryzenai_corelib_model_metadata*) → ryzenai_corelib_status` | Read vocab size, layer count, hidden size, context length |
| `ryzenai_corelib_model_get_method` | `(ryzenai_corelib_model_session_ptr, size_t index, ryzenai_corelib_model_method_info*) → ryzenai_corelib_status` | Read forward method's input names, types and flags |
| `ryzenai_corelib_model_run` | `(ryzenai_corelib_model_session_ptr, const char* method, ryzenai_corelib_model_value* inputs, size_t n_in, ryzenai_corelib_model_value* outputs, size_t n_out, uint32_t flags) → ryzenai_corelib_status` | Run one forward pass |

### Types

| Type | Kind | Fields used by hip-ep |
|---|---|---|
| `ryzenai_corelib_model_ptr` | Opaque handle | Passed to `create_session`, `object_release` |
| `ryzenai_corelib_model_session_ptr` | Opaque handle | Passed to `get_metadata`, `get_method`, `model_run`, `object_release` |
| `ryzenai_corelib_object_ptr` | Opaque base handle | Passed to `object_release` |
| `ryzenai_corelib_model_session_options` | Struct | `size`, `max_sequence_length`, `enabled_backends` |
| `ryzenai_corelib_model_metadata` | Struct | `size`, `vocab_size`, `layer_count`, `hidden_size`, `context_length` |
| `ryzenai_corelib_model_method_info` | Struct | `input_count`, `inputs[]` (array of `ryzenai_corelib_model_value_info`) |
| `ryzenai_corelib_model_value_info` | Struct | `name`, `data_type`, `accepted_types`, `flags`, `dims` |
| `ryzenai_corelib_model_value` | Struct | `size`, `name`, `data_type`, `rank`, `dims`, `data`, `capacity` |
| `ryzenai_corelib_status` | Enum | hip-ep checks for `ryzenai_corelib_status_success` |
| `ryzenai_corelib_data_type` | Enum | `int32`, `int64`, `fp16`, `fp32` variants used |
| `ryzenai_corelib_feature` | Enum | hip-ep tests `ryzenai_corelib_feature_npu_aiem` |

### Constants and macros

| Symbol | Source | Use |
|---|---|---|
| `RYZENAI_CORELIB_VERSION_MAJOR/MINOR/PATCH` | `base.h` | Compile-time version; compared against `get_version` at runtime |
| `RYZENAI_CORELIB_MODEL_MAX_DIMS` | `model.h` | Array size for `dims` in value structs |
| `RYZENAI_CORELIB_MODEL_VALUE_OPTIONAL` | `model.h` | Flag on `value_info.flags`; optional inputs may be absent from the graph |
| `RYZENAI_CORELIB_DATA_TYPE_BIT(type)` | `model.h` | Bitmask helper; tests whether a type is in `value_info.accepted_types` |
| `ryzenai_corelib_backend_npu` | `base.h` | Backend flag; set in `session_options.enabled_backends` |
| `ryzenai_corelib_backend_cpu` | `base.h` | Backend flag; set alongside NPU (token pick and gathers run on CPU) |
