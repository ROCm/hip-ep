<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# llama.cpp's ggml graphs: compiled splits

llama.cpp stays the host: model loading, KV cache, batching, sampling and the
server are its own. Its ROCm backend, ggml-hip, built with `GGML_HIP_HIPEP`,
cuts each split the ggml scheduler hands it into runs: what hip-ep compiles
runs as code hip-ep compiled for it, the rest - and a run whose code is not
ready or fails - by ggml-hip's own kernels. The device is still ROCm0.
Everything of hip-ep ggml-hip needs is one library, `libhipep_backend.so`,
behind one C header, `hipep_backend.h`.

```
llama.cpp ── ggml ── ggml-hip (llama.cpp's tree: ggml/src/ggml-cuda/hipep)
                          │  hipep_backend.h: descriptions in, code out
                          ▼
                  libhipep_backend.so (lib/HipepBackend)
                    split frontend + hip-ep's compiler (ggml-to-hip pipeline)
                    + the EP's JIT loader
                          │
                  HIP, hipBLASLt; the kernel libraries beside it
```

## Building

hip-ep, with ggml graphs (off by default):

```bash
python3 build.py --cmake_extra_defines HIPDNN_EP_BUILD_GGML=ON
```

| Option | Default | |
|---|---|---|
| `HIPDNN_EP_BUILD_GGML` | OFF | the `ggml` dialect, `convert-ggml-to-hip` and the `ggml-to-hip-pipeline`, the split frontend and `libhipep_backend` |

The install has `lib/libhipep_backend.so` (it exports `hipep_backend_*`
alone and depends only on HIP), `include/hipep_backend.h`, the CMake package
`lib/cmake/hipep_backend`, and beside it the kernel library the compiled
code links against, `libcustom_kernels_<arch>.so`.

hip-ep needs neither llama.cpp nor ggml to build: a split arrives as plain C
(`hipep_backend.h`), and the few facts of ggml's the frontend needs - its
type numbers, which GGUF files store, the bytes a block of each type takes,
and where each operation keeps its parameters (`GgmlOps.td`) - are written
down in hip-ep (`GgmlTensor.h`, `GgmlBlockLayout.h`). Parameters are read as
ggml 1.0 lays them out, and the library refuses a description of another ggml
minor version.

llama.cpp, from the AMD-Ecosystem fork's `ggml-hipep` branch, finds the
library with `find_package(hipep_backend)` and includes `hipep_backend.h`
alone:

```bash
cmake -B build -DGGML_HIP=ON -DGGML_HIP_HIPEP=ON \
  -Dhipep_backend_DIR=<hip-ep-install>/lib/cmake/hipep_backend
llama-cli -m model.gguf -ngl 99 -fa on
```

Nothing else changes for llama.cpp's tools: the device is ROCm0, and every
option means what it meant. `GGML_HIP_HIPEP=0` in the environment runs every
split by ggml-hip alone.

## What it takes

This first version takes what a Llama-style dense decoder needs with
`-fa on` (it is checked on Llama 3.2 3B Q4_K_S): weights Q4_K, Q5_K or Q6_K,
F16 KV caches, and the operations `lib/HipepBackend/Ops.cpp` lists - `ADD`,
`MUL`, `RMS_NORM`, `MUL_MAT`, `ROPE`, `GLU` (SwiGLU), `GET_ROWS` (F32),
`SET_ROWS` (F32 rows into an F16 cache), `FLASH_ATTN_EXT` (masked, head
widths 40 to 128 and 256, on llama.cpp's tile kernel) and the views
(`VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`). Several sequences are taken
over one KV cache (`-kvu`); a cache of a stream per sequence is refused.
Whatever it does not compile, ggml-hip runs
(`HIPEP_BACKEND_OPS` narrows what it compiles, as a model it takes in part).

## The library

| call | |
|---|---|
| `hipep_backend_abi_version()` | the header's `HIPEP_BACKEND_ABI_VERSION`, or the backend registers no device |
| `hipep_backend_open(device, cache_dir)` | a context per HIP device; the cache: `cache_dir`, else `HIPEP_BACKEND_CACHE`, else `~/.cache/hipep` |
| `hipep_backend_supports(ctx, node)` | whether it compiles an operation, its sources and what it views |
| `hipep_backend_find(ctx, split)` | code compiled already, here or in the cache, for the split's sizes or with dynamic shapes that fit it; null otherwise |
| `hipep_backend_compile(ctx, split, wait)` | the split compiled on this thread (`wait`) or in the background; a split refused is not tried again |
| `hipep_backend_run(code, data, NULL)` | the code on the split's tensors' memory, `data[i]` tensor `i`'s |
| `hipep_backend_close(ctx)` | waits for a compile under way |
| `hipep_backend_last_error(ctx)` | why the last call on this thread returned nothing |

A split is described in plain C (`hipep_backend_split_desc`, an array of
`hipep_backend_tensor_desc`): each tensor's operation by ggml's name, its
name, type, extents, strides, sources, view and `op_params`, and what ggml's
tensor does not say of itself - its role (a weight, a cache, an input),
whether the split computes it, whether it is read after the split
(`HIPEP_BACKEND_ABI_VERSION` 5). The library rebuilds the description as ggml tensors of its own
(`lib/HipepBackend/Graph.cpp`, metadata only). A description is the
caller's, valid during the call: a compile in the background works on a copy.

The split's tensors are numbered sources first - for each node in order, the
tensors it reads not described yet (a view of a tensor outside the split with
what it views, depth first), then the node -, and the library checks it;
`data` is indexed the same.

## A split in the library

1. **Find**: code compiled with static shapes for its signature (the
   description without the names: equal splits are one) in this process or
   the cache (`<signature>/`), else code compiled with dynamic shapes for its
   *dynamic signature* - the signature without extents, strides and view
   offsets but for the weights' and the written caches' - in
   `d<dynamic signature>-<n>/`, its `n`th generation, when its shapes
   (`shapes.json`) fit and it hands out every node the split does. A dynamic
   signature keeps a code for each assignment of symbols it was compiled
   with: a split no code fits is compiled with the symbols derived at its
   sizes, where a size of one is no symbol (a decode step's code is for one
   token). A split that could be compiled neither with dynamic shapes nor
   with its own (`d<dynamic signature>.failed`) refuses its structure: a
   split of it at another size runs on ggml-hip without a compile. A view
   of storage the split does not compute (a cache's window another backend
   attends over) is not handed out: ggml points it at that storage. A split
   whose last nodes have no
   elements (a chunk of a prompt that returns no logits) runs the code of one
   row: its padded inputs zeros, its padded outputs written aside.
2. **Compile**: the split captured as a whole graph (every leaf a runtime
   input, `lib/Frontend/Ggml/SplitCompiler.cpp`), emitted as a `ggml`
   module, its extents derived as symbols (`ggml-symbolize-shapes`), and
   compiled by hip-ep's compiler in this process: a module of the `ggml`
   dialect (`ggml.schema`) takes the `ggml-to-hip-pipeline`
   (`convert-ggml-to-hip`, to `hip` IR, then the tail ONNX shares), to LLVM
   bitcode
   (`model.bc`, and `constants.bin` when the code has constants to keep
   apart; the weights are inputs). `bindings.json` says
   which described tensor each of the code's inputs and outputs is, in order,
   and which outputs may be written into an input's memory (a cache updated
   in place, which `ProveStateDonation.cpp` proves from the graph).
3. **Run**: the bitcode is JIT-loaded as the execution provider loads its
   models (`LlvmIrJit`: the runtime's bitcode linked in, the kernel library
   found beside `libhipep_backend.so`), and run
   through the artifact ABI (`inference_init`, `inference_compute`, the output
   allocator) on the split's tensors where ggml put them - inputs read in
   place, outputs written to their tensors' memory (first to a buffer of the
   library's when ggml put an output over one of the inputs). A cache updated
   in place is handed its memory, and only its new rows are written. A packed
   weight is a `u8` input of its blocks, which the K-quant kernels read
   where llama.cpp loaded them.

Which of a split's sizes vary the split alone says (`SplitCompiler.h`). Each
extent of what it reads, and each length of a view of what it reads that runs
along an extent of it, is a candidate - a guess of its own; a candidate of one
is one (a broadcast; a decode step's code is for one token).
`ggml-symbolize-shapes` derives the split from the guesses in its order, as
it derives every compile, and learns as it goes: what each operation
requires of its operands (`constrainOperands`: a product's shared dimension,
the keys as many as the mask is wide and its rows a query each, a position
per token, a written row per index, elementwise operands alike) and a
reshape's elements make two guesses one size, or a guess the number the
other side leaves it (a weight's, a cache's, one an operation makes: `24 *
h = 3072`). What cannot be derived is fixed at what the capture saw. Each
size left that an input carries is a symbol, bounded by the room a view of a
cache has; the pass derives the rest from them, and the guard checks every
extent before code runs another split. No name, and nothing outside the
split, means anything to it. For Llama that is
two symbols - the step's tokens and the KV cells it attends -, so prefill of
any length is one dynamic-shape code and decode another (llama.cpp's reserved
graphs compile both while the context is made). The llama.cpp side is the
`ggml-hipep` branch: ggml-hip's use of the library
(`ggml/src/ggml-cuda/hipep`), the scheduler marking each split's outputs
(`GGML_TENSOR_FLAG_SPLIT_OUTPUT`) for the backends that ask for them, and the
reserved graphs' splits compiled while the context is made.

## Layout

Built with `HIPDNN_EP_BUILD_GGML`, where hip-ep keeps the like of it:

| | |
|---|---|
| `include/hip/Dialect/Ggml`, `lib/Dialect/Ggml`, `tools/ggml-tblgen` | the `ggml` dialect and its transforms (shape symbolization, the state donation proof) |
| `include/hip/Conversion/GgmlToHip`, `lib/Conversion/GgmlToHip` | `convert-ggml-to-hip`; the `ggml-to-hip-pipeline` is in `lib/Dialect/Transforms/Pipelines.cpp` |
| `include/hip/Frontend/Ggml`, `lib/Frontend/Ggml` | the split frontend: a split's tensors (`GgmlTensor.h`) captured and emitted, compiled in-process |
| `lib/HipepBackend` | `libhipep_backend` |
| `test/lit/{Dialect/Ggml,Conversion/ggml-to-hip,Pipeline/ggml}`, `test/lib/Dialect/Ggml`, `test/unit/hipep_backend` | its tests |

## The hip operations it lowers to

`convert-ggml-to-hip` emits `hip` IR only. Besides hip-ep's element-wise,
`rms_norm`, `rope`, `gather`, `cast`, `swiglu` and `scatter_nd` (a cache's
new rows), the ggml splits use these, defined with the rest of the dialect
(`include/hip/Dialect/IR/HipOps.td`, `HipKQuantOps.cpp`) and lowered to the
runtime (`lib/Runtime/real/kquant.cpp`, `token_major_gqa.cpp`) and its kernels
(`lib/Runtime/Kernels/hip/kquant`, `token_major_gqa`):

| op | |
|---|---|
| `hip.kquant_matmul` | an F32 activation times packed Q4_K/Q5_K/Q6_K weights read in place |
| `hip.quantize_q8_1` | an activation in Q8_1 blocks, for several projections to share |
| `hip.skip_rms_norm_q8_1` | `hip.skip_rms_norm` whose normalized value only projections read, in Q8_1 |
| `hip.kquant_matmul_q8_1` | a projection of a Q8_1 activation (MMVQ, or llama.cpp's MMQ tiles) |
| `hip.kquant_swiglu_q8_1` | the gate and up projections of a Q8_1 activation and their SwiGLU |
| `hip.token_major_gqa` | grouped-query attention over a token-major F16 KV cache under an additive mask (llama.cpp's tile kernel) |

## Checking it

| test | |
|---|---|
| `MorphizenMLIRLitTests` (lit feature `hip_ggml`) | the `ggml` dialect, `convert-ggml-to-hip` and the `ggml-to-hip-pipeline` |
| `HipepBackendUnitTest` | `hipep-backend-test`: descriptions of CPU-built graphs, the sizes their operations make one, dynamic signatures and fits |
| `ctest -L custom-kernels-gpu` (`test/runtime/custom_kernels`) | the K-quant, Q8_1 and token-major GQA kernels on a gfx1151 GPU (built with hip-ep, without `HIPDNN_EP_BUILD_GGML`) |

End to end, against ggml-hip in the same llama.cpp build: the same greedy
text, and `llama-perplexity --kl-divergence` (`-b` equal to `-c`: one stream)
with every split run by compiled code.

| Environment | |
|---|---|
| `HIPEP_BACKEND_LOG=1` | each split (compiled or ggml-hip, and why), each compile, and a summary |
| `GGML_HIP_HIPEP=0` | ggml-hip runs every split alone |
| `HIPEP_BACKEND_WAIT=1` | compiles on the calling thread (ggml-hip) |
| `HIPEP_BACKEND_OPS=MUL_MAT,ADD` | only these operations (bisecting) |
| `HIPEP_BACKEND_PROFILE=1` | where the time went (ggml-hip) |
| `HIPEP_BACKEND_COMPILE=0` | code from the cache only |
