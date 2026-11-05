<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# RyzenAI CoreLib (NPU) Backend

hip-ep can run selected operators on the AMD NPU through RyzenAI CoreLib
(`ryzenai-corelib`, C API under `include/ryzenai/corelib/`; hip-ep is built
against CoreLib 0.66). Every other
operator stays on the GPU, and each NPU-routed call falls back to the GPU at
runtime when CoreLib cannot run it.

## Hardware

CoreLib's operator API runs prebuilt kernels on an AIE4 (AIEM) NPU only.
`ryzenai_corelib_npu_usable()` is false on AIE2/AIE2P parts (Phoenix, Strix,
Krackan), where CoreLib runs only raw kernels through `stream_dispatch`, and on
an AIE4 part whose XRT is older than 2.26.0. hip-ep then logs
`[corelib] no usable AIE4 NPU (...); corelib ops run on the GPU` once per
session and every routed op runs on the GPU.

The backend runs inside the GPU execution provider, so the machine also needs
a working HIP GPU: CoreLib staging uses the session's HIP stream, and every
unsupported op runs on the GPU.

## Enabling

| Setting | Where | Effect |
|---|---|---|
| `use_ryzenai_corelib` = `1`/`true` | EP provider option | Compile with the CoreLib lowering |
| `HIPEP_USE_RYZENAI_CORELIB` = `1`/`true` | Environment, read at compile time | Same, when the provider option is unset |
| `RYZENAICORELIB_ROOT` | Environment, build and run time | CoreLib install tree (or repo root; `install/` is searched). Build: locates `include/ryzenai/corelib/base.h` (CoreLib 0.66 or later). Run: loads `bin/ryzenai_corelib.dll` (`lib/libryzenai_corelib.so` on Linux), else the default library search path |
| `RYZENAICORELIB_PDI` = `<prefill PDIs>;<decode PDIs>` | Environment, run time | The stream's PDIs, each phase a comma-separated priority list (for example `9;19,35`); must match the kernels available (see below). Default `1;16` |
| `RYZENAICORELIB_KERNEL_BUNDLE` = `<model>` | Environment, run time | Enables the kernel bundle linked into that CoreLib model library (e.g. `qwen3` from `ryzenai_corelib_model_qwen3.dll`) |
| `RYZENAI_CORELIB_KERNELS_ROOT` | Environment, run time (read by CoreLib) | NPU kernel store. With neither a store nor a bundle every call goes to the GPU |
| `RYZENAI_CORELIB_DISABLE_GPU` = `1` | Environment, run time (read by CoreLib) | Keeps CoreLib from loading HIP for its own GPU tier. hip-ep never lets CoreLib dispatch to its GPU or CPU paths; this also avoids the HIP load |
| `HIPDNN_EP_STRICT` = `1` | Environment, run time | A routed call the NPU refuses at a shape a kernel was listed for fails inference with `HIPDNN_EP_ERR_CORELIB_OFF_NPU` instead of running on the GPU. Shapes with no listed kernel still run on the GPU |

Both artifact formats are supported: `LLVM_IR` resolves the wrappers from the
host process, and `NATIVE` model DLLs link `hipdnn-ep-corelib`. An EPContext
artifact keeps the setting it was compiled with.

## Building, verifying and deploying

`RYZENAICORELIB_ROOT` names a CoreLib install tree, for example
`<ryzenai-corelib>/install` with `include/`, `bin/` and `lib/`; the repository
root also works. The build needs only the header. `ryzenai_corelib.lib` is
not linked: the EP loads `ryzenai_corelib.dll` itself on the first CoreLib
call, so it still loads on machines without CoreLib. Build CoreLib for hip-ep
with `python build.py --install-dir <prefix> --gpu-targets none`: without its
GPU kernels CoreLib never loads HIP, so it cannot contend with the EP's own HIP
runtime. XRT normally comes from
the NPU driver; if CoreLib cannot find it, add
`<install>/xrt_package/xrt` to `PATH`.

Whether a build contains the real wrappers or the stub:

| Check | Real wrappers | Stub |
|---|---|---|
| CMake configure output | `-- RyzenAI CoreLib <version>: <include dir>` | `-- RyzenAI CoreLib not found (set RYZENAICORELIB_ROOT); ...` |
| `RyzenAICorelib_INCLUDE_DIR` in `CMakeCache.txt` | the include path | `...-NOTFOUND` |
| `findstr /m /c:"ryzenai_corelib_create_stream" hipgpu.dll` | prints the file name | no output |
| stderr with `HIPEP_USE_RYZENAI_CORELIB=1` | at least one `[corelib]` line | nothing |

A found include path is cached; delete the cache entry or reconfigure when
CoreLib moves.

CoreLib is below 1.0, so the EP refuses a library whose major or minor
version differs from the header it was built against, and logs
`[corelib] <dll> is version ..., but hip-ep was built against ...`; everything
then runs on the GPU. Rebuild hip-ep after updating CoreLib, and point
`RYZENAICORELIB_ROOT` at the install it was built against.

The EP can be built on one machine and run on another (for example built on
an AIE2 machine, run on an AIE4 machine). Nothing in the build is specific to
the NPU: the PDIs, kernels and CoreLib library are all chosen at run time.
The target must match in:

- GPU architecture: build with `--hip_arch` (or `HIP_ARCH`) set to the
  target's GPU.
- CoreLib version: build against a header whose major/minor version matches
  the target's `ryzenai_corelib.dll`; copying the target's `install/include`
  is enough.
- Runtime libraries: HIP, hipBLASLt and MIOpen DLLs on the target from the
  same TheRock distribution the build used.

Deploy `hipgpu.dll` and `custom_kernels_<gfx>.dll` from the same build, in
the same folder: the EP's compiler emits the `wrap_corelib_*` calls, its
embedded runtime defines the matching `RuntimeState` layout, and it loads the
kernel library only from its own folder, never from `PATH`. This holds under
whatever name an application registers the EP DLL.

## Operator coverage

| HIP op | CoreLib call | Runs on CoreLib when |
|---|---|---|
| `hip.matmul_nbits` | `ryzenai_corelib_matmul_bf16` (`mladf_v4` weights) | `bits == 4`, no `g_idx`, `block_size` a multiple of 32 dividing K, K a multiple of 128 (block size divisible by 64) or of 64 (otherwise), zero points absent or packed uint8, and a kernel ships for (K, N, group, bias) |
| `hip.rms_norm` | `ryzenai_corelib_rmsnorm_bf16` | the scale spans exactly the normalized width, and a kernel ships for it |
| `hip.skip_rms_norm` | `ryzenai_corelib_add_rmsnorm_bf16` | no bias input, and an `add_rmsnorm` kernel ships for the width |

Activations may be f32, f16 or bf16; CoreLib computes in BF16. Weights are
packed once per op instance from the model constants. Attention (`hip.gqa`)
stays on the GPU: CoreLib's `flat_mha_bf16` owns its KV cache on the NPU,
which does not match ONNX Runtime GenAI's past/present tensors.

Kernels are named by PDI, and the stream reaches only those on its PDIs: a
shape no listed PDI ships runs on the GPU. A kernel bundle
(`RYZENAICORELIB_KERNEL_BUNDLE`) holds exactly the shapes its CoreLib model
dispatches, so it covers an ONNX export of that same model; a full kernel
store (`RYZENAI_CORELIB_KERNELS_ROOT`) covers more shapes. Set
`RYZENAICORELIB_PDI` to the PDIs of the CoreLib model matching the ONNX model,
as declared in its `models/<Model>.cpp` (`kPrefillPdi`, `kTokenPdi` or
`kTokenPdis`). At CoreLib 0.66:

| CoreLib model | `RYZENAICORELIB_PDI` |
|---|---|
| Qwen3, Llama, Phi4Mini | `1;16` (default) |
| Gemma4 | `8;17,33` |
| LFM2.5 | `4;18` |
| Qwen3.5, Qwen3.6 | `9;19,35` |
| Qwen3.8, Qwen4 | `9;37` |

No shipped PDI carries a biased plain prefill matmul kernel, so a
`MatMulNBits` with a bias runs its prefill on the GPU.

Matmul group size and shipped row counts are read from CoreLib's kernel
enumeration, so they need no configuration. An ONNX int4 export at block size
64 or 128 reaches g64 kernels; block size 32 selects g32 kernels.

## Model coverage

What each model in `test/python` can run on the NPU, with
`RYZENAI_CORELIB_KERNELS_ROOT` set to DynamicDispatch's kernel store
(`transaction/mds`) and no kernel bundle. Found by matching each model's
`MatMulNBits` (K, N, block size) and RMS norm widths against the store's
`mladfmatmulbias`, `mladfrmsnorm` and `addrmsnorm` kernel names at
DynamicDispatch tag `corelib/kernels/20261029.0`; a store at another revision
can differ. Prefill PDIs ship row counts 1 to 4096, decode PDIs one row.
Everything not listed runs on the GPU, as do attention, RoPE and every other
op.

The tests download each model into `models/<name>/` (git-ignored) and write
fixed-shape variants of it (`model_fixed_kv<N>_sq<M>.onnx`) into
`models/<name>-fixed/`, beside the same weights file. A fixed variant has the
model's `MatMulNBits` and norms unchanged, so its row in the table applies to
both folders; for Phi-4 both `models/phi-4-onnx-gpu-int4-rtn-block-32/` and
`models/phi-4-onnx-gpu-int4-rtn-block-32-fixed/` were checked.

| Model (`test/python`) | Matmuls on the NPU | Norms on the NPU | `RYZENAICORELIB_PDI` | Run on AIE4 |
|---|---|---|---|---|
| Llama 3.1 8B AWQ (`test_llama8b.py`, block 128) | All 225: Q/K/V/O, MLP, `lm_head` (decode) | All 65: final norm (PDIs 1/16), the 64 skip-norms (PDIs 9/19) | `1,9;16,19` (`1;16` leaves the skip-norms on the GPU) | Yes: every matmul and norm on the NPU, no fallback; logits match the GPU-only run (cosine 0.99995, top-1 49 of 51, the rest near-ties) |
| Llama 3.2 1B (`test_llama1b.py`, block 32) | All 112 | Final norm; the 32 skip-norms only on PDIs 9/19 | `1;16` | No |
| Gemma 3 4B (`test_gemma3_4b.py`, block 128) | All 239 | QK norms (k=256) and layer norms (k=2560); skip-norms on decode (PDI 33) | `2;17,33` | No |
| Mistral 7B v0.3 (`test_mistral7b_v3.py`, block 32) | 128 of 225: Q/K/V/O; MLP and `lm_head` stay on the GPU | Final norm; skip-norms only on PDIs 9/19 | `1;16` | No |
| Qwen2.5 14B, Qwen2.5-Coder 14B (`test_qwen2_5*_14b.py`, block 128, zero points) | 192 of 337: Q/O on PDIs 1/16, K/V on PDIs 9/37; MLP and `lm_head` stay on the GPU | All, on PDIs 9/37 | `1,9;16,37` | No |
| gpt-oss 20B (`test_gptoss20b.py`, block 32) | `lm_head` only, on decode | Final norm | `1;16` | No |
| Phi-4 14B (`test_phi4_14b.py`, block 32; `microsoft/phi-4-onnx` `gpu/gpu-int4-rtn-block-32`) | None of 281: the store has Phi-4's shapes (K/N 5120, 1280, 17920, 100352) at g64 only | The 81 norms (k=5120), on PDIs 9/37 only | — (no matmul offload; norms alone are not worth the per-call staging) | No |
| DeepSeek-R1 70B (`test_deepseek_r1_70b.py`) | None: no K=8192 kernels | None | — | No |
| Llama 3.1 8B asym, gpt-oss 120B | Not checked: their Hugging Face repos were not accessible | | | No |

A model bundle covers fewer shapes than the store: with
`RYZENAICORELIB_KERNEL_BUNDLE=llama`, Llama 3.1 8B runs Q/K/V/O, `lm_head` and
the final norm on the NPU but its MLP on the GPU, since the bundle carries only
CoreLib's fused MLP kernel. Phi-4 would reach the NPU for every matmul from an
export at block size 64 or 128. A PDI list naming more than one PDI per phase
(such as `1,9;16,19` to reach the Llama skip-norms) makes the NPU switch PDIs
between ops; for Llama 3.1 8B its throughput was within the run-to-run spread
of `1;16`.

The kernel store must be at the revision the table was found at or later: an
older DynamicDispatch checkout lacked the width-4096 `addrmsnorm` kernels, and
CoreLib reads the plain norms from `mladfrmsnorm/` and the fused ones from
`addrmsnorm/`. A sparse checkout for these three ops needs
`transaction/mds/mladfmatmulbias`, `transaction/mds/mladfrmsnorm` and
`transaction/mds/addrmsnorm`.

## Generated code

With `use-ryzenai-corelib`, `convert-hip-to-llvm` emits the CoreLib wrapper
first and guards the GPU wrapper on its result
(`lib/Conversion/HipToLLVM/RyzenAICorelibLowering.cpp`):

```mlir
%rc = llvm.call @wrap_corelib_matmul_nbits(%state, %slot, ..., %dtype)
%gpu = llvm.icmp "eq" %rc, %c1 : i32          // HIPDNN_EP_CORELIB_NOT_HANDLED
llvm.cond_br %gpu, ^gpu, ^cont
^gpu:
  llvm.call @wrap_matmul_nbits(%state, %slot, ...)
  llvm.br ^cont
```

Each `wrap_corelib_*` takes its GPU wrapper's exact arguments plus the
activation data type (`HIPDNN_EP_DATATYPE_*`), so the fallback needs no
argument translation.

## Runtime

`lib/Runtime/real/ryzenaicorelib.cpp` is compiled natively into the
`hipdnn-ep-corelib` static library (not `runtime.bc`) and reaches the JIT as
absolute symbols registered in `LlvmIrJit`, like the current-stream accessors.
When the CoreLib header is absent at build time, or in mock builds,
`lib/Runtime/mock/ryzenaicorelib_stub.cpp` takes its place and every call
returns `HIPDNN_EP_CORELIB_NOT_HANDLED`.

CoreLib is loaded with `LoadLibrary`/`dlopen` on first use rather than linked,
so the EP loads on machines without it. The library's version must match the
header's major (and, below 1.0, minor) version.

Per session (`RuntimeState::corelib_ctx`, released through
`corelib_ctx_destroy` in `hipdnn_ep_state_cleanup`):

- One CoreLib stream restricted to the NPU
  (`stream_set_enabled_backends(npu)`), on the configured PDIs, with the
  configured kernel bundle enabled. No usable NPU, no loadable library, a
  malformed setting, a bundle that fails to enable, or neither a bundle nor a
  kernel store makes the context unusable, and every call returns
  not-handled.
- Per op instance, keyed by its constant weight pointer (stable for the
  session), op kind and epsilon: the packed weights, the row counts that ship
  for the shape (from `*_enum_kernels`), and BF16 device tensors per padded
  row count.

Per call:

1. Rows (`M * batch`, or elements / width for the norms) are padded up to the
   smallest shipped row count; none means not-handled.
2. Inputs are copied GPU → host on the session stream, written into CoreLib
   tensors, run, synchronized, read back, and copied host → GPU. CoreLib
   exposes no device pointer, so this staging is unavoidable.
3. The stream may use only the NPU, so CoreLib never runs a call on its own
   GPU or CPU paths. Where the NPU cannot run it (no kernel on the stream's
   PDIs, reported as `unsupported` or `missing_kernel`, or a failed NPU run)
   the call fails. hip-ep then runs it on the GPU wrapper and routes that
   padded row count to the GPU for the rest of the session. Under
   `HIPDNN_EP_STRICT=1` the device error flag is set instead, so
   `inference_compute` fails.

Pad rows are zeroed when a row count's tensors are created, since a CoreLib
device tensor is undefined until written.

The last session to close calls `ryzenai_corelib_cleanup`, which CoreLib
requires before process exit.

## Testing

| Where | What it covers |
|---|---|
| LIT (`test/lit/Conversion/hip-to-llvm/test_ryzenai_corelib.mlir`) | Call-then-fallback IR with the option; GPU-only IR without it |
| Any machine, option unset | No regression: generated code is unchanged and CoreLib is never loaded |
| Any machine without an AIE4 NPU, option set | Compile option, fallback branch, library loading and version check; results match the GPU run |
| AIE4 machine, option set | NPU execution: weight packing, row padding, BF16 accuracy, off-NPU detection |

On an AIE4 machine, run an int4 MatMulNBits ONNX export of a model CoreLib
supports with `HIPEP_USE_RYZENAI_CORELIB=1`, `RYZENAICORELIB_ROOT`, the
model's `RYZENAICORELIB_PDI`, `RYZENAI_CORELIB_KERNELS_ROOT` (for example
`<DynamicDispatch>/transaction/mds`) or `RYZENAICORELIB_KERNEL_BUNDLE`, and
`HIPDNN_EP_STRICT=1`. Expect `[corelib] NPU stream: prefill PDIs ...` on
stderr. Only ONNX models are compiled; CoreLib's own GGUF model files are not
used.

The simplest demonstration is an int4 ONNX export of Qwen3 (block size 128)
with `RYZENAICORELIB_KERNEL_BUNDLE=qwen3` and the default PDIs, `1;16`: the
bundle holds the matmul and norm shapes CoreLib's Qwen3 driver dispatches.
That driver runs the MLP as one fused `ssmlp` kernel, so the bundle has no
plain matmul at the gate, up and down shapes, and an ONNX export that fuses
Q, K and V into one projection has no match either. Each such op logs
`no matmul kernel ...` once and runs on the GPU; the attention projections,
`lm_head` and the norms run on the NPU. `HIPDNN_EP_STRICT=1` fails the run
only when the NPU refuses a shape that a kernel was listed for.

## Limitations

- Every routed call stages through host memory and synchronizes the GPU
  stream, so per-op offload is expected to be slower than the GPU path; it
  establishes the plumbing and numerical correctness.
- CoreLib kernels ship only at specific shapes; anything else falls back.
- Attention, the fused MLP (`ssmlp_bf16`) and RoPE are not routed.
- AIE4 only; AIE2/AIE2P machines always use the GPU.
- Validated on AIE4 hardware with one model so far (see Model coverage).
