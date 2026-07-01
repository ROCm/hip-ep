<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen EP: Technical Pipeline Walkthrough

## Overview

This document walks through the complete execution flow of the MorphiZen Execution Provider (EP),
from ONNX Runtime loading the EP DLL, through MLIR compilation, to GPU kernel execution.

The running example is `conv_test_hybrid.onnx` — a small model with three ops:
**Conv → MatMul → Sigmoid** on a `1x1x8x8` f32 tensor.

---

## 1. The DLL Landscape

Five DLLs participate in the end-to-end flow:

| DLL | Built by | Role |
|-----|----------|------|
| `onnxruntime.dll` | ORT build | The inference framework |
| `onnxruntime_morphizen_ep.dll` | This repo (MorphiZen + ort-bridge) | Execution Provider — ORT's plugin interface; embeds `runtime.bc` and JIT-loads per-model bitcode in-process via `LlvmIrJit` |
| `hip-compiler.dll` | This repo (MLIR pipeline) | Compiles ONNX MLIR → OS-portable per-model LLVM bitcode (`.bc`) |
| `hipdnn_graph_runtime.dll` | This repo (runtime lib) | C ABI for hipDNN graph execution at inference time |
| `hipdnn_backend.dll` | TheRock SDK | The actual hipDNN implementation (wraps MIOpen) |

Plus a **per-model LLVM bitcode artifact** (`.bc`, stored in the EPContext tar)
generated at session creation. It is OS-portable (no triple, no data layout, no
MSVC/glibc symbols); the EP DLL JITs it in-process at session creation alongside
its embedded `runtime.bc`. No per-model DLL is written to disk and no
`LoadLibrary` call is made at inference time — a hard requirement for shipping
under Microsoft's signed-DLL-only loading policy.

Per-arch GPU kernels ship as side-by-side `custom_kernels_<arch>.{dll,so}` next
to the EP binary; `LlvmIrJit` `dlopen`s the matching variant at JIT init based
on the device's `gcnArchName`.

---

## 2. The Invocation Chain

The flow has three distinct phases: **EP Loading**, **Session Creation (MLIR Compilation)**,
and **Inference Execution**.

### Phase A: Loading the EP (ORT startup)

```
onnxruntime.dll
  │  (1) LoadLibrary("onnxruntime_morphizen_ep.dll")
  │  (2) GetProcAddress("CreateEpFactories")
  │  (3) Calls CreateEpFactories()
  ▼
onnxruntime_morphizen_ep.dll
```

ORT uses the **EP V2 API** to discover and load execution providers. The EP DLL exports
a standard entry point:

**File:** `3rd-party/morphizen/ort-bridge/src/ort-bridge.cpp`
```cpp
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories,
                             size_t* num_factories) {
    // ...
    std::unique_ptr<OrtEpFactory> factory =
        std::make_unique<morphizen::MorphiZenEpFactory>(
            registration_name, morphizen::ApiPtrs{*ort_api, *ort_ep_api},
            *default_logger);
    factories[0] = factory.release();
    // ...
}
```

ORT then calls `MorphiZenEpFactory::CreateEpImpl` which constructs a `MorphiZenEP` instance:

**File:** `3rd-party/morphizen/ort-bridge/src/morphizen-ep-factory.cpp`
```cpp
OrtStatus* MorphiZenEpFactory::CreateEpImpl(..., OrtEp** ep) noexcept {
    auto morphizen_ep = std::make_unique<MorphiZenEP>(
        *factory, factory->ep_name_, ep_metadata, *session_options, *logger);
    *ep = morphizen_ep.release();
    return nullptr;
}
```

This is what produces the log message:
```
[Plugin EP] EP Device [Index: 3, Name: MorphiZenExecutionProvider] has been added to session.
```

### Phase B: Session Creation (MLIR Compilation)

When ORT creates a session, MorphiZen's **Level-1 pass** takes the ONNX model
(as MLIR bytecode) and invokes the MLIR compiler via a second plugin load:

```
onnxruntime_morphizen_ep.dll
  │  Level1MlirPass::process()
  │  → morphizen::Plugin::get("hip-compiler")
  │  → LoadLibrary("hip-compiler.dll")
  │  → GetProcAddress("hip_compile_with_fs")
  ▼
hip-compiler.dll
  │  hip_compile_with_fs(mlir_bytecode, output_path, options, fs)
  │  → CompilerDriver::compile()
  │      ├── parseSourceFile        (MLIR bytecode → ONNX dialect)
  │      ├── runMLIRPasses          (the full pipeline — see Section 3)
  │      ├── translateToLLVMIR      (LLVM dialect → LLVM IR)
  │      ├── optimizeLLVMIR         (target-independent PerModule O0-O3)
  │      ├── stripTargetMetadata    (clear triple + data layout for OS portability)
  │      └── emit model.bc          (+ model.constants.bin sidecar)
  ▼
Per-model LLVM bitcode  (written via MorphiZen FileSystem into the EPContext tar)
```

**File:** `backend-mlir-compiler/level-1-pass/src/MlirCompiler.cpp`
```cpp
auto plugin = morphizen::Plugin::get("hip-compiler");
if (!plugin) {
    LOG(ERROR) << "Failed to load hip-compiler plugin";
    return std::nullopt;
}

auto func = plugin->get_method<CompilerErrorCode, const void*, size_t,
                               const char*, const char*, CompilerError*, void*>(
    "hip_compile_with_fs");

auto result = func(mlir_bytecode.data(), mlir_bytecode.size(),
                   temp_output_path.c_str(), options_json.c_str(), &error, fs);
```

**File:** `lib/CInterface/CompilerAPI.cpp`
The `hip_compile_with_fs` export delegates to `CompilerDriver`, which runs the
full MLIR pass pipeline and emits the per-model LLVM bitcode. The producer does
NOT link `runtime.bc` into the per-model module — `runtime.bc` is OS-specific
(MSVC vs glibc CRT) and is JIT-loaded as a separate module on the consumer side
inside the EP DLL, which keeps the per-model artifact OS-portable.

The `GenerateInterfacePass` (the last MLIR pass) emits four exported symbols
in the per-model bitcode:

| Export | Purpose |
|--------|---------|
| `inference_init(state**, fs*)` | Allocate GPU memory, load constants, set up buffer pool |
| `inference_compute(state*, inputs*, outputs*)` | Run the graph on GPU |
| `inference_cleanup(state*)` | Free GPU resources |
| `inference_get_metadata_json()` | Return embedded model metadata as JSON (used by `hip-test`/`hip-inspect`) |

### Phase C: Inference Execution (per-run)

After compilation, MorphiZen's custom op JIT-loads the per-model bitcode in
the EP DLL's address space (no temp file, no `LoadLibrary`) and calls into the
JITted code on every inference request:

```
ORT calls MlirCustomOp::Compute()
  │  marshal ORT input tensors → span_t (pointer + shape + size)
  │  install EP output allocator (hipdnn_ep_set_output_allocator)
  ▼
InferenceState::compute(inputs)
  │  calls inference_compute(state, &inputs) (JITted symbol)
  ▼
JITted in-memory module  (per-model bitcode + runtime.bc, single LLJIT JITDylib)
  │  inference_compute:
  │    ├── hipdnn_ep_tensor_prepare_input   → hipMemcpyH2D (host → GPU)
  │    ├── main_graph():
  │    │     ├── hip.get_pool        → get pre-allocated GPU scratch memory
  │    │     ├── hip.get_constant(0) → conv weights on GPU
  │    │     ├── hip.get_constant(1) → matmul weights on GPU
  │    │     ├── hipdnn_graph_execute        → Conv via hipDNN backend
  │    │     ├── wrap_hipblasLtMatmul        → MatMul via hipBLASLt
  │    │     ├── wrap_miopenActivationForward → Sigmoid via MIOpen
  │    │     └── hipdnn_ep_alloc_output       → EP allocator callback (in-graph output)
  │    ├── hipdnn_ep_stream_sync             → all GPU writes complete
  │    └── hipdnn_ep_tensor_free_input       → cleanup
  ▼
EP copies any host outputs D2H, then results back to ORT
```

**File:** `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
```cpp
// During session init — JIT the per-model bitcode and call inference_init
auto jit = LlvmIrJit::create(bitcode_bytes, "model_compiled");
auto init_fn = jit->get_method<int, void**, void*>("inference_init");
int ret = init_fn(&state, static_cast<void*>(fs));

// Per-run — call inference_compute (JITted symbol, 2-arg output-allocator ABI)
int InferenceState::compute(span_t* inputs) const {
    auto compute_fn =
        jit_->get_method<int, void*, span_t*>("inference_compute");
    return compute_fn(state_, inputs);
}
```

**File:** `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`
`marshal_input_tensors` converts ORT's `OrtKernelContext` input tensors into
`tensor_t` / `span_t` structs that the generated code expects. Outputs are
allocated in-graph via the EP's `output_allocate_cb` callback, which calls
`ctx.GetOutput()` with the DLL's computed shape.

---

## 3. The MLIR Compilation Pipeline (IR Dump Walkthrough)

Setting `HIPDNN_EP_IR_DUMP_PATH` dumps the IR before/after every pass. The pipeline
for our example model (`conv_test_hybrid.onnx`) proceeds through these stages:

### Stage 1: Input — ONNX Dialect

Pure ONNX ops with tensor semantics. No GPU concepts, no memory management.

```mlir
func.func @main_graph(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32> {
    %1 = "onnx.Constant"() {value = dense<...> : tensor<1x1x3x3xf32>}   // conv weights
    %2 = "onnx.Constant"() {value = dense<...> : tensor<8x8xf32>}        // matmul weights
    %3 = "onnx.Conv"(%arg0, %1) {group = 1 : si64, kernel_shape = [3,3],
                                  pads = [1,1,1,1], strides = [1,1]}
    %4 = "onnx.MatMul"(%3, %2)
    %5 = "onnx.Sigmoid"(%4)
    "onnx.Return"(%5)
}
```

### Stage 2: HipAddContextArgPass

Inserts `!hip.context` as the first function argument. This opaque handle carries
all runtime state (HIP stream, MIOpen handle, hipBLASLt handle, buffer pool).

```mlir
func.func @main_graph(%arg0: !hip.context, %arg1: tensor<1x1x8x8xf32>)
    -> tensor<1x1x8x8xf32> {
    // ... same ops, but %arg0 → %arg1 for the input tensor
}
```

### Stage 3: OutlineOnnxToHipDNNPass

Identifies ONNX ops that can be fused into a hipDNN graph (here, `onnx.Conv`)
and wraps them in a `hip.hipdnn_graph_outline` region:

```mlir
%3 = hip.hipdnn_graph_outline ins(%arg1, %1 : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
    -> tensor<1x1x8x8xf32> {
  ^bb0(%arg2: tensor<1x1x8x8xf32>, %arg3: tensor<1x1x3x3xf32>):
    %6 = "onnx.Conv"(%arg2, %arg3) {group = 1 : si64, kernel_shape = [3,3], ...}
    hip.yield %6 : tensor<1x1x8x8xf32>
}
%4 = "onnx.MatMul"(%3, %2)    // remains outside — not a hipDNN graph candidate
%5 = "onnx.Sigmoid"(%4)       // remains outside
```

The Conv is now isolated in its own region, ready for hipDNN graph compilation.
MatMul and Sigmoid remain as standalone ONNX ops.

### Stage 4: CompileHipDNNGraphsPass

Calls into `hipdnn_backend.dll` to build and compile an actual hipDNN graph
from the outlined region. On success, replaces the outline with a compiled
`hip.hipdnn_graph` op:

```mlir
%3 = tensor.empty() : tensor<1x1x8x8xf32>
%4 = hip.hipdnn_graph(%arg0) graph_id(0)
    ins(%arg1, %1 : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
    outs(%3 : tensor<1x1x8x8xf32>)
    {input_uids = [0, 1], output_uids = [2]} : tensor<1x1x8x8xf32>
```

The ONNX Conv op is gone — replaced by an opaque graph execution reference.
If compilation fails (e.g., unsupported op combination), the pass un-outlines
back to ONNX ops and the standard lowering path handles them.

### Stage 5: ConvertOnnxToHipPass

Converts remaining ONNX ops to HIP dialect ops, externalizes constants, and
attaches module-level metadata:

```mlir
module attributes {
    hip.constants_file = "model.constants.bin",
    hipdnn.constant_offsets = array<i64: 0, 64>,
    hipdnn.constant_sizes = array<i64: 36, 256>,
    hipdnn.input_count = 1 : i64,
    hipdnn.output_count = 1 : i64,
    hipdnn.pool_size = 512 : i64,
    // ...
} {
  memref.global @hip_ext_constant_0 : memref<1x1x3x3xf32>  // conv weights → file offset 0
  memref.global @hip_ext_constant_1 : memref<8x8xf32>       // matmul weights → file offset 64

  func.func @main_graph(%arg0: !hip.context, %arg1: tensor<1x1x8x8xf32>)
      -> tensor<1x1x8x8xf32> {
    %0 = memref.get_global @hip_ext_constant_0    // load conv weights reference
    %1 = memref.get_global @hip_ext_constant_1    // load matmul weights reference
    %4 = hip.hipdnn_graph(%arg0) graph_id(0) ...  // Conv (compiled hipDNN graph)
    %7 = hip.matmul(%arg0) ins(%5, %3) outs(%6)   // MatMul → hipBLASLt
    %9 = hip.sigmoid(%arg0) ins(%7) outs(%8)       // Sigmoid → MIOpen activation
    return %9
  }
}
```

Key transformations:
- `onnx.Constant` values → externalized to `model.constants.bin` (written via FileSystem)
- `onnx.MatMul` → `hip.matmul` (will lower to `wrap_hipblasLtMatmul`)
- `onnx.Sigmoid` → `hip.sigmoid` (will lower to `wrap_miopenActivationForward`)
- Module attributes describe the model's memory layout for the runtime

### Stage 6: Bufferization (tensor → memref)

`OneShotBufferize` converts tensor semantics to buffer (memref) semantics.
Tensors become explicit memory allocations; the graph output stays a returned
memref:

```mlir
func.func @main_graph(%arg0: !hip.context, %arg1: memref<1x1x8x8xf32>)
    -> memref<1x1x8x8xf32> {
    %alloc = memref.alloc() : memref<1x1x8x8xf32>         // scratch for conv output
    hip.hipdnn_graph(%arg0) ... outs(%alloc)                // Conv writes to alloc
    %alloc_0 = memref.alloc() : memref<1x1x8x8xf32>       // scratch for matmul output
    hip.matmul(%arg0) ins(%alloc, %1) outs(%alloc_0)        // MatMul
    %out = memref.alloc() : memref<1x1x8x8xf32>           // output buffer
    hip.sigmoid(%arg0) ins(%alloc_0) outs(%out)             // Sigmoid writes to output
    return %out
}
```

Buffer deallocation passes insert `memref.dealloc` for intermediate buffers
(the returned `%out` is owned, so no clone). `hip-use-output-allocator` then
rewrites the returned `memref.alloc` into `hip.alloc_output`, so the output is
allocated in-graph via the EP callback rather than passed as an out-param.

### Stage 7: PoolAllocsPass (memory optimization)

Replaces individual `memref.alloc`/`memref.dealloc` pairs with views into a single
pre-allocated GPU memory pool:

```mlir
module attributes { hipdnn.pool_size = 512 : i64, hipdnn.buffer_offsets = [0, 256], ... } {
  func.func @main_graph(%arg0: !hip.context, %arg1: memref<1x1x8x8xf32>,
                         %arg2: memref<1x1x8x8xf32>) {
    %pool = hip.get_pool(%arg0, %c512) : memref<?xi8>       // one pool allocation
    %view = memref.view %pool[%c0][] : ... to memref<1x1x8x8xf32>    // offset 0
    hip.hipdnn_graph(%arg0) ... outs(%view)
    %view_0 = memref.view %pool[%c256][] : ... to memref<1x1x8x8xf32> // offset 256
    hip.matmul(%arg0) ins(%view, %1) outs(%view_0)
    hip.sigmoid(%arg0) ins(%view_0) outs(%arg2)
    return
  }
}
```

Two 256-byte buffers (1x1x8x8 x 4 bytes = 256) are packed into a single 512-byte pool.
The pool is allocated once during `inference_init` and reused across all inference calls.

### Stage 8: ResolveExternConstantsPass

Replaces `memref.get_global` references with runtime calls to `hip.get_constant`,
which retrieve GPU pointers from the pre-loaded constants buffer:

```mlir
%0 = hip.get_constant(%arg0, %c0_i64) : memref<1x1x3x3xf32>  // constant index 0
%1 = hip.get_constant(%arg0, %c1_i64) : memref<8x8xf32>       // constant index 1
```

### Stage 9: ConvertHipToLLVMPass

Lowers all HIP dialect ops to LLVM dialect — function calls to the EP runtime:

| HIP op | LLVM call target | Resolution |
|--------|-----------------|-----------------|
| `hip.hipdnn_graph` | `hipdnn_graph_execute(ctx, graph_id, num_tensors, uids, ptrs)` | hipdnn_graph_runtime.dll (process search generator) |
| `hip.matmul` | `wrap_hipblasLtMatmul(ctx, A, B, C, M, N, K, batch, elem_size)` | runtime.bc (JIT-loaded into the same JITDylib) |
| `hip.sigmoid` | `wrap_miopenActivationForward(ctx, in, out, num_elems, alpha, beta)` | runtime.bc |
| `hip.get_pool` | `hipdnn_ep_get_pool_base(ctx, domain_id, size)` | runtime.bc |
| `hip.get_constant` | `hipdnn_ep_constant_get(ctx, index)` | runtime.bc |

The function signature is lowered to the C ABI with explicit memref descriptors
(base pointer, aligned pointer, offset, sizes, strides).

### Stage 10: GenerateInterfacePass

Generates the three public entry points (`inference_init`, `inference_compute`,
`inference_cleanup`) that wrap the lowered `main_graph` function.

**`inference_init`:**
- Parses the embedded metadata blob (constants file path, shapes, pool size)
- Calls `hipdnn_ep_state_init_with_fs` to set up `RuntimeState`
  (creates HIP stream, MIOpen handle, hipBLASLt handle, loads constants to GPU)
- Calls `hipdnn_ep_pool_init` to pre-allocate the GPU buffer pool (512 bytes here)

**`inference_compute`** (2-arg output-allocator ABI):
- Calls `hipdnn_ep_tensor_prepare_input` — copies host input to GPU (`hipMemcpyH2D`)
- Calls `main_graph(context, input_memref)` — the compiled graph; each output is
  allocated in-graph via `hip.alloc_output` → `hipdnn_ep_alloc_output` → the EP
  allocator callback (zero-copy for GPU outputs)
- Calls `hipdnn_ep_stream_sync` — ensures all GPU writes are complete on return
- Calls `hipdnn_ep_tensor_free_input` — releases input staging buffer

**`inference_cleanup`:**
- Calls `hipdnn_ep_state_cleanup` — destroys handles, frees GPU memory

---

## 4. Post-MLIR: LLVM IR to OS-Portable Bitcode

The IR dump ends after `GenerateInterfacePass`. The remaining producer-side steps
happen inside `CompilerDriver::compileImpl`; the JIT-load + symbol-resolve steps
happen later, on the consumer side, inside the EP DLL:

```
MLIR LLVM Dialect IR  (what ir_dump.mlir shows)
        │
        │  (1) translateToLLVMIR     — MLIR LLVM dialect → actual LLVM IR (llvm::Module)
        ▼
   LLVM IR  (in-memory; external decls for wrap_*, hipdnn_ep_*, hip_* kept)
        │
        │  (2) optimizeLLVMIR        — target-independent PerModule O0-O3
        ▼
   Optimized LLVM IR
        │
        │  (3) emitLlvmIr           — strip triple/datalayout for OS portability,
        │                              then llvm::WriteBitcodeToFile
        ▼
   model.bc  (OS-portable, stored in the EPContext tar)

   --- runtime side, inside the signed EP DLL ---
        │
        │  (4) LlvmIrJit::create    — parseBitcodeFile (per-model AND embedded
        │                              runtime.bc) into one shared LLVMContext;
        │                              addIRModule both into one LLJIT JITDylib;
        │                              install ROCm + per-arch kernel
        │                              DynamicLibrarySearchGenerators for
        │                              external symbols (libamdhip64, MIOpen,
        │                              hipBLASLt, hipdnn_backend, hipdnn_graph_runtime,
        │                              custom_kernels_<arch>); jit->initialize()
        │                              runs @llvm.global_ctors.
        ▼
   JITted in-memory module
        │
        │  (5) lookup_raw            — resolve `inference_init`, `inference_compute`,
        │                              `inference_cleanup`, `inference_get_metadata_json`,
        │                              and the optional `hipdnn_ep_runtime_begin_compute`
        │                              hook.
        ▼
   ready to run
```

**File:** `lib/Compiler/CompilerDriver.cpp`
```cpp
bool CompilerDriver::compileImpl(mlir::ModuleOp module,
                                 const std::string &output_path,
                                 const mlir::hip::CompilationOptionsT &options,
                                 std::string &error_message) {
  if (!runMLIRPasses(module, options, error_message))                       // MLIR pipeline
    return false;

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);  // (1)
  if (!llvmModule) return false;

  // runtime.bc is JIT-loaded as a separate module by LlvmIrJit on the
  // consumer side (no producer-time link merge -> OS-portable artifact).

  optimizeLLVMIR(llvmModule.get(), options.opt_level);                      // (2)

  return emitLlvmIr(llvmModule.get(), output_path, error_message);         // (3)
}
```

The LLVM IR produced by the MLIR pipeline contains only **external declarations**
for `wrap_*` / `hipdnn_ep_*` / `hip_*` symbols. The producer does NOT link in
`runtime.bc` — that step (formerly `LLVMBackend::linkRuntimeModule`, now removed)
would have baked the build-host's CRT into every per-model artifact and broken
OS portability. Resolution of the external decls is deferred to the consumer-side
JIT, where `runtime.bc` is added as a sibling module in the same JITDylib and the
process search generator binds `hip_*` kernels and ROCm symbols.

### 4.1 Two-Stage Runtime Compilation

The runtime wrappers in `lib/Runtime/real/` (e.g., `matmul.cpp`, `miopen.cpp`, `gqa.cpp`,
`activation.cpp`, etc.) use a two-stage compilation strategy. Stage 1 produces an
OS-specific `runtime.bc` at EP-build time; stage 2 is the consumer-side JIT load that
stitches `runtime.bc` and the per-model bitcode into one JITDylib.

**Stage 1 — EP Build Time: C++ to Embedded Bitcode**

```
 lib/Runtime/real/*.cpp  (matmul.cpp, miopen.cpp, gqa.cpp, activation.cpp, ...)
         │
         │  clang -c -emit-llvm -std=c++17 -O2 \
         │         -fno-threadsafe-statics -fno-sized-deallocation -fno-rtti
         │  (LLVM bitcode, NOT native object code; the -fno-* flags shrink the
         │   MSVC-only symbol surface the consumer-side JIT must resolve.)
         ▼
 Individual .bc files  (runtime_matmul.bc, runtime_miopen.bc, runtime_gqa.bc, ...)
         │
         │  llvm-link   (combines all ~50 modules into one)
         ▼
 runtime.bc  (per-OS: MSVC ABI on Windows, glibc ABI on Linux)
         │
         │  cmake/xxd.py   (converts binary to C byte array)
         ▼
 runtime_ir_data.cpp
   extern "C" const unsigned char runtime_bc_data[] = { 0x42, 0x43, ... };
   extern "C" const size_t runtime_bc_data_size = 123456;
         │
         │  Compiled into morphizen-custom-op-mlir static archive,
         │  WHOLE_ARCHIVE-linked into onnxruntime_morphizen_ep.dll
         ▼
 Embedded as a byte array inside the EP DLL
```

The key build command is `clang -c -emit-llvm`, which compiles C++ to **LLVM bitcode**
instead of native machine code. This is why Clang is a hard build requirement — MSVC
cannot produce LLVM bitcode.

**File:** `lib/Runtime/CMakeLists.txt` — the `compile_to_bitcode` macro:
```cmake
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
    COMMAND ${CLANG_EXECUTABLE}
            -c -emit-llvm -std=c++17
            -fno-threadsafe-statics -fno-sized-deallocation -fno-rtti
            $<$<NOT:$<CONFIG:Debug>>:-O2>
            ${COMPILE_FLAGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
    ...
)
```

**File:** `backend-mlir-compiler/custom-op-mlir/CMakeLists.txt` — the embed step
(consumer side; `runtime.bc` lives in the EP DLL, not in `hip-compiler.dll`):
```cmake
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var runtime_bc_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
          ${RUNTIME_BC_PATH}
  ...
)
```

**Stage 2 — Session Creation Time: Consumer-Side JIT Load**

When the EP creates a session, `InferenceState::create` reads the per-model bitcode
from the EPContext tar, hands it to `LlvmIrJit::create`, which adds **two modules**
into a single ORC `LLJIT` JITDylib (one shared `LLVMContext` so opaque-struct identity
holds across the boundary):

**File:** `backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp`
```cpp
extern "C" const unsigned char runtime_bc_data[];
extern "C" const std::size_t runtime_bc_data_size;

// 1. Parse runtime.bc (embedded in the EP DLL) and the per-model bitcode
//    into one shared LLVMContext.
auto context = std::make_unique<llvm::LLVMContext>();
auto module = llvm::parseBitcodeFile(per_model_bytes, *context);
auto runtime = llvm::parseBitcodeFile(runtime_bc_data, *context);

// 2. The per-model bitcode has empty triple/datalayout (OS-portable);
//    stamp the JIT host values onto it before addIRModule.
module->setDataLayout(jit->getDataLayout());
module->setTargetTriple(jit->getTargetTriple());

// 3. Add runtime first so the per-model module's external `hipdnn_ep_*`
//    references resolve in the JITDylib's symbol table.
jit->addIRModule(ThreadSafeModule(std::move(runtime), tsc));
jit->addIRModule(ThreadSafeModule(std::move(module),  tsc));
```

This is **not** producer-time linking. Cross-module inlining between generated code
and runtime functions does not happen at producer time — the per-model bitcode keeps
external `wrap_*` declarations. The ORC `IRCompileLayer` lazily codegens both modules
on first symbol lookup; cross-module inlining within a single module is preserved, but
between modules it requires explicit IR-level merging that we deliberately avoid here
(merging would re-introduce the OS-specific CRT symbols into the per-model artifact).

**Why bitcode at rest, JIT at runtime.** The two-module split is what makes the
per-model artifact OS-portable: `runtime.bc` carries the host-OS CRT/ABI baggage and
stays inside the per-OS EP DLL; the per-model `.bc` is target-neutral and the same
on-disk bytes work on Windows or Linux. It also satisfies Microsoft's signed-DLL-only
loading policy — there is no per-model DLL written to disk at inference time.

### 4.2 The `!hip.context` Type and RuntimeState Lifetime

In the HIP dialect IR (Section 3), every function receives `%arg0: !hip.context` as its
first argument. This opaque type threads GPU execution state through the entire compute
graph.

**MLIR definition** (`include/hip/Dialect/IR/HipTypes.td`):
```tablegen
def Hip_ContextType : Hip_Type<"Context", "context"> {
  let summary = "Opaque HIP execution context";
  let description = [{ Lowered to !llvm.ptr. }];
}
```

During `ConvertHipToLLVMPass`, `!hip.context` is lowered to `!llvm.ptr`. At runtime, that
pointer points to a `RuntimeState` struct:

**File:** `lib/Runtime/runtime_state_internal.h`
```cpp
struct RuntimeState {
  hipStream_t stream;              // GPU command queue
  miopenHandle_t miopen_handle;    // MIOpen library handle (conv, activation, norm)
  hipblasLtHandle_t hipblas_handle;// hipBLASLt handle (matmul/GEMM)

  void *gpu_constants_blob;        // Model weights on GPU (single allocation, VRAM)
  void **gpu_constants;            // Per-constant pointers into the blob
  size_t num_constants;

  void *pool_base;                 // Pre-allocated GPU memory pool
  size_t pool_size;                // Total pool size in bytes
  size_t *buffer_offsets;          // Per-buffer offsets within pool
  size_t num_buffers;

  void *workspace;                 // Shared scratch buffer (MatMul, GQA, Conv)
  size_t workspace_size;           // Current workspace size (lazily grown)

  void *hipdnn_handle;             // hipDNN graph compilation handle
  void *hipdnn_graph_registry;     // hipDNN compiled graph registry
};
```

**What it carries:**

| Field | Purpose |
|-------|---------|
| `stream` | Single HIP stream all ops execute on |
| `miopen_handle` | MIOpen library handle (conv, activation, layernorm) |
| `hipblas_handle` | hipBLASLt handle (matmul/GEMM) |
| `gpu_constants_blob` / `gpu_constants` | Model weights on GPU (loaded at init) |
| `pool_base` / `buffer_offsets` | Pre-allocated memory pool for intermediate tensors |
| `workspace` | Shared scratch buffer for library workspace needs |
| `hipdnn_handle` / `hipdnn_graph_registry` | hipDNN graph compilation support |

**Lifetime** — managed by the three exported functions in `model.dll`:

```
inference_init(state**, fs)
  │  Allocates RuntimeState, creates stream + library handles,
  │  loads constants from disk to GPU, allocates memory pool
  │
  ▼  state is now live
inference_compute(state, inputs, outputs)    ← called N times per session
  │  state passed as %arg0 to every wrap_* runtime function
  │  Each wrapper extracts what it needs internally:
  │    stream   = state->stream
  │    handle   = state->miopen_handle  (or hipblas_handle)
  │    constant = state->gpu_constants[index]
  │    pool_ptr = state->pool_base + offset
  ▼
inference_cleanup(state)
     Destroys handles, frees GPU memory, frees struct
```

**Key design principle:** Generated code never dereferences `RuntimeState` directly — it
only passes the pointer to runtime wrapper functions (e.g., `wrap_miopenConvolutionForward`,
`wrap_hipblasLtMatmul`), which access the fields internally. This opaque-pointer pattern
allows the `RuntimeState` struct to evolve without recompiling models.

---

## 5. Visual Summary

```
                    SESSION CREATION                          INFERENCE
                    ════════════════                          ═════════

  ORT ──→ morphizen_ep.dll ──→ hip-compiler.dll              ORT
               │                      │                        │
               │     ┌────────────────┘                        │
               │     │ MLIR Pipeline:                    MlirCustomOp
               │     │  ONNX → HIP → LLVM → strip-triple     │
               │     │                                   InferenceState
               │     ▼                                        │
               │  model.bc        ─────────────────────►  LlvmIrJit::create
               │  model.constants.bin                     (model.bc + embedded
               │                                           runtime.bc → LLJIT)
               │                                              │
               │                                              ▼
               │                                       inference_compute()
               │                                              │
               │                                    ┌─────────┼──────────┐
               │                                    ▼         ▼          ▼
               │                              hipdnn_graph  hipBLASLt  MIOpen
               │                              _execute()    Matmul()   Sigmoid()
               │                                    │
               │                                    ▼
               └──────────────────────────► hipdnn_graph_runtime.dll
                                                    │
                                                    ▼
                                            hipdnn_backend.dll (TheRock)
                                                    │
                                                    ▼
                                              amdhip64.dll (HIP)
                                                    │
                                                    ▼
                                               AMD GPU
```

---

## 6. Plugin Loading Architecture

There are **three distinct binding steps**, each using a different mechanism:

| Step | Loader | Target | API |
|------|--------|--------|-----|
| 1 | ORT | `onnxruntime_morphizen_ep.dll` | EP V2 (`CreateEpFactories`) |
| 2 | MorphiZen | `hip-compiler.dll` | MorphiZen Plugin (`morphizen::Plugin::get`) |
| 3 | EP DLL (in-process) | per-model `.bc` (no DLL load) | `LlvmIrJit::create` over ORC `LLJIT` |

This three-level architecture keeps ORT, the compiler, and the generated code
cleanly separated with stable C ABI boundaries. Step 3 was previously a `LoadLibrary`
of a per-model DLL; replacing it with an in-process JIT was the change that lets
the per-model artifact stay OS-portable and avoids the signed-DLL gate at inference
time.

---

## 7. Key Source Files

| Concern | Path |
|---------|------|
| **EP Loading** | |
| ORT EP factory exports (DEF file) | `3rd-party/morphizen/morphizen-core/onnxruntime_morphizen_ep_with_ort_bridge.def` |
| `CreateEpFactories` implementation | `3rd-party/morphizen/ort-bridge/src/ort-bridge.cpp` |
| EP factory / `CreateEpImpl` | `3rd-party/morphizen/ort-bridge/src/morphizen-ep-factory.cpp` |
| **Compilation** | |
| Load `hip-compiler` plugin | `backend-mlir-compiler/level-1-pass/src/MlirCompiler.cpp` |
| `hip_compile_with_fs` export | `lib/CInterface/CompilerAPI.cpp` |
| MLIR pipeline + bitcode emit | `lib/Compiler/CompilerDriver.cpp` |
| Triple/datalayout strip + bitcode write | `lib/Target/LLVM/LLVMBackend.cpp` |
| **MLIR Passes** | |
| Add context argument | `lib/Conversion/Passes.cpp` (HipAddContextArgPass) |
| Outline ops for hipDNN | `lib/Conversion/OnnxToHipDNN/` (OutlineOnnxToHipDNNPass) |
| Compile hipDNN graphs | `lib/HipDNNGraph/HipDNNGraph.cpp` (CompileHipDNNGraphsPass) |
| ONNX → HIP dialect | `lib/Conversion/OnnxToHip/OnnxToHip.cpp` (ConvertOnnxToHipPass) |
| Memory pool allocation | `lib/Dialect/Transforms/` (PoolAllocsPass) |
| Resolve external constants | `lib/Dialect/Transforms/` (ResolveExternConstantsPass) |
| HIP → LLVM lowering | `lib/Conversion/HipToLLVM/HipToLLVM.cpp` (ConvertHipToLLVMPass) |
| Generate init/compute/cleanup | `lib/Dialect/Transforms/GenerateInterface.cpp` |
| **Runtime (Bitcode)** | |
| Bitcode build system | `lib/Runtime/CMakeLists.txt` |
| MIOpen convolution wrapper | `lib/Runtime/real/miopen.cpp` |
| hipBLASLt matmul wrapper | `lib/Runtime/real/matmul.cpp` |
| Activation (sigmoid, relu, etc.) | `lib/Runtime/real/activation.cpp` |
| GQA fused attention | `lib/Runtime/real/gqa.cpp` |
| LayerNorm / SkipLayerNorm | `lib/Runtime/real/simplified_layer_norm.cpp`, `skip_simplified_layer_norm.cpp` |
| HIP stream/device wrappers | `lib/Runtime/real/hip.cpp` |
| Memory alloc/free wrappers | `lib/Runtime/real/memory.cpp` |
| EP state management | `lib/Runtime/hipdnn_ep_runtime_state.cpp` |
| EP tensor marshalling (GPU) | `lib/Runtime/hipdnn_ep_runtime_tensor.cpp` |
| **Runtime Bitcode Embedding (consumer side)** | |
| Embed `runtime.bc` into EP DLL via `xxd.py` | `backend-mlir-compiler/custom-op-mlir/CMakeLists.txt` |
| `runtime.bc` link target | `lib/Runtime/CMakeLists.txt` (`RuntimeBitcode`) |
| **Runtime (Host-side)** | |
| In-process ORC JIT for per-model bitcode | `backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp` |
| JIT bitcode + call init/compute | `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp` |
| ORT tensor marshalling | `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` |
| hipDNN graph runtime API | `lib/HipDNNGraphRuntime/hipdnn_graph_runtime.h`, `.cpp` |

---

## 8. Enabling IR Dumps

To see the full MLIR pass pipeline output for any model:

```powershell
$env:HIPDNN_EP_IR_DUMP_PATH = "C:\path\to\ir_dump.mlir"
```

Then run `onnxruntime_perf_test.exe` (or any ORT session creation). The file will contain
before/after IR for every pass in the pipeline.
