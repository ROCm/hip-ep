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
| `onnxruntime_morphizen_ep.dll` | This repo (MorphiZen + ort-bridge) | Execution Provider — ORT's plugin interface |
| `hip-compiler.dll` | This repo (MLIR pipeline) | Compiles ONNX MLIR → native model DLL |
| `hipdnn_graph_runtime.dll` | This repo (runtime lib) | C ABI for hipDNN graph execution at inference time |
| `hipdnn_backend.dll` | TheRock SDK | The actual hipDNN implementation (wraps MIOpen) |

Plus a **temporary `model.dll`** generated at session creation time, containing the
JIT-compiled inference code for the specific model.

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
  │      ├── link EP runtime        (hipdnn_ep_runtime + hipdnn_graph_runtime)
  │      └── emit model.dll         (+ model.constants.bin)
  ▼
Temporary model.dll  (written to disk via MorphiZen FileSystem)
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
full MLIR pass pipeline and emits the model DLL.

The `GenerateInterfacePass` (the last MLIR pass) emits three exported functions
in the model DLL:

| Export | Purpose |
|--------|---------|
| `inference_init(state**, fs*)` | Allocate GPU memory, load constants, set up buffer pool |
| `inference_compute(state*, inputs*, outputs*)` | Run the graph on GPU |
| `inference_cleanup(state*)` | Free GPU resources |

### Phase C: Inference Execution (per-run)

After compilation, MorphiZen's custom op loads the model DLL and calls into it
on every inference request:

```
ORT calls MlirCustomOp::Compute()
  │  marshal ORT tensors → span_t (pointer + shape + size)
  ▼
InferenceState::compute(inputs, outputs)
  │  calls inference_compute(state, &inputs, &outputs) in model.dll
  ▼
model.dll  (the generated code from the MLIR pipeline)
  │  inference_compute:
  │    ├── hipdnn_ep_tensor_prepare_input   → hipMemcpyH2D (host → GPU)
  │    ├── hipdnn_ep_tensor_prepare_output  → allocate GPU output buffer
  │    ├── main_graph():
  │    │     ├── hip.get_pool        → get pre-allocated GPU scratch memory
  │    │     ├── hip.get_constant(0) → conv weights on GPU
  │    │     ├── hip.get_constant(1) → matmul weights on GPU
  │    │     ├── hipdnn_graph_execute        → Conv via hipDNN backend
  │    │     ├── wrap_hipblasLtMatmul        → MatMul via hipBLASLt
  │    │     └── wrap_miopenActivationForward → Sigmoid via MIOpen
  │    ├── hipdnn_ep_tensor_finalize_output  → hipMemcpyD2H (GPU → host)
  │    └── hipdnn_ep_tensor_free_input       → cleanup
  ▼
Results back to ORT
```

**File:** `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp`
```cpp
// During session init — load model DLL and call inference_init
auto init_fn = plugin->get_method<int, void**, void*>("inference_init");
int ret = init_fn(&state, static_cast<void*>(fs));

// Per-run — call inference_compute
int InferenceState::compute(span_t* inputs, span_t* outputs) const {
    auto compute_fn =
        plugin_->get_method<int, void*, span_t*, span_t*>("inference_compute");
    return compute_fn(state_, inputs, outputs);
}
```

**File:** `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`
`marshal_input_tensors` / `marshal_output_tensors` convert ORT's `OrtKernelContext`
tensors into `tensor_t` / `span_t` structs that the generated code expects.

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
Tensors become explicit memory allocations:

```mlir
func.func @main_graph(%arg0: !hip.context, %arg1: memref<1x1x8x8xf32>,
                       %arg2: memref<1x1x8x8xf32> {bufferize.result}) {
    %alloc = memref.alloc() : memref<1x1x8x8xf32>         // scratch for conv output
    hip.hipdnn_graph(%arg0) ... outs(%alloc)                // Conv writes to alloc
    %alloc_0 = memref.alloc() : memref<1x1x8x8xf32>       // scratch for matmul output
    hip.matmul(%arg0) ins(%alloc, %1) outs(%alloc_0)        // MatMul
    hip.sigmoid(%arg0) ins(%alloc_0) outs(%arg2)            // Sigmoid writes to output param
    return
}
```

`BufferResultsToOutParams` converts the return value into an output parameter (`%arg2`).
Buffer deallocation passes insert `memref.dealloc` for intermediate buffers.

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

| HIP op | LLVM call target | Runtime library |
|--------|-----------------|-----------------|
| `hip.hipdnn_graph` | `hipdnn_graph_execute(ctx, graph_id, num_tensors, uids, ptrs)` | hipdnn_graph_runtime.dll |
| `hip.matmul` | `wrap_hipblasLtMatmul(ctx, A, B, C, M, N, K, batch, elem_size)` | EP runtime (linked into model.dll) |
| `hip.sigmoid` | `wrap_miopenActivationForward(ctx, in, out, num_elems, alpha, beta)` | EP runtime (linked into model.dll) |
| `hip.get_pool` | `hipdnn_ep_get_pool_base(ctx, domain_id, size)` | EP runtime |
| `hip.get_constant` | `hipdnn_ep_constant_get(ctx, index)` | EP runtime |

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

**`inference_compute`:**
- Calls `hipdnn_ep_tensor_prepare_input` — copies host input to GPU (`hipMemcpyH2D`)
- Calls `hipdnn_ep_tensor_prepare_output` — allocates GPU output buffer
- Calls `main_graph(context, input_memref, output_memref)` — the compiled graph
- Calls `hipdnn_ep_tensor_finalize_output` — copies GPU result to host (`hipMemcpyD2H`)
- Calls `hipdnn_ep_tensor_free_input` — releases input staging buffer

**`inference_cleanup`:**
- Calls `hipdnn_ep_state_cleanup` — destroys handles, frees GPU memory

---

## 4. Post-MLIR: LLVM IR to Native DLL

The IR dump ends after `GenerateInterfacePass`, but there are 5 more steps that happen
**outside** the MLIR pass pipeline, inside `CompilerDriver::compileImpl`:

```
MLIR LLVM Dialect IR  (what ir_dump.mlir shows)
        │
        │  (1) translateToLLVMIR     — MLIR LLVM dialect → actual LLVM IR (llvm::Module)
        ▼
   LLVM IR  (in-memory)
        │
        │  (2) linkRuntime           — link EP runtime bitcode (function bodies for
        │                              wrap_miopenConvolutionForward, wrap_hipblasLtMatmul,
        │                              hipdnn_ep_tensor_prepare_input, etc.)
        ▼
   LLVM IR + runtime
        │
        │  (3) optimizeLLVMIR        — LLVM optimization passes (O2/O3)
        ▼
   Optimized LLVM IR
        │
        │  (4) compileToObject       — LLVM TargetMachine emits native x86-64 COFF
        ▼
   model.obj
        │
        │  (5) linkToDLL             — system linker produces Windows DLL,
        │                              exporting: inference_init, inference_compute,
        │                              inference_cleanup, inference_get_metadata_json
        ▼
   model.dll  (temporary, loaded at inference time)
```

**File:** `lib/Compiler/CompilerDriver.cpp`
```cpp
bool CompilerDriver::compileImpl(mlir::ModuleOp module,
                                 const std::string &output_path,
                                 const mlir::hip::CompilationOptionsT &options,
                                 std::string &error_message) {
  if (!runMLIRPasses(module, options, error_message))       // MLIR pipeline
    return false;

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message); // (1)
  if (!llvmModule) return false;

  if (!linkRuntime(llvmModule.get(), error_message))        // (2)
    return false;

  optimizeLLVMIR(llvmModule.get(), options.opt_level);      // (3)

  if (!compileToObject(llvmModule.get(), obj_path, error_message))  // (4)
    return false;

  if (!linkToDLL(obj_path, output_path, libraries, library_paths,   // (5)
                 export_symbols, error_message))
    return false;

  cleanupIntermediates(base_path);  // removes .ll and .obj
  return true;
}
```

Step (2) is critical: the LLVM IR from the MLIR pipeline only contains **declarations**
(e.g., `llvm.func @wrap_hipblasLtMatmul(...)`) without function bodies. `linkRuntime`
links in the **pre-compiled bitcode** of the EP runtime (`lib/Runtime/`), which provides
the actual implementations that call HIP, MIOpen, and hipBLASLt APIs.

**File:** `lib/Compiler/CompilerDriver.cpp` — the bridge to `LLVMBackend`:
```cpp
bool CompilerDriver::linkRuntime(llvm::Module *llvmModule,
                                 std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.linkRuntimeModule(llvmModule)) {
    error_message = "Failed to link runtime module";
    return false;
  }
  return true;
}
```

This delegates to `LLVMBackend::linkRuntimeModule` (see Section 4.1 Stage 2), which
reads the embedded `runtime_bc_data[]` byte array and merges it into the generated
module via `llvm::Linker`.

The intermediate `.ll` and `.obj` files are cleaned up after the DLL is produced.

### 4.1 Two-Stage Runtime Compilation

The runtime wrappers in `lib/Runtime/real/` (e.g., `matmul.cpp`, `miopen.cpp`, `gqa.cpp`,
`activation.cpp`, etc.) use a two-stage compilation strategy that enables cross-module
inlining between generated code and runtime code.

**Stage 1 — Build Time: C++ to Embedded Bitcode**

```
 lib/Runtime/real/*.cpp  (matmul.cpp, miopen.cpp, gqa.cpp, activation.cpp, ...)
         │
         │  clang -c -emit-llvm -std=c++17 -O2   (LLVM bitcode, NOT native object code)
         ▼
 Individual .bc files  (runtime_matmul.bc, runtime_miopen.bc, runtime_gqa.bc, ...)
         │
         │  llvm-link   (combines all ~19 modules into one)
         ▼
 runtime.bc
         │
         │  xxd.py   (Python script converts binary to C byte array)
         ▼
 runtime_ir_data.cpp
   extern "C" const unsigned char runtime_bc_data[] = { 0x42, 0x43, ... };
   extern "C" const size_t runtime_bc_data_size = 123456;
         │
         │  MSVC compiles into HipTargetLLVM.lib → linked into hip-compiler.dll
         ▼
 Embedded as a byte array inside hip-compiler.dll
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
            $<$<NOT:$<CONFIG:Debug>>:-O2>
            ${COMPILE_FLAGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
    ...
)
```

**Stage 2 — Model Compilation Time: Bitcode Linking**

When `CompilerDriver` compiles a model, `linkRuntime` reads the embedded byte array back
as LLVM bitcode and merges it into the generated IR using `llvm::Linker`:

**File:** `lib/Target/LLVM/LLVMBackend.cpp`
```cpp
extern "C" const unsigned char runtime_bc_data[];
extern "C" const size_t runtime_bc_data_size;

bool LLVMBackend::linkRuntimeModule(llvm::Module *destModule) {
  auto MemBuf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(runtime_bc_data), bcSize),
      "runtime.bc", /*RequiresNullTerminator=*/false);

  auto ModuleOrErr = llvm::parseBitcodeFile(MemBuf->getMemBufferRef(),
                                            destModule->getContext());
  // ...
  llvm::Linker linker(*destModule);
  linker.linkInModule(std::move(RuntimeModule));
  return true;
}
```

After linking, the LLVM optimizer (step 3) can **inline** runtime function bodies directly
into the generated compute graph code, eliminating call overhead. This is why bitcode
linking is used instead of a traditional static library — it enables cross-module
optimization that would be impossible at the native object code level.

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
               │     │  ONNX → HIP → LLVM → link             │
               │     │                                   InferenceState
               │     ▼                                        │
               │  model.dll  ◄─────────────────────────►  model.dll
               │  model.constants.bin                    inference_compute()
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

There are **three distinct plugin-loading steps**, each using a different mechanism:

| Step | Loader | Loaded DLL | API |
|------|--------|-----------|-----|
| 1 | ORT | `onnxruntime_morphizen_ep.dll` | EP V2 (`CreateEpFactories`) |
| 2 | MorphiZen | `hip-compiler.dll` | MorphiZen Plugin (`morphizen::Plugin::get`) |
| 3 | MorphiZen | `model.dll` (temporary) | MorphiZen Plugin (`morphizen::Plugin::create`) |

This three-level architecture keeps ORT, the compiler, and the generated code
cleanly separated with stable C ABI boundaries.

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
| MLIR pipeline + model DLL linking | `lib/Compiler/CompilerDriver.cpp` |
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
| **Bitcode Embedding & Linking** | |
| LLVM IR translation + bitcode linking | `lib/Target/LLVM/LLVMBackend.cpp` |
| DLL linking (lld-based) | `lib/Target/LLVM/DLLLinker.cpp` |
| Bitcode embed build (xxd.py) | `lib/Target/LLVM/CMakeLists.txt` |
| **Runtime (Host-side)** | |
| Load model DLL + call init/compute | `backend-mlir-compiler/custom-op-mlir/src/InferenceState.cpp` |
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
