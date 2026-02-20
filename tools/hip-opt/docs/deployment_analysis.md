# Deployment Analysis: From Toy GEMM to LLM Inference

This document analyzes the deployment strategies for integrating the HIP MLIR pipeline
into ONNX Runtime as an Execution Provider, covering the journey from a standalone
GEMM demo to full LLM model inference.

---

## 1. Background: Why AOT Produces Unsigned Artifacts

### What is AOT?

AOT (Ahead-of-Time) means the code is fully compiled before execution. The current
pipeline uses `hip-compiler` to compile MLIR directly to a DLL:

```
MLIR → hip-compiler → model.dll + model.lib
```

The `hip-compiler` tool performs all steps internally (MLIR lowering, LLVM IR
translation, native code generation, and linking with `hip_runtime_static.lib`).
The resulting `.dll` is loaded by a driver program at runtime.

This contrasts with JIT (Just-in-Time), where compilation and execution happen in the
same process at runtime.

### Why is it unsigned?

On Windows, code signing works as follows:

1. You need a **code signing certificate** (purchased from a CA like DigiCert, or issued
   by an enterprise PKI)
2. You sign `.exe` / `.dll` files using `signtool.exe`
3. Windows, antivirus (AV), and EDR software check this signature to determine trust

Our pipeline uses `hip-compiler` to produce a DLL from MLIR. There is no signing step,
so the output is **unsigned**. Compare:

| Binary | Signed? | Why |
|---|---|---|
| `amdhip64.dll` | Yes (AMD-signed) | AMD signs it with their certificate |
| `onnxruntime.dll` | Yes (Microsoft-signed) | Microsoft signs it |
| Our `model.dll` | **No (unsigned)** | We only compiled and linked -- no signing step |

Unsigned executables on enterprise machines may be:
- Flagged or quarantined by antivirus
- Blocked by EDR (CrowdStrike, Carbon Black, etc.)
- Rejected by AppLocker / WDAC policies

---

## 2. From Standalone EXE to ONNX Runtime Integration

### Current state: DLL-based compilation

The current pipeline uses `hip-compiler` to produce a `.dll` from MLIR. Each test has
its own C++ driver that links against the generated DLL's import library. The driver
declares the MLIR entry function with `__declspec(dllimport)` and is compiled/linked
via `cl.exe` against the generated `.lib`.

### Target state: ONNX Runtime Execution Provider

ONNX Runtime's architecture loads Execution Providers (EPs) into its own process:

```
Application (.exe)
  └── onnxruntime.dll
        └── Execution Providers
              ├── CUDAExecutionProvider    (NVIDIA)
              ├── DmlExecutionProvider     (DirectML)
              └── HipDnnExecutionProvider  (ours)
```

Our EP must run **inside** the ONNX Runtime process, not as a separate `.exe`. When the
EP needs to execute a subgraph, the AOT approach would mean:

```
Runtime discovers subgraph → MLIR compile → hip-compiler → kernel.dll → LoadLibrary() → call
```

This dynamically generates an **unsigned `.dll`** at runtime, which triggers the same
AV/EDR concerns.

---

## 3. Do We Actually Need Runtime Code Generation?

### The key question

Can we pre-compile everything before deployment, or must we generate code at runtime?

### For the current GEMM demo: no runtime generation needed

Our generated code is just host-side orchestration calling hipDNN library functions:

```llvm
define void @run_gemm(ptr %A, ptr %B, ptr %C, i64 %M, i64 %K, i64 %N) {
  %handle = call ptr @hipCreateHandle()
  call void @hip_gemm_f32(ptr %A, ptr %B, ptr %C, i64 %M, i64 %K, i64 %N)
  call void @hipDestroyHandle(ptr %handle)
  ret void
}
```

The shapes (`M`, `K`, `N`) are already parameterized. The actual GPU kernel selection
happens inside hipDNN (`amdhip64.dll`), which is AMD-signed. There is no need to
generate different code for different shapes — the same compiled function handles all
of them.

### For LLM inference (e.g., LLaMA): still mostly static

A Transformer model like LLaMA has a fixed, repetitive graph structure:

```
Each Transformer Block (×32 for LLaMA-7B):
  ├── RMSNorm
  ├── Attention
  │   ├── Q/K/V Projection  (GEMM)
  │   ├── RoPE
  │   ├── Softmax(Q·Kᵀ/√d)  (GEMM + Softmax)
  │   └── Output Projection (GEMM)
  ├── RMSNorm
  └── FFN
      ├── Gate Projection (GEMM)
      ├── Up Projection   (GEMM)
      ├── SiLU activation
      └── Down Projection (GEMM)
```

The graph structure is **fully known at model load time**. What varies at inference
time is only the shapes (batch size, sequence length), but as long as shapes are
passed as parameters (as we already do), the compiled code can handle all of them.

Even LLM-specific concerns do not require runtime code generation:

| Concern | Dynamic? | Solution |
|---|---|---|
| Prefill vs Decode phase | Different shapes (seq_len=N vs seq_len=1) | Parameterized shapes, or pre-compile 2 versions |
| Variable batch size | Different M dimension | Already parameterized |
| KV cache growth | Size changes, but the operation is the same | Same kernel, different pointer/size arguments |
| Different models (7B vs 70B) | Different hidden dimensions | Known at model load time |
| Mixture of Experts (MoE) | Dynamic routing | Each expert's kernel is pre-compilable; only routing is dynamic |

---

## 4. Graph-Level Fusion and Compile-Once-Cache Strategy

### When runtime compilation becomes useful

If the MLIR frontend performs **graph-level fusion** (e.g., fusing RMSNorm + Q_proj
into a single kernel), the compiled code depends on the specific model's graph structure.
Different models produce different fused subgraphs, so we cannot pre-compile a universal
set of kernels.

### Compile-once + cache solves this

The model graph is fully determined at load time, so we can compile on first load and
cache the result:

```
First load of LLaMA-7B:
  1. Read ONNX model → get full computation graph
  2. Run optimization passes → discover fusion opportunities
  3. MLIR codegen → generate code for fused subgraphs
  4. Compile → cache to disk
     ~/.cache/hip_ep/llama7b_<graph_hash>.bc

Second load (same model):
  1. Read ONNX model → compute graph hash → match
  2. Cache hit → load .bc directly
  3. Skip compilation, proceed to execution
```

The cache key is a hash of the model's graph structure. Recompilation is only needed when:
- The model changes
- The compiler (`hip-compiler`) is upgraded (optimization strategies may differ)
- The target GPU changes (different tiling strategies)
- Compilation options are modified

All of these are **low-frequency events** that do not affect normal inference.

---

## 5. LLVM Bitcode (.bc) Files

### What is a .bc file?

LLVM IR has two equivalent representations:

| Format | Extension | Content | Analogy |
|---|---|---|---|
| Text (human-readable) | `.ll` | Readable LLVM IR assembly | Like `.json` |
| Bitcode (binary) | `.bc` | Compact binary encoding of the same IR | Like `.bson` |

They are interconvertible:

```bash
llvm-as gemm.ll -o gemm.bc    # text → binary
llvm-dis gemm.bc -o gemm.ll   # binary → text
```

A `.bc` file is **not machine code**. It cannot be executed directly by the CPU. It is
a data file containing an intermediate representation — no different from a configuration
file or a serialized model from the OS's perspective.

---

## 6. Three Execution Strategies for Compiled IR

Dynamic compilation (compiling at model load time) does not dictate the execution
method. The compilation output and execution method are independent choices:

```
                      Compilation Output
              ┌──────────────┬──────────────┬──────────────┐
              │  .dll / .exe │  Native code │  .bc (IR)    │
              │  (on disk)   │  (in memory) │  (on disk)   │
              └──────┬───────┴──────┬───────┴──────┬───────┘
                     │              │              │
              Execution:     Execution:      Execution:
              OS loads &     CPU jumps to    Interpreter
              runs binary    memory page     simulates IR
              (LoadLibrary)  (VirtualProtect)  (switch-case)
                     │              │              │
                = AOT to disk   = JIT        = Interpreter
```

### Strategy A: Compile to DLL (current approach for standalone tests)

```cpp
// hip-compiler does this internally:
void compile(const char* mlirFile, const char* outputDll) {
    auto module = parseMLIR(mlirFile);
    runPassPipeline(module);              // --convert-hip-to-llvm, etc.
    injectDllExport(module);              // mark entry functions
    auto llvmIR = translateToLLVMIR(module);
    auto obj = compileToObject(llvmIR);   // LLVM TargetMachine
    linkToDLL(obj, outputDll);            // lld-link or link.exe + hip_runtime_static.lib
}

// Driver code (e.g. main_attention.cpp):
extern "C" __declspec(dllimport) void attention(...);
// Compiled and linked against attention.lib + amdhip64.lib by cl.exe
```

**Pros:** Best execution performance (native speed). Clean separation between model DLL
and driver. Each test has its own driver `.exe` linked against the model `.lib`.
**Cons:** Requires a linker on target machine. Produces unsigned DLL. `LoadLibrary` of
unsigned DLL may be blocked by security policies.

### Strategy B: JIT (LLVM ORC / MLIR ExecutionEngine)

Compiles LLVM IR to native machine code in memory and executes it directly.

**Pros:** No files on disk. Native execution speed.
**Cons:** Calls `VirtualProtect(PAGE_EXECUTE_*)` to make memory pages executable.
AV/EDR treats this as highly suspicious (same technique used by malware).

### Strategy C: Interpreter

Reads `.bc` as data and simulates each IR instruction via a C++ switch-case loop:

```cpp
// Pseudocode: how the interpreter works
while (hasNextInstruction()) {
    Instruction &I = getNext();
    switch (I.getOpcode()) {
        case Instruction::Call:
            // Look up function pointer for @hip_gemm_f32
            // Call the pre-linked C++ function directly
            break;
        case Instruction::Ret:
            return;
        // ... other opcodes
    }
}
```

From the OS's perspective, this is just a normal program reading a data file and doing
computation — identical to reading a JSON config file.

**Pros:** No unsigned binaries. No executable memory. Lowest AV/EDR risk. No linker
needed on target. `.bc` files are harmless data.
**Cons:** 100-1000x slower per interpreted instruction (irrelevant for our use case —
see below).

### Why interpreter performance does not matter for us

The interpreted code is only host-side orchestration — a handful of function calls
(`hipCreateHandle`, `hip_gemm_f32`, `hipDestroyHandle`). The actual matrix computation
runs on the GPU, completely independent of how the host code executes.

| Phase | Where it runs | Time |
|---|---|---|
| Interpreting ~10-20 IR instructions | CPU (interpreter) | ~microseconds |
| GPU GEMM computation | GPU hardware | ~milliseconds |
| Total overhead from interpreter | | < 0.1% |

---

## 7. Recommended Architecture

### Summary of tradeoffs

```
Performance:       AOT DLL ≈ JIT >> Interpreter (per instruction)
Actual inference:  AOT DLL ≈ JIT ≈ Interpreter  (GPU-bound, host overhead negligible)
Deployment:        Interpreter > JIT > AOT DLL
Security/AV:       Interpreter > AOT DLL > JIT
```

### Recommended approach: Compile-once + Cache + Interpreter

```
Model Load (first time):
  ONNX model → graph optimization/fusion → MLIR → hip-compiler → .bc → cache to disk

Model Load (subsequent):
  ONNX model → compute graph hash → cache hit → load .bc

Inference:
  Load .bc → Interpreter executes host orchestration → GPU runs actual computation
```

This approach:
- Produces **no unsigned executables** (`.bc` is pure data)
- Requires **no executable memory pages** (no `VirtualProtect`)
- Has **negligible performance overhead** (GPU-bound workload)
- Supports **graph-level fusion** (different models produce different `.bc` files)
- Works with **compile-once + cache** (recompilation only on model/compiler/GPU change)
- Is **deployment-friendly** (single signed executable + data files)
