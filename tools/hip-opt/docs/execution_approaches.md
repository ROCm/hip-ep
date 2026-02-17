# Understanding the Problem

Your current pipeline is AOT (Ahead-of-Time):

```
MLIR → hip-opt → mlir-translate → llc → .obj → link.exe → .exe
```

The two alternative in-memory execution paths you've considered both have problems:

- **Dropping unsigned DLLs** - suspicious to AV/endpoint protection, may fail on locked-down machines
- **JIT from memory** (e.g., MLIR ExecutionEngine / ORC JIT) - requires `VirtualProtect(PAGE_EXECUTE_*)`, which is equally suspicious since you're making data memory executable

The LLVM interpreter is a legitimate third path. Here's the full picture.

---

## Three Execution Approaches

### Approach 1: AOT Compiler (Current)

What you have now. Produces `.obj` → `.exe` on disk.

### Approach 2: JIT (MLIR ExecutionEngine / LLVM ORC JIT)

MLIR has `mlir::ExecutionEngine` which wraps LLVM's ORC JIT. It compiles LLVM IR to native machine code in memory and runs it. This does **NOT** solve your problem because ORC JIT calls `VirtualProtect` to make memory pages executable (RWX → RX), which is exactly concern #2.

### Approach 3: LLVM Interpreter (`lli -force-interpreter`)

LLVM has a built-in bytecode interpreter (`llvm::Interpreter` class) that executes LLVM IR instruction-by-instruction without ever generating native code. All execution happens in regular data memory. No executable pages needed.

---

## Detailed Comparison

| Aspect | AOT Compiler | JIT (ExecutionEngine) | LLVM Interpreter |
|---|---|---|---|
| Executable on disk | Yes (.exe/.dll) | No | No |
| Executable memory pages | N/A (runs as normal exe) | Yes (PAGE_EXECUTE_*) | No (pure data memory) |
| Unsigned artifacts | .exe/.dll (unsigned) | None on disk | .bc file (data only, not executable) |
| AV/EDR suspicion | Medium (unsigned binary) | High (RWX memory) | Low (just reads data) |
| Performance overhead | None (native speed) | ~1-2x (one-time JIT cost) | 100-1000x on interpreted code |
| Practical GEMM perf impact | None | Negligible | Negligible (see below) |
| External function support | Full (native linking) | Full (symbol resolution) | Needs libffi on Windows |
| Maintenance burden | Low (standard toolchain) | Medium | Medium-High |
| LLVM instruction coverage | Full | Full | Partial (not all intrinsics) |

---

## The Critical Performance Insight

For your specific use case, the interpreter performance penalty is almost irrelevant. Here's why - look at what the MLIR code actually does:

```mlir
// Test: hip.gemm lowering
// ...
module {
  func.func @run_gemm(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr,
                      %M: index, %K: index, %N: index) {
    %handle = hip.create_handle() : !hip.handle
    hip.gemm(%handle, %A, %B, %C, %M, %K, %N)
        : (!hip.handle, !llvm.ptr, !llvm.ptr, !llvm.ptr, index, index, index)
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
```

The LLVM IR being interpreted is just **host-side orchestration** — a handful of function calls (`hipCreateHandle`, `hip_gemm_f32`, `hipDestroyHandle`). The actual matrix math runs on the GPU hardware, completely independent of how the host code executes. Even if the interpreter is 1000x slower per instruction, you're interpreting maybe 10-20 instructions total, which takes microseconds. The GPU GEMM takes milliseconds.

---

## Implementation: Custom LLVM Bitcode Runner

The most practical approach is to build a single trusted executable (`hip-runner.exe`) that:

1. Embeds the LLVM Interpreter
2. Statically links your runtime functions (`hip_gemm_f32`, etc.)
3. Dynamically loads `amdhip64.dll` (AMD-signed, safe)
4. Reads `.bc` files as data input and interprets them

The deployment model becomes:

```
┌─────────────────────────────────────────┐
│  hip-runner.exe  (signed, trusted)      │
│  ├── LLVM Interpreter (linked in)       │
│  ├── hip_gemm_f32()  (linked in)        │
│  └── links to amdhip64.dll (AMD-signed) │
│                                         │
│  Input: gemm.bc  (data file, not code)  │
└─────────────────────────────────────────┘
```

No unsigned DLLs. No executable memory. The `.bc` file is just data — like a configuration or model file.

---

## New Pipeline

```
MLIR → hip-opt → mlir-translate → llvm-as → gemm.bc (bitcode, data)
                                                │
                              hip-runner.exe loads & interprets
```
