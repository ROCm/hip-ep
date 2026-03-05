<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native DLL vs LLVM IR Storage - Detailed Comparison

**Date:** 2026-02-13
**Document Type:** Design
**Review Status:** Self-Reviewed

---

## Overview

This document compares two approaches for storing compiled models in [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html):

1. **Native DLL:** Compile to platform-specific machine code
2. **LLVM IR:** Store LLVM Intermediate Representation, JIT compile at runtime

Both approaches are viable. This comparison helps inform the architectural decision.

Key portability difference:
- **Native DLL:** OS-specific binaries (Windows .dll vs Linux .so)
- **LLVM IR:** OS-portable bitcode (same file works on Linux and Windows)

Note: Both approaches are GPU-portable. Generated code calls ROCm library APIs (MIOpen, rocBLAS), which handle GPU architecture differences. No custom HIP kernels are embedded.

---

## Quick Comparison

| Aspect | Native DLL | LLVM IR |
|--------|-----------|---------|
| **OS Portability** | ❌ OS-specific (.dll vs .so) | ✅ OS-portable (same bitcode) |
| **GPU Portability** | ✅ Calls ROCm APIs (arch-independent) | ✅ Calls ROCm APIs (arch-independent) |
| **Runtime Dependencies** | ✅ [MemoryModule](https://github.com/fancycode/MemoryModule) only | ❌ LLVM JIT (~5-10 MB per [research](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg)) |
| **Runtime Memory** | ~2x currently, ~1x possible ([optimization](../EPCONTEXT-MEMORY-OPTIMIZATION.md)) | ~2x (IR + compiled code, no optimization path) |
| **Startup Time** | ✅ No JIT delay | ❌ JIT compilation required |
| **JIT Overhead** | ✅ Zero (pre-compiled) | ❌ Recompiles every process start |
| **Code Optimization** | ✅ Full LLVM passes at compile-time | ⚠️ Limited by JIT time budget |
| **Primary Advantage** | Performance (eliminate JIT overhead) | OS portability (single EPContext for Linux/Windows) |
| **Primary Disadvantage** | OS-specific deployment | Runtime overhead (JIT + memory) |

**Key Finding:** OS portability is the only differentiator. Both are GPU-portable (call ROCm library APIs).

---

## Approach 1: Native DLL

### How It Works

```
Compile-time:
  ONNX → MLIR → LLVM IR → Native x64 Machine Code → DLL
  Store DLL bytes in EPContext

Runtime:
  Extract DLL bytes → Load via MemoryModule → Execute directly
```

### Trade-offs

| Aspect | Native DLL |
|--------|-----------|
| **Portability** | OS-specific (.dll vs .so) |
| **Runtime deps** | [MemoryModule](https://github.com/fancycode/MemoryModule) only |
| **Runtime memory** | ~2x currently, ~1x optimization possible (per [EPCONTEXT-MEMORY-OPTIMIZATION.md](../EPCONTEXT-MEMORY-OPTIMIZATION.md)) |
| **Startup time** | No JIT compilation delay |
| **JIT overhead** | Zero (pre-compiled) |
| **Code optimization** | Full LLVM optimizations at compile-time |
| **Validation** | Requires OS detection at runtime |

### Strengths

- **Eliminates JIT overhead** - Primary goal per ARCHITECTURE.md
- **Minimal runtime dependencies** - MemoryModule only vs LLVM JIT libraries (per [ClamAV comparison](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg))
- **No compilation delay** - Pre-compiled machine code loads directly
- **Full optimizations** - All LLVM optimization passes run at compile-time
- **Potential memory optimization** - Can execute directly from mmap without copy (~1x vs ~2x, per [EPCONTEXT-MEMORY-OPTIMIZATION.md](../EPCONTEXT-MEMORY-OPTIMIZATION.md))

### Weaknesses

- **OS-specific binaries** - Separate builds for Windows (.dll) and Linux (.so)
- **Runtime memory consumption** - Current MemoryModule approach uses ~2x memory (optimization to ~1x possible but not implemented, per [EPCONTEXT-MEMORY-OPTIMIZATION.md](../EPCONTEXT-MEMORY-OPTIMIZATION.md))
- **Runtime validation** - Must detect OS at runtime

### Sub-Decision: Memory DLL Loading vs Disk Files

If Native DLL is chosen, there's a secondary decision about how to load the DLL:

#### Option A: Memory Loading (via MemoryModule)

**How it works:**
- DLL stored as bytes in EPContext
- Load directly from memory buffer using [MemoryModule](https://github.com/fancycode/MemoryModule)
- No disk I/O required

**Trade-offs:**

| Aspect | Impact |
|--------|--------|
| **Deployment** | Single ONNX file (DLL embedded) |
| **Startup** | No disk I/O overhead |
| **Dependencies** | Requires [MemoryModule](https://github.com/fancycode/MemoryModule) library |
| **Security** | DLL embedded in ONNX (user must trust model file) |
| **WebNN compatibility** | Works with no-disk-access constraint |

**Strengths:**
- Cleaner deployment: No temporary files, no disk permissions needed
- Fast loading: Parse PE and map to memory
- WebNN compatibility: Required for no-disk-access constraint
- Simple integration: MemoryModule is lightweight, [MPL 2.0 license](https://github.com/fancycode/MemoryModule/blob/master/LICENSE.txt)

**Weaknesses:**
- Additional dependency (though small)
- DLL embedded in model file (harder for security scanning)

#### Option B: Disk Files

**How it works:**
- DLL stored as separate file on disk
- Load using standard OS loader
- ONNX model references DLL path

**Trade-offs:**

| Aspect | Impact |
|--------|--------|
| **Deployment** | ONNX file + separate DLL file(s) |
| **Startup** | Requires disk access |
| **Dependencies** | OS loader (zero additional deps) |
| **Security** | Separate DLL (easier to scan/verify) |
| **WebNN compatibility** | Violates no-disk-access constraint |

**Strengths:**
- Zero additional dependencies (use OS loader)
- Easier security scanning (separate DLL file)
- Simpler implementation (standard LoadLibrary)

**Weaknesses:**
- More complex deployment (multiple files)
- Requires disk permissions
- Path management complexity
- Incompatible with WebNN no-disk-access requirement

#### Recommendation

**Use Memory Loading (Option A)** for WebNN compatibility (required - no disk access allowed).

Option B (Disk Files) only viable for non-WebNN deployments where disk access is permitted and security scanning of separate DLL files is required.

---

## Approach 2: LLVM IR

### How It Works

```
Compile-time:
  ONNX → MLIR → LLVM IR (bitcode)
  Store LLVM IR in EPContext

Runtime:
  Extract IR → JIT compile to native code → Execute
```

### Trade-offs

| Aspect | LLVM IR |
|--------|---------|
| **Portability** | OS-portable (same bitcode on Linux/Windows) |
| **Runtime deps** | LLVM JIT statically linked (~5-10 MB per [ClamAV example](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg), 50-150 MB if dynamically linked) |
| **Runtime memory** | ~2x (IR bitcode + JIT compiled code both in memory) |
| **Startup time** | JIT compilation required (time depends on model size) |
| **JIT overhead** | Recompiles on every process start |
| **Code optimization** | JIT optimizations limited by time budget |
| **Validation** | None (ROCm runtime handles GPU detection) |

### Strengths

- **OS portability** - Single bitcode works on Linux and Windows (per [LLVM bitcode format](https://en.wikipedia.org/wiki/LLVM))
- **No OS detection** - Same bitcode loads on any OS
- **Flexibility** - Could apply runtime optimizations

### Weaknesses

- **Reintroduces JIT overhead** - Contradicts primary goal of EPContext (eliminate recompilation)
- **Runtime memory consumption** - IR bitcode + JIT compiled code both in memory (~2x, IR cannot be freed while EPContext loaded)
- **Increased deployment binary size** - Statically linking LLVM JIT adds ~5-10 MB to executable ([ClamAV example: +5-8 MB](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg), dynamically linked requires 50-150 MB of libraries)
- **Slower startup** - Compilation delay on every process start
- **Limited JIT optimizations** - Time budget constraints prevent full optimization

---

## Hybrid Approach: Support Both

### How It Works

EPContext stores a format tag indicating DLL or IR:

```
Runtime:
  Read format tag from EPContext
  if (format == NATIVE_DLL):
    Load via MemoryModule
  else if (format == LLVM_IR):
    JIT compile then execute
```

### Trade-offs

| Aspect | Impact |
|--------|--------|
| **Flexibility** | Maximum - choose format per model |
| **Runtime complexity** | Doubled - maintain both code paths |
| **Testing burden** | Both paths need coverage |
| **Deployment size** | Full - ship MemoryModule + LLVM libraries |
| **Use cases** | Development (IR) vs Production (DLL) |

### Potential Use Cases

- **Development builds:** Use IR for flexibility during testing
- **Production builds:** Use Native DLL for performance
- **Cross-platform testing:** Use IR to test logic on different GPUs
- **Deployed applications:** Use Native DLL for optimal performance

---

## Decision Criteria

### Choose Native DLL if:

1. **Performance is critical** - Eliminate JIT compilation overhead
2. **Minimal runtime dependencies** - Avoid shipping LLVM JIT libraries
3. **EPContext philosophy alignment** - Goal is to eliminate recompilation overhead

### Choose LLVM IR if:

1. **OS portability is critical** - Need single EPContext file for Linux and Windows
2. **JIT overhead acceptable** - Startup delay is tolerable

### Choose Hybrid if:

1. **Different use cases exist** - Development vs production have different needs
2. **Migration path needed** - Want to transition gradually
3. **Maximum flexibility required** - Let users choose per-model

---

## Recommendation Framework

Consider your primary goal:

| Primary Goal | Recommendation |
|--------------|----------------|
| Eliminate JIT overhead (per ARCHITECTURE.md "Problem") | **Native DLL** |
| OS portability (Linux/Windows from single EPContext) | **LLVM IR** |
| Support both development and production | **Hybrid** |
| Minimize runtime dependencies | **Native DLL** |
| Minimize runtime memory (if optimization implemented) | **Native DLL** |
| Fastest inference startup | **Native DLL** |

---

## Implementation Impact

### If Native DLL Chosen:

**Required Components:**
- MemoryModule integration for in-memory DLL loading
- GPU architecture detection at runtime
- Architecture-specific compilation pipeline

**Optional Enhancements:**
- Multi-architecture DLL packaging (single EPContext with multiple DLLs)
- Fallback to CPU if architecture mismatch detected

### If LLVM IR Chosen:

**Required Components:**
- LLVM JIT infrastructure at runtime
- LLVM library deployment strategy
- JIT compilation error handling

**Optional Enhancements:**
- JIT compilation caching to disk
- Incremental compilation to reduce startup time

### If Hybrid Chosen:

**Required Components:**
- Format detection in EPContext loading
- Both Native DLL and LLVM IR code paths
- Dual testing infrastructure

**Optional Enhancements:**
- Auto-selection based on environment (dev vs prod)
- Performance telemetry to guide format choice

---

## Open Questions

1. **Storage budget:** What is the acceptable EPContext size limit?
2. **Startup time budget:** What is the acceptable inference initialization delay?
3. **Deployment scenarios:** Single GPU type or multiple?
4. **Development workflow:** How important is cross-architecture testing?
5. **Runtime environment:** Can we assume LLVM libraries available, or must we minimize dependencies?

---

## References

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Main architecture document
- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
- [MemoryModule](https://github.com/fancycode/MemoryModule) - In-memory DLL loading
- [LLVM JIT Tutorial](https://llvm.org/docs/tutorial/) - LLVM JIT compilation

---
