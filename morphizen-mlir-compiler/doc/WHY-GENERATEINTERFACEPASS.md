<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Why GenerateInterfacePass is Necessary

## TL;DR

The pass cannot be eliminated because MLIR's type system requires compile-time rank information for memref structs. Moving the functions to runtime would require hardcoding MLIR's internal memref layout in C++ code, creating fragile ABI coupling and losing type safety.

**Recommendation:** Keep the pass—it serves as an adapter layer that correctly isolates MLIR knowledge from runtime code.

---

## Architecture Context: Runtime as Embedded Bitcode

**IMPORTANT:** The runtime is NOT a separate library. The runtime is compiled to LLVM bitcode and **merged into model.dll** at compilation time.

**How it works:**
1. Runtime C++ code compiled to bitcode (runtime.bc) at EP build time
2. Bitcode embedded in EP DLL as binary resource
3. During model compilation: `llvm::Linker::linkInModule()` merges runtime IR with generated IR
4. LLVM optimization inlines everything
5. Final model.dll contains both generated code and runtime code (fully optimized together)

**Key point:** Both "generated code" and "runtime code" end up in the **same DLL** after IR merging.

For complete architecture details, see [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md).

---

## Problem Statement

**Question:** Can we eliminate GenerateInterfacePass and move the three interface functions (`inference_init`, `inference_cleanup`, `inference_compute`) to runtime?

**Motivation:** Reduce per-model code generation, simplify the compilation pipeline, and centralize interface logic in maintainable C++ code instead of generated IR.

---

## Analysis: What is Model-Specific vs Generic?

The three interface functions have very different characteristics:

| Function | Lines | Generic % | Model-Specific Dependencies |
|----------|-------|-----------|----------------------------|
| `inference_init` | 40 | **95%** | Only `get_constant_count()` call |
| `inference_cleanup` | 32 | **100%** | Nothing—already pure runtime delegation |
| `inference_compute` | 490 | **40%** | Memref struct types (rank-dependent) |

### `inference_init` - Already 95% Generic

Current pattern is a thin wrapper that delegates to runtime:

```llvm
define i32 @inference_init(ptr %out_state) {
    %registry = call ptr @get_constant_registry()  ; Only model-specific call
    %result = call i32 @hipdnn_ep_state_init(ptr %out_state, ptr %registry)
    ret i32 %result
}
```

**Could easily move to runtime** if we expose `get_constant_registry()` address as metadata.

### `inference_cleanup` - Already 100% Generic

Pure delegation to runtime with zero model-specific logic:

```llvm
define i32 @inference_cleanup(ptr %state) {
    %result = call i32 @hipdnn_ep_state_cleanup(ptr %state)
    ret i32 %result
}
```

**Already runtime code**, just wrapped for C ABI export.

### `inference_compute` - 40% Generic, 60% Constrained

Breakdown of the 490 lines:

**Generic portions (could move to runtime):**
- Parsing `span_t`/`tensor_t` structures (lines 465-566)
- GPU buffer allocation (lines 612-677)
- Host-to-device transfer (lines 679-711)
- Device-to-host transfer + sync (lines 878-912)

**Model-specific portions (type-system constrained):**
- **Memref struct building (lines 712-876)** ← The blocker

---

## The Core Constraint: LLVM Type System

### Why Memref Building Cannot Move to Runtime

MLIR's memref type is a compile-time construct with rank-dependent struct layout:

```llvm
// 4D tensor memref
%memref_4d = type { ptr, ptr, i64, [4 x i64], [4 x i64] }
                                    ^^^^^^^^^^  ^^^^^^^^^^
                                    Size arrays depend on rank

// 3D tensor memref
%memref_3d = type { ptr, ptr, i64, [3 x i64], [3 x i64] }
                                    ^^^^^^^^^^  ^^^^^^^^^^
                                    Different type!
```

**Consequences:**

1. **Different ranks = different types** in LLVM IR
2. `@main` signature is statically typed:
   ```llvm
   declare i32 @main(ptr, ptr %memref_4x3x256x256xf32, ptr %memref_1000xf32)
   ```
3. Cannot use runtime dispatch for different ranks without abandoning type safety
4. LLVM's type checker enforces memref struct layout at compile time

### What the Type System Enforces

**With the pass (current):**
```llvm
; Type system ensures correctness
%input = alloca %memref_4d
%output = alloca %memref_2d
call i32 @main(ptr %state, ptr %input, ptr %output)
; ✅ Compiler verifies memref layout matches @main signature
```

**Without the pass (runtime construction):**
```cpp
// Runtime must manually construct memref (C++ pseudocode)
void* input_memref = malloc(memref_size_for_rank(4));
void* output_memref = malloc(memref_size_for_rank(2));

// Must hardcode MLIR's internal layout:
struct memref {
    void* allocated;     // offset 0
    void* aligned;       // offset 8
    int64_t offset;      // offset 16
    int64_t sizes[N];    // offset 24
    int64_t strides[N];  // offset 24 + 8*N
};

main(state, input_memref, output_memref);
// ❌ No compiler verification—manual pointer arithmetic
```

---

## Alternative Approach: Move init/cleanup to Runtime Bitcode

### How It Would Work

Since runtime bitcode is merged into model.dll anyway, we could move `inference_init` and `inference_cleanup` from the pass to runtime bitcode:

**Runtime bitcode would implement:**
```cpp
// In hipdnn_ep_runtime.cpp (compiled to bitcode)
extern "C" ConstantRegistry* get_constant_registry();  // Provided by generated code

extern "C" int32_t inference_init(void** out_state) {
    ConstantRegistry* registry = get_constant_registry();  // Call into generated code
    return hipdnn_ep_state_init(out_state, registry);
}

extern "C" int32_t inference_cleanup(void* state) {
    return hipdnn_ep_state_cleanup(state);  // Already pure delegation
}
```

**GenerateInterfacePass would only generate:**
```cpp
@inference_compute(...)   // Still needs pass (rank-specific memref building)
@get_constant_count()     // NEW - expose metadata for runtime
```

After `llvm::Linker::linkInModule()` → Same single model.dll with all functions merged

### Why Not Worth It?

Moving init/cleanup to runtime bitcode would create a **bidirectional dependency**:

**Current (one-way dependency):**
```
┌─────────────────────┐
│ Generated Code      │
│  - @main            │
│  - inference_*      │ ────calls───> Runtime bitcode
│  - get_constant_*   │                (helpers)
└─────────────────────┘
```

**Proposed (bidirectional dependency):**
```
┌─────────────────────┐
│ Generated Code      │ ────calls───> Runtime bitcode
│  - @main            │               (inference_init/cleanup)
│  - inference_compute│ <───calls────
│  - get_constant_*   │               Runtime needs get_constant_count()
└─────────────────────┘
```

**Problems:**
1. **Bidirectional coupling** - Runtime bitcode assumes `get_constant_count()` exists
2. **Split ownership** - 2 functions in runtime, 1 in pass (confusing)
3. **No benefit** - Both end up in same model.dll after IR merging anyway
4. **Same performance** - LLVM inlines everything regardless of where it comes from

---

## Trade-offs Analysis

### Moving init/cleanup to Runtime Bitcode

#### What Gets SIMPLER

✅ **Less OpBuilder code** - ~72 lines of IR generation eliminated from pass
✅ **C++ instead of OpBuilder** - Easier to read/modify init/cleanup logic
✅ **Single implementation** - Init/cleanup bugs fixed once in runtime.cpp

#### What Gets MORE COMPLEX

❌ **Bidirectional dependency** - Runtime bitcode calls back to generated code
❌ **Split ownership** - 2 interface functions in runtime, 1 in pass (confusing)
❌ **Larger generated API** - Must export `get_constant_count()` for runtime to call
❌ **No actual benefit** - Both end up in same model.dll after IR merging anyway

### Moving ALL Functions to Runtime (Including compute)

This would require runtime to build memref structs, which creates:

❌ **Runtime depends on MLIR internals** - Must hardcode memref layout:
```cpp
// Runtime must know this MLIR implementation detail:
struct memref {
    void* allocated;     // offset 0
    void* aligned;       // offset 8
    int64_t offset;      // offset 16
    int64_t sizes[N];    // offset 24
    int64_t strides[N];  // offset 24 + 8*N
};
```

❌ **Tight ABI coupling**
- Currently: Type system enforces memref layout correctness
- New: Manual pointer arithmetic (off-by-one errors = subtle bugs)
- If MLIR changes memref layout → runtime breaks silently

❌ **Leaky abstraction** - Memref layout is MLIR implementation detail
❌ **Lost type safety** - No compiler enforcement of layout correctness
❌ **Fragile maintenance** - MLIR updates require manual runtime C++ changes

### Architectural Clarity

**Current design (clean separation):**
```
┌─────────────────────────┐
│ GenerateInterfacePass   │ ← Knows MLIR types
│  - All 3 interface fns  │ ← Bridges MLIR ↔ Runtime
│  - Memref building      │ ← Type-safe IR generation
└─────────────────────────┘
           ↓ (one-way calls)
┌─────────────────────────┐
│ Runtime Bitcode         │ ← Knows GPU ops, NOT MLIR internals
│  - hipdnn_ep_state_*    │ ← Pure helpers
│  - wrap_* GPU ops       │
└─────────────────────────┘
           ↓ (both merged via llvm::Linker)
┌─────────────────────────┐
│    model.dll            │ ← Single DLL, fully inlined
└─────────────────────────┘
```

**Proposed design (split ownership):**
```
┌─────────────────────────┐
│ GenerateInterfacePass   │ ← Only inference_compute
│  - 1 interface function │ ← Exports get_constant_count
└─────────────────────────┘
           ↕ (bidirectional calls!)
┌─────────────────────────┐
│ Runtime Bitcode         │ ← inference_init/cleanup
│  - Calls get_constant_* │ ← 2 interface functions here
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│    model.dll            │ ← Same result, more complexity
└─────────────────────────┘
```

---

## Recommendation: Keep All 3 Functions in the Pass

### Why the Current Design is Correct

**1. Clear Ownership**
- GenerateInterfacePass owns ALL interface functions (not split between pass and runtime)
- Clean responsibility: Pass = interface layer, Runtime = helpers
- Easy to understand: all interface generation in one place

**2. One-Way Dependency**
- Generated code calls runtime helpers (simple, stable contract)
- No callbacks from runtime to generated code
- Runtime bitcode is truly generic (doesn't assume specific generated functions exist)

**3. Same Final Result**
- Both approaches end up in same model.dll after `llvm::Linker::linkInModule()`
- LLVM optimization inlines everything regardless of source
- Zero performance difference

**4. Minimal Cost**
- ~570 lines of generated IR per model (40 init + 490 compute + 32 cleanup)
- Trivial compared to full model IR (thousands of lines)
- OpBuilder code is ~150 lines in the pass (not a maintenance burden)

**5. Type Safety for compute**
- `inference_compute` MUST stay in pass (memref building requires MLIR type system)
- If compute stays, keeping init/cleanup with it makes sense (cohesive interface)

### What the Pass Actually Does

Think of GenerateInterfacePass as a **type-safe adapter**:

```
┌─────────────────────────────────────────┐
│ C API World (Runtime)                   │
│ - Dynamic span_t/tensor_t structures    │
│ - Runtime rank information              │
│ - Generic GPU operations                │
└─────────────────────────────────────────┘
                    ↕
        ┌───────────────────────┐
        │ GenerateInterfacePass │  ← Adapter Layer
        │ - Parses dynamic data │
        │ - Builds typed memrefs│
        │ - Type-safe bridge    │
        └───────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│ MLIR World (Generated Code)             │
│ - Statically-typed memref structs       │
│ - Compile-time rank information         │
│ - Type-checked @main signature          │
└─────────────────────────────────────────┘
```

**This is the right architectural pattern.**

### Potential Refinement (Optional)

If you want to reduce generated IR size, consider factoring generic helpers into runtime bitcode:

**Move to runtime:**
```cpp
// Runtime provides (as bitcode functions):
TensorMetadata parse_first_tensor(span_t* span);
GpuBuffer allocate_and_transfer_h2d(void* host, size_t bytes, hipStream_t);
void transfer_d2h_and_free(GpuBuffer gpu, void* host, size_t bytes, hipStream_t);
```

**Pass generates only:**
- Rank-specific memref struct building (unavoidable)
- Call to `@main` with typed memrefs
- Thin coordination logic

**Impact:**
- Reduces `inference_compute` from 490 lines → ~150 lines
- Keeps type safety intact
- Still zero runtime overhead after LLVM inlining
- Cleaner separation of generic vs. model-specific code

**But:** Even this refinement is optional—the current design is already sound.

---

## Conclusion

**Answer to "Can we eliminate GenerateInterfacePass?"**

**No.** The pass is necessary because:

1. **MLIR is a statically-typed IR** - Memref structs require compile-time rank information
2. **Type safety is valuable** - Compiler enforces correctness automatically
3. **Clean architecture** - Pass owns all interface generation (not split with runtime)
4. **Minimal cost** - ~570 lines of IR per model is negligible
5. **One-way dependency** - Generated code → Runtime helpers (simple contract)

**Answer to "Can we move init/cleanup to runtime bitcode?"**

**Not worth it.** Even though both end up in the same model.dll:

1. **Bidirectional dependency** - Runtime would call back to `get_constant_count()`
2. **Split ownership** - 2 functions in runtime, 1 in pass (confusing)
3. **No benefit** - Both merge into same DLL, same inlining, same performance
4. **Current design cleaner** - All interface functions in one place

**Final Recommendation:** Keep all 3 interface functions in GenerateInterfacePass. It's a well-designed adapter layer that:
- Serves as the bridge between MLIR types and C API
- Maintains clear separation (MLIR knowledge in pass, GPU ops in runtime)
- Uses type system for compile-time verification
- Results in zero runtime overhead (LLVM inlines everything)
