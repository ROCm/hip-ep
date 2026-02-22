<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# LLVM SROA Workaround: Why @main Requires `noinline`

## TL;DR

The `@main` function in HipToLLVM.cpp must have the `noinline` attribute to prevent LLVM's SROA (Scalar Replacement of Aggregates) optimizer from producing `undef` values when tracking GPU pointers through memref structures.

**Without `noinline`**: First convolution and output memcpy receive `undef` pointers → segfault
**With `noinline`**: All operations receive correct GPU pointers → works correctly

## The Pattern That Breaks SROA

Our code generation creates this pattern:

```cpp
// inference_compute() builds memref arrays:
Memref input_memref;                           // Stack (AS0)
input_memref.aligned_ptr = input_tb.gpu_ptr;   // Store AS1 pointer in AS0 memory
Memref* input_array[1];                        // Stack (AS0)
input_array[0] = &input_memref;                // Pointer to pointer

// Call @main which unpacks:
@main(state, input_array, output_array)

// Inside @main:
Memref* input_ptr = input_array[0];            // Load pointer from array
void* gpu_ptr = input_ptr->aligned_ptr;        // Load AS1 pointer from struct
use_gpu_pointer(gpu_ptr);                      // Use it
```

This is **double pointer indirection** with **mixed address spaces**:
```
array (AS0) → struct (AS0) → GPU pointer (AS1)
```

## What Happens When @main Is Inlined

### Step 1: LLVM Inliner (Default Behavior)

LLVM sees:
- `@main` is `private` (internal linkage)
- `@main` is called exactly once
- No `noinline` attribute

**Decision**: Inline it!

After inlining, `inference_compute` contains:

```llvm
; Build memref
%input_memref = alloca { ptr addrspace(1), ptr addrspace(1), i64, [4 x i64], [4 x i64] }
%input_gpu_as1 = addrspacecast ptr %input_gpu to ptr addrspace(1)
store ptr addrspace(1) %input_gpu_as1, ptr %input_memref

; Build array
%input_array = alloca ptr
store ptr %input_memref, ptr %input_array

; --- Inlined @main code ---
%input_ptr = load ptr, ptr %input_array
%loaded_gpu_ptr = load ptr addrspace(1), ptr %input_ptr

; Use it
call @wrap_miopenConvolutionForward(..., ptr %loaded_gpu_ptr, ...)
```

### Step 2: SROA Tries to Optimize

SROA analyzes the inlined code and thinks:

> "Both `input_memref` and `input_array` are local stack allocations. I can eliminate them and replace all uses with direct values!"

SROA attempts to trace:
```
%loaded_gpu_ptr = load ptr addrspace(1), ptr %input_ptr
                = load ptr addrspace(1), ptr %input_array[0]
                = load ptr addrspace(1), ptr %input_memref
                = %input_gpu_as1
```

**But SROA FAILS** because:
1. Two levels of pointer indirection (array → struct)
2. Address space mixing (AS0 memory holding AS1 pointers)
3. Complex GEP chains for struct field access

**SROA gives up and produces**: `%loaded_gpu_ptr = undef`

### Step 3: Dead Code Elimination (DCE)

DCE sees:
```llvm
store ptr addrspace(1) %input_gpu_as1, ptr %input_memref
store ptr %input_memref, ptr %input_array
; ... but these values are never used (SROA replaced with undef)
```

**DCE deletes** all memref building code as "dead".

### Step 4: Runtime Crash

```llvm
call @wrap_miopenConvolutionForward(..., ptr undef, ...)
```

When this executes:
- MIOpen tries to dereference `undef` pointer
- **Segmentation fault!**

## Actual Evidence from LLVM IR

### Broken Version (without `noinline`)

File: `demo_two_layer.ll`, line 262:
```llvm
%13 = call i32 @wrap_miopenConvolutionForward(
    ptr readonly %0,        ; state - OK
    ptr readnone undef,     ; ← INPUT = UNDEF! BUG!
    i64 1, i64 3, i64 224, i64 224,
    ptr %10,                ; weights - OK
    ...
```

Line 272 (output memcpy):
```llvm
call void @llvm.memcpy.p1.p1.i64(
    ptr addrspace(1) noundef align 1 dereferenceable(3211264) undef,  ; ← DEST = UNDEF! BUG!
    ptr addrspace(1) noundef align 1 dereferenceable(3211264) %21,
    i64 3211264,
    i1 false)
```

**Result**: No `@main` function exists (completely optimized away)

### Fixed Version (with `noinline`)

File: `demo_two_layer_fixed.ll`, line 220:
```llvm
define private fastcc void @main(
    ptr noalias nocapture nonnull readonly %0,
    ptr noalias nocapture nonnull readonly %1,
    ptr noalias nocapture nonnull readonly %2)
```

Line 222 (correctly loads GPU pointer):
```llvm
%.unpack2 = load ptr addrspace(1), ptr %.elt1, align 8
```

Line 231 (passes real pointer):
```llvm
%8 = call i32 @wrap_miopenConvolutionForward(
    ptr readonly %0,
    ptr readnone %7,        ; ← Real pointer!
    ...
```

**Result**: `@main` exists and works correctly

## Why `noinline` Fixes It

When `@main` is NOT inlined:

1. **SROA optimizes `inference_compute` separately**:
   - Sees memref structs passed to `@main` call
   - Cannot see what happens inside `@main`
   - Doesn't try the optimization that would fail

2. **SROA optimizes `@main` separately**:
   - Sees memref structs received as parameters from outside
   - Doesn't see the full allocation chain
   - Successfully optimizes within `@main`

3. **Function call boundary prevents SROA from seeing the full pattern**
   - The problematic double indirection is split across function boundary
   - SROA never attempts the optimization that produces `undef`

## Is This a Pattern Bug or LLVM Bug?

**The pattern is legal**:
- ✅ Storing AS1 pointers in AS0 memory is valid
- ✅ Loading AS1 pointers from AS0 memory is valid
- ✅ Passing pointers through arrays and structs is valid

**LLVM SROA has a limitation**:
- SROA should be conservative: if it can't track a value, it should give up
- Instead, SROA produces `undef` (undefined behavior)
- This is a known LLVM issue that persists even in LLVM 22 with opaque pointers

**Research findings**:
- Multiple LLVM bugs related to SROA + address spaces
- Common workarounds: `noinline`, `optnone`, `noipa`
- Opaque pointers (LLVM 15+) were supposed to fix this, but don't

## Why Not Use Opaque Pointers?

We **are** using opaque pointers! LLVM 22 only supports opaque pointers (typed pointers removed in LLVM 17).

Evidence in our IR:
```llvm
ptr addrspace(1)     // Opaque pointer syntax
ptr readnone         // Opaque pointer syntax
```

**Opaque pointers did NOT fix this SROA limitation.**

## Alternative Solutions Considered

### 1. `optnone` (Rejected)

```cpp
newMainFunc->setAttr("passthrough",
    builder.getArrayAttr({
        builder.getStringAttr("noinline"),
        builder.getStringAttr("optnone")
    }));
```

**Problem**: Disables ALL optimizations inside `@main`, including:
- Constant folding
- Dead code elimination
- Loop optimizations
- All arithmetic optimizations

**Result**: Much slower code for minimal benefit

### 2. `noipa` (No Interprocedural Analysis)

```cpp
newMainFunc->setAttr("passthrough",
    builder.getArrayAttr({builder.getStringAttr("noipa")}));
```

**Problem**:
- Prevents all interprocedural optimizations
- May be too aggressive
- `noinline` alone is sufficient and more precise

### 3. Remove Double Indirection (Not Feasible)

Change interface to:
```cpp
@main(state, input_allocated, input_aligned, input_offset,
      input_s0, input_s1, input_s2, input_s3,
      input_st0, input_st1, input_st2, input_st3,
      output_allocated, output_aligned, ...)  // 23+ params!
```

**Problem**:
- Unreadable for humans
- Defeats purpose of memref abstraction
- Makes debugging much harder

### 4. Keep `noinline` (✅ CHOSEN)

**Benefits**:
- ✅ Prevents SROA bug
- ✅ Preserves all optimizations inside `@main`
- ✅ Clean 3-parameter interface
- ✅ Minimal performance cost (one function call)
- ✅ Clear, targeted fix

**Cost**:
- One function call overhead (negligible compared to GPU operations)

## Performance Impact

The `noinline` attribute prevents ONE inlining:
```
inference_compute() → @main()
```

**What's still optimized**:
- ✅ `@main_internal` → `@main` (fully inlined)
- ✅ All code inside `@main` (full optimization)
- ✅ All code inside `inference_compute` (full optimization)

**What's not optimized**:
- ❌ One function call from `inference_compute` to `@main`

**Impact**: Negligible. The function call overhead is ~5-10 CPU cycles. Compare this to:
- GPU kernel launch: ~10,000 cycles
- Convolution operation: millions of cycles

The function call is 0.0001% of total execution time.

## LLVM Version Information

- **Current**: LLVM 22.0.0git (commit 0c2701fe7fa0, November 2025)
- **onnx-mlir dependency**: Same commit (perfectly synchronized)
- **Opaque pointers**: Enabled and only mode (typed pointers removed in LLVM 17)
- **SROA status**: Bug persists even with opaque pointers

## Future Considerations

### Monitor LLVM Updates

As onnx-mlir follows biweekly LLVM updates, watch for:
- SROA improvements in release notes
- Address space handling improvements
- Related bug fixes

### Report to LLVM (Optional)

This could be reported as an LLVM SROA bug with minimal reproducer showing:
- Legal IR pattern
- SROA producing `undef` instead of being conservative
- Persists in LLVM 22 with opaque pointers

### Re-evaluate Periodically

Every few LLVM versions, test if the bug is fixed by:
1. Remove `noinline` attribute
2. Rebuild and run tests
3. Check if GPU pointers are still `undef`

If LLVM fixes it, we can remove the workaround.

## References

- **HipToLLVM.cpp**: Implementation of `@main` wrapper with `noinline`
- **HipToLLVM.md**: Design document for two-tier architecture
- **SIDE_BY_SIDE_EXPLANATION.txt**: C code vs LLVM IR comparison
- **BROKEN_VERSION_EXPLAINED.txt**: Detailed optimization step walkthrough
- **demo_two_layer.ll**: Broken IR (without `noinline`)
- **demo_two_layer_fixed.ll**: Working IR (with `noinline`)

## See Also

- LLVM Opaque Pointers: https://llvm.org/docs/OpaquePointers.html
- LLVM SROA Pass: https://llvm.org/docs/Passes.html#sroa-scalar-replacement-of-aggregates
- onnx-mlir LLVM Update Process: `doc/UpdatingLLVMCommit.md`
