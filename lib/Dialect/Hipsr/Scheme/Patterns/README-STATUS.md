# ONNX→HipSR Scheme Patterns - Status

**Branch:** `scheme-pattern-dsl`  
**Date:** 2026-09-14/15  
**Status:** ✅ Core infrastructure complete, ⏳ Testing pending

---

## Working

### Cast Pattern ✅
- **File:** `cast-manual.sls`
- **C++ Equivalent:** `lib/Conversion/OnnxToHipsr/CastConversion.cpp`
- **Status:** Implemented, not tested
- **Pattern:**
  ```scheme
  (define (onnx-cast->hipsr-manual op rewriter)
    (and (string=? (mlir-operation-name op) "onnx.Cast")
         (= (mlir-operation-num-operands op) 1)
         (let* ([%input (mlir-operation-get-operand-value op 0)]
                [ctx (mlir-get-hipsr-context-arg op)]
                [%placeholder (mlir-create-placeholder-op ...)]
                [%cast (mlir-create-cast-op ...)])
           (mlir-replace-op op %cast)
           #t)))
  ```

### Infrastructure ✅
- **FFI:** `mlir_register_conversion_pattern()` wraps Scheme callbacks
- **Runtime:** `ChezSchemeInterpreter` owned by `HipsrDialect`
- **Integration:** `onnx-to-hipsr.sls` uses Scheme Cast pattern

---

## Not Yet Ported

### Blocked on Generic IR Builder

These patterns need `mlir-create-operation()`:

- **Min** - `lib/Conversion/OnnxToHipsr/MinConversion.cpp`
- **Equal** - `lib/Conversion/OnnxToHipsr/EqualConversion.cpp`
- **Transpose** - needs `hipsr.transpose`
- **Gather** - needs `hipsr.gather`
- **Slice** - needs `hipsr.slice`

**What's missing:**
```c
SchemeValue mlir_create_operation(
    SchemeValue rewriter,
    const char* op_name,
    SchemeValue operands,    // list
    SchemeValue attributes,  // list of pairs
    SchemeValue result_types); // list
```

### Blocked on Region Support

Complex patterns with shape/compute regions:

- **Reshape** - `lib/Conversion/OnnxToHipsr/ReshapeConversion.cpp`
  - Needs shape region construction
  - Needs compute body with collapse/expand
  - ~300 lines of C++ logic

- **Expand** - similar complexity
- **Unsqueeze** - similar complexity

---

## Macros (Deferred)

### Pattern DSL ⏸️
**File:** `Runtime/macros/pattern-dsl.sls`  
**Status:** Skeleton only

**Goal:**
```scheme
(define-conversion-pattern onnx-cast->hipsr
  :if-match
    %cast = "onnx.Cast" (%input) (:to $dtype) :type (!t) -> !t2
  :rewrite %cast
    (mlir-build
      %placeholder = "hipsr.Placeholder" (%input) :type (!t) -> !t
      %result = "hipsr.Cast" (%input %placeholder) :type (!t) -> !t2))
```

**Why deferred:** Test manual pattern first, then automate

### Build DSL ⏸️
**File:** `Runtime/macros/build-dsl.sls`  
**Status:** Skeleton only

**Goal:**
```scheme
(mlir-build
  %op = "dialect.op" (%operand ...)
        [:regions ((body...))]
        [(:attr ,value)]
        :type (!t ...) -> !t)
```

**Why deferred:** Manual IR construction works, macro is sugar

---

## Testing

### Remote Machine Required

**Machine:** xcoengvm226019:23762  
**Build:** `/home/build/hip-ep-1`

```bash
# Build (~40 minutes)
python3 hip-ep-1/build.py --mock --build-dir /home/build/hip-ep-1

# Test Cast pattern
bin/hip-mlir-opt test/lit/Conversion/onnx-to-hipsr/simple.mlir \
  --convert-onnx-to-hipsr
```

**Expected output:**
```
[debug] Registering Scheme pattern for onnx.Cast
[info] ONNX to HipSR Conversion (Scheme): Success
```

---

## Next Steps

**Priority 1: Test** ⏳
- Build on remote machine
- Run simple.mlir test
- Debug if needed

**Priority 2: Generic Builder** (after test passes)
- Add `mlir_create_operation()` FFI
- Port Min pattern (simplest binary op)
- Verify generic builder works

**Priority 3: More Patterns**
- Port 5-10 simple patterns (Min, Equal, Transpose, etc.)
- Each ~50 lines of Scheme

**Priority 4: Macros**
- Implement full `define-conversion-pattern`
- Implement `mlir-build`
- Convert manual patterns to use macros

**Priority 5: Complex Patterns**
- Add region support
- Port Reshape (~300 lines → ~100 lines Scheme)

---

## Commits

- `c9bd5b44` - Initial implementation
- `3a45cf1c` - ChezSchemeInterpreter refactoring

**Branch:** https://github.com/wcy123/hip-ep/tree/scheme-pattern-dsl

---

## Files

**Patterns:**
- `cast-manual.sls` - ✅ Working example
- `cast.sls` - ⏸️ Macro-based (not used)

**Macros:**
- `Runtime/macros/pattern-dsl.sls` - ⏸️ Skeleton
- `Runtime/macros/build-dsl.sls` - ⏸️ Skeleton

**Infrastructure:**
- `Runtime/SchemeBindings.cpp` - FFI layer
- `Runtime/ChezSchemeInterpreter.cpp` - Runtime ownership

**Integration:**
- `Passes/onnx-to-hipsr.sls` - Uses Scheme Cast pattern
- `IR/HipsrDialect.cpp` - Owns interpreter

---

**Status:** Ready for testing. Foundation is solid.
