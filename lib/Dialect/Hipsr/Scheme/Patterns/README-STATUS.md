# ONNX→HipSR Scheme Patterns - Status

**Branch:** `scheme-pattern-dsl`  
**Date:** 2026-09-14/15  
**Status:** ✅ Core infrastructure complete + 3 patterns, ⏳ Testing pending

---

## Completed ✅

### Infrastructure
- ✅ FFI layer (`mlir_register_conversion_pattern`)
- ✅ ChezSchemeInterpreter (owned by HipsrDialect)
- ✅ Generic IR builder (`mlir_create_generic_op`)
- ✅ Pattern DSL macro (`define-simple-pattern`)
- ✅ Build DSL helpers (`hipsr-builders` library)

### Patterns Ported (3/16)

#### 1. Cast Pattern ✅
- **Files:** `cast-manual.sls`, `cast-with-simple-macro.sls`
- **C++ Equivalent:** `lib/Conversion/OnnxToHipsr/CastConversion.cpp`
- **Status:** Implemented with and without macros
- **Pattern (with macro):**
  ```scheme
  (define-simple-pattern onnx-cast->hipsr-with-macro "onnx.Cast"
    (lambda (op rewriter)
      (and (= (mlir-operation-num-operands op) 1)
           (let* ([%input (mlir-operation-get-operand-value op 0)]
                  [ctx (mlir-get-hipsr-context-arg op)]
                  [%placeholder (create-hipsr-placeholder ctx %input input-type)]
                  [%cast (create-hipsr-cast ctx %input %placeholder output-type)])
             (mlir-replace-op op %cast)
             #t))))
  ```

#### 2. Min Pattern ✅
- **Files:** `min-manual.sls`, `min-with-builders.sls`
- **C++ Equivalent:** `lib/Conversion/OnnxToHipsr/MinConversion.cpp`
- **Status:** Implemented with generic builder and helpers
- **Features:** Handles single input (identity) and multiple inputs (fold)

#### 3. Equal Pattern ✅
- **File:** `equal-simple.sls`
- **C++ Equivalent:** `lib/Conversion/OnnxToHipsr/EqualConversion.cpp`
- **Status:** Simplified version (device operands only)
- **Note:** Full version needs host constant handling FFI

---

## Not Yet Ported (13/16)

### Easy to Port (have generic builder)

These can use `mlir-create-generic-op` + `hipsr-builders`:

- **Add** - binary op like Min
- **Mul** - binary op like Min
- **Transpose** - needs `hipsr.transpose`
- **Gather** - needs `hipsr.gather`
- **Slice** - needs `hipsr.slice`
- **MatMul** - needs `hipsr.matmul`
- **ScatterND** - needs `hipsr.scatternd`
- **NonZero** - needs `hipsr.nonzero`
- **Shape** - needs `hipsr.shape`

**Estimated:** ~30 minutes each

### Blocked on Region Support

Complex patterns with shape/compute regions:

- **Reshape** - `lib/Conversion/OnnxToHipsr/ReshapeConversion.cpp`
  - Needs shape region construction
  - Needs compute body with collapse/expand
  - ~300 lines of C++ logic

- **Expand** - similar complexity
- **Unsqueeze** - similar complexity

---

## Macros and Helpers ✅

### Pattern DSL ✅
**File:** `Runtime/macros/pattern-dsl-simple.sls`  
**Status:** Working!

**Macro:**
```scheme
(define-simple-pattern pattern-name "op-name"
  (lambda (op rewriter)
    ;; Match and rewrite logic
    ...))
```

**Reduces boilerplate:** No need to manually check operation name

### Builder Helpers ✅
**File:** `Runtime/hipsr-builders.sls`  
**Status:** Working library of common operations

**Helpers:**
- `create-hipsr-placeholder` - Creates placeholder
- `create-hipsr-cast` - Creates cast
- `create-hipsr-min` - Creates min (binary op)
- `create-hipsr-equal` - Creates equal
- `create-hipsr-add` - Creates add
- `create-hipsr-mul` - Creates mul

**More can be added easily**

### Full Pattern DSL (Future)
**File:** `Runtime/macros/pattern-dsl.sls`  
**Status:** Skeleton (not needed yet)

**Would enable:**
```scheme
(define-conversion-pattern onnx-cast->hipsr
  :if-match
    %cast = "onnx.Cast" (%input) :type (!t) -> !t2
  :rewrite %cast
    ...)
```

**Current approach works well** - full DSL is optional polish

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

**Priority 2: Port More Patterns** ✅ (generic builder done!)
- ✅ Min pattern
- ✅ Equal pattern  
- ⏳ Add, Mul, Transpose, Gather, Slice, etc. (easy with helpers)

**Priority 3: Test Everything**
- Build on remote machine
- Test Cast, Min, Equal patterns
- Verify all work correctly

**Priority 4: Complete Pattern Library**
- Port remaining 13 simple patterns
- Each ~30-50 lines with helpers
- **Estimate:** 1-2 days

**Priority 5: Complex Patterns** (future)
- Add region support FFI
- Port Reshape, Expand, Unsqueeze
- **Estimate:** 2-3 days

---

## Commits

- `c9bd5b44` - Initial implementation (FFI, Cast manual)
- `3a45cf1c` - ChezSchemeInterpreter refactoring
- `df72d8c5` - Status documentation
- `34b6374f` - Generic IR builder + Min pattern
- `2f4dfacc` - Macros, builders, 3 patterns complete

**Branch:** https://github.com/wcy123/hip-ep/tree/scheme-pattern-dsl

**Progress:** 3/16 patterns ported (19%)

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
