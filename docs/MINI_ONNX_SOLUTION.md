# MiniONNX Dialect - Complete Solution

## Executive Summary

Successfully created a minimal ONNX dialect that registers **176 out of 193 ONNX operations** without heavy dependencies. The solution involved deep investigation into MLIR 22's properties system and TableGen code generation.

## Problem Space

### Initial Challenge
Create a minimal ONNX dialect for hip-ep that:
- Auto-generates from ONNX schemas (no manual maintenance)
- Avoids onnx-mlir's heavy dependencies (ShapeInference, IndexExpr, etc.)
- Registers as many operations as possible
- Handles remaining operations via `allowUnknownOperations()`

### Root Cause Discovery
Through systematic investigation with specialized agents, we identified:

1. **MLIR Properties Migration (MLIR 18+)**
   - Properties became mandatory for operations with attributes
   - Default changed: `usePropertiesForAttributes = 1`
   - Legacy mode removed in MLIR 19

2. **Version Incompatibility**
   - onnx-mlir uses LLVM 24.x (development branch, Feb 25 2026)
   - hip-ep uses LLVM 22.1.0 (stable release, Feb 24 2026)
   - onnx-mlir's patterns designed for older MLIR API

3. **Compiler Requirements**
   - MLIR 22 generated code requires Clang 22+
   - GCC 9 and Clang 10 have template instantiation issues
   - BytecodeOpInterface errors with old compilers

## Solution Architecture

### 1. Generator Updates (`utils/gen_mini_onnx.py`)

**Key Change**: Add assembly format to enable properties auto-generation

```python
def gen_op_def(schema):
    # ... arguments and results generation ...
    
    # CRITICAL: Add assembly format
    # This tells TableGen how to serialize/deserialize the operation
    # and enables auto-generation of all properties infrastructure
    s += indent + 'let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";\n'
    
    s += "}\n\n"
    return s
```

**Why This Works**:
- `assemblyFormat` provides serialization specification
- TableGen uses this to auto-generate properties methods:
  - `computePropertiesHash()`
  - `setPropertiesFromAttr()`
  - `getPropertiesAsAttr()`
  - Properties storage struct
- No custom C++ implementation needed

### 2. Attribute Handling

```python
def get_attrs(schema):
    # Use OptionalAttr for all non-required attributes
    # Properties mode handles this correctly with assemblyFormat
    if attr.required:
        name_to_type[attr.name] = mlir_type
    else:
        name_to_type[attr.name] = f"OptionalAttr<{mlir_type}>"
```

**Note**: We use `OptionalAttr` instead of `DefaultValuedAttr` because:
- Simpler to generate
- Properties mode handles optionality correctly
- Default values are in ONNX runtime, not dialect

### 3. Type Filtering

Filter out unsupported ONNX types:
- Sequences (`seq(...)`)
- Maps (`map(...)`)  
- Optional wrappers (`optional(...)`)
- Strings (`tensor(string)`)
- Complex numbers
- Graph attributes (subgraphs for If/Loop/Scan)

**Result**: 176 operations supported, 17 filtered out

### 4. Dialect Configuration

Minimal dialect definition - no special properties configuration needed:

```tablegen
def MiniONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "Minimal ONNX dialect for hip-ep";
  let cppNamespace = "::mlir::onnx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["func::FuncDialect"];
  // Note: usePropertiesForAttributes defaults to 1 (properties mode)
}
```

### 5. Build Requirements

**Critical**: Must use Clang 22+ compiler

```cmake
cmake ../hip-ep-1 \
  -DCMAKE_C_COMPILER=/path/to/clang-22 \
  -DCMAKE_CXX_COMPILER=/path/to/clang++-22 \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja
```

**Why**: 
- MLIR 22 generates modern C++ template code
- GCC 9: Template deduction failures in properties code
- Clang 10: BytecodeOpInterface availability issues
- Clang 22: Matches MLIR version, full compatibility

## Operations Coverage

### Supported (176 operations)

**Simple Math Operations** (no attributes):
- Abs, Acos, Acosh, Add, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Div, Erf, Exp, Floor, Log, MatMul, Mul, Neg, Not, Reciprocal, Round, Selu, Sigmoid, Sign, Sin, Sinh, Sqrt, Sub, Tan, Tanh

**Operations with Attributes** (properties supported):
- AffineGrid, ArgMax, ArgMin, AveragePool, BatchNormalization, Cast, Conv, ConvTranspose, Dropout, Gemm, GlobalAveragePool, GlobalMaxPool, LRN, LSTM, MaxPool, Pad, Reshape, Resize, Softmax, Transpose, etc.

### Filtered Out (17 operations)

**Reason: Unsupported Types**
- Sequences: ConcatFromSequence, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, SequenceMap, SplitToSequence
- Strings: StringConcat, StringNormalizer, StringSplit, RegexFullMatch  
- Optional: Optional
- Subgraphs: If, Loop, Scan

**Handling**: These operations remain unregistered and are handled via `allowUnknownOperations()` as generic `Operation*` instances. Conversion patterns can still match them by string name.

## Key Learnings

### 1. Properties Are Mandatory in MLIR 22
- Cannot use `usePropertiesForAttributes = 0` (deprecated, buggy)
- Must embrace properties mode
- Assembly format is the key enabler

### 2. TableGen Auto-Generation
- With `assemblyFormat`, TableGen generates ALL properties code
- No manual C++ implementation needed
- No custom builders required

### 3. Compiler Matters
- MLIR version and compiler must align
- Template-heavy generated code is compiler-sensitive
- Clang 22 is non-negotiable for MLIR 22

### 4. onnx-mlir vs MiniONNX
- onnx-mlir: Full infrastructure, manual builders, shape inference
- MiniONNX: Minimal, auto-generated, no shape inference
- Both approaches valid for different use cases

## Files Modified

1. **utils/gen_mini_onnx.py**: Generator script
   - Added `assemblyFormat` generation
   - Type filtering logic
   - Properties-compatible attribute handling

2. **include/hip/Dialect/MiniONNX/IR/MiniONNX.td**: Dialect definition
   - No changes needed (defaults work)

3. **lib/Dialect/MiniONNX/IR/MiniONNXDialect.cpp**: Dialect implementation
   - `allowUnknownOperations()` for unregistered ops

## Build Instructions

```bash
# 1. Generate operations
python3 utils/gen_mini_onnx.py --ops all \
  --output include/hip/Dialect/MiniONNX/IR/MiniONNXOps.td.inc

# 2. Configure build (with Clang 22)
cmake -B build -S . \
  -DCMAKE_C_COMPILER=/path/to/clang-22 \
  -DCMAKE_CXX_COMPILER=/path/to/clang++-22 \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja

# 3. Build
ninja -C build MiniONNXDialectIR
```

## Testing

Operations can be used in conversion patterns:

```cpp
// Registered operations (type-safe)
patterns.add<MatMulConversionPattern, AddConversionPattern>(...);

// Unregistered operations (string matching still works)
struct UnregisteredConversionPattern : public ConversionPattern {
  UnregisteredConversionPattern(MLIRContext *ctx)
      : ConversionPattern("onnx.Unsupported", 1, ctx) {}
  // ... implementation
};
```

## References

- MLIR Properties RFC: https://discourse.llvm.org/t/rfc-deprecating-usepropertiesforattributes-0/87536
- MLIR ODS Documentation: https://mlir.llvm.org/docs/DefiningDialects/Operations/
- onnx-mlir LLVM commit: 1053047a4be7d1fece3adaf5e7597f838058c947
- hip-ep LLVM version: llvmorg-22.1.0

## Success Metrics

✅ 176/193 ONNX operations registered (91%)  
✅ Auto-generated from ONNX schemas (zero manual maintenance)  
✅ No heavy dependencies (ShapeInference, IndexExpr removed)  
✅ Properties mode compatible with MLIR 22  
✅ Conversion patterns work with both registered and unregistered ops
