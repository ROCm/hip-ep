<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Adding a New Operator to HIP Execution Provider

This guide provides step-by-step instructions for adding a new ONNX operator to the HIP execution provider. It focuses on **mock runtime implementation** (not real GPU kernels) and includes complete LIT test requirements.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Critical Constraints Checklist](#critical-constraints-checklist)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Analyze the ONNX Operator Specification](#step-1-analyze-the-onnx-operator-specification)
  - [Step 2: Classify the Operator](#step-2-classify-the-operator)
  - [Step 3: ONNX-to-HIP Conversion Layer](#step-3-onnx-to-hip-conversion-layer)
  - [Step 4: HIP Dialect Definition](#step-4-hip-dialect-definition)
  - [Step 5: HIP-to-LLVM Lowering Layer](#step-5-hip-to-llvm-lowering-layer)
  - [Step 6: Mock Runtime Implementation](#step-6-mock-runtime-implementation)
  - [Step 7: CMake and Configuration](#step-7-cmake-and-configuration)
  - [Step 8: LIT Tests](#step-8-lit-tests)
  - [Step 9: Build and Validate](#step-9-build-and-validate)
- [Common Patterns](#common-patterns)
- [Troubleshooting Checklist](#troubleshooting-checklist)
- [Complete Examples](#complete-examples)
- [Quick Reference Card](#quick-reference-card)

## Overview

The HIP execution provider uses a **three-layer architecture**:

```
ONNX (.onnx) → ONNX-to-HIP → HIP Dialect → HIP-to-LLVM → LLVM IR → model.dll
```

**Three conversion layers**:
1. **ONNX → HIP Conversion** (`lib/Conversion/OnnxToHip/`) - Pattern-matches ONNX ops and emits HIP dialect ops
2. **HIP Dialect** (`lib/Dialect/`) - Custom MLIR dialect with DPS (Destination-Passing Style) ops
3. **HIP → LLVM Lowering** (`lib/Conversion/HipToLLVM/`) - Lowers HIP ops to LLVM IR calls into runtime

**Operator classification**:
- **Zero-cost metadata ops** (Shape, Reshape, Squeeze) - No HIP dialect ops, direct lowering to standard MLIR
- **GPU compute ops** (Sqrt, MatMul, Conv) - Full three-layer implementation with runtime functions

**Mock vs Real runtime**:
- **Mock runtime** (`lib/Runtime/mock/`) - CPU stubs for GPU-free development/testing
- **Real runtime** (`lib/Runtime/real/`) - GPU execution via MIOpen, hipBLASLt, custom kernels

This guide focuses on **mock runtime** implementation.

## Prerequisites

Before adding a new operator, you should understand:

- **MLIR concepts**: Patterns, dialects, conversion, bufferization
- **TableGen syntax**: MLIR's domain-specific language for defining operations
- **DPS (Destination-Passing Style)**: Output buffers passed as operands, not results
- **Basic C++ and CMake**: Standard build system knowledge

## Critical Constraints Checklist

Before starting, ensure you understand these **mandatory constraints**:

### ✓ ONNX Spec Consistency
- [ ] Input/output names **exactly match** ONNX spec
- [ ] Optional properties (inputs/outputs/attributes) match ONNX spec
- [ ] Parameter order matches ONNX spec
- [ ] For Microsoft contrib ops, check `function_name` and `domain_name` attributes

### ✓ TableGen Definition
- [ ] **Do NOT use `OptionalAttr`** (doesn't exist) - use `DefaultValuedAttr` with sentinel values
- [ ] All DPS ops inherit from `Hip_DpsOp<"mnemonic">`
- [ ] First parameter **must be** `Hip_ContextType:$ctx`
- [ ] Manually implement `getDpsInitsMutable()` and `getEffects()` in `HipDialect.cpp`
- [ ] Register bufferization interface in `tools/hip-mlir-opt/hip-mlir-opt.cpp`

### ✓ Runtime Functions
- [ ] Declare in `hipdnn_ep_runtime.h` **inside `extern "C" { }` block**
- [ ] Function name constant defined in `HipToLLVMUtils.h` (don't hardcode strings)
- [ ] Signature **exactly matches** lowering layer's generated call
- [ ] First parameter always `RuntimeState *state`
- [ ] Integer params use `int64_t`, floating point use `double`

### ✓ CMake Configuration
- [ ] Add new `.cpp` files to corresponding `CMakeLists.txt`
- [ ] Keep file organization consistent between OnnxToHip and HipToLLVM layers
- [ ] Mock mode: no CMake changes needed for runtime (all in `mock_gpu.cpp`)

### ✓ LIT Tests
- [ ] Cover static/dynamic shapes, multiple dtypes, multi-dimensional tensors
- [ ] Test different attribute values (if applicable)
- [ ] Include edge cases (scalar, empty tensor, etc.)
- [ ] Create **three test files**: onnx-to-hip, hip-to-llvm, e2e

---

## Step-by-Step Guide

### Step 1: Analyze the ONNX Operator Specification

Before implementation, fully understand the ONNX operator spec.

**1. Find ONNX Spec**
- Standard operators (`ai.onnx`): https://onnx.ai/onnx/operators/
- Microsoft contrib (`com.microsoft`): https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
- Search for operator name and version

**2. Record Key Information**
- **Inputs**: Names, types, optional/required, order
- **Outputs**: Names, types, shape inference rules
- **Attributes**: Names, types, required/optional, default values
- **Type constraints**: Supported data types (f32, f16, i32, etc.)
- **Shape inference**: How output shape is derived from inputs

**3. Special Considerations**
- Optional inputs handling (ONNX uses empty string `""` or missing)
- Variadic inputs (e.g., Sum, Concat)
- Broadcasting rules (e.g., Add, Mul)
- Default attribute values **must match ONNX exactly**

**Example (Sqrt)**:
```
ONNX Spec: onnx.Sqrt
- Inputs: X (T, required)
- Outputs: Y (T, required)
- Attributes: None
- Type constraints: T = tensor(float16), tensor(float), tensor(double)
- Shape: Y.shape = X.shape
```

---

### Step 2: Classify the Operator

Choose implementation strategy based on operator nature.

**Decision Tree**:
```
┌─ Does operator only change shape/layout without computing values?
│  └─ YES → Zero-cost operation
│     - Lower to tensor.expand_shape/collapse_shape/extract_slice
│     - Examples: Shape, Reshape, Squeeze, Unsqueeze, Split
│     - **Skip Steps 4, 5, 6** (no HIP dialect, lowering, runtime needed)
│
└─ NO → GPU compute operation
   - Requires full three-layer implementation
   - Examples: Sqrt, MatMul, Conv, GQA
   - Continue to Step 3
```

**Classification Criteria**:
- **Zero-cost**: Operation completes via tensor descriptor adjustment, no data copy/compute
- **GPU compute**: Requires GPU kernel for element-wise computation, reduction, or complex algorithm

---

### Step 3: ONNX-to-HIP Conversion Layer

**Files to create/modify**:
- `lib/Conversion/OnnxToHip/<OpName>Conversion.cpp` - Create new file
- `lib/Conversion/OnnxToHip/OnnxToHip.cpp` - Register pattern
- `lib/Conversion/OnnxToHip/OnnxToHipUtils.h` - Declare populate function
- `lib/Conversion/OnnxToHip/CMakeLists.txt` - Add new file to build

**Basic Template** (simple element-wise op):

```cpp
// File: lib/Conversion/OnnxToHip/NewOpConversion.cpp

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct NewOpToHip : public mlir::RewritePattern {
  NewOpToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NewOp", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    // 1. Validate inputs/outputs
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");
    }

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "output must be ranked tensor");
    }

    // 2. Get !hip.context parameter
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure)) {
      return rewriter.notifyMatchFailure(op, "failed to get context");
    }
    mlir::Value context = *ctxOrFailure;

    // 3. (Optional) Extract ONNX attributes
    // auto attr = op->getAttrOfType<mlir::IntegerAttr>("attr_name");
    // if (!attr) return rewriter.notifyMatchFailure(op, "missing attr");

    // 4. Create output tensor (DPS)
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

    // 5. Create HIP op
    auto hipOp = mlir::hip::NewOpOp::create(
        rewriter, loc, mlir::TypeRange{resultType}, context, input, init);

    // 6. Replace original op
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populateNewOpConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<NewOpToHip>(ctx);
}

} // namespace hip
} // namespace mlir
```

**Complete Template** (with optional inputs, attributes, dynamic dims):

```cpp
// File: lib/Conversion/OnnxToHip/NewOpConversion.cpp

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct NewOpToHip : public mlir::RewritePattern {
  NewOpToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NewOp", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    // 1. Validate basic inputs/outputs
    if (op->getNumOperands() < 1) {
      return rewriter.notifyMatchFailure(op, "expected at least 1 input");
    }
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "expected 1 output");
    }

    // 2. Extract required inputs
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    // 3. Extract optional inputs
    auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
      if (idx >= op->getNumOperands()) {
        return mlir::Value{};
      }
      mlir::Value v = op->getOperand(idx);
      if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
        return mlir::Value{};
      }
      return v;
    };
    mlir::Value optionalInput = getOptionalInput(1);

    // 4. Extract ONNX attributes

    // 4a. Integer attribute (required)
    auto someAttr = op->getAttrOfType<mlir::IntegerAttr>("some_attr");
    if (!someAttr) {
      return rewriter.notifyMatchFailure(op, "missing some_attr");
    }
    auto someAttrHip = rewriter.getI64IntegerAttr(someAttr.getSInt());

    // 4b. Integer attribute (optional with default)
    int64_t optionalAttrValue = -1;  // Default value
    if (auto optAttr = op->getAttrOfType<mlir::IntegerAttr>("optional_attr")) {
      optionalAttrValue = optAttr.getSInt();
    }
    auto optionalAttrHip = rewriter.getI64IntegerAttr(optionalAttrValue);

    // 4c. Float attribute
    auto floatAttr = op->getAttrOfType<mlir::FloatAttr>("threshold");
    double threshold = floatAttr ? floatAttr.getValueAsDouble() : 0.0;
    auto thresholdHip = rewriter.getF64FloatAttr(threshold);

    // 4d. String attribute
    auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode");
    std::string mode = modeAttr ? modeAttr.getValue().str() : "NONE";
    auto modeHip = rewriter.getStringAttr(mode);

    // 5. Handle output type and dynamic dimensions
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "output must be ranked tensor");
    }

    // 5a. Extract dynamic dimensions
    llvm::SmallVector<mlir::Value> dynamicDims;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
        auto dimI64 = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(), dim);
        dynamicDims.push_back(dimI64);
      }
    }

    // 6. Get !hip.context parameter
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure)) {
      return rewriter.notifyMatchFailure(op, "failed to get context");
    }
    mlir::Value context = *ctxOrFailure;

    // 7. Create output tensor (DPS)
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

    // 8. Create HIP op
    auto hipOp = mlir::hip::NewOpOp::create(
        rewriter, loc,
        mlir::TypeRange{resultType},
        context,
        input,
        optionalInput,  // May be empty
        init,
        someAttrHip,
        optionalAttrHip,
        thresholdHip,
        modeHip);

    // 9. Replace original op
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

// (Optional) Microsoft contrib domain ops
struct NewOpMicrosoftToHip : public mlir::RewritePattern {
  NewOpMicrosoftToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check function_name and domain_name
    auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
    if (!funcNameAttr || funcNameAttr.getValue() != "NewOp") {
      return rewriter.notifyMatchFailure(op, "not a NewOp custom op");
    }
    auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
      return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
    }

    // Rest similar to standard ONNX op...
    return mlir::success();
  }
};

} // namespace

void populateNewOpConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<NewOpToHip>(ctx);
  // If Microsoft contrib op:
  // patterns.add<NewOpMicrosoftToHip>(ctx);
}

} // namespace hip
} // namespace mlir
```

**Register Pattern** in `lib/Conversion/OnnxToHip/OnnxToHip.cpp`:

```cpp
// Add to populateOnnxToHipConversionPatterns function
populateNewOpConversionPatterns(patterns, ctx);
```

**Declare Populate Function** in `lib/Conversion/OnnxToHip/OnnxToHipUtils.h`:

```cpp
void populateNewOpConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
```

**Key Patterns**:

**Get Context**:
```cpp
auto ctxOrFailure = getContextArg(op, rewriter);
if (mlir::failed(ctxOrFailure)) return mlir::failure();
mlir::Value context = *ctxOrFailure;
```

**Handle Optional Inputs**:
```cpp
auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
  if (idx >= op->getNumOperands()) return mlir::Value{};
  mlir::Value v = op->getOperand(idx);
  if (!v || mlir::isa<mlir::NoneType>(v.getType())) return mlir::Value{};
  return v;
};
```

**Extract Attributes**:
```cpp
// Integer (required)
auto attr = op->getAttrOfType<mlir::IntegerAttr>("attr_name");
if (!attr) return mlir::failure();
int64_t value = attr.getInt();

// Integer (optional with default)
int64_t optValue = -1;
if (auto opt = op->getAttrOfType<mlir::IntegerAttr>("opt_attr")) {
  optValue = opt.getSInt();
}

// Float
auto floatAttr = op->getAttrOfType<mlir::FloatAttr>("threshold");
double threshold = floatAttr ? floatAttr.getValueAsDouble() : 0.0;

// String
auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode");
std::string mode = modeAttr ? modeAttr.getValue().str() : "NONE";
```

**Handle Dynamic Dimensions**:
```cpp
llvm::SmallVector<mlir::Value> dynamicDims;
for (unsigned i = 0; i < resultType.getRank(); ++i) {
  if (resultType.isDynamicDim(i)) {
    auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
    auto dimI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), dim);
    dynamicDims.push_back(dimI64);
  }
}
```

**Reference Examples**:
- Simple ops: `lib/Conversion/OnnxToHip/PowerConversion.cpp` (Sqrt, Reciprocal)
- Attributes + dynamic output: `lib/Conversion/OnnxToHip/RangeConversion.cpp`
- Microsoft contrib: `lib/Conversion/OnnxToHip/MatMulNBitsConversion.cpp`

---

### Step 4: HIP Dialect Definition

**Files to modify**:
- `include/hip/Dialect/IR/HipOps.td` - TableGen definition
- `lib/Dialect/IR/HipDialect.cpp` - Manual interface implementations
- `tools/hip-mlir-opt/hip-mlir-opt.cpp` - Bufferization registration

**4.1 TableGen Definition** (`HipOps.td`):

```tablegen
def Hip_NewOpOp : Hip_DpsOp<"newop"> {
  let summary = "Brief description of the operation";
  let description = [{
    Detailed description with semantics.

    This operation performs ... on the input tensor.
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
    // Optional attributes:
    // I64Attr:$some_attr,
    // DefaultValuedAttr<F64Attr, "0.0">:$threshold,
    // DefaultValuedAttr<StrAttr, "\"NONE\"">:$mode
  );

  let results = (outs Variadic<AnyTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 0;
}
```

**Key Constraints**:
- **Must inherit from `Hip_DpsOp<"mnemonic">`**
- **First argument must be `Hip_ContextType:$ctx`**
- **DPS outputs as operands**, not results
- **Do NOT use `OptionalAttr`** - use `DefaultValuedAttr` with sentinel values:
  - Integer: `-1` for "not set"
  - Float: `0.0` or `-1.0` for "not set" or "auto"
  - String: `"NONE"` or `""` for "not set"
- Optional operands: `Optional<Hip_TensorOrMemRef>:$optional_input`

**4.2 Manual Interface Implementations** (`HipDialect.cpp`):

Add at end of file (around line 285+), after other op implementations:

```cpp
// In lib/Dialect/IR/HipDialect.cpp

MutableOperandRange NewOpOp::getDpsInitsMutable() {
  return getOutputMutable();  // Or getOutsMutable(), depending on TableGen arg name
}

void NewOpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}
```

**4.3 Bufferization Registration** (`tools/hip-mlir-opt/hip-mlir-opt.cpp`):

In `registerHipBufferizableOpInterfaceModels` function (around line 89-148):

```cpp
void registerHipBufferizableOpInterfaceModels(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::hip::HipDialect *) {
    mlir::hip::ConvOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ConvOp>>(*ctx);
    // ... existing ops ...

    // Add new op (insert alphabetically)
    mlir::hip::NewOpOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::NewOpOp>>(*ctx);
  });
}
```

---

### Step 5: HIP-to-LLVM Lowering Layer

**Files to create/modify**:
- `lib/Conversion/HipToLLVM/<OpName>Lowering.cpp` - Create new file
- `lib/Conversion/HipToLLVM/HipToLLVM.cpp` - Register pattern
- `lib/Conversion/HipToLLVM/HipToLLVMUtils.h` - Declare populate function + function name constant
- `lib/Conversion/HipToLLVM/CMakeLists.txt` - Add new file

**5.1 Define Runtime Function Name** (`HipToLLVMUtils.h`):

Add at end of file (around line 38-78), with other `kWrap*` constants:

```cpp
// In lib/Conversion/HipToLLVM/HipToLLVMUtils.h
inline constexpr const char *kWrapNewOp = "wrap_newop";
```

**Naming convention**: `kWrap` + CamelCaseOpName

**5.2 Basic Lowering Template**:

```cpp
// File: lib/Conversion/HipToLLVM/NewOpLowering.cpp

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct NewOpLowering : public ConvertOpToLLVMPattern<NewOpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NewOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // 1. Get LLVM types
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // 2. Lambda helpers for constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // 3. Get RuntimeState and operands
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // 4. Compute num_elements (supports dynamic shapes)
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);  // Runtime value
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));  // Compile-time constant
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // 5. Get data type enum
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    Value dataTypeVal = createI64Const(dataType);

    // 6. Create runtime function call
    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNewOp, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {statePtr, inputPtr, outputPtr, numElements, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // 7. Delete original op
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNewOpLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<NewOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
```

**5.3 Complete Template** (with all advanced patterns):

```cpp
// File: lib/Conversion/HipToLLVM/NewOpLowering.cpp

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// (Optional) String → Enum conversion
static int64_t convertNewOpMode(llvm::StringRef mode) {
  return llvm::StringSwitch<int64_t>(mode)
      .Case("default", kNewOpModeDefault)
      .Case("fast", kNewOpModeFast)
      .Default(-1);
}

struct NewOpLowering : public ConvertOpToLLVMPattern<NewOpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // (Optional) Inline helper for complex logic
  Value computeSomeMetadata(MemRefType type, Value descriptor,
                            ConversionPatternRewriter &rewriter,
                            Location loc) const {
    // Complex logic...
    return someValue;
  }

  LogicalResult
  matchAndRewrite(NewOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // 1. Get LLVM types
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    // 2. Lambda helpers for constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF64Const = [&](double value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f64Type,
                                      rewriter.getF64FloatAttr(value));
    };

    // 3. Get RuntimeState and required operands
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // 4. (Optional) Handle optional operands
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref) return nullPtr;
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };
    Value optionalPtr = getMemRefPtrOrNull(adaptor.getOptionalInput());

    // 5. Extract shape information
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // 5a. Compute total elements (supports dynamic dims)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());
    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // 6. Get data type enum
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    Value dataTypeVal = createI64Const(dataType);

    // 7. (Optional) Extract attributes

    // 7a. Integer attribute
    auto someAttr = op.getSomeAttr();  // I64Attr
    Value attrVal = createI64Const(someAttr);

    // 7b. Float attribute with default
    float someFloat = 1.0f;
    if (auto floatAttr = op.getSomeFloatAttr()) {
      double floatValue = floatAttr.convertToFloat();
      if (floatValue >= 0.0) {  // Check for sentinel value
        someFloat = floatValue;
      }
    }
    Value floatVal = createF64Const(someFloat);

    // 7c. String attribute → enum
    auto modeAttr = op.getModeAttr();  // StrAttr
    int64_t modeEnum = kNewOpModeDefault;
    if (modeAttr) {
      std::string modeStr = modeAttr.getValue().str();
      if (modeStr != "NONE") {  // Check for sentinel value
        modeEnum = convertNewOpMode(modeAttr.getValue());
        if (modeEnum < 0) {
          return rewriter.notifyMatchFailure(op, "invalid mode value");
        }
      }
    }
    Value modeVal = createI64Const(modeEnum);

    // 8. Create runtime function call

    // 8a. Define parameter types
    SmallVector<Type, 10> paramTypes = {
      ptrType,    // RuntimeState*
      ptrType,    // input
      ptrType,    // output
      ptrType,    // optional (nullable)
      i64Type,    // num_elements
      i64Type,    // data_type
      i64Type,    // some_attr
      f64Type,    // some_float
      i64Type     // mode_enum
    };

    // 8b. Lookup/create function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNewOp, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // 8c. Prepare arguments (order must match runtime function signature)
    SmallVector<Value, 10> args = {
      statePtr,
      inputPtr,
      outputPtr,
      optionalPtr,
      numElements,
      dataTypeVal,
      attrVal,
      floatVal,
      modeVal
    };

    // 8d. Call
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // 9. Delete original op
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNewOpLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<NewOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
```

**Register Pattern** in `lib/Conversion/HipToLLVM/HipToLLVM.cpp`:

```cpp
// Add to populateHipToLLVMConversionPatterns function
populateNewOpLoweringPatterns(converter, patterns);
```

**Declare Populate Function** in `lib/Conversion/HipToLLVM/HipToLLVMUtils.h`:

```cpp
void populateNewOpLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
```

**Reference Examples**:
- Template-based lowering: `lib/Conversion/HipToLLVM/PowerLowering.cpp`
- Optional operands: `lib/Conversion/HipToLLVM/GqaLowering.cpp`
- String→enum, batch computation: `lib/Conversion/HipToLLVM/MatMulNBitsLowering.cpp`
- Template with enum: `lib/Conversion/HipToLLVM/ElementwiseLowering.cpp`

---

### Step 6: Mock Runtime Implementation

**Files to modify**:
- `lib/Runtime/hipdnn_ep_runtime.h` - Add function declaration
- `lib/Runtime/mock/mock_gpu.cpp` - Add implementation
- `lib/Runtime/CMakeLists.txt` - (No changes needed for mock mode)

**6.1 Declare Function** (`hipdnn_ep_runtime.h`):

Add at end of file (around line 600+), **inside `extern "C" { }` block**, before final `#ifdef __cplusplus`:

```cpp
// In lib/Runtime/hipdnn_ep_runtime.h

// NewOp: element-wise operation with optional parameters
int wrap_newop(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type);

// If with attributes:
int wrap_newop_with_attrs(RuntimeState *state, void *input, void *output,
                          int64_t num_elements, int64_t data_type,
                          int64_t some_attr, double some_float);
```

**Critical**:
- Declaration **must be inside `extern "C" { }`** - otherwise C++ name mangling causes link failure
- Signature **must exactly match** lowering layer's call
- First parameter always `RuntimeState *state`
- Return `int`: 0 = success, non-zero = failure

**6.2 Implement Function** (`mock/mock_gpu.cpp`):

Add at end of file (around line 920+), alongside other `wrap_*` functions:

**Template 1: Simple No-Op** (just print):

```cpp
int wrap_newop(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_newop\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_newop(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements,
             hipdnn_ep_datatype_name(data_type),
             (long long)data_type);

  return 0;  // Success
}
```

**Template 2: CPU Fallback** (simple computation):

```cpp
int wrap_newop_cpu_fallback(RuntimeState *state, void *input, void *output,
                            int64_t num_elements, int64_t data_type) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_newop_cpu_fallback: null pointer\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_newop_cpu_fallback(num_elements=%lld, dtype=%s)\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type));

  // Simple CPU implementation (for mock verification only)
  if (data_type == HIPDNN_EP_DATATYPE_FLOAT) {
    float *in = (float *)input;
    float *out = (float *)output;
    for (int64_t i = 0; i < num_elements; ++i) {
      out[i] = in[i] * 2.0f;  // Example operation
    }
  } else if (data_type == HIPDNN_EP_DATATYPE_HALF) {
    // For f16, can choose:
    // 1. Don't implement (return 0, no crash)
    // 2. Use __fp16/_Float16 (compiler support required)
    // 3. Simple memcpy (for identity ops)
    memcpy(output, input, num_elements * 2);  // f16 = 2 bytes
  } else {
    MOCK_PRINT("[MOCK]   Unsupported dtype, skipping computation\n");
  }

  return 0;
}
```

**Template 3: With Validation**:

```cpp
int wrap_newop_complex(RuntimeState *state, void *input, void *output,
                       int64_t num_elements, int64_t data_type,
                       int64_t mode, double threshold) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_newop_complex: null tensor argument\n");
    return -1;
  }

  // Validation (same as real runtime to reject same bad inputs)
  if (mode < 0 || mode > 2) {
    fprintf(stderr, "wrap_newop_complex: invalid mode %lld\n", (long long)mode);
    return -1;
  }
  if (threshold < 0.0) {
    fprintf(stderr, "wrap_newop_complex: threshold must be non-negative\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_newop_complex(num_elements=%lld, dtype=%s, "
             "mode=%lld, threshold=%g)\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)mode, threshold);

  return 0;
}
```

**Key Principles**:
1. **`extern "C"` linkage** - handled by header declaration
2. **Parameter validation** - match real runtime validation
3. **Return 0 for success**, non-zero for failure
4. **CPU fallback is optional** - mock mode is for compile/lowering verification, not correctness
5. **Use helper macros/functions**:
   - `MOCK_PRINT(...)` - Platform-agnostic logging
   - `hipdnn_ep_datatype_name(data_type)` - Enum to string
   - `hipdnn_ep_datatype_size(data_type)` - Element byte size

---

### Step 7: CMake and Configuration

**Files to modify**:
1. `lib/Conversion/OnnxToHip/CMakeLists.txt`
2. `lib/Conversion/HipToLLVM/CMakeLists.txt`
3. `tools/hip-mlir-opt/hip-mlir-opt.cpp` (already covered in Step 4.3)
4. `lib/Runtime/CMakeLists.txt` (Mock: no changes needed)

**7.1 OnnxToHip CMake**:

```cmake
# lib/Conversion/OnnxToHip/CMakeLists.txt
add_library(OnnxToHip STATIC
  ...
  NewOpConversion.cpp  # Add alphabetically
)
```

**7.2 HipToLLVM CMake**:

```cmake
# lib/Conversion/HipToLLVM/CMakeLists.txt
add_library(HipToLLVM STATIC
  ...
  NewOpLowering.cpp  # Add alphabetically
)
```

**7.3 Runtime CMake**:

**Mock mode**: No changes needed (`mock_gpu.cpp` already in sources)

**Real mode** (future): Add headers to `IMPL_DEPS` list

**Critical Gotcha** (from CLAUDE.md):
> **Bitcode DEPENDS list must include all headers.** Every header a runtime `.cpp` file `#include`s must be in DEPENDS, otherwise editing header won't rebuild bitcode.

**Modification Checklist**:
- [ ] `lib/Conversion/OnnxToHip/CMakeLists.txt`
- [ ] `lib/Conversion/HipToLLVM/CMakeLists.txt`
- [ ] `tools/hip-mlir-opt/hip-mlir-opt.cpp` (Step 4.3)
- [ ] `lib/Runtime/CMakeLists.txt` (Mock: skip)

---

### Step 8: LIT Tests

Create **three test files** to verify each layer.

**8.1 ONNX-to-HIP Conversion Test**

File: `test/lit/Conversion/onnx-to-hip/test_newop.mlir`

```mlir
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // CHECK-LABEL: func @test_newop_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<128xf32>)
  func.func @test_newop_static(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    // CHECK-NOT: onnx.NewOp
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
    // CHECK: %[[RESULT:.*]] = hip.newop(%[[CTX]]) ins(%[[INPUT]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
    // CHECK: return %[[RESULT]]
    %0 = "onnx.NewOp"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  // Test dynamic shapes
  // CHECK-LABEL: func @test_newop_dynamic
  func.func @test_newop_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "onnx.NewOp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // Test different dtypes
  // CHECK-LABEL: func @test_newop_f16
  func.func @test_newop_f16(%arg0: tensor<64xf16>) -> tensor<64xf16> {
    %0 = "onnx.NewOp"(%arg0) : (tensor<64xf16>) -> tensor<64xf16>
    return %0 : tensor<64xf16>
  }

  // Test multi-dimensional
  // CHECK-LABEL: func @test_newop_multidim
  func.func @test_newop_multidim(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = "onnx.NewOp"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}
```

**8.2 HIP-to-LLVM Lowering Test**

File: `test/lit/Conversion/hip-to-llvm/test_newop.mlir`

```mlir
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK: llvm.func @wrap_newop(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

module {
  // Test 1: Static 1D f32
  func.func @newop_static_1d_f32(
      %ctx: !hip.context,
      %input: memref<128xf32, 1>,
      %output: memref<128xf32, 1>) {
    // CHECK-LABEL: llvm.func @newop_static_1d_f32

    hip.newop(%ctx) ins(%input : memref<128xf32, 1>)
                    outs(%output : memref<128xf32, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK: llvm.call @wrap_newop({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 2: Static 2D f16
  func.func @newop_static_2d_f16(
      %ctx: !hip.context,
      %input: memref<128x512xf16, 1>,
      %output: memref<128x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @newop_static_2d_f16

    hip.newop(%ctx) ins(%input : memref<128x512xf16, 1>)
                    outs(%output : memref<128x512xf16, 1>)

    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.mlir.constant(512 : i64)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_newop({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic 2D f32
  func.func @newop_dynamic_2d_f32(
      %ctx: !hip.context,
      %input: memref<?x?xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @newop_dynamic_2d_f32

    hip.newop(%ctx) ins(%input : memref<?x?xf32, 1>)
                    outs(%output : memref<?x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @wrap_newop({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }
}
```

**8.3 End-to-End Pipeline Test**

File: `test/lit/e2e/test_newop_model.mlir`

```mlir
// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_newop
// CHECK-NOT: onnx.NewOp

module {
  func.func @main_graph(%arg0: tensor<1x128x512xf16>) -> (tensor<1x128x512xf16>) {
    %0 = "onnx.NewOp"(%arg0) : (tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    "onnx.Return"(%0) : (tensor<1x128x512xf16>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
```

**Coverage Requirements**:
- [ ] Static shapes
- [ ] Dynamic shapes (`?`)
- [ ] Multiple data types (f32, f16, bf16)
- [ ] Multi-dimensional tensors
- [ ] Edge cases (scalar, empty tensor, if applicable)
- [ ] Different attribute values (if operator has attributes)

**Run Tests**:

```bash
# All LIT tests
ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests

# Individual test files
hip-mlir-opt test/lit/Conversion/onnx-to-hip/test_newop.mlir \
  --hip-add-context-arg --convert-onnx-to-hip | FileCheck test/lit/Conversion/onnx-to-hip/test_newop.mlir

hip-mlir-opt test/lit/Conversion/hip-to-llvm/test_newop.mlir \
  --convert-hip-to-llvm | FileCheck test/lit/Conversion/hip-to-llvm/test_newop.mlir

hip-mlir-opt test/lit/e2e/test_newop_model.mlir \
  --hipdnn-pipeline | FileCheck test/lit/e2e/test_newop_model.mlir
```

---

### Step 9: Build and Validate

**Build**:

```bash
# Clean build (if needed)
python build.py --clean

# Normal build
python build.py

# Run LIT tests
ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests
```

**Validation Checklist**:
- [ ] Build completes without errors
- [ ] All three LIT tests pass (onnx-to-hip, hip-to-llvm, e2e)
- [ ] Manual `hip-mlir-opt` invocations work
- [ ] (Optional) End-to-end test with `hip-onnx-runner` on real ONNX model

---

## Common Patterns

Reusable code patterns extracted from existing implementations.

### Pattern: Get Context Parameter

```cpp
auto ctxOrFailure = getContextArg(op, rewriter);
if (mlir::failed(ctxOrFailure))
  return mlir::failure();
mlir::Value context = *ctxOrFailure;
```

### Pattern: Create Empty Tensor (DPS)

```cpp
mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
```

### Pattern: Handle Dynamic Dimensions (ONNX-to-HIP)

```cpp
llvm::SmallVector<mlir::Value> dynamicDims;
for (unsigned i = 0; i < resultType.getRank(); ++i) {
  if (resultType.isDynamicDim(i)) {
    auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
    auto dimI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), dim);
    dynamicDims.push_back(dimI64);
  }
}
```

### Pattern: Extract ONNX Attributes

```cpp
// Integer (required)
auto attr = op->getAttrOfType<mlir::IntegerAttr>("attribute_name");
if (!attr) return mlir::failure();
int64_t value = attr.getInt();

// Float (optional with default)
auto floatAttr = op->getAttrOfType<mlir::FloatAttr>("float_attr");
double value = floatAttr ? floatAttr.getValueAsDouble() : 0.0;

// String
auto strAttr = op->getAttrOfType<mlir::StringAttr>("str_attr");
std::string value = strAttr ? strAttr.getValue().str() : "NONE";

// Array
auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>("array_attr");
for (auto elem : arrayAttr) { /* process */ }
```

### Pattern: Lambda Helpers (HIP-to-LLVM)

```cpp
auto createI64Const = [&](int64_t value) -> Value {
  return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                  rewriter.getI64IntegerAttr(value));
};

auto createF64Const = [&](double value) -> Value {
  return LLVM::ConstantOp::create(rewriter, loc, f64Type,
                                  rewriter.getF64FloatAttr(value));
};
```

### Pattern: Optional Operand Handling

**In Conversion Layer**:
```cpp
auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
  if (idx >= op->getNumOperands()) return mlir::Value{};
  mlir::Value v = op->getOperand(idx);
  if (!v || mlir::isa<mlir::NoneType>(v.getType())) return mlir::Value{};
  return v;
};
```

**In Lowering Layer**:
```cpp
Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
auto getMemRefPtrOrNull = [&](Value memref) -> Value {
  if (!memref) return nullPtr;
  return extractContiguousMemRefPtr(memref, rewriter, loc);
};
```

### Pattern: String to Enum Conversion

```cpp
// Define mapping (top of file or anonymous namespace)
static int64_t convertModeToEnum(llvm::StringRef mode) {
  return llvm::StringSwitch<int64_t>(mode)
      .Case("default", kModeDefault)
      .Case("fast", kModeFast)
      .Default(-1);
}

// Use in matchAndRewrite
int64_t modeEnum = convertModeToEnum(modeAttr.getValue());
if (modeEnum < 0) {
  return rewriter.notifyMatchFailure(op, "invalid mode");
}
```

### Pattern: Compute Total Elements (Dynamic Dims)

```cpp
Value numElements = createI64Const(1);
MemRefDescriptor outputDesc(adaptor.getOutput());

for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
  Value dimSize;
  if (outputType.isDynamicDim(dimIdx)) {
    dimSize = outputDesc.size(rewriter, loc, dimIdx);  // Runtime
  } else {
    dimSize = createI64Const(outputType.getDimSize(dimIdx));  // Compile-time
  }
  numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
}
```

### Pattern: Batch Dimension Computation

```cpp
// For MatMul-like ops: flatten batch dimensions (all dims except last 2)
Value batch = createI64Const(1);
for (int64_t i = 0; i < rank - 2; ++i) {
  if (inputType.isDynamicDim(i)) {
    Value dimSize = MemRefDescriptor(adaptor.getA()).size(rewriter, loc, i);
    batch = LLVM::MulOp::create(rewriter, loc, batch, dimSize);
  } else {
    batch = LLVM::MulOp::create(rewriter, loc, batch,
                                createI64Const(inputType.getDimSize(i)));
  }
}
```

### Pattern: Element Size Calculation

```cpp
Type elemType = outputType.getElementType();
unsigned elemBits = elemType.getIntOrFloatBitWidth();  // f32→32, f16→16
unsigned elemBytes = elemBits / 8;
Value elemSizeVal = createI64Const(elemBytes);
```

---

## Troubleshooting Checklist

Common errors and solutions (based on CLAUDE.md gotchas).

### Compilation Errors

**1. `undefined reference to 'wrap_newop'`**
- **Cause**: Runtime function not declared as `extern "C"`
- **Fix**: Add declaration in `hipdnn_ep_runtime.h` inside `extern "C" { }` block

**2. `use of undeclared identifier 'populateNewOpConversionPatterns'`**
- **Cause**: Populate function not declared in header
- **Fix**: Add declaration in `OnnxToHipUtils.h`

**3. CMake can't find new file**
- **Cause**: Not added to CMakeLists.txt
- **Fix**: Add `.cpp` file to corresponding `add_library()` in CMakeLists.txt

**4. `error: no member named 'getDpsInitsMutable'`**
- **Cause**: Forgot to manually implement interface methods
- **Fix**: Add `getDpsInitsMutable()` and `getEffects()` in `HipDialect.cpp`

**5. Bufferization fails**
- **Cause**: Op not registered for bufferization
- **Fix**: Add `attachInterface` call in `tools/hip-mlir-opt/hip-mlir-opt.cpp`

**6. `error: 'OptionalAttr' does not name a type`**
- **Cause**: Used non-existent `OptionalAttr` in TableGen
- **Fix**: Use `DefaultValuedAttr<Type, "sentinel_value">` instead

### LIT Test Failures

**7. `CHECK` pattern doesn't match**
- **Cause**: MLIR IR format differs from expectation
- **Fix**: Run `hip-mlir-opt` manually, inspect actual output, adjust CHECK pattern

**8. `onnx.NewOp` not replaced**
- **Cause**: Pattern not registered or match logic incorrect
- **Fix**: Verify `populateNewOpConversionPatterns` called in `OnnxToHip.cpp`

**9. `llvm.func @wrap_newop` not found**
- **Cause**: Lowering pattern not registered
- **Fix**: Verify `populateNewOpLoweringPatterns` called in `HipToLLVM.cpp`

### Runtime Errors (with hip-onnx-runner)

**10. Segfault or crash**
- **Cause**: Pointer not passed correctly or signature mismatch
- **Fix**: Check lowering layer parameter order matches runtime function signature

**11. Output incorrect (mock mode)**
- **Cause**: Mock implementation logic error
- **Fix**: Mock mode is for compile/lowering verification, not correctness. Use real runtime for accuracy validation.

**12. C++ name mangling error**
- **Cause**: Missing `extern "C"` in runtime function declaration
- **Fix**: Ensure declaration in `hipdnn_ep_runtime.h` is inside `extern "C" { }` block

---

## Complete Examples

### Example 1: Sqrt (Simple Element-wise Operation)

**Reference Files**:
- `lib/Conversion/OnnxToHip/PowerConversion.cpp` (lines 12-91)
- `lib/Conversion/HipToLLVM/PowerLowering.cpp` (lines 22-118)
- `include/hip/Dialect/IR/HipOps.td` (search for `Hip_SqrtOp`)
- `lib/Runtime/mock/mock_gpu.cpp` (search for `wrap_power`)
- `test/lit/Conversion/onnx-to-hip/test_sqrt.mlir`
- `test/lit/Conversion/hip-to-llvm/test_sqrt.mlir`
- `test/lit/e2e/test_sqrt_model.mlir`

**Key Characteristics**:
- Simple element-wise operation
- No attributes
- Template-based lowering (`PowerOpLowering` shared with Reciprocal)
- Runtime: `wrap_power(..., alpha=0.0, beta=1.0, gamma=0.5)`

### Example 2: Range (Complex with Attributes and Dynamic Output)

**Reference Files**:
- `lib/Conversion/OnnxToHip/RangeConversion.cpp` (lines 174-241)
- `lib/Conversion/HipToLLVM/RangeLowering.cpp`
- `include/hip/Dialect/IR/HipOps.td` (search for `Hip_RangeOp`)
- `lib/Runtime/mock/mock_gpu.cpp` (search for `wrap_range`)
- `test/lit/Conversion/onnx-to-hip/test_range.mlir`
- `test/lit/e2e/test_range_model.mlir`

**Key Characteristics**:
- Dynamic output shape (computed from start/limit/delta)
- Attributes validation
- Scalar inputs extracted via tensor.extract

### Example 3: MatMulNBits (Microsoft Contrib Domain)

**Reference Files**:
- `lib/Conversion/OnnxToHip/MatMulNBitsConversion.cpp`
- `include/hip/Dialect/IR/HipOps.td` (search for `Hip_MatMulNBitsOp`)
- `lib/Runtime/mock/mock_gpu.cpp` (search for `wrap_matmul_nbits`)

**Key Characteristics**:
- Microsoft contrib domain (`com.microsoft`)
- Checks `function_name` and `domain_name` attributes
- Optional inputs handling (zero_points, g_idx, bias)
- Zero-points element type validation

---

## Quick Reference Card

### File Checklist

When adding `NewOp`, modify these files:

**ONNX → HIP Layer**:
- [ ] `lib/Conversion/OnnxToHip/NewOpConversion.cpp` (create)
- [ ] `lib/Conversion/OnnxToHip/OnnxToHip.cpp` (register pattern)
- [ ] `lib/Conversion/OnnxToHip/OnnxToHipUtils.h` (declare populate func)
- [ ] `lib/Conversion/OnnxToHip/CMakeLists.txt` (add .cpp)

**HIP Dialect**:
- [ ] `include/hip/Dialect/IR/HipOps.td` (TableGen definition)
- [ ] `lib/Dialect/IR/HipDialect.cpp` (getDpsInitsMutable + getEffects)
- [ ] `tools/hip-mlir-opt/hip-mlir-opt.cpp` (bufferization registration)

**HIP → LLVM Layer**:
- [ ] `lib/Conversion/HipToLLVM/NewOpLowering.cpp` (create)
- [ ] `lib/Conversion/HipToLLVM/HipToLLVM.cpp` (register pattern)
- [ ] `lib/Conversion/HipToLLVM/HipToLLVMUtils.h` (declare populate func + kWrapNewOp)
- [ ] `lib/Conversion/HipToLLVM/CMakeLists.txt` (add .cpp)

**Runtime**:
- [ ] `lib/Runtime/hipdnn_ep_runtime.h` (declare wrap_newop in extern "C")
- [ ] `lib/Runtime/mock/mock_gpu.cpp` (implement wrap_newop)

**Tests**:
- [ ] `test/lit/Conversion/onnx-to-hip/test_newop.mlir` (create)
- [ ] `test/lit/Conversion/hip-to-llvm/test_newop.mlir` (create)
- [ ] `test/lit/e2e/test_newop_model.mlir` (create)

### Command Quick Reference

```bash
# Build
python build.py [--clean]

# Run all LIT tests
ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests

# Run specific test
hip-mlir-opt <test.mlir> <passes> | FileCheck <test.mlir>

# Example passes
--hip-add-context-arg --convert-onnx-to-hip
--convert-hip-to-llvm
--hipdnn-pipeline
```

### Key Constraints Summary

1. **TableGen**: Inherit from `Hip_DpsOp`, first arg is `Hip_ContextType:$ctx`, use `DefaultValuedAttr` not `OptionalAttr`
2. **Runtime**: Declare in `extern "C" { }`, first param `RuntimeState *state`, return `int`
3. **Function names**: Define constants in `HipToLLVMUtils.h` as `kWrap*`
4. **Bufferization**: Register in `tools/hip-mlir-opt/hip-mlir-opt.cpp`
5. **LIT tests**: Three files (onnx-to-hip, hip-to-llvm, e2e)

---

## Additional Resources

- **ONNX Operators**: https://onnx.ai/onnx/operators/
- **Microsoft Contrib Operators**: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
- **MLIR Documentation**: https://mlir.llvm.org/
- **Project CLAUDE.md**: Build gotchas and critical constraints
- **TableGen Language Reference**: https://mlir.llvm.org/docs/DefiningDialects/Operations/

---

**End of Document**
