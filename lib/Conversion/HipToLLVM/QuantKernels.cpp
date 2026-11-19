/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Quantized kernel table: one entry per op kind with a quantized runtime
// kernel. Adding a kernel means adding an entry here; the ops stay unchanged.

#include "hip/Conversion/HipToLLVM/QuantKernels.h"
#include "hip/Dialect/IR/HipQuantTypes.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

bool isUniformOfWidth(Type type, ArrayRef<unsigned> widths) {
  auto quantType = dyn_cast<quant::UniformQuantizedType>(type);
  return quantType &&
         llvm::is_contained(widths, quantType.getStorageTypeIntegralWidth());
}

// wrap_qelementwise: per-tensor scales, one storage type for lhs, rhs and
// output (i8/ui8/i16/ui16/i32).
bool qElementwise(TypeRange inputs, Type result) {
  auto out = dyn_cast<quant::UniformQuantizedType>(result);
  if (!out || inputs.size() != 2)
    return false;
  IntegerType storage = hip::getHipStorageType(out);
  unsigned width = storage.getWidth();
  if (width != 8 && width != 16 && !(width == 32 && storage.isSignless()))
    return false;
  return llvm::all_of(inputs, [&](Type type) {
    auto in = dyn_cast<quant::UniformQuantizedType>(type);
    return in && hip::getHipStorageType(in) == storage;
  });
}

// wrap_qmatmul, or wrap_qgemm with a bias: per-tensor scales. A and the result
// are 8 or 16 bit, B is 8 bit (the kernel splits only A into byte planes), and
// the bias may also be 32 bit (the int32 an ONNX quantizer emits).
bool qMatmul(TypeRange inputs, Type result) {
  size_t numInputs = inputs.size();
  if (numInputs < 2 || numInputs > 3)
    return false;
  if (!isUniformOfWidth(inputs[0], {8, 16}) ||
      !isUniformOfWidth(inputs[1], {8}) || !isUniformOfWidth(result, {8, 16}))
    return false;
  return numInputs == 2 || isUniformOfWidth(inputs[2], {8, 16, 32});
}

struct QuantKernel {
  llvm::StringLiteral opName;
  bool (*supports)(TypeRange inputs, Type result);
};

constexpr QuantKernel kQuantKernels[] = {
    {"hip.add", qElementwise}, // wrap_qelementwise
    {"hip.matmul", qMatmul},   // wrap_qmatmul, wrap_qgemm (bias)
    {"hip.gemm", qMatmul},     // wrap_qgemm
};

} // namespace

bool mlir::hip::hasQuantKernel(Operation *op, TypeRange inputElementTypes,
                               Type resultElementType) {
  StringRef name = op->getName().getStringRef();
  for (const QuantKernel &kernel : kQuantKernels)
    if (kernel.opName == name)
      return kernel.supports(inputElementTypes, resultElementType);
  return false;
}
