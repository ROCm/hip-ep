//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_QUANT_CAPABLE_OP_INTERFACE
#define HIP_QUANT_CAPABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

// An op quantized or not by its element types (`!quant.uniform` vs float).
// Front end only: every method is default-implemented, and kernel support is
// the backend's (hasQuantKernel). DPS ops that do not declare it get it at
// dialect load (attachQuantCapableInterface, with a deny-list); the verifier
// below does not run for those, so hip-legalize-quant-types checks them.
// See docs/design/quant-types.md.
def HipQuantCapableOpInterface : OpInterface<"HipQuantCapableOpInterface"> {
  let cppNamespace = "::mlir::hip";

  let methods = [
    InterfaceMethod<
      [{
        Tensors whose element types carry the quantization, in `quant_types`
        order: the shaped inputs (optional ones only when present), then the
        output. Defaults to the DPS inputs and inits. Integer operands
        (indices) are listed but never quantized.
      }],
      "::llvm::SmallVector<::mlir::Value>", "getQuantOperands", (ins), [{}], [{
        return ::mlir::hip::getDefaultQuantOperands($_op.getOperation());
      }]>,

    // Queries: read `quant_types` after legalization, tensor types before.
    InterfaceMethod<
      [{
        Element type per quant operand (getQuantOperands order): a quant type
        if quantized, else the float type.
      }],
      "::llvm::SmallVector<::mlir::Type>", "getOperandQuantTypes", (ins), [{}], [{
        return ::mlir::hip::getDefaultOperandQuantTypes(
            $_op.getOperation(), $_op.getQuantOperands());
      }]>,
    InterfaceMethod<"True if any quant operand is quantized.",
      "bool", "isQuantized", (ins), [{}], [{
        return ::llvm::any_of($_op.getOperandQuantTypes(), [](::mlir::Type type) {
          return ::llvm::isa<::mlir::quant::QuantizedType>(type);
        });
      }]>,
    InterfaceMethod<"Element type of input `index` (getQuantOperands order).",
      "::mlir::Type", "getInputQuantType", (ins "unsigned":$index), [{}], [{
        ::llvm::SmallVector<::mlir::Type> types = $_op.getOperandQuantTypes();
        assert(index + 1 < types.size() && "not an input");
        return types[index];
      }]>,
    InterfaceMethod<"Element type of the result.",
      "::mlir::Type", "getResultQuantType", (ins), [{}], [{
        return $_op.getOperandQuantTypes().back();
      }]>,
    InterfaceMethod<
      [{
        Per-tensor quant type of quant operand `index` (the result is last);
        null if float or per-axis.
      }],
      "::mlir::quant::UniformQuantizedType", "getUniformQuantType",
      (ins "unsigned":$index), [{}], [{
        return ::llvm::dyn_cast<::mlir::quant::UniformQuantizedType>(
            $_op.getOperandQuantTypes()[index]);
      }]>
  ];

  // All quantized or none; after legalization, `quant_types` matches the
  // storage types.
  let verify = [{
    return ::mlir::hip::verifyQuantTypes(
        $_op, ::llvm::cast<HipQuantCapableOpInterface>($_op)
                  .getQuantOperands());
  }];
}

#endif // HIP_QUANT_CAPABLE_OP_INTERFACE
