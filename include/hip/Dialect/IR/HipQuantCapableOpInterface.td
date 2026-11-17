//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_QUANT_CAPABLE_OP_INTERFACE
#define HIP_QUANT_CAPABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

// An op that is quantized or not by its element types: `!quant.uniform`
// tensors mean it dequantizes its inputs and quantizes its output, float
// tensors mean the float op. See docs/design/quant-types.md.
def HipQuantCapableOpInterface : OpInterface<"HipQuantCapableOpInterface"> {
  let cppNamespace = "::mlir::hip";

  let methods = [
    InterfaceMethod<
      [{
        Tensors whose element types carry the quantization, in `quant_types`
        order: the data inputs (optional ones only when present), then the
        output.
      }],
      "::llvm::SmallVector<::mlir::Value>", "getQuantOperands">,
    InterfaceMethod<
      [{
        True if a kernel takes these quantized element types (inputs in
        getQuantOperands order, then the result).
      }],
      "bool", "supportsQuantTypes",
      (ins "::mlir::TypeRange":$inputElementTypes,
           "::mlir::Type":$resultElementType)>
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
