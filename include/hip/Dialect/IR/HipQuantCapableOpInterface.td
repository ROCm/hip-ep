//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_QUANT_CAPABLE_OP_INTERFACE
#define HIP_QUANT_CAPABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

// An op quantized or not by its element types (`!quant.uniform` vs float).
// Front end only: every method is default-implemented, and kernel support is
// the backend's (hasQuantKernel). See docs/design/quant-types.md.
def HipQuantCapableOpInterface : OpInterface<"HipQuantCapableOpInterface"> {
  let cppNamespace = "::mlir::hip";

  let methods = [
    InterfaceMethod<
      [{
        Tensors whose element types carry the quantization, in `quant_types`
        order: the shaped inputs (optional ones only when present), then the
        output. Defaults to the DPS inputs and inits; override if an op has
        shaped operands that are not quantized (e.g. indices).
      }],
      "::llvm::SmallVector<::mlir::Value>", "getQuantOperands", (ins), [{}], [{
        ::mlir::Operation *op = $_op.getOperation();
        ::llvm::SmallVector<::mlir::Value> operands;
        auto shaped = [](::mlir::Value value) {
          return ::llvm::isa<::mlir::ShapedType>(value.getType());
        };
        if (auto dps = ::llvm::dyn_cast<::mlir::DestinationStyleOpInterface>(op)) {
          ::llvm::append_range(operands,
                               ::llvm::make_filter_range(dps.getDpsInputs(), shaped));
          ::llvm::append_range(operands, dps.getDpsInits());
          return operands;
        }
        ::llvm::append_range(operands,
                             ::llvm::make_filter_range(op->getOperands(), shaped));
        ::llvm::append_range(operands, op->getResults());
        return operands;
      }]>,

    // Queries: read `quant_types` after legalization, tensor types before.
    InterfaceMethod<
      [{
        Element type per quant operand (getQuantOperands order): a quant type
        if quantized, else the float type.
      }],
      "::llvm::SmallVector<::mlir::Type>", "getOperandQuantTypes", (ins), [{}], [{
        ::mlir::Operation *op = $_op.getOperation();
        ::llvm::SmallVector<::mlir::Type> types;
        if (auto attr = op->getAttrOfType<::mlir::ArrayAttr>("quant_types")) {
          for (auto type : attr.getAsRange<::mlir::TypeAttr>())
            types.push_back(type.getValue());
          return types;
        }
        for (::mlir::Value value : $_op.getQuantOperands())
          types.push_back(::mlir::getElementTypeOrSelf(value.getType()));
        return types;
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
