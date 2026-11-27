//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===- GgmlShapeRule.td - Deriving a result's extents ------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLSHAPERULE_TD
#define HIP_DIALECT_GGML_IR_GGMLSHAPERULE_TD

include "mlir/IR/OpBase.td"

def Ggml_ShapeRule : OpInterface<"GgmlShapeRule"> {
  let description = [{
    How a result's extents follow from its operands' is a property of the
    operation, so each operation states it here rather than in a pass that
    matches on what the operation happens to be.

    An operation without this interface has no rule, and a graph that uses one
    is refused rather than left at whatever extents the capture observed.
  }];
  let cppNamespace = "::mlir::ggml";
  let methods = [
    InterfaceMethod<
      [{Derives this operation's result extents from its operands'.}],
      "::mlir::LogicalResult", "deriveResultShape",
      (ins "::mlir::ggml::ShapeDerivation &":$derivation)>,
    InterfaceMethod<
      [{
        States which of this operation's operand extents GGML requires to be
        equal. Most operations require nothing beyond what their rule already
        derives, so this says nothing by default.
      }],
      "void", "constrainOperands",
      (ins "::mlir::ggml::ShapeConstraints &":$constraints),
      /*methodBody=*/"", /*defaultImplementation=*/[{ return; }]>
  ];
}

#endif // HIP_DIALECT_GGML_IR_GGMLSHAPERULE_TD
