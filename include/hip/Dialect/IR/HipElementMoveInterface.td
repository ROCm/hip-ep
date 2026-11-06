//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_ELEMENT_MOVE_INTERFACE
#define HIP_ELEMENT_MOVE_INTERFACE

include "mlir/IR/OpBase.td"

// Interface for ops whose every result element is a copy of one element of
// their moved operands, or one selected among them by comparison, with no
// arithmetic. A per-tensor quantize or an exact widening cast then commutes
// with the op:
//   q(f(x1, x2, ...)) == f(q(x1), q(x2), ...)
// so hip-fuse-rocmlir can run the op on the quantized values, and a cast
// round trip around it folds away (see docs/design/element-move-interface.md).
//
// An op may implement the interface only when every element type that a
// caller rebuilds it on (8-bit integers for the quantize users, f16 and bf16
// for the cast fold) lowers both inside a rocMLIR kernel and on the HIP
// runtime, since the rebuilt op may stay outside a kernel.
def ElementMoveOpInterface : OpInterface<"ElementMoveOpInterface"> {
  let description = [{
    Marks ops that only move or select elements, and rebuilds them on another
    element type.
  }];

  let cppNamespace = "::mlir::hip";

  let methods = [
    InterfaceMethod<
      [{
        Whether this instance only moves or selects elements. An op whose
        other modes compute (a linear resize) returns false for them.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"movesElements">,
    InterfaceMethod<
      [{
        The operands whose elements reach the result. Other operands (shapes,
        indices, offsets) are kept as they are.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpOperand *>",
      /*methodName=*/"getMovedOperands">,
    InterfaceMethod<
      [{
        Build the same op before this one, on `moved` (one value per
        getMovedOperands entry, in order) and with result element type
        `elementType`. Returns the single result.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"cloneWithElementType",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
                    "::mlir::ValueRange":$moved,
                    "::mlir::Type":$elementType)>
  ];
}

#endif // HIP_ELEMENT_MOVE_INTERFACE
