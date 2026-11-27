//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_TD
#define HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_TD

include "hip/Dialect/Ggml/IR/GgmlDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// Shape expressions
//===----------------------------------------------------------------------===//
//
// A GGML graph is captured at one request size, so every extent, stride, and
// storage offset arrives as a concrete number. The compiler recovers how each
// one depends on the request and records that dependence here.
//
// Today every recovered dependence is affine, and the frontend rejects a graph
// whose shapes are not -- an integer division that does not divide evenly, or a
// product of two varying extents, has no expression in this language. Should a
// future architecture need one (a padded extent, say), it becomes another
// attribute implementing this interface rather than a new field on this one.
// Readers go through `ShapeExprAttrInterface` and the free functions beside it,
// so adding a form does not touch the code that consumes expressions.

def Ggml_ShapeExprAttrInterface : AttrInterface<"ShapeExprAttrInterface"> {
  let cppNamespace = "::mlir::ggml";
  let description = [{
    One request-dependent quantity: an extent, a byte stride, or an offset.

    An implementation must be a pure function of the symbols it names, so that
    evaluating it twice with the same assignment yields the same result and the
    compiler may evaluate it at any request size it wants to check.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Evaluates the expression under `lookup`, which maps a symbol name to
        its value. Returns failure when `lookup` does not know a symbol this
        expression names, or when the arithmetic overflows.
      }],
      "::llvm::FailureOr<int64_t>", "evaluate",
      (ins "::llvm::function_ref<::std::optional<int64_t>(::llvm::StringRef)>":$lookup)
    >,
    InterfaceMethod<
      [{
        Appends every symbol this expression names, without duplicates and in a
        stable order.
      }],
      "void", "getSymbols",
      (ins "::llvm::SmallVectorImpl<::llvm::StringRef> &":$symbols)
    >,
    InterfaceMethod<
      [{
        Returns the value this expression always takes, or nullopt when it
        depends on a symbol.
      }],
      "::std::optional<int64_t>", "getConstantValue"
    >,
  ];
}

#endif // HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_TD
