//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef ONNX_DIALECT
#define ONNX_DIALECT

include "mlir/IR/DialectBase.td"

def Onnx_Dialect : Dialect {
  let name = "onnx";
  let summary = "ONNX operations as imported from an ONNX Runtime graph";
  let description = [{
    Models the ONNX operations the compiler consumes, so the conversions that
    lower them can match a typed operation and read a typed accessor instead of
    a name string and an attribute lookup.

    Every standard ONNX operation the dialect can spell has a definition in
    `Ops/`, one file per operation, alongside the hand-written assist
    operations. What it cannot spell is an operation whose signature needs an
    ONNX sequence, optional or map container, or a string tensor, since none of
    those has an MLIR type here. The dialect permits unknown operations, so
    those, a contrib-domain operation, and an `onnx.Custom` keep flowing
    through unmodeled.

    Element types come from the schema, as they do upstream, and an ONNX signed
    integer is a signless MLIR integer. Rank is never constrained, because the
    importer emits `tensor<*xT>` for a value whose shape it does not know; see
    `docs/design/unranked-tensor-handling.md` for the cross-repo contract.

    An ONNX bool is `i1`. The importer writes `ui8` for BOOL to match ONNX's
    byte storage, so that mapping has to change before its output parses here.

    A modeled operation carries every operand and result its schema declares,
    with `onnx.NoValue` standing in for an omitted operand and the `none` type
    for an omitted result. The stub dialect registers no operation, so under it
    an omitted argument may instead be absent.

    Inherent attributes are properties, as they are upstream, so a modeled
    operation prints them in a `<{...}>` group and leaves the trailing
    dictionary to the attributes no schema describes. The stub dialect claims
    the same namespace without modeling anything, so under it every attribute
    stays in the dictionary.
  }];
  let cppNamespace = "::mlir::onnx";
}

#endif // ONNX_DIALECT
