//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_TRANSFORMS_PASSES_TD
#define HIP_DIALECT_GGML_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def GgmlSymbolizeShapesPass
    : Pass<"ggml-symbolize-shapes", "mlir::ModuleOp"> {
  let summary = "Recover request-dependent extents from one captured graph";
  let description = [{
    Turns a module whose extents are the ones a single capture happened to
    observe into one whose extents say how they vary with the request.

    The graph reaches the compiler with every extent already evaluated, because
    GGML records the value a dimension had rather than the expression that
    produced it. The request size is therefore recoverable only from where it
    entered -- the runtime inputs the module's `ggml.symbols` name -- and from
    what each operation does to its operands' extents. This pass seeds the
    former and deduces the latter.

    Every derived extent is checked against the capture it came from: evaluating
    it at that capture's symbol values must reproduce the observed extent
    exactly. A wrong rule therefore fails on the graph that produced it, naming
    the operation and dimension, rather than silently baking one request size
    into the artifact. A value the rules do not resolve is an error for the same
    reason.

    The tensor `#ggml.layout` encoding receives the derived extents, byte
    strides, and storage offset. Tensor shapes gain `?` wherever an extent
    varies, and the entry signature follows its argument and returned value
    types. Storage spans are derived from the root's layout rather than written
    as per-value attributes.

    A module marked `ggml.infer_symbols` asks which of its declared symbols
    are one size instead: each is a guess, and in IR order each operation's
    requirements of its operands (`constrainOperands`) and a reshape's
    elements make two guesses one, or a guess the number the other side
    leaves it; what cannot be derived is fixed at what the capture saw. The
    module gets `ggml.inferred` (each guess's class, or its number) and keeps
    its layouts.
  }];
  let dependentDialects = [
    "::mlir::ggml::GgmlDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // HIP_DIALECT_GGML_TRANSFORMS_PASSES_TD
