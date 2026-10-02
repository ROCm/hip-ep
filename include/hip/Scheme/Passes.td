//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Pass definitions for the Scheme pass runner infrastructure.
//

#ifndef SCHEME_PASSES
#define SCHEME_PASSES

include "mlir/Pass/PassBase.td"

def SchemePass : Pass<"scheme-pass", "mlir::ModuleOp"> {
  let summary = "Run a pass implemented in Scheme";
  let description = [{
    General-purpose pass runner that loads a Scheme library and calls its
    run-pass function. The Scheme module is specified via the 'module' option
    using slash-separated R6RS library name notation.

    Examples:
      --scheme-pass="module=passes/onnx-to-hipsr"
      --scheme-pass="module=passes/hip-fusion"
      --scheme-pass="module=passes/print"
      --scheme-pass="module=passes/onnx-to-hipsr,log-level=debug"
  }];
  let options = [
    Option<"moduleName", "module", "std::string", /*default=*/"\"\"",
           "Scheme module to load (slash-separated, e.g. 'passes/onnx-to-hipsr')">,
    Option<"logLevel", "log-level", "std::string", /*default=*/"\"warning\"",
           "Logging level: trace, debug, info, warning, error, fatal">
  ];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::onnx::OnnxDialect",
    "::mlir::func::FuncDialect",
    "::mlir::shape::ShapeDialect"
  ];
}

#endif // SCHEME_PASSES
