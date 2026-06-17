//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT
#define HIP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def Hip_Dialect : Dialect {
  let name = "hip";
  let summary = "HIP runtime dialect";
  let description = [{
    A dialect for HIP (Heterogeneous-compute Interface for Portability) operations.
    Provides operations for memory management and GPU compute dispatch.
  }];
  let cppNamespace = "::mlir::hip";
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    // Public op-registration shim for out-of-tree plugins.
    //
    // `Dialect::addOperations` is protected, so a dynamically-loaded plugin
    // (.so loaded via hip-mlir-opt --load-dialect-plugin) cannot inject real
    // `hip.*` ops without this hook. This shim exposes it so a plugin's
    // DialectRegistry extension can register its ops at dialect-load time,
    // making plugin-contributed ops genuine `hip.*` ops that reuse the in-tree
    // bufferization and lowering machinery. See plugins/fusion/.
    template <typename... Ops>
    void registerPluginOps() { addOperations<Ops...>(); }
  }];
}

#endif // HIP_DIALECT
