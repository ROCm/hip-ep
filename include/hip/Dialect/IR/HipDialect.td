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
}

// Marks an op whose result EXTENT depends on input *data values* (not just
// input shapes) — e.g. NonZero (trailing dim = count of non-zero elements).
// The C++ trait class is declared in HipDialect.h. Generic passes / the EP
// metadata builder gate data-dependent-output handling on this trait via
// `op->hasTrait<mlir::OpTrait::hip::DataDependentResult>()`.
def DataDependentResult : NativeOpTrait<"DataDependentResult"> {
  let cppNamespace = "::mlir::OpTrait::hip";
}

#endif // HIP_DIALECT
