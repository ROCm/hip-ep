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

#endif // HIP_DIALECT
