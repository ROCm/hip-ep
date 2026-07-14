//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_DIALECT
#define HIPSR_DIALECT

include "mlir/IR/DialectBase.td"

def Hipsr_Dialect : Dialect {
  let name = "hipsr";
  let summary = "Declarative AMD GPU compute dialect with first-class shape regions";
  let description = [{
    An AMD GPU compute dialect that represents tensor computations
    declaratively and models dynamic shapes as first-class shape regions.
  }];
  let cppNamespace = "::mlir::hipsr";

  // Lets the dialect parse/print `#hipsr.mem<...>`.
  let useDefaultAttributePrinterParser = 1;
}

#endif // HIPSR_DIALECT
