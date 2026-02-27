//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_TYPES
#define HIP_TYPES

include "hip/Dialect/IR/HipDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Hip_Type<string name, string typeMnemonic>
    : TypeDef<Hip_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Hip_HandleType : Hip_Type<"Handle", "handle"> {
  let summary = "HIP runtime handle type";
  let description = [{
    Represents an opaque HIP runtime handle used for managing HIP operations.
  }];
}

#endif // HIP_TYPES
