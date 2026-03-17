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

def Hip_ContextType : Hip_Type<"Context", "context"> {
  let summary = "Opaque HIP execution context";
  let description = [{
    Represents an opaque HIP execution context. Lowered to !llvm.ptr.
  }];
}

#endif // HIP_TYPES
