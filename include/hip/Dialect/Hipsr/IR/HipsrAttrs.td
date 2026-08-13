//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_ATTRS
#define HIPSR_ATTRS

include "hip/Dialect/Hipsr/IR/HipsrDialect.td"
include "hip/Dialect/Hipsr/IR/HipsrEnums.td"
include "mlir/IR/AttrTypeBase.td"

def Hipsr_MemorySpaceAttr :
    EnumAttr<Hipsr_Dialect, Hipsr_MemorySpace, "mem"> {
  let description = [{
    Says where a buffer lives, used as a `memref` memory space:
    `device` (VRAM), `host` (pageable), `pinned` (page-locked host),
    `managed` (unified). Example: `memref<4xf32, #hipsr.mem<device>>`.

    Every hipsr `memref` must have one of these memory spaces; a memref
    with no memory space is not allowed. The `Hipsr_*MemRef` type
    constraints in `HipsrTypes.td` enforce this on op operands and results.
  }];
  let assemblyFormat = "`<` $value `>`";
}

def Hipsr_PlaceholderTypeAttr :
    EnumAttr<Hipsr_Dialect, Hipsr_PlaceholderType, "placeholder_type"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // HIPSR_ATTRS
