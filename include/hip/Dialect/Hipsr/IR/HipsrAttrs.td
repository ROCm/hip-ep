//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_ATTRS
#define HIPSR_ATTRS

include "hip/Dialect/Hipsr/IR/HipsrDialect.td"
include "hip/Dialect/Hipsr/IR/HipsrEnums.td"
include "mlir/IR/AttrTypeBase.td"

def Hipsr_MemorySpaceAttr : AttrDef<Hipsr_Dialect, "MemorySpace"> {
  let mnemonic = "mem";
  let summary = "hipsr memory space attribute";
  let description = [{
    Says where a buffer lives, used as a `memref` memory space:
    `device` (VRAM), `host` (pageable), `pinned` (page-locked host),
    `managed` (unified). Example: `memref<4xf32, #hipsr.mem<device>>`.

    Every hipsr `memref` must have one of these memory spaces; a memref
    with no memory space is not allowed. The `Hipsr_*MemRef` type
    constraints in `HipsrTypes.td` check this once ops start using them.
  }];
  let parameters = (ins EnumParameter<Hipsr_MemorySpaceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

#endif // HIPSR_ATTRS
