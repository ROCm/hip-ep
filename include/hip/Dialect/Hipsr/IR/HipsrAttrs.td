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

// A constant whose bytes live in an external file: absolute path + byte offset
// + byte size. Read on demand (streaming) via the dialect's file-map cache.
def Hipsr_FileSourceAttr : AttrDef<Hipsr_Dialect, "FileSource"> {
  let mnemonic = "file_source";
  let summary = "File-backed constant source (path, offset, size)";
  let parameters = (ins "::mlir::StringAttr":$path,
                        "int64_t":$offset,
                        "int64_t":$size);
  let assemblyFormat = "`<` $path `,` $offset `,` $size `>`";
}

// A constant whose bytes live at a raw memory address (an ORT tensor pointer,
// zero-copy). Valid only within the compiling process/session.
def Hipsr_MemSourceAttr : AttrDef<Hipsr_Dialect, "MemSource"> {
  let mnemonic = "mem_source";
  let summary = "Memory-backed constant source (address, size)";
  let parameters = (ins "int64_t":$address, "int64_t":$size);
  let assemblyFormat = "`<` $address `,` $size `>`";
}

#endif // HIPSR_ATTRS
