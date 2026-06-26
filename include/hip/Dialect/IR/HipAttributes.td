//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_ATTRIBUTES
#define HIP_ATTRIBUTES

include "hip/Dialect/IR/HipDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Memory space kind enum
//===----------------------------------------------------------------------===//

def HIP_Device  : I32EnumAttrCase<"Device",  0, "device">;
def HIP_Host    : I32EnumAttrCase<"Host",    1, "host">;
def HIP_Pinned  : I32EnumAttrCase<"Pinned",  2, "pinned">;
def HIP_Managed : I32EnumAttrCase<"Managed", 3, "managed">;

def HIP_MemorySpaceKind : I32EnumAttr<"MemorySpaceKind",
    "HIP memory space kind",
    [HIP_Device, HIP_Host, HIP_Pinned, HIP_Managed]> {
  let cppNamespace = "::mlir::hip";
  // We wrap the enum in the custom MemorySpaceAttr below (so the syntax is
  // `#hip.mem<device>`, not a bare enum attribute); do not emit a specialized
  // enum-attribute class.
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Memory space attribute  (#hip.mem<device|host|pinned|managed>)
//===----------------------------------------------------------------------===//

def HIP_MemorySpaceAttr : AttrDef<Hip_Dialect, "MemorySpace"> {
  let mnemonic = "mem";
  let summary = "HIP memory space attribute";
  let description = [{
    Records where a HIP buffer lives, used as a `memref` memory space:

    - `device`:  GPU VRAM (hipMalloc), device-only access
    - `host`:    pageable host memory (malloc/stack), CPU-only access
    - `pinned`:  pinned host memory (hipHostMalloc), fast transfers / zero-copy
    - `managed`: unified memory (hipMallocManaged), auto-migration

    Syntax: `#hip.mem<device>` (likewise `host`, `pinned`, `managed`). Used as
    `memref<4xf32, #hip.mem<device>>`.

    See `docs/design/memory-space-design.md`.
  }];
  let parameters = (ins EnumParameter<HIP_MemorySpaceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

#endif // HIP_ATTRIBUTES
