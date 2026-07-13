//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_ENUMS
#define HIPSR_ENUMS

include "mlir/IR/EnumAttr.td"

// Numeric values match the AMDGPU address spaces: host = 0 (flat),
// device = 1 (global), pinned = 2, managed = 3.
def Hipsr_Host    : I32EnumAttrCase<"Host",    0, "host">;
def Hipsr_Device  : I32EnumAttrCase<"Device",  1, "device">;
def Hipsr_Pinned  : I32EnumAttrCase<"Pinned",  2, "pinned">;
def Hipsr_Managed : I32EnumAttrCase<"Managed", 3, "managed">;

def Hipsr_MemorySpaceKind : I32EnumAttr<"MemorySpaceKind",
    "hipsr memory space kind",
    [Hipsr_Host, Hipsr_Device, Hipsr_Pinned, Hipsr_Managed]> {
  let cppNamespace = "::mlir::hipsr";
  // The enum is wrapped by MemorySpaceAttr (HipsrAttrs.td), so the syntax
  // is `#hipsr.mem<device>`. Don't generate a separate attribute for the enum.
  let genSpecializedAttr = 0;
}

#endif // HIPSR_ENUMS
