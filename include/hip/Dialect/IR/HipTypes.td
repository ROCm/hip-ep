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

def Hip_LoopFrameType : Hip_Type<"LoopFrame", "loop_frame"> {
  let summary = "Per-invocation storage for shape-changing loop carriers";
  let description = [{
    Opaque handle owned by the loop runtime for one invocation of one
    `hip.loop`. The handle gives an outlined body access to the dedicated
    double-buffered carrier allocator without using a process-global or
    RuntimeState-global carrier slot. Lowered to !llvm.ptr.
  }];
}

#endif // HIP_TYPES
