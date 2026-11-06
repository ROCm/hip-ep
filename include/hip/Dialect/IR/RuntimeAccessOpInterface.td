/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_RUNTIME_ACCESS_OP_INTERFACE
#define HIP_RUNTIME_ACCESS_OP_INTERFACE

include "mlir/IR/OpBase.td"

def RuntimeAccessOpInterface : OpInterface<"RuntimeAccessOpInterface"> {
  let cppNamespace = "::mlir::hip";
  let description = [{
    Backend-provided access guarantees. Payload accesses use the context
    stream; host reads use an ordered copy and wait, except for the operand
    returned by getEarlyHostRead. This does not make the op a completion barrier.
    MemoryEffectOpInterface describes which operands are read or written.
    Alternative lowerings must preserve these guarantees to use this interface.
  }];
  let methods = [
    InterfaceMethod<"Return the operand read on the host before waiting, or null.",
                    "::mlir::Value", "getEarlyHostRead">,
    InterfaceMethod<"Whether read-only inputs support host-mapped storage.",
                    "bool", "supportsHostMappedInputs">
  ];
}

#endif
