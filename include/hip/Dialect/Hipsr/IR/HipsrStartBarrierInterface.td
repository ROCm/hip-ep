//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// StartBarrierInterface: marks an op that can start a new pool domain.
//

#ifndef HIPSR_START_BARRIER_INTERFACE
#define HIPSR_START_BARRIER_INTERFACE

include "mlir/IR/OpBase.td"

def StartBarrierInterface : OpInterface<"StartBarrierInterface"> {
  let cppNamespace = "::mlir::hipsr";

  let description = [{
    Marks an op that starts a new pool domain because its output shape comes
    from its input data (e.g. Expand). A dedicated `hipsr.placeholder` that
    produces the op's DPS tensor init may precede it in that domain.

    This is an interface method rather than a trait because barrier status can
    depend on the operation instance: an Expand with a compile-time shape
    attribute is not a barrier (its output shape is static), while one with a
    runtime shape operand is. The default returns true; an op overrides
    isStartBarrier() to return false when its shape is static.
  }];

  let methods = [
    InterfaceMethod<
      "Return true if this op instance starts a new pool domain.",
      "bool", "isStartBarrier", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]
    >,
  ];
}

#endif // HIPSR_START_BARRIER_INTERFACE
