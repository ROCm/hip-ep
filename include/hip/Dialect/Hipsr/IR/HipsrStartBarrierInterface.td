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
    Marks an op that starts a new pool domain because determining its output
    shape requires reading runtime tensor elements that encode dimensions
    (e.g. Expand's shape operand). A dedicated `hipsr.placeholder` that produces
    the op's DPS tensor init may precede it in that domain.

    This is an interface method rather than a trait because barrier status can
    depend on the operation instance: an Expand with a compile-time shape
    attribute does not require that runtime read and is not a barrier, even if
    its result type remains dynamic. An Expand with a runtime shape operand is
    a barrier. The default returns true; an op overrides isStartBarrier() when
    it can determine its output shape without reading runtime shape data.
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
