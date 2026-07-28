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
    Marks an op whose output shape requires reading runtime tensor elements
    that encode dimensions, such as Expand's shape operand.

    Barrier status is instance-specific. A compile-time shape attribute avoids
    the runtime read even when the result type is dynamic; a runtime shape
    operand still requires it. The default returns true, and ops override
    `isStartBarrier()` when no runtime shape-data read is needed.
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
