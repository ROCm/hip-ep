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
    Marks an op that reads tensor values at runtime to compute its output shape,
    such as Expand reading its shape operand.

    An op may start a domain in some cases but not others. Expand does not start
    one when its shape is an attribute, even if the result type is dynamic. It
    does start one when the shape comes from a tensor. The default is true. Ops
    return false from `isStartBarrier()` when they do not need runtime shape
    data.
  }];

  let methods = [
    InterfaceMethod<
      "Return true if this op starts a new pool domain.",
      "bool", "isStartBarrier", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]
    >,
  ];
}

#endif // HIPSR_START_BARRIER_INTERFACE
