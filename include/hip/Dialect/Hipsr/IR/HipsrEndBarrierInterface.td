//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// EndBarrierInterface: marks an op that can end the current pool domain.
//

#ifndef HIPSR_END_BARRIER_INTERFACE
#define HIPSR_END_BARRIER_INTERFACE

include "mlir/IR/OpBase.td"

def EndBarrierInterface : OpInterface<"EndBarrierInterface"> {
  let cppNamespace = "::mlir::hipsr";

  let description = [{
    Marks an op that ends the current pool domain. The op stays in the current
    domain; the next domain starts at the first op that uses this op's result.
    Its output shape comes from its output data (e.g. NonZero), so the real
    shape is only known after the op runs.

    isEndBarrier() usually returns true (NonZero always ends a domain). It is
    an interface method for symmetry with StartBarrierInterface, so an op can
    opt out per instance if needed.
  }];

  let methods = [
    InterfaceMethod<
      "Return true if this op instance ends the current pool domain.",
      "bool", "isEndBarrier", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]
    >,
  ];
}

#endif // HIPSR_END_BARRIER_INTERFACE
