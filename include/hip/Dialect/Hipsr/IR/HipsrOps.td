//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_OPS
#define HIPSR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrAttrs.td"
include "hip/Dialect/Hipsr/IR/HipsrDialect.td"
include "hip/Dialect/Hipsr/IR/HipsrTypes.td"

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

// Common base for all ops in the hipsr dialect. Downstream PRs define concrete
// ops (shape-region terminators, shaped compute ops, ...) on top of this.
class Hipsr_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hipsr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Destination-passing style op base class
//===----------------------------------------------------------------------===//

// Base for HIPSR destination-passing operations. Each `outs` operand is tied
// to the result at the same position.
class Hipsr_DpsOp<string mnemonic, list<Trait> traits = []> :
    Hipsr_Op<mnemonic, !listconcat([
        DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                                  ["getDpsInitsMutable"]>
      ], traits)> {
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
}

include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.td"
include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.td"
include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.td"
include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.td"
include "hip/Dialect/Hipsr/IR/HipsrComputeOp.td"
include "hip/Dialect/Hipsr/IR/HipsrComputeYieldOp.td"
include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.td"
include "hip/Dialect/Hipsr/IR/HipsrExpandOp.td"
include "hip/Dialect/Hipsr/IR/HipsrCastOp.td"
include "hip/Dialect/Hipsr/IR/HipsrAddOp.td"
include "hip/Dialect/Hipsr/IR/HipsrMulOp.td"
include "hip/Dialect/Hipsr/IR/HipsrEqualOp.td"
include "hip/Dialect/Hipsr/IR/HipsrTransposeOp.td"
include "hip/Dialect/Hipsr/IR/HipsrGatherOp.td"
include "hip/Dialect/Hipsr/IR/HipsrSliceOp.td"
include "hip/Dialect/Hipsr/IR/HipsrScatterNDOp.td"
include "hip/Dialect/Hipsr/IR/HipsrConstantOp.td"
include "hip/Dialect/Hipsr/IR/HipsrGetPoolOp.td"
include "hip/Dialect/Hipsr/IR/HipsrPreserveShapeOp.td"
include "hip/Dialect/Hipsr/IR/HipsrAllocOutputOp.td"

#endif // HIPSR_OPS
