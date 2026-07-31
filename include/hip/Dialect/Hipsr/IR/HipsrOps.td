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

// Common base for all operations in the hipsr dialect.
class Hipsr_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hipsr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Shaped DPS op base class
//===----------------------------------------------------------------------===//

// Base for shaped compute ops (MatMul, Add, ...). The `outs` inits are the
// outputs; their placeholders own the corresponding shape regions.
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
include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.td"
include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.td"
include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.td"
include "hip/Dialect/Hipsr/IR/HipsrExpandOp.td"
include "hip/Dialect/Hipsr/IR/HipsrCastOp.td"
include "hip/Dialect/Hipsr/IR/HipsrAddOp.td"
include "hip/Dialect/Hipsr/IR/HipsrConstantOp.td"
include "hip/Dialect/Hipsr/IR/HipsrGetPoolOp.td"

#endif // HIPSR_OPS
