//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The base class every onnx operation is defined on.
//

#ifndef ONNX_BASE
#define ONNX_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Onnx/IR/OnnxDialect.td"

// The mnemonic is the ONNX operator type verbatim (`onnx.MatMul`), which is
// what the importer emits and what the name-matched conversions expect.
//
// No operation declares an `assemblyFormat`, so the generic form stays the
// printed form.
class Onnx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Onnx_Dialect, mnemonic, traits>;

#endif // ONNX_BASE
