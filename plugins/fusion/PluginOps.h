/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PLUGIN_FUSION_OPS_H
#define HIP_PLUGIN_FUSION_OPS_H

// Pull in the in-tree `hip` dialect + op declarations (HipDialect, ContextType,
// AddOp, MulOp, ...). The plugin's own op below is generated into the same
// ::mlir::hip namespace so it becomes a real hip.* op.
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "plugins/fusion/FusedMulAddOps.h.inc"

#endif // HIP_PLUGIN_FUSION_OPS_H
