/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

namespace morphizen {
namespace mlir_impl {

class MLIRGraph;

// Legalize ONNX INT4/UINT4 initializer constants (when GBQ has no explicit
// zero_points) and annotate every GatherBlockQuantized with storage semantics
// (`unsigned_quant_storage`) and quantize_axis (inferred from shapes when
// omitted). When zero_points is present, keep logical ONNX data/scales/zp shapes.
void legalizeGatherBlockQuantizedInt4Constants(MLIRGraph &graph,
                                               mlir::Block &block);
void legalizeGatherBlockQuantizedInt4Constants(MLIRGraph &graph,
                                               mlir::func::FuncOp func);

} // namespace mlir_impl
} // namespace morphizen
