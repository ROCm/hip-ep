/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_ONNXDIMPARAMS_H
#define HIP_CONVERSION_ONNXTOHIP_ONNXDIMPARAMS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::hip {

inline constexpr llvm::StringLiteral kOnnxDimParamsModuleAttr =
    "hipdnn.onnx_dim_params_v1";
inline constexpr llvm::StringLiteral kBroadcastDimSourcesAttr =
    "hipdnn.broadcast_dim_sources";

class OnnxDimParams {
public:
  static FailureOr<OnnxDimParams> parse(ModuleOp module);

  LogicalResult annotateBroadcastDimSources(func::FuncOp funcOp) const;

private:
  llvm::StringMap<llvm::SmallVector<std::string>> byValueName;
};

/// Return one canonical operand index per result axis. `-1` means that the
/// exact runtime broadcast merge remains required.
llvm::SmallVector<int64_t> getBroadcastDimSources(Operation *onnxOp);

} // namespace mlir::hip

#endif // HIP_CONVERSION_ONNXTOHIP_ONNXDIMPARAMS_H
