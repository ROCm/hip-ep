/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShapeProvenanceAnalysis.h - Host shape dataflow ----------*- C++ -*-===//
//
// Function-level sparse dataflow for host-side shape payloads and canonical
// tensor dimensions used by Reshape shape materialization.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_SHAPEPROVENANCEANALYSIS_H
#define HIP_CONVERSION_ONNXTOHIP_SHAPEPROVENANCEANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace hip {

enum class ShapeValueProof {
  Positive,
  NonNegative,
  MinusOne,
  Unknown,
  Invalid,
};

struct ShapeProvenanceExpr {
  enum class Kind {
    Constant,
    TensorDim,
    HostScalar,
  };

  Kind kind;
  int64_t constant = 0;
  mlir::Value value;
  int64_t dimension = 0;
  ShapeValueProof proof = ShapeValueProof::Unknown;
};

/// Function-level analysis of shape payload and dimension provenance.
///
/// Each instance is intended for one successful solve over a stable function.
/// Query before mutating the analyzed IR; copied descriptors contain non-owning
/// `mlir::Value` handles and do not extend the lifetime of referenced IR.
class ShapeProvenanceAnalysis {
public:
  explicit ShapeProvenanceAnalysis(mlir::func::FuncOp funcOp);
  ~ShapeProvenanceAnalysis();

  ShapeProvenanceAnalysis(const ShapeProvenanceAnalysis &) = delete;
  ShapeProvenanceAnalysis &operator=(const ShapeProvenanceAnalysis &) = delete;

  mlir::LogicalResult run();

  mlir::FailureOr<llvm::SmallVector<ShapeProvenanceExpr>>
  getPayload(mlir::Value value) const;

  /// Return whether \p expr and the selected dimension of \p value have the
  /// same canonical dimension root.
  bool isEquivalentToDimension(const ShapeProvenanceExpr &expr,
                               mlir::Value value, int64_t dimension) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Return whether \p funcOp contains a dynamic onnx.Reshape whose rank-1 i64
/// target operand could benefit from shape-provenance analysis.
bool hasEligibleReshapeShapeProvenanceCandidate(mlir::func::FuncOp funcOp);

/// Consume one completed analysis and materialize all provable onnx.Reshape
/// target shapes. This preserves the proof-marker contract consumed by
/// ReshapeConversion.
mlir::LogicalResult
materializeReshapeShapeOperands(mlir::func::FuncOp funcOp,
                                const ShapeProvenanceAnalysis &analysis);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_SHAPEPROVENANCEANALYSIS_H
