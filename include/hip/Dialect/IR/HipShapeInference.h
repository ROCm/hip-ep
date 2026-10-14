/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIPSHAPEINFERENCE_H
#define HIP_DIALECT_IR_HIPSHAPEINFERENCE_H

#include "mlir/IR/DialectInterface.h"

namespace mlir {
namespace hip {

/// Opts a target dialect into `--hip-infer-shapes`.
///
/// The pass considers only operations that also implement
/// DestinationStyleOpInterface and ReifyRankedShapedTypeOpInterface. Merely
/// implementing reification is intentionally insufficient because refining a
/// result also requires updating its destination operand.
class ShapeInferenceDialectInterface
    : public DialectInterface::Base<ShapeInferenceDialectInterface> {
public:
  explicit ShapeInferenceDialectInterface(Dialect *dialect) : Base(dialect) {}
};

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIPSHAPEINFERENCE_H
