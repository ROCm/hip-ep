/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_COMMON_DIM_SPEC_MATERIALIZER_H
#define HIPDNN_EP_COMMON_DIM_SPEC_MATERIALIZER_H

#include "hip/Dialect/IR/HipShapeInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace hip {

//===----------------------------------------------------------------------===//
// DimSpecMaterializer: emit MLIR SSA that evaluates a DimSpec at runtime.
//===----------------------------------------------------------------------===//
//
// Most DimSpec evaluation happens host-side in the EP (Categories A, B, D)
// or implicitly via runtime slot bookkeeping (Category C). This helper
// exists for the *future* case where a consumer-side lowering (e.g. a HIP
// op consuming a NonZero output) needs the actual slot value at
// codegen-time to compute its own buffer size or kernel grid.
//
// Strategy: walk the DimSpec tree and emit an i64 SSA value in `builder`
// for each node:
//   - Static          → `arith.constant <value> : i64`
//   - InputDim        → `arith.constant <static_value>` when the consumer
//                       has a statically-known operand dim; otherwise
//                       emit a call into a "read_input_dim" helper
//                       (caller-supplied lambda).
//   - InputValueI64   → caller-supplied lambda; the consumer typically
//                       has the host-side i64 value in a constant or has
//                       loaded it via tensor.extract earlier in the IR.
//   - RuntimeSlot     → `call @hipdnn_ep_state_read_dim(state, slot_id)`.
//   - Binary nodes    → emit the matching `arith.*` op.
//
// The helper is intentionally generic: the caller passes lambdas for the
// two operand-style leaves so the materializer remains agnostic to
// surrounding IR (memref descriptors vs SSA constants).

struct DimSpecMaterializerCallbacks {
  // Emit an i64 SSA value for the dim extent of the consumer's operand
  // `input_index` at `dim_index`. Typically the consumer holds the memref
  // descriptor of that operand and reads `desc.sizes[dim_index]`.
  std::function<mlir::Value(unsigned input_index, unsigned dim_index)>
      readInputDim;

  // Emit an i64 SSA value for the i64 element at `flat_offset` of the
  // consumer's operand `input_index`. Most consumers do not need this
  // path; provide nullptr if the caller is certain InputValueI64 cannot
  // appear in the spec.
  std::function<mlir::Value(unsigned input_index, int64_t flat_offset)>
      readInputValueI64;

  // The runtime state pointer (RuntimeState*) — used as the first arg of
  // the `hipdnn_ep_state_read_dim` call when materialising RuntimeSlot.
  mlir::Value statePtr;
};

// Emit IR to evaluate `spec` and return the resulting i64 SSA value. The
// caller is responsible for ensuring `cbacks.statePtr` is valid when the
// spec contains a RuntimeSlot leaf.
mlir::Value materializeDimSpec(mlir::OpBuilder &builder, mlir::Location loc,
                               const DimSpec &spec,
                               const DimSpecMaterializerCallbacks &cbacks);

} // namespace hip
} // namespace mlir

#endif // HIPDNN_EP_COMMON_DIM_SPEC_MATERIALIZER_H
