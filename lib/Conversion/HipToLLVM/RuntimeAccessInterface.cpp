/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/RuntimeAccessOpInterface.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::hip;

namespace {
template <typename Op, bool Mapped = false>
struct OrderedAccess
    : RuntimeAccessOpInterface::ExternalModel<OrderedAccess<Op, Mapped>, Op> {
  Value getEarlyHostRead(Operation *) const { return {}; }
  bool supportsHostMappedInputs(Operation *) const { return Mapped; }
};

template <typename Op> using MappedInputAccess = OrderedAccess<Op, true>;

struct TopKAccess
    : RuntimeAccessOpInterface::ExternalModel<TopKAccess, TopKOp> {
  Value getEarlyHostRead(Operation *op) const {
    return cast<TopKOp>(op).getK();
  }
  bool supportsHostMappedInputs(Operation *) const { return false; }
};
} // namespace

template <template <typename> class Model, typename... Ops>
static void attachAll(MLIRContext *ctx) {
  (Ops::template attachInterface<Model<Ops>>(*ctx), ...);
}

void mlir::hip::registerRuntimeAccessOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipDialect *) {
    // Register only guarantees audited against the native lowering/runtime.
    // An unregistered op provides no ordering or mapped-input guarantee.
    attachAll<MappedInputAccess, CastOp, AddOp, PadOp, SliceOp>(ctx);
    TopKOp::attachInterface<TopKAccess>(*ctx);
    attachAll<OrderedAccess, MulOp, MinOp, MaxOp, SubOp, GqaOp, NonZeroOp,
              AndOp, OrOp, NotOp, EqualOp, LessOp, WhereOp, ExpandOp, GatherOp,
              GatherElementsOp, ScatterElementsOp, CumSumOp, GeluOp, GegluOp,
              SigmoidOp, TanhOp, SoftplusOp, SwishOp, SqrtOp, ReciprocalOp,
              ReduceMeanOp, ReduceSumOp, RmsNormOp, RmsNormAddOp, SkipRmsNormOp,
              RopeOp, MatmulOp, MatMulNBitsOp, QMoEOp, MiopenSoftmaxOp>(ctx);
  });
}
