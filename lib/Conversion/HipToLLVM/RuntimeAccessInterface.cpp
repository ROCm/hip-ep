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

struct TopKAccess
    : RuntimeAccessOpInterface::ExternalModel<TopKAccess, TopKOp> {
  Value getEarlyHostRead(Operation *op) const {
    return cast<TopKOp>(op).getK();
  }
  bool supportsHostMappedInputs(Operation *) const { return false; }
};
} // namespace

void mlir::hip::registerRuntimeAccessOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipDialect *) {
    // Register only guarantees audited against the native lowering/runtime.
    // An unregistered op provides no ordering or mapped-input guarantee.
    CastOp::attachInterface<OrderedAccess<CastOp, true>>(*ctx);
    AddOp::attachInterface<OrderedAccess<AddOp, true>>(*ctx);
    PadOp::attachInterface<OrderedAccess<PadOp, true>>(*ctx);
    SliceOp::attachInterface<OrderedAccess<SliceOp, true>>(*ctx);
    TopKOp::attachInterface<TopKAccess>(*ctx);
    MulOp::attachInterface<OrderedAccess<MulOp>>(*ctx);
    MinOp::attachInterface<OrderedAccess<MinOp>>(*ctx);
    MaxOp::attachInterface<OrderedAccess<MaxOp>>(*ctx);
    SubOp::attachInterface<OrderedAccess<SubOp>>(*ctx);
    GqaOp::attachInterface<OrderedAccess<GqaOp>>(*ctx);
    NonZeroOp::attachInterface<OrderedAccess<NonZeroOp>>(*ctx);
    AndOp::attachInterface<OrderedAccess<AndOp>>(*ctx);
    OrOp::attachInterface<OrderedAccess<OrOp>>(*ctx);
    NotOp::attachInterface<OrderedAccess<NotOp>>(*ctx);
    EqualOp::attachInterface<OrderedAccess<EqualOp>>(*ctx);
    LessOp::attachInterface<OrderedAccess<LessOp>>(*ctx);
    WhereOp::attachInterface<OrderedAccess<WhereOp>>(*ctx);
    ExpandOp::attachInterface<OrderedAccess<ExpandOp>>(*ctx);
    GatherOp::attachInterface<OrderedAccess<GatherOp>>(*ctx);
    GatherElementsOp::attachInterface<OrderedAccess<GatherElementsOp>>(*ctx);
    ScatterElementsOp::attachInterface<OrderedAccess<ScatterElementsOp>>(*ctx);
    CumSumOp::attachInterface<OrderedAccess<CumSumOp>>(*ctx);
    GeluOp::attachInterface<OrderedAccess<GeluOp>>(*ctx);
    SigmoidOp::attachInterface<OrderedAccess<SigmoidOp>>(*ctx);
    TanhOp::attachInterface<OrderedAccess<TanhOp>>(*ctx);
    SoftplusOp::attachInterface<OrderedAccess<SoftplusOp>>(*ctx);
    SwishOp::attachInterface<OrderedAccess<SwishOp>>(*ctx);
    SqrtOp::attachInterface<OrderedAccess<SqrtOp>>(*ctx);
    ReciprocalOp::attachInterface<OrderedAccess<ReciprocalOp>>(*ctx);
    ReduceMeanOp::attachInterface<OrderedAccess<ReduceMeanOp>>(*ctx);
    ReduceSumOp::attachInterface<OrderedAccess<ReduceSumOp>>(*ctx);
    RmsNormOp::attachInterface<OrderedAccess<RmsNormOp>>(*ctx);
    SkipRmsNormOp::attachInterface<OrderedAccess<SkipRmsNormOp>>(*ctx);
    RopeOp::attachInterface<OrderedAccess<RopeOp>>(*ctx);
    MatmulOp::attachInterface<OrderedAccess<MatmulOp>>(*ctx);
    MatMulNBitsOp::attachInterface<OrderedAccess<MatMulNBitsOp>>(*ctx);
    QMoEOp::attachInterface<OrderedAccess<QMoEOp>>(*ctx);
    MiopenSoftmaxOp::attachInterface<OrderedAccess<MiopenSoftmaxOp>>(*ctx);
  });
}
