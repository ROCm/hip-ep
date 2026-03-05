/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna-compiler/Dialect/Hip/IR/HipBufferize.h"
#include "udna-compiler/Dialect/Hip/IR/HipDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace hip {

void registerHipBufferizableOpInterfaceModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipDialect *dialect) {
    // All 14 non-elementwise compute ops use the standard DPS model.
    ConvOp::attachInterface<HipDstBufferizableModel<ConvOp>>(*ctx);
    GemmOp::attachInterface<HipDstBufferizableModel<GemmOp>>(*ctx);
    MaxPoolOp::attachInterface<HipDstBufferizableModel<MaxPoolOp>>(*ctx);
    AvgPoolOp::attachInterface<HipDstBufferizableModel<AvgPoolOp>>(*ctx);
    MatMulOp::attachInterface<HipDstBufferizableModel<MatMulOp>>(*ctx);
    GroupQueryAttentionOp::attachInterface<
        HipDstBufferizableModel<GroupQueryAttentionOp>>(*ctx);
    MulOp::attachInterface<HipDstBufferizableModel<MulOp>>(*ctx);
    SubOp::attachInterface<HipDstBufferizableModel<SubOp>>(*ctx);
    SigmoidOp::attachInterface<HipDstBufferizableModel<SigmoidOp>>(*ctx);
    GatherOp::attachInterface<HipDstBufferizableModel<GatherOp>>(*ctx);
    ReduceSumOp::attachInterface<HipDstBufferizableModel<ReduceSumOp>>(*ctx);
    RotaryEmbeddingOp::attachInterface<
        HipDstBufferizableModel<RotaryEmbeddingOp>>(*ctx);
    SimplifiedLayerNormOp::attachInterface<
        HipDstBufferizableModel<SimplifiedLayerNormOp>>(*ctx);
    SkipSimplifiedLayerNormOp::attachInterface<
        HipDstBufferizableModel<SkipSimplifiedLayerNormOp>>(*ctx);

    // Elementwise ops (relu, cast) use the elementwise model which allows
    // in-place aliasing of input and output buffers.
    ReluOp::attachInterface<HipElementwiseBufferizableModel<ReluOp>>(*ctx);
    CastOp::attachInterface<HipElementwiseBufferizableModel<CastOp>>(*ctx);
  });
}

} // namespace hip
} // namespace mlir
