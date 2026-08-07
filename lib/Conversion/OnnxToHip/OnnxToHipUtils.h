/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipUtils.h - Shared helpers for ONNX-to-HIP patterns --------===//
//
// Shared utility functions and forward declarations used by per-operator
// conversion files.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_UTILS_H
#define HIP_CONVERSION_ONNXTOHIP_UTILS_H

#include "ReadbackScalar.h"

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir {
namespace hip {

inline constexpr llvm::StringLiteral kHostShapeOperandAttr =
    "hip.host_shape_operand";
inline constexpr llvm::StringLiteral kHostShapeNoMinusOneAttr =
    "hip.host_shape_no_minus_one";
inline constexpr llvm::StringLiteral kHostShapeInputDimMapAttr =
    "hip.host_shape_input_dim_map";

/// Return whether an arith.index_cast preserves every nonnegative value under
/// the generated ABI's 64-bit index model.
bool isLosslessShapeIndexCast(mlir::arith::IndexCastOp op);

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Sanitize an arbitrary string (typically an ONNX node name) into a valid
/// MLIR bare identifier fragment.  Non-alphanumeric characters are replaced
/// with '_', consecutive underscores are collapsed, and leading/trailing
/// underscores are stripped.  Returns an empty string if the input yields
/// no usable characters.
inline std::string sanitizeForMlirIdentifier(llvm::StringRef raw) {
  std::string sanitized;
  sanitized.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  // Collapse runs of underscores and trim leading/trailing ones.
  std::string result;
  result.reserve(sanitized.size());
  bool lastWasUnderscore = true; // suppress leading '_'
  for (char c : sanitized) {
    if (c == '_') {
      if (!lastWasUnderscore)
        result.push_back(c);
      lastWasUnderscore = true;
    } else {
      result.push_back(c);
      lastWasUnderscore = false;
    }
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result;
}

/// Create a tensor.empty for a DPS init operand.  Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
/// Suitable for ops where the output shape aligns positionally with one input
/// (e.g., softmax, element-wise).
inline mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value source) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(builder, loc, source, dimIdx));
  }
  return mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

enum class CompileTimeConstantScope {
  InlineOnly,
  IncludeExternalized,
};

/// Return the dense payload of a compile-time tensor constant. Inline matching
/// covers `arith.constant` and pre-externalization ops carrying a `value`
/// attribute. `IncludeExternalized` additionally recognizes
/// `to_tensor(get_global)` when the global retains an initializer.
///
/// Externalized-global lookup remains conversion-side because generic HIP
/// dialect shape code must not depend on bufferization/global policy. Integer
/// payload parsing is nevertheless shared through
/// `mlir::hip::parseDenseIntElements`.
mlir::DenseElementsAttr
getCompileTimeConstantTensor(mlir::Value value,
                             CompileTimeConstantScope scope =
                                 CompileTimeConstantScope::IncludeExternalized);

/// Recognize \p value as a compile-time rank-0/rank-1 integer tensor, covering
/// the inline and optional externalized forms selected by \p scope.
bool extractConstantIntTensor(
    mlir::Value value, llvm::SmallVectorImpl<int64_t> &out,
    std::optional<int64_t> expectedRank = std::nullopt,
    CompileTimeConstantScope scope =
        CompileTimeConstantScope::IncludeExternalized);

/// Rank-1 convenience wrapper around `extractConstantIntTensor`.
///
/// This must recognize at least everything the reification helpers can
/// introspect (an inlined `arith.constant`). A converter that saw *fewer*
/// constants than reification would build its destination from a weaker rule
/// than the shape consumers observe, and the two would disagree.
bool extractConstantIntVector(
    mlir::Value value, llvm::SmallVectorImpl<int64_t> &out,
    CompileTimeConstantScope scope =
        CompileTimeConstantScope::IncludeExternalized);

/// Conversion-side RotaryEmbedding attributes after applying ONNX defaults and
/// the shared shape-based inference rules used by the standard and contrib
/// spellings of the op.
struct RotaryEmbeddingConfig {
  int64_t interleaved;
  int64_t numHeads;
  int64_t rotaryDim;
};

/// Resolve RotaryEmbedding's optional attributes from `input` and `cosCache`.
///
/// This deliberately remains conversion-side: the two ONNX spellings share
/// attribute defaults and layout inference, while hip.rope's result shape is
/// simply its input shape. Failures are reported through the pattern rewriter.
mlir::FailureOr<RotaryEmbeddingConfig>
resolveRotaryEmbeddingConfig(mlir::PatternRewriter &rewriter,
                             mlir::Operation *op, mlir::Value input,
                             mlir::Value cosCache);

/// Resolve the reduced axis list of an ONNX reduction op.
///
/// The axes arrive either as an `axes` attribute (opset < 13) or as an operand
/// (opset 13+) that may still be a compile-time constant. ONNX's empty-axes
/// semantics are applied here: with `noop_with_empty_axes = 0` an absent or
/// empty list reduces every axis, and with 1 it reduces nothing.
///
/// Returns `std::nullopt` when the axes are only known at runtime. The axis
/// mapping is then data-dependent and no shape rule applies, so both
/// destination construction and reification fall back to the `outs` shape.
/// Deciding this from one predicate is what keeps those two paths in agreement
/// -- gating the converter on the operand *count* while reification gates on
/// the operand being *constant* is exactly how they drift apart.
///
/// \p storage is caller-owned scratch that backs the returned view; it must
/// outlive the result and must not be modified while the view is in use.
std::optional<llvm::ArrayRef<int64_t>>
resolveReductionAxes(mlir::Operation *op, mlir::Value data,
                     int64_t noopWithEmptyAxes,
                     llvm::SmallVectorImpl<int64_t> &storage);

/// Resolve the ranked result type of an ONNX reduction op (ReduceMax / Sum /
/// Mean / Prod / ...).
///
/// Usually this is just the op's own result type. The ONNX importer can,
/// however, leave a reduction result UNRANKED (`tensor<*xT>`) when the input
/// carries dynamic symbolic dims and the axes are not explicit -- e.g. Phi's
/// `ReduceMax(position_ids)` feeding `GreaterOrEqual`. A bare
/// `mlir::cast<RankedTensorType>` on an unranked type is unchecked in release
/// builds and dereferences garbage -> crash. In that case, infer the result
/// type from the (ranked) input + statically-known reduced axes + keepdims,
/// per ONNX reduction shape semantics:
///   keepdims=1: reduced axes become size 1, other dims preserved.
///   keepdims=0: reduced axes are dropped.
///
/// The shape rule itself is shared with destination construction and
/// `reifyResultShapes` through `mlir::hip::inferReductionShape`.
///
/// \p reducedAxes reduced axis indices (may be negative; normalized by the
///                shared helper), or `std::nullopt` when the axes are only
///                known at runtime, in which case an unranked result cannot be
///                inferred.
/// Returns failure only when the result is unranked AND cannot be inferred
/// (unranked/absent input type, or runtime-only axes).
mlir::FailureOr<mlir::RankedTensorType>
inferReduceResultType(mlir::Operation *op, mlir::Value data,
                      std::optional<llvm::ArrayRef<int64_t>> reducedAxes,
                      int64_t keepdims);

/// Build a tensor.empty with the imported result type and the dynamic sizes
/// described by `reifiedShape`. Static reified dimensions are materialized as
/// constant index operands when the imported type keeps that dimension
/// dynamic; the existing `hip-infer-shapes` pass owns later type refinement.
mlir::FailureOr<mlir::Value> createEmptyTensorFromReifiedShape(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::RankedTensorType resultType,
    llvm::ArrayRef<mlir::OpFoldResult> reifiedShape);

/// Create a DPS init whose shape comes from one named semantic source operand.
/// This is the converter-side bridge for `Hip_DpsOp_SameShape`: both paths use
/// `reifyElementwiseSameShape`, and contradictory static result metadata is
/// rejected before any destination IR is emitted.
mlir::FailureOr<mlir::Value>
createSameShapeEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::Value source);

/// Derive a tensor type for a synthesized intermediate from a reified shape.
/// Constant dimensions become static; all other dimensions stay dynamic.
mlir::RankedTensorType
getTensorTypeFromReifiedShape(llvm::ArrayRef<mlir::OpFoldResult> reifiedShape,
                              mlir::Type elementType,
                              mlir::Attribute encoding = {});

/// Create a tensor.empty for a DPS init whose shape is the NumPy-style
/// broadcast of \p operands. Converter destination construction delegates to
/// the same dialect helper used by ReifyRankedShapedTypeOpInterface.
///
/// Before:
///   %init = tensor.empty(%lhs_dim) : tensor<?xf32>
/// After:
///   %lhs_is_one = arith.cmpi eq, %lhs_dim, %c1 : index
///   %extent = arith.select %lhs_is_one, %rhs_dim, %lhs_dim : index
///   %init = tensor.empty(%extent) : tensor<?xf32>
mlir::FailureOr<mlir::Value>
createBroadcastEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::ValueRange operands);

/// Create a tensor.empty for the DPS init of an ONNX reduction op.
///
/// When \p reducedAxes carries a value the extents come from
/// `reifyReductionResultShape` — the same helper that backs
/// `reifyResultShapes` — so the destination and the shape observed by
/// consumers implement one ONNX reduction shape function. This matters for
/// `keepdims=0`, where the output dimension order is not positional in the
/// input:
///
/// Before (positional, wrong once a reduced axis precedes a kept one):
///   %d0 = tensor.dim %data, %c0
///   %d1 = tensor.dim %data, %c1
///   %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
/// After (axes = [1, 2], keepdims = 0 on a rank-4 input):
///   %d0 = tensor.dim %data, %c0
///   %d3 = tensor.dim %data, %c3
///   %init = tensor.empty(%d0, %d3) : tensor<?x?xf32>
///
/// `std::nullopt` means the axes are only known at runtime: the mapping is then
/// data-dependent and no shape function applies, so the destination falls back
/// to a positional copy from `data`. `reifyReductionShape` bails on the same
/// condition and lifts the `outs` shape instead, so the two still agree.
mlir::FailureOr<mlir::Value>
createReductionEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType, mlir::Value data,
                           std::optional<llvm::ArrayRef<int64_t>> reducedAxes,
                           int64_t keepdims);

/// Return the ONNX reduction's axes operand, or materialize the resolved
/// attribute/default axes as one rank-1 i64 constant.
mlir::Value materializeReductionAxes(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Operation *op,
                                     llvm::ArrayRef<int64_t> resolvedAxes);

/// Get !hip.context from function argument 0. Returns failure if the
/// function has no arguments or the first argument is not !hip.context.
inline mlir::FailureOr<mlir::Value>
getContextArg(mlir::Operation *op, mlir::PatternRewriter &rewriter) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp)
    return rewriter.notifyMatchFailure(op, "not inside a function");
  auto &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0)
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  mlir::Value ctx = entry.getArgument(0);
  if (!mlir::isa<mlir::hip::ContextType>(ctx.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hip.context");
  return ctx;
}

/// Map an MLIR element type onto the HIPDNN_EP_DATATYPE_* enum that runtime
/// wrappers take as an `input_data_type` argument. Only the subset needed by
/// the converters that scan a raw buffer (hip.nonzero, and the Compress
/// selected-count scan built on top of it) is enumerated; any other element
/// type returns -1 so the caller fails conversion explicitly instead of
/// silently mis-classifying the buffer.
inline int64_t getHipdnnInputDataType(mlir::Type elemType) {
  if (elemType.isF32())
    return 0; // HIPDNN_EP_DATATYPE_FLOAT
  if (elemType.isF16())
    return 1; // HIPDNN_EP_DATATYPE_HALF
  if (elemType.isInteger(32))
    return 3; // HIPDNN_EP_DATATYPE_INT32
  if (elemType.isInteger(64))
    return 4; // HIPDNN_EP_DATATYPE_INT64
  if (elemType.isUnsignedInteger(8))
    return 7; // HIPDNN_EP_DATATYPE_UINT8 (ORT bool, ui8)
  if (elemType.isInteger(1) || elemType.isSignedInteger(8) ||
      elemType.isSignlessInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8 (bool/i1, signed/signless i8)
  return -1;
}

/// Shared ONNX reduction conversion skeleton. All six supported reductions
/// differ only by their HIP op type; axes/default resolution, unranked result
/// recovery, destination construction, and attributes stay centralized here.
template <typename HipOpTy>
class OnnxReductionToHip final : public mlir::RewritePattern {
public:
  OnnxReductionToHip(mlir::MLIRContext *ctx, llvm::StringRef onnxOpName)
      : RewritePattern(onnxOpName, /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduction expects at least one input and one output");

    auto context = getContextArg(op, rewriter);
    if (mlir::failed(context))
      return mlir::failure();

    mlir::Value data = op->getOperand(0);
    int64_t noopWithEmptyAxes = 0;
    if (auto attr =
            op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes"))
      noopWithEmptyAxes = attr.getSInt();
    int64_t keepdims = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
      keepdims = attr.getSInt();

    llvm::SmallVector<int64_t> axesStorage;
    std::optional<llvm::ArrayRef<int64_t>> reducedAxes =
        resolveReductionAxes(op, data, noopWithEmptyAxes, axesStorage);
    auto resultType = inferReduceResultType(op, data, reducedAxes, keepdims);
    if (mlir::failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "cannot infer unranked reduction result without ranked data and "
              "compile-time axes");

    mlir::Location loc = op->getLoc();
    auto init = createReductionEmptyTensor(rewriter, loc, *resultType, data,
                                           reducedAxes, keepdims);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "result type is incompatible with the reduction shape");

    mlir::Value axes = materializeReductionAxes(rewriter, loc, op, axesStorage);
    auto hipOp = HipOpTy::create(rewriter, loc, *context, data, axes, *init,
                                 rewriter.getI64IntegerAttr(keepdims),
                                 rewriter.getI64IntegerAttr(noopWithEmptyAxes));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// Lower a variadic ONNX elementwise op to a left-associated chain of pairwise
/// broadcasting HIP ops:
///
///   Max(a, b, c) -> hip.max(hip.max(a, b), c)
///
/// Each step's destination comes from `reifyBroadcastResultShape`, the same
/// helper backing `Hip_DpsOp_Broadcast`'s `reifyResultShapes`, so no step's
/// `outs` shape can disagree with the shape its consumers observe. Intermediate
/// steps take the type of their own broadcast shape; only the final step has to
/// match the imported ONNX result type. A single operand is the identity.
template <typename HipOpTy>
mlir::LogicalResult
lowerVariadicBroadcastChain(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) {
  llvm::StringRef opName = op->getName().getStringRef();
  unsigned numInputs = op->getNumOperands();
  if (numInputs == 0)
    return rewriter.notifyMatchFailure(op, llvm::Twine(opName) +
                                               " requires at least 1 input");

  if (numInputs == 1) {
    rewriter.replaceOp(op, op->getOperand(0));
    return mlir::success();
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  mlir::Value accumulate = op->getOperand(0);
  for (unsigned i : llvm::seq<unsigned>(1, numInputs)) {
    mlir::Value rhs = op->getOperand(i);
    mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> stepShape =
        mlir::hip::reifyBroadcastResultShape(rewriter, loc, {accumulate, rhs},
                                             [&]() { return op->emitError(); });
    if (mlir::failed(stepShape))
      return mlir::failure();

    bool isFinal = i == numInputs - 1;
    mlir::RankedTensorType stepResultType =
        isFinal ? resultType
                : getTensorTypeFromReifiedShape(*stepShape,
                                                resultType.getElementType());
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, stepResultType, *stepShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, llvm::Twine(opName) +
                  " result type is incompatible with the broadcast shape");

    accumulate = HipOpTy::create(rewriter, loc, context, accumulate, rhs, *init)
                     ->getResult(0);
  }

  rewriter.replaceOp(op, accumulate);
  return mlir::success();
}

/// Build a hip.gqa op for the Whisper-MHA / Whisper-encoder-Attention paths.
///
/// Emits one `hip.gqa` op with the 19-slot AttrSizedOperandSegments layout
/// and replaces \p op's results positionally with the hip.gqa's results
/// (callers with fewer than 3 results — e.g. cross-attn or fused-QKV
/// Attention — simply drop the trailing present_* outputs on the floor).
///
/// Defined in HipGqaBuilder.cpp.  Shared between
/// MultiHeadAttentionConversion.cpp and AttentionConversion.cpp.
///
/// \param numHeads        N (== kv_num_heads; HPG=1, multi-head, not GQA)
/// \param scale           ONNX `scale` attr, or 0.0 sentinel for "auto-compute
///                        1/sqrt(head_size) at runtime"
/// \param noCausal        true for bidirectional (encoder / cross-attn);
///                        false for causal (decoder self-attn)
/// \param pastKey/pastValue  Past KV-cache buffers; nullptr when absent
///                        (encoder / cross-attn).  When null the corresponding
///                        operand_segment_size is 0.
/// \param seqlensK        1-D [B] i32 (must already be in scope; either a
///                        runtime arg or an arith.constant)
/// \param totalSeqLen     scalar i32 (must already be in scope)
/// \param outputType      hip.gqa output[0] type (== query type)
/// \param presentKeyType  hip.gqa present_key type (BNSH); for cross-attn or
///                        encoder paths an unused-but-required DPS init buffer
/// \param presentValueType ditto for present_value
mlir::LogicalResult
buildHipGqaCall(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                mlir::Value context, mlir::Value query, mlir::Value key,
                mlir::Value value, mlir::Value pastKey, mlir::Value pastValue,
                mlir::Value seqlensK, mlir::Value totalSeqLen, int64_t numHeads,
                float scale, bool noCausal, mlir::RankedTensorType outputType,
                mlir::RankedTensorType presentKeyType,
                mlir::RankedTensorType presentValueType);

/// `readbackScalarToHost` recovering !hip.context from `op`. Falls back to a
/// bare `tensor.extract` when the function has no context arg (utility funcs /
/// pre-context-arg conversions) -- the only case where the unsynchronized load
/// is acceptable because there is no runtime to sync against.
inline mlir::Value
readbackScalarToHostOrExtract(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Operation *op,
                              mlir::Value rank0Tensor) {
  auto ctx = getContextArg(op, rewriter);
  if (mlir::failed(ctx))
    return mlir::tensor::ExtractOp::create(rewriter, loc, rank0Tensor,
                                           mlir::ValueRange{})
        .getResult();
  return readbackScalarToHost(rewriter, loc, *ctx, rank0Tensor);
}

inline mlir::Value
readbackScalarToIndexOrExtract(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Operation *op,
                               mlir::Value rank0Tensor) {
  mlir::Value scalar =
      readbackScalarToHostOrExtract(rewriter, loc, op, rank0Tensor);
  if (scalar.getType().isIndex())
    return scalar;
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), scalar);
}

/// `readbackShapeEntryToHost` recovering !hip.context from `op`. Falls back to
/// a bare `tensor.extract %shape[idx]` when there is no context arg.
inline mlir::Value
readbackShapeEntryToHostOrExtract(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Operation *op,
                                  mlir::Value shape, int64_t idx) {
  auto ctx = getContextArg(op, rewriter);
  if (mlir::failed(ctx)) {
    mlir::Value cidx = mlir::arith::ConstantIndexOp::create(rewriter, loc, idx);
    return mlir::tensor::ExtractOp::create(rewriter, loc, shape,
                                           mlir::ValueRange{cidx})
        .getResult();
  }
  return readbackShapeEntryToHost(rewriter, loc, *ctx, shape, idx);
}

/// Dynamic sequence extents used by GQA destination construction.
///
/// A null field means the corresponding result dimension is static.
struct GqaSequenceExtents {
  mlir::Value logical;
  mlir::Value presentKey;
  mlir::Value presentValue;
};

/// Validate and materialize the payload-dependent GQA sequence extents.
///
/// When any dynamic present-cache or QK extent needs `total_seq_len`, this
/// helper performs one synchronized readback and reuses the resulting
/// nonnegative index. A dynamic present-cache extent is the unsigned maximum
/// of that logical extent and its matching past-cache dim 2 when past exists;
/// without past it is the logical extent directly. Optional QK always uses the
/// logical extent, never cache capacity.
///
/// Pair/rank validation happens before any IR is emitted.
mlir::FailureOr<GqaSequenceExtents>
resolveGqaSequenceExtents(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Operation *op, mlir::Value totalSeqLen,
                          mlir::Value pastKey, mlir::Value pastValue,
                          mlir::RankedTensorType presentKeyType,
                          mlir::RankedTensorType presentValueType,
                          mlir::RankedTensorType outputQkType = nullptr);

/// Build an exact GQA present-cache destination in BNSH layout.
///
/// The caller supplies the sequence capacity resolved by
/// `resolveGqaSequenceExtents`.
/// Dynamic batch and head-size extents are derived from the query/KV operands;
/// all static extents remain authoritative in `resultType`.
mlir::FailureOr<mlir::Value>
createGqaPresentEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::RankedTensorType resultType, mlir::Value query,
                      mlir::Value key, mlir::Value totalSeqExtent,
                      int64_t numHeads, int64_t kvNumHeads);

/// Build an exact optional GQA QK destination `[B, H, S_q, S_kv]`.
mlir::FailureOr<mlir::Value>
createGqaQkEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::RankedTensorType resultType, mlir::Value query,
                 mlir::Value totalSeqExtent, int64_t numHeads);

// Pattern population functions (one per operator file)
void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateTransposeConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateElementwiseConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populatePowerConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateActivationConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateBiasGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateFastGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateCastConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateReduceMeanConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateReduceL2ConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateMatMulNBitsConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateGatherBlockQuantizedPreparePatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx);
void populateGatherBlockQuantizedConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx);
void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateConvTransposeConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx);
void populateNormConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateOnnxRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx);
void populateGqaConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMultiHeadAttentionConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx);
void populateAttentionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateOnnxAttentionConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx);
void populateGatherConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateCompressConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateOneHotConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateGatherElementsConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx);
void populateTopKConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateScatterElementsConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateShapeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateReshapeConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateCausalConvWithStateConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx);
void populateGemmConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateWhereConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateLinearAttentionConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateRangeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateEqualConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateDivConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMaxConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateReduceMaxConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateReduceMinConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateNotConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCosConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCeilConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateExpConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateLogConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCumSumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populatePadConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateTileConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateExpandConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateReduceProdConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateLessConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateGreaterConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateGreaterOrEqualConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx);
void populateLessOrEqualConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populateGatherNDConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateSignConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateModConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateConstantOfShapeConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateSliceConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateScatterNDConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateIdentityConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateOrConversionPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx);
void populateAndConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateAbsConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSizeConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateNonZeroConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateConcatConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateReluConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateLeakyReluConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateClipConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populatePoolConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateResizeConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateGlobalPoolConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateFlattenConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);

/// Pre-lowering pattern set: collapse the Gather(Shape(x), const_idx)
/// idiom into tensor.from_elements over a tensor.dim of x. Must run
/// BEFORE lowerOnnxConstants so the index value is still inline in the
/// onnx.Constant `value` attribute. See GatherShapeFold.cpp for the
/// dynseqlen-regression rationale.
void populateGatherShapeFoldPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);

/// Pre-lowering pattern set: stamp `onnx.Pad`'s compile-time `pads` (and
/// optional `axes`) constant onto the op as `hipdnn.pad_amounts` /
/// `hipdnn.pad_axes` attributes so PadConversion can compute the dynamic
/// output shape from them without reading the (by-then externalized) operand.
/// Sibling of GatherShapeFold; must run BEFORE lowerOnnxConstants. See
/// PadShapeFold.cpp.
void populatePadShapeFoldPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx);

/// Pre-lowering pattern set: stamp compile-time `onnx.Slice` starts/ends/axes/
/// steps onto the op as `hipdnn.slice_*` attributes so SliceDecompose can
/// rewrite to `tensor.extract_slice` after `lowerOnnxConstants` externalizes
/// the operand constants. Sibling of PadShapeFold; must run BEFORE
/// lowerOnnxConstants. See SliceShapeFold.cpp.
void populateSliceShapeFoldPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);

/// Pre-lowering pattern set: stamp a compile-time Tile `repeats` vector onto
/// the op before constant externalization so conversion and reification can
/// compute exact dynamic result extents without runtime readback.
void populateTileShapeFoldPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);

/// Pre-lowering pattern set: collapse ORT's inlined `FastGelu` primitive
/// chain (Pow / Mul / Sum / Tanh) back into a single
/// `onnx.Gelu(approximate="tanh")`. ORT inlines the Gelu function body
/// for some loading paths (notably dynamic-shape models) and the inlined
/// primitives have no MorphiZen converters. Must run BEFORE
/// `lowerOnnxConstants` so the literal float values of the embedded
/// constants (3.0, 0.044715, sqrt(2/π), 1.0, 0.5) are still inline.
/// See FastGeluFusion.cpp.
void populateFastGeluFusionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);

/// Pre-lowering pattern set: decompose `onnx.Pow(x, c)` (constant scalar
/// exponent c, optionally wrapped in `onnx.Cast`/`onnx.CastLike`) into ONNX
/// primitives (`onnx.Mul` / `onnx.Sqrt` / `onnx.Reciprocal`), which then flow
/// through their own ONNX→HIP converters in `convertComputeOps`. Must run
/// BEFORE `lowerOnnxConstants` because production builds externalize every
/// `onnx.Constant` (incl. 1-element scalars) into a memref.global with the
/// value moved to the constants file — at that point the exponent is no
/// longer recoverable from IR. See PowerConversion.cpp.
void populatePowDecompositionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
/// Pre-lowering pattern set: collapse the inlined erf-form `Gelu` primitive
/// chain (Div / Erf / Sum / Mul, optionally wrapped in CastLike scalars and
/// `Sqrt(2.0)`) back into a single `onnx.Gelu(approximate="none")`. Some
/// exports (e.g. ConvNeXt) inline the exact erf-based Gelu definition
/// `0.5 * x * (1 + erf(x / sqrt(2)))` as primitives that have no MorphiZen
/// converters. Must run BEFORE `lowerOnnxConstants` so the literal float
/// values (1.0, 2.0, 0.5) of the wrapped constants are still inline.
/// See ErfGeluFusion.cpp.
void populateErfGeluFusionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);

/// Pre-lowering pattern set: decompose vision/projector ops that have no
/// direct MorphiZen converter into supported primitives — patch-embed
/// Conv-ND → Reshape/Gemm/Reshape, AveragePool(kernel==stride) →
/// Reshape/Transpose/ReduceMean, Pow(x, c) → repeated Mul, ReduceMean →
/// ReduceSum·(1/N), and broadcasting Div → Mul(x, Reciprocal). Emits
/// `onnx.*` ops with result types built explicitly from the dims the
/// rewriter already knows (no separate shape pass needed at emission;
/// `--hip-infer-shapes` resolves any residual dynamic dims post-conversion).
/// Runs in the same ExistingOps pre-lowering set as FastGeluFusion. See
/// ProjectorOpsRewrites.cpp.
void populateProjectorOpsRewritePatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);

/// Pre-lowering pattern set: decompose `onnx.LpNormalization` into a small
/// chain of already-supported ONNX primitives (Mul / Sqrt / ReduceSum /
/// Div). Lives in the pre-lowering loop next to FastGeluFusion /
/// ProjectorOpsRewrites so the freshly-emitted onnx.* primitives become
/// "existing" ops by the next round and reach `convertComputeOps`'s
/// converters as ordinary onnx.* ops. See LpNormalizationConversion.cpp.
void populateLpNormalizationConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_UTILS_H
