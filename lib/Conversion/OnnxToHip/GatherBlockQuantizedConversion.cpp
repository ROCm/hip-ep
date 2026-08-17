/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hip {
namespace gbq {

inline bool isUnsignedMlirElementType(Type elemType) {
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return intTy.isUnsigned();
  return false;
}

inline bool isGatherBlockQuantizedOp(Operation *op) {
  if (!op || op->getName().getStringRef() != "onnx.Custom")
    return false;
  auto fn = op->getAttrOfType<StringAttr>("function_name");
  if (!fn || fn.getValue() != "GatherBlockQuantized")
    return false;
  auto dom = op->getAttrOfType<StringAttr>("domain_name");
  return dom && dom.getValue() == "com.microsoft";
}

inline int64_t getGbqIntAttr(Operation *op, StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
    if (attr.getType().isSignedInteger())
      return attr.getSInt();
    if (attr.getType().isSignlessInteger())
      return attr.getInt();
    return static_cast<int64_t>(attr.getUInt());
  }
  return fallback;
}

inline int normalizeGbqAxis(int64_t axis, int64_t rank) {
  int a = static_cast<int>(axis);
  if (a < 0)
    a += static_cast<int>(rank);
  return a;
}

inline bool resolveUnsignedQuantStorage(bool hasUnsignedAttr,
                                        Type dataElemType) {
  if (isUnsignedMlirElementType(dataElemType))
    return true;
  if (auto intTy = dyn_cast<IntegerType>(dataElemType))
    return intTy.isSignless() && hasUnsignedAttr;
  return false;
}

inline bool isAlreadyPackedByteTensor(RankedTensorType dataType,
                                      RankedTensorType scalesType, int axis,
                                      int64_t blockSize) {
  if (dataType.getElementTypeBitWidth() != 8)
    return false;
  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (axis < 0 || axis >= static_cast<int>(dataShape.size()) ||
      axis >= static_cast<int>(scalesShape.size()))
    return false;
  return scalesShape[axis] * blockSize == dataShape[axis] * 2;
}

inline bool needsLogicalInt4Legalize(RankedTensorType dataType,
                                     RankedTensorType scalesType, int axis,
                                     int64_t blockSize) {
  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (axis < 0 || axis >= static_cast<int>(dataShape.size()) ||
      axis >= static_cast<int>(scalesShape.size()))
    return false;
  if (dataShape[axis] % 2 != 0)
    return false;
  return scalesShape[axis] * blockSize == dataShape[axis];
}

inline bool quantizeAxisMatches(ArrayRef<int64_t> dataShape,
                                ArrayRef<int64_t> scalesShape, int axis,
                                int64_t blockSize, int64_t bits) {
  if (blockSize <= 0 || axis < 0 ||
      axis >= static_cast<int>(dataShape.size()) ||
      axis >= static_cast<int>(scalesShape.size()))
    return false;

  for (int i = 0; i < static_cast<int>(dataShape.size()); ++i) {
    if (i == axis)
      continue;
    if (dataShape[i] != scalesShape[i])
      return false;
  }

  const int64_t dataDim = dataShape[axis];
  const int64_t scaleDim = scalesShape[axis];
  if (bits == 4) {
    if (scaleDim * blockSize == dataDim)
      return true;
    if (scaleDim * blockSize == dataDim * 2)
      return true;
    return false;
  }
  return scaleDim * blockSize == dataDim;
}

inline FailureOr<int64_t> inferQuantizeAxis(RankedTensorType dataType,
                                            RankedTensorType scalesType,
                                            int64_t blockSize, int64_t bits) {
  if (!dataType || !scalesType || blockSize <= 0)
    return failure();

  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (dataShape.size() != scalesShape.size())
    return failure();

  llvm::SmallVector<int, 4> matches;
  for (int axis = 0; axis < static_cast<int>(dataShape.size()); ++axis) {
    if (quantizeAxisMatches(dataShape, scalesShape, axis, blockSize, bits))
      matches.push_back(axis);
  }
  if (matches.size() != 1)
    return failure();
  return matches.front();
}

inline FailureOr<int64_t>
resolveQuantizeAxis(RankedTensorType dataType, RankedTensorType scalesType,
                    int64_t blockSize, int64_t bits,
                    std::optional<int64_t> explicitAxis) {
  if (explicitAxis.has_value()) {
    int64_t axis = *explicitAxis;
    int rank = static_cast<int>(dataType.getRank());
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return failure();
    return axis;
  }
  return inferQuantizeAxis(dataType, scalesType, blockSize, bits);
}

} // namespace gbq

namespace {

//===----------------------------------------------------------------------===//
// GatherBlockQuantized ONNX legalize (called from convert-onnx-to-hip)
//===----------------------------------------------------------------------===//

IntegerType packedI8ElementType(MLIRContext *ctx) {
  return IntegerType::get(ctx, 8);
}

Operation *recreateExternalConstant(PatternRewriter &rewriter,
                                    Operation *oldConst,
                                    RankedTensorType newType) {
  if (oldConst->getAttr("value"))
    return nullptr;

  OperationState state(oldConst->getLoc(), "onnx.Constant");
  state.addTypes(newType);
  for (StringRef name : {"location", "offset", "size", "onnx.element_type",
                         "onnx_node_name", "node_outputs"}) {
    if (auto attr = oldConst->getAttr(name))
      state.addAttribute(name, attr);
  }

  auto *newConst = rewriter.create(state);
  rewriter.replaceOp(oldConst, newConst->getResults());
  return newConst;
}

bool annotateGbqSemantics(Operation *gbq, OpBuilder &builder) {
  if (gbq->getNumOperands() < 3)
    return false;

  const int64_t bits = gbq::getGbqIntAttr(gbq, "bits", 0);
  const int64_t blockSize = gbq::getGbqIntAttr(gbq, "block_size", 0);
  if ((bits != 4 && bits != 8) || blockSize <= 0)
    return false;

  auto dataType = dyn_cast<RankedTensorType>(gbq->getOperand(0).getType());
  auto scalesType = dyn_cast<RankedTensorType>(gbq->getOperand(2).getType());
  if (!dataType || !scalesType)
    return false;

  bool changed = false;
  // Preserve importer-provided unsignedness for signless legalized storage.
  // Explicit MLIR ui8 is also canonicalized to the marker. Never infer
  // unsignedness from `bits`: bits describes width, not signedness.
  if (gbq::isUnsignedMlirElementType(dataType.getElementType()) &&
      !gbq->hasAttr("unsigned_quant_storage")) {
    gbq->setAttr("unsigned_quant_storage", UnitAttr::get(builder.getContext()));
    changed = true;
  }

  if (!gbq->getAttr("quantize_axis")) {
    auto axisOr = gbq::inferQuantizeAxis(dataType, scalesType, blockSize, bits);
    if (succeeded(axisOr)) {
      gbq->setAttr("quantize_axis", builder.getI64IntegerAttr(*axisOr));
      changed = true;
    }
  }
  return changed;
}

bool legalizeInt4ConstantIfNeeded(Operation *gbq, PatternRewriter &rewriter) {
  const int64_t bits = gbq::getGbqIntAttr(gbq, "bits", 0);
  if (bits != 4 || gbq->getNumOperands() < 3)
    return false;

  Value data = gbq->getOperand(0);
  Operation *constOp = data.getDefiningOp();
  if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
    return false;

  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto scalesType = dyn_cast<RankedTensorType>(gbq->getOperand(2).getType());
  if (!dataType || !scalesType)
    return false;

  const int64_t blockSize = gbq::getGbqIntAttr(gbq, "block_size", 0);
  if (blockSize <= 0)
    return false;

  int qa = gbq::normalizeGbqAxis(gbq::getGbqIntAttr(gbq, "quantize_axis", -1),
                                 dataType.getRank());
  if (gbq::isAlreadyPackedByteTensor(dataType, scalesType, qa, blockSize))
    return false;
  if (!gbq::needsLogicalInt4Legalize(dataType, scalesType, qa, blockSize))
    return false;

  auto shape = llvm::to_vector(dataType.getShape());
  shape[qa] /= 2;
  auto packedType =
      RankedTensorType::get(shape, packedI8ElementType(rewriter.getContext()));

  rewriter.setInsertionPoint(constOp);
  return recreateExternalConstant(rewriter, constOp, packedType) != nullptr;
}

// zero_points shares scales' rank but may carry ONNX UINT4 logical shapes whose
// external `size` is half the ui8 element count; data/scales invariants do not
// detect that case.
bool legalizeInt4ZeroPointsConstantIfNeeded(Operation *gbq,
                                            PatternRewriter &rewriter) {
  const int64_t bits = gbq::getGbqIntAttr(gbq, "bits", 0);
  if (bits != 4 || gbq->getNumOperands() < 4)
    return false;

  Value zeroPoints = gbq->getOperand(3);
  if (!zeroPoints || isa<NoneType>(zeroPoints.getType()))
    return false;

  Operation *constOp = zeroPoints.getDefiningOp();
  if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
    return false;
  if (constOp->getAttr("value"))
    return false;
  auto sizeAttr = constOp->getAttrOfType<IntegerAttr>("size");
  if (!sizeAttr || sizeAttr.getInt() <= 0)
    return false;

  auto dataType = dyn_cast<RankedTensorType>(gbq->getOperand(0).getType());
  auto zpType = dyn_cast<RankedTensorType>(zeroPoints.getType());
  if (!dataType || !zpType)
    return false;

  int qa = gbq::normalizeGbqAxis(gbq::getGbqIntAttr(gbq, "quantize_axis", -1),
                                 dataType.getRank());
  if (qa < 0 || qa >= zpType.getRank())
    return false;
  if (zpType.getShape()[qa] % 2 != 0)
    return false;

  int64_t numElements = 1;
  for (int64_t dim : zpType.getShape()) {
    if (llvm::MulOverflow(numElements, dim, numElements))
      return false;
  }
  if (sizeAttr.getInt() * 2 != numElements)
    return false;

  auto shape = llvm::to_vector(zpType.getShape());
  shape[qa] /= 2;
  auto packedType =
      RankedTensorType::get(shape, packedI8ElementType(rewriter.getContext()));

  rewriter.setInsertionPoint(constOp);
  return recreateExternalConstant(rewriter, constOp, packedType) != nullptr;
}

struct GatherBlockQuantizedPreparePattern : public RewritePattern {
  GatherBlockQuantizedPreparePattern(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!gbq::isGatherBlockQuantizedOp(op))
      return failure();
    bool changed = annotateGbqSemantics(op, rewriter);
    changed |= legalizeInt4ConstantIfNeeded(op, rewriter);
    changed |= legalizeInt4ZeroPointsConstantIfNeeded(op, rewriter);
    return changed ? success() : failure();
  }
};

//===----------------------------------------------------------------------===//
// ONNX com.microsoft.GatherBlockQuantized -> HIP gather_block_quantized
//===----------------------------------------------------------------------===//
//
// Before:
//   %out = "onnx.Custom"(%data, %indices, %scales, %zero_points)
//       {function_name = "GatherBlockQuantized",
//        domain_name = "com.microsoft",
//        bits = 4 : si64, block_size = 16 : si64,
//        gather_axis = 0 : si64, quantize_axis = 1 : si64}
//       : (tensor<2048x96xui8>, tensor<8xi64>,
//          tensor<2048x12xf16>, tensor<2048x12xui8>) -> tensor<8x192xf16>
//
// After:
//   %init = tensor.empty() : tensor<8x192xf16>
//   %out = hip.gather_block_quantized(%ctx)
//       ins(%data, %indices, %scales :
//           tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>)
//       zero_points(%zp : tensor<2048x12xui8>)
//       outs(%init : tensor<8x192xf16>)
//       {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
//       : tensor<8x192xf16>
//
// Output shape derivation applies Gather to the logical dequantized shape:
//   logical_data[quantize_axis] = 2 * data[quantize_axis]  // packed int4
//   out.shape = logical_data[0:gather_axis]
//             ++ indices.shape
//             ++ logical_data[gather_axis+1:]
// The multiplication is omitted when Gather removes `quantize_axis` itself.
//
// Storage semantics come from ONNX T1, not `bits`. Explicit si8/ui8 retain
// their signedness. For signless i8 produced while legalizing INT4/UINT4,
// `unsigned_quant_storage` is the authoritative UINT marker; absence means
// signed storage.
//
// quantize_axis is taken from the ONNX attribute when present; otherwise it is
// inferred from (data, scales, block_size, bits) shape invariants.

struct GatherBlockQuantizedToHip : public mlir::RewritePattern {
  GatherBlockQuantizedToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult GatherBlockQuantizedToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GatherBlockQuantized")
    return rewriter.notifyMatchFailure(op,
                                       "not a GatherBlockQuantized custom op");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 3)
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs (data, indices, scales)");
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected exactly 1 result (output)");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Value data = op->getOperand(0);
  mlir::Value indices = op->getOperand(1);
  mlir::Value scales = op->getOperand(2);

  // ONNX models that omit `zero_points` may either drop the operand entirely
  // (only 3 operands present) or pass an `onnx.NoValue` of !NoneType — match
  // MatMulNBitsConversion's handling so both forms produce a null Value.
  mlir::Value zeroPoints;
  if (op->getNumOperands() >= 4) {
    mlir::Value v = op->getOperand(3);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoints = v;
  }

  // Attribute extraction. The import path materializes the required bits and
  // block_size values. gather_axis defaults to 0; when quantize_axis is absent,
  // the prepare/conversion path resolves it from the data/scales block grid.
  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  if (!bitsIntAttr)
    return rewriter.notifyMatchFailure(op, "missing required `bits` attribute");
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  if (!blockSizeIntAttr)
    return rewriter.notifyMatchFailure(
        op, "missing required `block_size` attribute");
  auto gatherAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("gather_axis");

  const int64_t bits = bitsIntAttr.getSInt();
  const int64_t blockSize = blockSizeIntAttr.getSInt();
  if (bits != 4 && bits != 8)
    return rewriter.notifyMatchFailure(op, "GBQ `bits` must be 4 or 8");
  if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0)
    return rewriter.notifyMatchFailure(
        op, "GBQ `block_size` must be a power of two and at least 16");

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
  auto scalesType = mlir::dyn_cast<mlir::RankedTensorType>(scales.getType());
  auto zeroPointsType =
      zeroPoints ? mlir::dyn_cast<mlir::RankedTensorType>(zeroPoints.getType())
                 : mlir::RankedTensorType();
  if (!resultType || !dataType || !indicesType || !scalesType ||
      (zeroPoints && !zeroPointsType))
    return rewriter.notifyMatchFailure(
        op,
        "GBQ data, indices, scales, zero_points, and output must be ranked");

  auto dataElementType =
      mlir::dyn_cast<mlir::IntegerType>(dataType.getElementType());
  auto indicesElementType =
      mlir::dyn_cast<mlir::IntegerType>(indicesType.getElementType());
  if (!dataElementType || dataElementType.getWidth() != 8)
    return rewriter.notifyMatchFailure(
        op, "GBQ data storage must use an 8-bit integer element type");
  if (dataElementType.isSigned() && op->hasAttr("unsigned_quant_storage")) {
    return op->emitError(
        "GBQ `unsigned_quant_storage` conflicts with explicitly signed data");
  }
  if (!indicesElementType || (indicesElementType.getWidth() != 32 &&
                              indicesElementType.getWidth() != 64))
    return rewriter.notifyMatchFailure(op, "GBQ indices must be i32 or i64");
  mlir::Type scalesElementType = scalesType.getElementType();
  if (!scalesElementType.isF16() && !scalesElementType.isF32() &&
      !scalesElementType.isBF16())
    return rewriter.notifyMatchFailure(op,
                                       "GBQ scales must use f16, f32, or bf16");
  if (resultType.getElementType() != scalesElementType)
    return rewriter.notifyMatchFailure(
        op, "GBQ output element type must match scales");
  if (zeroPointsType &&
      zeroPointsType.getElementType() != dataType.getElementType())
    return rewriter.notifyMatchFailure(
        op, "GBQ zero_points element type must match data");

  std::optional<int64_t> explicitQuantAxis;
  if (op->getAttr("quantize_axis"))
    explicitQuantAxis = gbq::getGbqIntAttr(op, "quantize_axis", 0);
  auto quantAxisOr = gbq::resolveQuantizeAxis(dataType, scalesType, blockSize,
                                              bits, explicitQuantAxis);
  if (mlir::failed(quantAxisOr))
    return rewriter.notifyMatchFailure(
        op, "could not resolve `quantize_axis` from GBQ data/scales shapes");

  bool unsignedQuantStorage = gbq::resolveUnsignedQuantStorage(
      op->hasAttr("unsigned_quant_storage"), dataType.getElementType());

  auto bitsAttr = rewriter.getI64IntegerAttr(bits);
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSize);
  auto gatherAxisAttr = rewriter.getI64IntegerAttr(
      gatherAxisIntAttr ? gatherAxisIntAttr.getSInt() : 0);
  auto quantAxisAttr = rewriter.getI64IntegerAttr(*quantAxisOr);

  bool bytePackedInt4 = bits == 4;
  bool uint8Storage = bits == 8 && unsignedQuantStorage;
  auto resultShape = mlir::hip::reifyGatherBlockQuantizedShape(
      rewriter, loc, data, indices, scales, zeroPoints, bits, blockSize,
      gatherAxisAttr.getInt(), quantAxisAttr.getInt(), bytePackedInt4,
      uint8Storage, [&]() { return op->emitError(); });
  if (mlir::failed(resultShape))
    return mlir::failure();
  auto init = createEmptyTensorFromReifiedShape(rewriter, loc, resultType,
                                                *resultShape);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "GBQ result type is incompatible with its logical gathered shape");

  auto hipOp = mlir::hip::GatherBlockQuantizedOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, context, data, indices,
      scales, zeroPoints, *init, bitsAttr, blockSizeAttr, gatherAxisAttr,
      quantAxisAttr);
  if (unsignedQuantStorage)
    hipOp->setAttr("unsigned_quant_storage", rewriter.getUnitAttr());
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateGatherBlockQuantizedPreparePatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.add<GatherBlockQuantizedPreparePattern>(ctx);
}

void populateGatherBlockQuantizedConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.add<GatherBlockQuantizedToHip>(ctx);
}

} // namespace hip
} // namespace mlir
