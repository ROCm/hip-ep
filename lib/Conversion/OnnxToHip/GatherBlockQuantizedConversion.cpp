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
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getSInt();
  return fallback;
}

inline int normalizeGbqAxis(int64_t axis, int64_t rank) {
  int a = static_cast<int>(axis);
  if (a < 0)
    a += static_cast<int>(rank);
  return a;
}

// `bits == 8` implies unsigned storage because the contrib schema constrains
// T1 to {tensor(int4), tensor(uint4), tensor(uint8)} -- uint8 is the only
// legal 8-bit storage type, and tensor(int8) is rejected at model load.
inline bool resolveUnsignedQuantStorage(int64_t bits, bool hasUnsignedAttr,
                                        Type dataElemType) {
  if (hasUnsignedAttr)
    return true;
  if (bits == 8)
    return true;
  return isUnsignedMlirElementType(dataElemType);
}

constexpr StringRef kQuantStorageBitsAttr = "quant_storage_bits";
constexpr StringRef kOnnxElementTypeAttr = "onnx.element_type";

inline std::optional<int64_t> quantStorageBitsFromOnnxType(int64_t onnxType) {
  switch (onnxType) {
  case 21: // TensorProto_DataType_UINT4
  case 22: // TensorProto_DataType_INT4
    return 4;
  case 2: // TensorProto_DataType_UINT8
  case 3: // TensorProto_DataType_INT8
    return 8;
  default:
    return std::nullopt;
  }
}

// Falls back to `bits` when the mark is absent (hand-written IR, or a
// non-constant `data`), reproducing the pre-existing behaviour rather than
// guessing a storage width.
inline int64_t resolveQuantStorageBits(Operation *gbq, int64_t bits) {
  if (auto attr = gbq->getAttrOfType<IntegerAttr>(kQuantStorageBitsAttr))
    return attr.getInt();
  if (gbq->getNumOperands() >= 1) {
    if (Operation *def = gbq->getOperand(0).getDefiningOp()) {
      if (auto attr = def->getAttrOfType<IntegerAttr>(kOnnxElementTypeAttr)) {
        if (auto width = quantStorageBitsFromOnnxType(attr.getInt()))
          return *width;
      }
    }
  }
  return bits;
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
    if (!quantizeAxisMatches(dataType.getShape(), scalesType.getShape(),
                             static_cast<int>(axis), blockSize, bits))
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

bool annotateGbqSemantics(Operation *gbq, PatternRewriter &rewriter) {
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

  // Decide everything up front, then apply as a single rewriter-tracked
  // mutation. Mutating an op inside a pattern without going through
  // `modifyOpInPlace` bypasses the driver's listener/worklist bookkeeping.
  const bool hasUnsignedAttr = gbq->hasAttr("unsigned_quant_storage");
  const bool unsignedStorage = gbq::resolveUnsignedQuantStorage(
      bits, hasUnsignedAttr, dataType.getElementType());
  const bool addUnsigned = unsignedStorage && !hasUnsignedAttr;
  const bool dropUnsigned = !unsignedStorage && hasUnsignedAttr;

  std::optional<int64_t> inferredAxis;
  if (!gbq->getAttr("quantize_axis")) {
    auto axisOr = gbq::inferQuantizeAxis(dataType, scalesType, blockSize, bits);
    if (succeeded(axisOr))
      inferredAxis = *axisOr;
  }

  // Pin the storage width now, while the defining constant still carries the
  // ONNX element type: legalizeInt4ConstantIfNeeded rewrites the constant
  // below, after which a 4-bit tensor is shape-indistinguishable from a
  // uint8 tensor holding packed nibbles.
  std::optional<int64_t> storageBits;
  if (!gbq->hasAttr(gbq::kQuantStorageBitsAttr))
    storageBits = gbq::resolveQuantStorageBits(gbq, bits);

  if (!addUnsigned && !dropUnsigned && !inferredAxis.has_value() &&
      !storageBits.has_value())
    return false;

  rewriter.modifyOpInPlace(gbq, [&] {
    if (addUnsigned)
      gbq->setAttr("unsigned_quant_storage", rewriter.getUnitAttr());
    else if (dropUnsigned)
      gbq->removeAttr("unsigned_quant_storage");
    if (inferredAxis.has_value())
      gbq->setAttr("quantize_axis", rewriter.getI64IntegerAttr(*inferredAxis));
    if (storageBits.has_value())
      gbq->setAttr(gbq::kQuantStorageBitsAttr,
                   rewriter.getI64IntegerAttr(*storageBits));
  });
  return true;
}

bool legalizeInt4ConstantIfNeeded(Operation *gbq, PatternRewriter &rewriter) {
  const int64_t bits = gbq::getGbqIntAttr(gbq, "bits", 0);
  if (bits != 4 || gbq->getNumOperands() < 3)
    return false;

  if (gbq->getNumOperands() >= 4) {
    Value zeroPoints = gbq->getOperand(3);
    if (zeroPoints && !isa<NoneType>(zeroPoints.getType()))
      return false;
  }

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

struct GatherBlockQuantizedPreparePattern : public RewritePattern {
  GatherBlockQuantizedPreparePattern(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!gbq::isGatherBlockQuantizedOp(op))
      return failure();
    bool changed = annotateGbqSemantics(op, rewriter);
    changed |= legalizeInt4ConstantIfNeeded(op, rewriter);
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
//          tensor<2048x12xf16>, tensor<2048x12xui8>) -> tensor<8x96xf16>
//
// After:
//   %init = tensor.empty() : tensor<8x96xf16>
//   %out = hip.gather_block_quantized(%ctx)
//       ins(%data, %indices, %scales :
//           tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>)
//       zero_points(%zp : tensor<2048x12xui8>)
//       outs(%init : tensor<8x96xf16>)
//       {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
//       : tensor<8x96xf16>
//
// Output shape derivation (mirrors plain Gather):
//   out.shape = data.shape[0:gather_axis]
//             ++ indices.shape
//             ++ data.shape[gather_axis+1:]
// The dequant block axis lives entirely inside `data`, so the output has
// no extra "blocks" dim — the runtime fans out per-element on the gathered
// rows during the dequantize step.
//
// Storage semantics (unsigned vs signed) come from ONNX T1 + bits, not from
// signless MLIR integer types alone. `unsigned_quant_storage` is set when:
//   - convert-onnx-to-hip prepare annotated the Custom op, or
//   - the data tensor element type is ui8 at conversion time.
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

  // Attribute extraction. Spec defaults:
  //   bits           — required, 4 or 8
  //   block_size     — required, power of 2 >= 16
  //   gather_axis    — optional, default 0
  //   quantize_axis  — optional, default 0
  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  if (!bitsIntAttr)
    return rewriter.notifyMatchFailure(op, "missing required `bits` attribute");
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  if (!blockSizeIntAttr)
    return rewriter.notifyMatchFailure(
        op, "missing required `block_size` attribute");
  auto gatherAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("gather_axis");
  auto quantAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("quantize_axis");

  const int64_t bits = bitsIntAttr.getSInt();
  const int64_t blockSize = blockSizeIntAttr.getSInt();
  if (bits != 4 && bits != 8)
    return rewriter.notifyMatchFailure(op, "GBQ `bits` must be 4 or 8");
  if (blockSize <= 0)
    return rewriter.notifyMatchFailure(op, "invalid `block_size`");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  auto scalesType = mlir::cast<mlir::RankedTensorType>(scales.getType());

  std::optional<int64_t> explicitQuantAxis;
  if (quantAxisIntAttr)
    explicitQuantAxis = quantAxisIntAttr.getSInt();
  auto quantAxisOr = gbq::resolveQuantizeAxis(dataType, scalesType, blockSize,
                                              bits, explicitQuantAxis);
  if (mlir::failed(quantAxisOr))
    return rewriter.notifyMatchFailure(
        op, "could not resolve `quantize_axis` from GBQ data/scales shapes");

  bool unsignedQuantStorage = gbq::resolveUnsignedQuantStorage(
      bits, op->hasAttr("unsigned_quant_storage"), dataType.getElementType());
  const int64_t quantStorageBits = gbq::resolveQuantStorageBits(op, bits);

  auto bitsAttr = rewriter.getI64IntegerAttr(bits);
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSize);
  auto gatherAxisAttr = rewriter.getI64IntegerAttr(
      gatherAxisIntAttr ? gatherAxisIntAttr.getSInt() : 0);
  auto quantAxisAttr = rewriter.getI64IntegerAttr(*quantAxisOr);

  // Output shape: [data[0:gather_axis], indices.shape, data[gather_axis+1:]].
  // Mirror GatherConversion's dim-mapping: walk output dims, source each
  // dynamic dim from either data or indices according to its position.
  int64_t gatherAxis = gatherAxisAttr.getInt();
  int64_t normalizedGatherAxis =
      gatherAxis < 0 ? gatherAxis + dataType.getRank() : gatherAxis;

  auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());
  llvm::SmallVector<mlir::Value> dynSizes;
  int64_t outDimIdx = 0;
  for (auto i : llvm::seq<int64_t>(0, normalizedGatherAxis)) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    outDimIdx++;
  }
  for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, indices, i));
    outDimIdx++;
  }
  for (auto i :
       llvm::seq<int64_t>(normalizedGatherAxis + 1, dataType.getRank())) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    outDimIdx++;
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto hipOp = mlir::hip::GatherBlockQuantizedOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, context, data, indices,
      scales, zeroPoints, init, bitsAttr, blockSizeAttr, gatherAxisAttr,
      quantAxisAttr,
      unsignedQuantStorage ? rewriter.getUnitAttr() : mlir::UnitAttr(),
      rewriter.getI64IntegerAttr(quantStorageBits));
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
