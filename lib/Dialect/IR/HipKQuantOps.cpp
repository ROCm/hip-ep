/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip.quantize_q8_1, hip.skip_rms_norm_q8_1, hip.kquant_matmul,
// hip.kquant_matmul_q8_1, hip.kquant_swiglu_q8_1 and hip.token_major_gqa: their
// destinations, memory effects and verifiers.

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipKQuant.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::hip;

namespace {

using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

bool isTensorMode(Value value) {
  return isa<RankedTensorType>(value.getType());
}

bool areCompatibleDimensions(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

bool hasIdentityMemRefLayout(Value value) {
  auto memref = dyn_cast<MemRefType>(value.getType());
  return !memref || memref.getLayout().isIdentity();
}

/// The runtime entry points take bare pointers, so every operand is a tensor
/// or every operand is a memref, and a tensor op has one result per init.
LogicalResult verifyDpsMode(Operation *operation) {
  auto dpsOperation = cast<DestinationStyleOpInterface>(operation);
  SmallVector<Value> dataOperands;
  for (Value operand : operation->getOperands()) {
    if (isa<RankedTensorType, MemRefType>(operand.getType()))
      dataOperands.push_back(operand);
  }
  const bool tensorMode = isTensorMode(dataOperands.front());
  for (Value operand : dataOperands) {
    if (isTensorMode(operand) != tensorMode) {
      return operation->emitOpError(
          "requires all data operands to be tensors or all to be memrefs");
    }
  }

  OperandRange inits = dpsOperation.getDpsInits();
  if (!tensorMode) {
    if (operation->getNumResults() != 0)
      return operation->emitOpError("requires zero results in memref mode");
    return success();
  }
  if (operation->getNumResults() != inits.size())
    return operation->emitOpError("requires one tensor result per output init");
  for (auto [result, init] : llvm::zip(operation->getResults(), inits)) {
    if (result.getType() != init.getType()) {
      return operation->emitOpError(
          "requires tensor result types to match the output init types");
    }
  }
  return success();
}

void emitDpsMemoryEffects(Operation *operation,
                          SmallVectorImpl<MemoryEffect> &effects) {
  auto dpsOperation = cast<DestinationStyleOpInterface>(operation);
  for (OpOperand *operand : dpsOperation.getDpsInputOperands()) {
    if (isa<MemRefType>(operand->get().getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), operand,
                           SideEffects::DefaultResource::get());
    }
  }
  for (OpOperand &operand : dpsOperation.getDpsInitsMutable()) {
    if (isa<MemRefType>(operand.get().getType())) {
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
    }
  }
}

FailureOr<int64_t> getQ8RowBytes(Operation *operation, int64_t k) {
  int64_t byteCount = 0;
  if (llvm::MulOverflow(k / kquant::kQ8_1BlockElements, kquant::kQ8_1BlockBytes,
                        byteCount)) {
    operation->emitOpError("Q8_1 row byte cardinality overflows int64");
    return failure();
  }
  return byteCount;
}

FailureOr<int64_t> getPackedByteCount(Operation *operation, int64_t rows,
                                      int64_t k, StringRef format,
                                      StringRef operandName) {
  std::optional<kquant::Format> packedFormat = kquant::parseFormat(format);
  if (!packedFormat) {
    operation->emitOpError()
        << "requires " << operandName << " format q4_k, q5_k or q6_k";
    return failure();
  }
  int64_t blockCount = 0;
  int64_t byteCount = 0;
  if (k <= 0 || k % kquant::kBlockElements != 0 ||
      llvm::MulOverflow(rows, k / kquant::kBlockElements, blockCount) ||
      llvm::MulOverflow(blockCount, kquant::getBlockBytes(*packedFormat),
                        byteCount)) {
    operation->emitOpError()
        << operandName << " byte cardinality overflows int64";
    return failure();
  }
  return byteCount;
}

/// The activation's leading dimensions are the output's, and static ones are
/// positive.
LogicalResult verifyRowPrefix(Operation *operation, ShapedType activation,
                              ShapedType output) {
  for (int64_t dimension = 0; dimension < activation.getRank() - 1;
       ++dimension) {
    const int64_t extent = activation.getDimSize(dimension);
    if (!areCompatibleDimensions(extent, output.getDimSize(dimension))) {
      return operation->emitOpError()
             << "requires output dimension " << dimension
             << " to match the activation";
    }
    if (!ShapedType::isDynamic(extent) && extent <= 0) {
      return operation->emitOpError(
          "requires static activation dimensions to be positive");
    }
  }
  return success();
}

LogicalResult verifyQ8Activation(Operation *operation, ShapedType type,
                                 int64_t k) {
  FailureOr<int64_t> rowBytes = getQ8RowBytes(operation, k);
  if (failed(rowBytes))
    return failure();
  if (type.getShape().back() != *rowBytes) {
    return operation->emitOpError()
           << "requires the Q8_1 innermost dimension to be " << *rowBytes
           << " bytes for K=" << k;
  }
  return success();
}

LogicalResult verifyPackedWeights(Operation *operation, ShapedType type,
                                  int64_t n, int64_t k, StringRef format,
                                  StringRef name) {
  if (!type.getElementType().isUnsignedInteger(8) || type.getRank() != 1 ||
      !type.hasStaticShape()) {
    return operation->emitOpError()
           << "requires a statically shaped rank-1 ui8 " << name;
  }
  FailureOr<int64_t> expectedBytes =
      getPackedByteCount(operation, n, k, format, name);
  if (failed(expectedBytes))
    return failure();
  if (type.getNumElements() != *expectedBytes) {
    return operation->emitOpError()
           << name << " has " << type.getNumElements() << " bytes but "
           << format << " [" << n << ", " << k << "] has " << *expectedBytes;
  }
  return success();
}

} // namespace

#define DEFINE_DPS_METHODS(OpType, Inits)                                      \
  MutableOperandRange OpType::getDpsInitsMutable() { return Inits(); }         \
  void OpType::getEffects(SmallVectorImpl<MemoryEffect> &effects) {            \
    emitDpsMemoryEffects(getOperation(), effects);                             \
  }

DEFINE_DPS_METHODS(QuantizeQ8_1Op, getOutputMutable)
DEFINE_DPS_METHODS(SkipRmsNormQ8_1Op, getOutputsMutable)
DEFINE_DPS_METHODS(KQuantMatmulOp, getOutputMutable)
DEFINE_DPS_METHODS(KQuantMatmulQ8_1Op, getOutputMutable)
DEFINE_DPS_METHODS(KQuantSwigluQ8_1Op, getOutputMutable)
DEFINE_DPS_METHODS(TokenMajorGqaOp, getOutputMutable)

#undef DEFINE_DPS_METHODS

LogicalResult QuantizeQ8_1Op::verify() {
  if (failed(verifyDpsMode(*this)))
    return failure();
  if (!hasIdentityMemRefLayout(getOutput()))
    return emitOpError("requires an identity-layout output memref");

  auto activationType = cast<ShapedType>(getActivation().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!activationType.getElementType().isF32() ||
      !outputType.getElementType().isUnsignedInteger(8))
    return emitOpError("requires an f32 activation and a ui8 output");
  if (activationType.getRank() < 2 ||
      outputType.getRank() != activationType.getRank())
    return emitOpError("requires a rank >= 2 activation and an equal-rank "
                       "output");

  const int64_t k = getK();
  if (k <= 0 || k % kquant::kQ8_1BlockElements != 0)
    return emitOpError("requires K to be a positive multiple of 32");
  if (activationType.getShape().back() != k)
    return emitOpError("requires the activation's innermost dimension to be K");
  if (failed(verifyQ8Activation(*this, outputType, k)) ||
      failed(verifyRowPrefix(*this, activationType, outputType)))
    return failure();
  return success();
}

LogicalResult SkipRmsNormQ8_1Op::verify() {
  if (getOutputs().size() != 2)
    return emitOpError("requires two outputs: the sum and its Q8_1 form");
  if (failed(verifyDpsMode(*this)))
    return failure();

  Value sum = getOutputs()[0];
  Value quantized = getOutputs()[1];
  auto inputType = cast<ShapedType>(getInput().getType());
  auto skipType = cast<ShapedType>(getSkip().getType());
  auto gammaType = cast<ShapedType>(getGamma().getType());
  auto sumType = cast<ShapedType>(sum.getType());
  auto quantizedType = cast<ShapedType>(quantized.getType());
  if (!inputType.getElementType().isF32() ||
      !skipType.getElementType().isF32() ||
      !gammaType.getElementType().isF32() || !sumType.getElementType().isF32())
    return emitOpError("requires f32 input, skip, gamma and sum");
  if (!quantizedType.getElementType().isUnsignedInteger(8))
    return emitOpError("requires a ui8 Q8_1 output");
  if (inputType.getRank() != 4 || quantizedType.getRank() != 4)
    return emitOpError("requires rank-4 input and outputs");
  if (skipType.getShape() != inputType.getShape() ||
      sumType.getShape() != inputType.getShape())
    return emitOpError("requires identical input, skip and sum shapes");

  const int64_t k = getK();
  if (k <= 0 || k % kquant::kBlockElements != 0)
    return emitOpError("requires K to be a positive multiple of 256");
  if (inputType.getShape().back() != k)
    return emitOpError("requires the input's innermost dimension to be K");
  if (gammaType.getRank() != 1 || gammaType.getDimSize(0) != k)
    return emitOpError("requires a rank-1 gamma of K values");
  const llvm::APFloat &epsilon = getEpsilonAttr().getValue();
  if (!epsilon.isFinite() || epsilon.isNegative())
    return emitOpError("requires epsilon to be finite and non-negative");
  if (!hasIdentityMemRefLayout(sum) || !hasIdentityMemRefLayout(quantized))
    return emitOpError("requires identity-layout output memrefs");
  if (failed(verifyQ8Activation(*this, quantizedType, k)) ||
      failed(verifyRowPrefix(*this, inputType, quantizedType)))
    return failure();
  return success();
}

LogicalResult KQuantMatmulOp::verify() {
  if (failed(verifyDpsMode(*this)))
    return failure();
  if (!hasIdentityMemRefLayout(getOutput()))
    return emitOpError("requires an identity-layout output memref");

  auto activationType = cast<ShapedType>(getActivation().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!activationType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return emitOpError("requires an f32 activation and output");
  if (activationType.getRank() < 2 ||
      outputType.getRank() != activationType.getRank())
    return emitOpError("requires a rank >= 2 activation and an equal-rank "
                       "output");
  const int64_t k = getK();
  const int64_t n = getN();
  if (n <= 0)
    return emitOpError("requires N to be positive");
  if (activationType.getShape().back() != k)
    return emitOpError("requires the activation's innermost dimension to be K");
  if (outputType.getShape().back() != n)
    return emitOpError("requires the output's innermost dimension to be N");
  if (failed(verifyPackedWeights(*this,
                                 cast<ShapedType>(getPackedWeights().getType()),
                                 n, k, getFormat(), "packed_weights")) ||
      failed(verifyRowPrefix(*this, activationType, outputType)))
    return failure();
  return success();
}

LogicalResult KQuantMatmulQ8_1Op::verify() {
  if (failed(verifyDpsMode(*this)))
    return failure();
  if (!hasIdentityMemRefLayout(getOutput()))
    return emitOpError("requires an identity-layout output memref");

  auto activationType = cast<ShapedType>(getQuantizedActivation().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!activationType.getElementType().isUnsignedInteger(8) ||
      !outputType.getElementType().isF32())
    return emitOpError("requires a ui8 Q8_1 activation and an f32 output");
  if (activationType.getRank() < 2 ||
      outputType.getRank() != activationType.getRank())
    return emitOpError("requires a rank >= 2 activation and an equal-rank "
                       "output");
  const int64_t k = getK();
  const int64_t n = getN();
  if (n <= 0)
    return emitOpError("requires N to be positive");
  if (outputType.getShape().back() != n)
    return emitOpError("requires the output's innermost dimension to be N");
  if (failed(verifyPackedWeights(*this,
                                 cast<ShapedType>(getPackedWeights().getType()),
                                 n, k, getFormat(), "packed_weights")) ||
      failed(verifyQ8Activation(*this, activationType, k)) ||
      failed(verifyRowPrefix(*this, activationType, outputType)))
    return failure();
  return success();
}

LogicalResult KQuantSwigluQ8_1Op::verify() {
  if (failed(verifyDpsMode(*this)))
    return failure();
  if (!hasIdentityMemRefLayout(getOutput()))
    return emitOpError("requires an identity-layout output memref");

  auto activationType = cast<ShapedType>(getQuantizedActivation().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!activationType.getElementType().isUnsignedInteger(8) ||
      !outputType.getElementType().isF32())
    return emitOpError("requires a ui8 Q8_1 activation and an f32 output");
  if (activationType.getRank() != 4 || outputType.getRank() != 4)
    return emitOpError("requires a rank-4 activation and output");
  for (StringRef format : {getGateFormat(), getUpFormat()}) {
    if (format != "q4_k" && format != "q6_k")
      return emitOpError("requires q4_k or q6_k gate and up formats");
  }
  const int64_t k = getK();
  const int64_t n = getN();
  if (n <= 0)
    return emitOpError("requires N to be positive");
  if (outputType.getShape().back() != n)
    return emitOpError("requires the output's innermost dimension to be N");
  if (failed(verifyPackedWeights(*this,
                                 cast<ShapedType>(getGateWeights().getType()),
                                 n, k, getGateFormat(), "gate_weights")) ||
      failed(verifyPackedWeights(*this,
                                 cast<ShapedType>(getUpWeights().getType()), n,
                                 k, getUpFormat(), "up_weights")) ||
      failed(verifyQ8Activation(*this, activationType, k)) ||
      failed(verifyRowPrefix(*this, activationType, outputType)))
    return failure();
  return success();
}

LogicalResult TokenMajorGqaOp::verify() {
  if (failed(verifyDpsMode(*this)))
    return failure();
  if (!hasIdentityMemRefLayout(getOutput()))
    return emitOpError("requires an identity-layout output memref");

  auto queryType = cast<ShapedType>(getQuery().getType());
  auto keyType = cast<ShapedType>(getKeyCache().getType());
  auto valueType = cast<ShapedType>(getValueCache().getType());
  auto maskType = cast<ShapedType>(getMask().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (!queryType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return emitOpError("requires an f32 query and output");
  if (!keyType.getElementType().isF16() ||
      !valueType.getElementType().isF16() || !maskType.getElementType().isF16())
    return emitOpError("requires f16 key_cache, value_cache and mask");
  if (queryType.getRank() != 4 || keyType.getRank() != 4 ||
      valueType.getRank() != 4 || maskType.getRank() != 4 ||
      outputType.getRank() != 4)
    return emitOpError("requires rank-4 query, caches, mask and output");

  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();
  ArrayRef<int64_t> maskShape = maskType.getShape();
  for (auto [shape, name] :
       {std::pair{queryShape, "query"}, std::pair{keyShape, "key_cache"}}) {
    for (int64_t dimension : {0, 2, 3}) {
      if (ShapedType::isDynamic(shape[dimension]) || shape[dimension] <= 0) {
        return emitOpError() << "requires static positive " << name
                             << " dimensions 0, 2 and 3";
      }
    }
  }
  const int64_t batch = queryShape[0];
  const int64_t sequence = queryShape[1];
  const int64_t cacheCapacity = keyShape[1];
  if (valueType.getShape() != keyShape)
    return emitOpError("requires identical key_cache and value_cache shapes");
  if (keyShape[0] != batch || keyShape[3] != queryShape[3])
    return emitOpError("requires the caches' B and D to be the query's");
  if (!isTokenMajorGqaHeadDimension(queryShape[3]))
    return emitOpError("requires D to be 40, 64, 72, 80, 96, 112, 128 or 256");
  if (queryShape[2] % keyShape[2] != 0)
    return emitOpError("requires query H to be a multiple of the caches' G");
  if (outputType.getShape() != queryShape)
    return emitOpError("requires the output shape to be the query's");
  if (ShapedType::isDynamic(maskShape[0]) ||
      (maskShape[0] != 1 && maskShape[0] != batch) || maskShape[1] != 1 ||
      maskShape[2] != sequence || maskShape[3] != cacheCapacity)
    return emitOpError("requires a [1|B, 1, S, C] mask");

  if (Value writtenRows = getWrittenRows()) {
    auto rowsType = cast<ShapedType>(writtenRows.getType());
    if (!rowsType.getElementType().isSignlessInteger(64) ||
        rowsType.getRank() != 4 || rowsType.getDimSize(0) != 1 ||
        rowsType.getDimSize(1) != 1 || rowsType.getDimSize(2) != 1 ||
        !areCompatibleDimensions(rowsType.getDimSize(3), sequence))
      return emitOpError("requires i64 written_rows [1, 1, 1, S]");
  }
  const llvm::APFloat &scale = getScaleAttr().getValue();
  if (!scale.isFinite() || !(scale.convertToDouble() > 0.0))
    return emitOpError("requires a finite positive scale");
  return success();
}
