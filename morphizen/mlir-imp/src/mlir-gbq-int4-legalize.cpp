/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./mlir-gbq-int4-legalize.hpp"
#include "./mlir-constants.hpp"
#include "./mlir-context-manager.hpp"
#include "./mlir-graph.hpp"
#include "./mlir-node-arg.hpp"

#include "glog/logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <optional>
#include <string>

namespace morphizen {
namespace mlir_impl {
namespace {

constexpr int kOnnxUint4 = 21;
constexpr int kOnnxInt4 = 22;
constexpr int kOnnxUint8 = 2;

static mlir::MLIRContext *contextFor(mlir::Block &block) {
  if (auto *parent = block.getParentOp())
    return parent->getContext();
  return &MLIRContextManager::getInstance().getContext();
}

bool isGatherBlockQuantized(mlir::Operation *op) {
  if (!op || op->getName().getStringRef() != "onnx.Custom")
    return false;
  auto fn =
      op->getAttrOfType<mlir::StringAttr>(attr_names::CUSTOM_OP_FUNCTION_NAME);
  if (!fn || fn.getValue() != "GatherBlockQuantized")
    return false;
  auto dom =
      op->getAttrOfType<mlir::StringAttr>(attr_names::CUSTOM_OP_DOMAIN_NAME);
  return dom && dom.getValue() == "com.microsoft";
}

int64_t getIntAttr(mlir::Operation *op, llvm::StringRef name,
                   int64_t fallback) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    // MorphiZen stores ONNX scalar attrs as si64 (see
    // MLIRNamedAttribute::create_int). IntegerAttr::getInt() requires
    // signless/index and will assert on si64.
    return attr.getSInt();
  return fallback;
}

int normalizeAxis(int64_t axis, int64_t rank) {
  int a = static_cast<int>(axis);
  if (a < 0)
    a += static_cast<int>(rank);
  return a;
}

bool isAlreadyPackedByteTensor(mlir::RankedTensorType dataType,
                               mlir::RankedTensorType scalesType, int qa,
                               int64_t blockSize) {
  if (dataType.getElementTypeBitWidth() != 8)
    return false;
  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (qa < 0 || qa >= static_cast<int>(dataShape.size()) ||
      qa >= static_cast<int>(scalesShape.size()))
    return false;
  return scalesShape[qa] * blockSize == dataShape[qa] * 2;
}

bool needsLogicalInt4Legalize(mlir::RankedTensorType dataType,
                              mlir::RankedTensorType scalesType, int qa,
                              int64_t blockSize) {
  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (qa < 0 || qa >= static_cast<int>(dataShape.size()) ||
      qa >= static_cast<int>(scalesShape.size()))
    return false;
  if (dataShape[qa] % 2 != 0)
    return false;
  return scalesShape[qa] * blockSize == dataShape[qa];
}

mlir::Type packedElementType(mlir::MLIRContext *ctx) {
  // MLIR dense constants and many tensor ops require signless integers.
  (void)ctx;
  return mlir::IntegerType::get(ctx, 8);
}

bool quantizeAxisMatches(llvm::ArrayRef<int64_t> dataShape,
                         llvm::ArrayRef<int64_t> scalesShape, int axis,
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

std::optional<int64_t> inferQuantizeAxis(mlir::RankedTensorType dataType,
                                         mlir::RankedTensorType scalesType,
                                         int64_t blockSize, int64_t bits) {
  if (!dataType || !scalesType || blockSize <= 0)
    return std::nullopt;
  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (dataShape.size() != scalesShape.size())
    return std::nullopt;
  llvm::SmallVector<int, 4> matches;
  for (int axis = 0; axis < static_cast<int>(dataShape.size()); ++axis) {
    if (quantizeAxisMatches(dataShape, scalesShape, axis, blockSize, bits))
      matches.push_back(axis);
  }
  if (matches.size() != 1)
    return std::nullopt;
  return matches.front();
}

mlir::Operation *traceToInitializerConstant(mlir::Value value) {
  mlir::Value cur = value;
  for (int depth = 0; depth < 8; ++depth) {
    mlir::Operation *op = cur.getDefiningOp();
    if (!op)
      break;
    const llvm::StringRef name = op->getName().getStringRef();
    if (name == "onnx.Constant")
      return op;
    if (op->getNumOperands() == 0)
      break;
    if (name == "onnx.Reshape" || name == "onnx.Transpose" ||
        name == "onnx.Squeeze" || name == "onnx.Unsqueeze" ||
        name == "onnx.Flatten" || name == "onnx.Gather") {
      cur = op->getOperand(0);
      continue;
    }
    break;
  }
  return nullptr;
}

bool inferSignedStorage(mlir::RankedTensorType dataType, MLIRGraph &graph,
                        mlir::Operation *constOp) {
  if (auto constTy = mlir::dyn_cast<mlir::RankedTensorType>(
          constOp->getResult(0).getType())) {
    if (constTy.getElementType().isUnsignedInteger(8))
      return false;
    if (auto intTy =
            mlir::dyn_cast<mlir::IntegerType>(constTy.getElementType())) {
      if (intTy.isUnsigned())
        return false;
    }
  }

  if (auto names =
          constOp->getAttrOfType<mlir::ArrayAttr>(attr_names::NODE_OUTPUTS)) {
    if (!names.empty()) {
      if (auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(names[0])) {
        auto idx = graph.get_node_arg_index(nameAttr.getValue().str());
        if (idx.is_valid()) {
          if (auto *nodeArg = graph.get_node_arg(idx)) {
            int elem = nodeArg->getElementType();
            if (elem == kOnnxInt4)
              return true;
            if (elem == kOnnxUint4 || elem == kOnnxUint8)
              return false;
          }
        }
      }
    }
  }

  // MorphiZen import may have fallen back to F32 before we preserved ONNX type
  // on the MLIR value. Assume signed INT4 storage when the ONNX type is lost.
  if (dataType.getElementType().isF32()) {
    LOG(WARNING) << "GBQ INT4 legalize: constant element type is F32 (ONNX "
                    "INT4/UINT4 type lost on import); assuming signed INT4 "
                    "storage";
    return true;
  }

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(dataType.getElementType()))
    return intTy.isSigned() || intTy.isSignless();

  return true;
}

void annotateGatherBlockQuantizedSemantics(MLIRGraph &graph,
                                           mlir::Operation *gbq,
                                           mlir::OpBuilder &builder) {
  if (gbq->getNumOperands() < 3)
    return;

  const int64_t bits = getIntAttr(gbq, "bits", 0);
  const int64_t blockSize = getIntAttr(gbq, "block_size", 0);
  if ((bits != 4 && bits != 8) || blockSize <= 0)
    return;

  auto dataType =
      mlir::dyn_cast<mlir::RankedTensorType>(gbq->getOperand(0).getType());
  auto scalesType =
      mlir::dyn_cast<mlir::RankedTensorType>(gbq->getOperand(2).getType());
  if (!dataType || !scalesType)
    return;

  bool unsignedStorage = gbq->hasAttr("unsigned_quant_storage");
  if (!unsignedStorage) {
    if (bits == 8) {
      unsignedStorage = true;
    } else if (auto *constOp = traceToInitializerConstant(gbq->getOperand(0))) {
      if (auto constTy = mlir::dyn_cast<mlir::RankedTensorType>(
              constOp->getResult(0).getType())) {
        if (constTy.getElementType().isUnsignedInteger(8))
          unsignedStorage = true;
        else
          unsignedStorage = !inferSignedStorage(dataType, graph, constOp);
      } else {
        unsignedStorage = !inferSignedStorage(dataType, graph, constOp);
      }
    } else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(
                   dataType.getElementType())) {
      unsignedStorage = intTy.isUnsigned();
    }
  }
  if (unsignedStorage) {
    gbq->setAttr("unsigned_quant_storage",
                 mlir::UnitAttr::get(builder.getContext()));
  } else {
    gbq->removeAttr("unsigned_quant_storage");
  }

  if (!gbq->getAttr("quantize_axis")) {
    if (auto inferred =
            inferQuantizeAxis(dataType, scalesType, blockSize, bits)) {
      gbq->setAttr("quantize_axis", builder.getI64IntegerAttr(*inferred));
    }
  }
}

mlir::Operation *recreateConstant(mlir::OpBuilder &builder,
                                  mlir::Operation *oldConst,
                                  mlir::RankedTensorType newType) {
  mlir::OperationState state(oldConst->getLoc(), "onnx.Constant");
  state.addTypes(newType);

  if (oldConst->getAttr("value")) {
    // Inline dense constants must already match the packed byte width; external
    // data is the common case for large GBQ tables.
    LOG(WARNING)
        << "Skipping INT4 legalize for inline onnx.Constant with dense "
           "value (expected external data)";
    return nullptr;
  } else {
    if (auto loc = oldConst->getAttr("location"))
      state.addAttribute("location", loc);
    if (auto offset = oldConst->getAttr("offset"))
      state.addAttribute("offset", offset);
    if (auto size = oldConst->getAttr("size"))
      state.addAttribute("size", size);
  }

  for (auto attrName : {attr_names::NODE_OUTPUTS, attr_names::ONNX_NODE_NAME}) {
    if (auto attr = oldConst->getAttr(attrName))
      state.addAttribute(attrName, attr);
  }

  auto *newConst = builder.create(state);
  oldConst->getResult(0).replaceAllUsesWith(newConst->getResult(0));
  oldConst->erase();
  return newConst;
}

void legalizeInBlock(MLIRGraph &graph, mlir::Block &block) {
  llvm::SmallVector<mlir::Operation *> gbqOps;
  for (mlir::Operation &op : block.getOperations()) {
    if (isGatherBlockQuantized(&op))
      gbqOps.push_back(&op);
  }

  if (gbqOps.empty())
    return;

  mlir::OpBuilder builder(contextFor(block));

  for (mlir::Operation *gbq : gbqOps) {
    annotateGatherBlockQuantizedSemantics(graph, gbq, builder);

    const int64_t bits = getIntAttr(gbq, "bits", 0);
    if (bits != 4)
      continue;
    if (gbq->getNumOperands() < 3)
      continue;

    // Explicit zero_points: keep logical INT4/UINT4 data shape; halving only
    // data (not zp/scales) breaks (q - zp) * scale block indexing.
    if (gbq->getNumOperands() >= 4) {
      mlir::Value zeroPoints = gbq->getOperand(3);
      if (zeroPoints && !mlir::isa<mlir::NoneType>(zeroPoints.getType()))
        continue;
    }

    mlir::Value data = gbq->getOperand(0);
    mlir::Value scales = gbq->getOperand(2);
    auto *constOp = data.getDefiningOp();
    if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
      continue;

    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto scalesType = mlir::dyn_cast<mlir::RankedTensorType>(scales.getType());
    if (!dataType || !scalesType)
      continue;

    const int64_t blockSize = getIntAttr(gbq, "block_size", 0);
    if (blockSize <= 0)
      continue;

    const int qa =
        normalizeAxis(getIntAttr(gbq, "quantize_axis", -1), dataType.getRank());
    if (isAlreadyPackedByteTensor(dataType, scalesType, qa, blockSize))
      continue;
    if (!needsLogicalInt4Legalize(dataType, scalesType, qa, blockSize))
      continue;

    auto shape = llvm::to_vector(dataType.getShape());
    shape[qa] /= 2;
    const bool signedStorage = inferSignedStorage(dataType, graph, constOp);
    auto packedType = mlir::RankedTensorType::get(
        shape, packedElementType(builder.getContext()));

    if (!signedStorage) {
      gbq->setAttr("unsigned_quant_storage",
                   mlir::UnitAttr::get(builder.getContext()));
    } else {
      gbq->removeAttr("unsigned_quant_storage");
    }

    builder.setInsertionPoint(constOp);
    mlir::Operation *newConst = recreateConstant(builder, constOp, packedType);
    if (!newConst)
      continue;

    if (auto names = newConst->getAttrOfType<mlir::ArrayAttr>(
            attr_names::NODE_OUTPUTS)) {
      if (!names.empty()) {
        if (auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(names[0])) {
          auto idx = graph.get_node_arg_index(nameAttr.getValue().str());
          if (idx.is_valid()) {
            if (auto *nodeArg = graph.get_node_arg(idx)) {
              nodeArg->setValue(newConst->getResult(0));
            }
          }
        }
      }
    }

    std::string shapeStr;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0)
        shapeStr += "x";
      shapeStr += std::to_string(shape[i]);
    }
    LOG(INFO) << "Legalized GatherBlockQuantized INT4 constant to packed i8"
              << (signedStorage ? "" : " (unsigned via unsigned_quant_storage)")
              << " shape=[" << shapeStr << "]";
  }
}

} // anonymous namespace

void legalizeGatherBlockQuantizedInt4Constants(MLIRGraph &graph,
                                               mlir::Block &block) {
  legalizeInBlock(graph, block);
}

void legalizeGatherBlockQuantizedInt4Constants(MLIRGraph &graph,
                                               mlir::func::FuncOp func) {
  func.walk([&](mlir::Block *block) {
    legalizeGatherBlockQuantizedInt4Constants(graph, *block);
  });
}

} // namespace mlir_impl
} // namespace morphizen
