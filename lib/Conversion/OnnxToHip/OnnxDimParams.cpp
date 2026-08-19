/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "OnnxDimParams.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hip {
namespace {

bool isBroadcastOp(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return llvm::is_contained({"onnx.Add", "onnx.And", "onnx.Div", "onnx.Equal",
                             "onnx.Greater", "onnx.GreaterOrEqual", "onnx.Less",
                             "onnx.LessOrEqual", "onnx.Mod", "onnx.Mul",
                             "onnx.Or", "onnx.Sub", "onnx.Where"},
                            name);
}

std::optional<std::string> getValueName(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto funcOp =
        dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!funcOp || argument.getOwner() != &funcOp.getBody().front())
      return std::nullopt;
    auto name = funcOp.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                                    "onnx.name");
    if (!name)
      return std::nullopt;
    return name.getValue().str();
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return std::nullopt;
  auto outputs =
      result.getDefiningOp()->getAttrOfType<ArrayAttr>("node.outputs");
  if (!outputs || result.getResultNumber() >= outputs.size())
    return std::nullopt;
  auto name = dyn_cast<StringAttr>(outputs[result.getResultNumber()]);
  if (!name || name.getValue().empty())
    return std::nullopt;
  return name.getValue().str();
}

bool isFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto funcOp =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  return funcOp && argument.getOwner() == &funcOp.getBody().front();
}

} // namespace

FailureOr<OnnxDimParams> OnnxDimParams::parse(ModuleOp module) {
  auto records = module->getAttrOfType<ArrayAttr>(kOnnxDimParamsModuleAttr);
  if (!records)
    return failure();

  OnnxDimParams result;
  llvm::StringSet<> seen;
  for (Attribute recordAttr : records) {
    auto record = dyn_cast<DictionaryAttr>(recordAttr);
    if (!record) {
      module.emitError("symbolic dimension record must be a dictionary");
      return failure();
    }
    auto scope = record.getAs<StringAttr>("scope");
    auto valueName = record.getAs<StringAttr>("value_name");
    auto dimensions = record.getAs<ArrayAttr>("dimensions");
    if (!scope || !valueName || !dimensions) {
      module.emitError("symbolic dimension record is missing required fields");
      return failure();
    }
    if (scope.getValue() != "main_graph")
      continue;
    if (valueName.getValue().empty()) {
      module.emitError("symbolic dimension value name is empty");
      return failure();
    }
    if (!seen.insert(valueName.getValue()).second) {
      module.emitError("duplicate symbolic dimension value record: ")
          << valueName.getValue();
      return failure();
    }

    llvm::SmallVector<std::string> params;
    params.reserve(dimensions.size());
    for (Attribute dimensionAttr : dimensions) {
      auto dimension = dyn_cast<StringAttr>(dimensionAttr);
      if (!dimension) {
        module.emitError("symbolic dimension entry must be a string");
        return failure();
      }
      params.push_back(dimension.getValue().str());
    }
    result.byValueName[valueName.getValue()] = std::move(params);
  }
  return result;
}

LogicalResult
OnnxDimParams::annotateBroadcastDimSources(func::FuncOp funcOp) const {
  llvm::StringMap<unsigned> inputBindings;
  llvm::StringSet<> liveNames;

  auto validateValue = [&](Value value) -> LogicalResult {
    auto name = getValueName(value);
    if (!name)
      return success();
    if (!liveNames.insert(*name).second)
      return funcOp.emitError("duplicate live ONNX value name: ") << *name;
    auto found = byValueName.find(*name);
    if (found == byValueName.end())
      return success();
    // ORT may retain tensor ValueInfo for an omitted optional output even
    // though the importer represents that result position as NoneType. Such a
    // value is an absence marker, not a live tensor, and cannot contribute a
    // symbolic broadcast proof.
    if (isa<NoneType>(value.getType()))
      return success();
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return funcOp.emitError("symbolic dimension metadata names a non-ranked "
                              "tensor value: ")
             << *name;
    if (static_cast<int64_t>(found->second.size()) != tensorType.getRank())
      return funcOp.emitError("symbolic dimension rank mismatch for ") << *name;
    return success();
  };

  for (BlockArgument argument : funcOp.getArguments()) {
    if (isa<hip::ContextType>(argument.getType()))
      continue;
    if (failed(validateValue(argument)))
      return failure();
    auto name = getValueName(argument);
    if (!name)
      continue;
    auto found = byValueName.find(*name);
    if (found == byValueName.end())
      continue;
    for (const std::string &symbol : found->second)
      if (!symbol.empty())
        ++inputBindings[symbol];
  }

  WalkResult validation = funcOp.walk([&](Operation *op) {
    if (op->getParentOp() != funcOp)
      return WalkResult::advance();
    for (Value result : op->getResults())
      if (failed(validateValue(result)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (validation.wasInterrupted())
    return failure();

  WalkResult annotation = funcOp.walk([&](Operation *op) {
    if (op->getParentOp() != funcOp)
      return WalkResult::advance();
    if (!isBroadcastOp(op) || op->getNumResults() != 1)
      return WalkResult::advance();
    if (op->hasAttr(kBroadcastDimSourcesAttr)) {
      op->emitError("broadcast operation already has a dimension-source plan");
      return WalkResult::interrupt();
    }
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return WalkResult::advance();

    llvm::SmallVector<int64_t> sources(resultType.getRank(), -1);
    bool hasSource = false;
    for (int64_t outputAxis : llvm::seq<int64_t>(resultType.getRank())) {
      if (!resultType.isDynamicDim(outputAxis))
        continue;

      llvm::SmallVector<std::pair<unsigned, std::string>> contributors;
      bool hasStaticNonUnit = false;
      bool complete = true;
      for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandType) {
          complete = false;
          break;
        }
        int64_t padding = resultType.getRank() - operandType.getRank();
        if (padding < 0) {
          complete = false;
          break;
        }
        if (outputAxis < padding)
          continue;
        int64_t operandAxis = outputAxis - padding;
        int64_t extent = operandType.getDimSize(operandAxis);
        if (extent == 1)
          continue;
        if (!ShapedType::isDynamic(extent)) {
          hasStaticNonUnit = true;
          continue;
        }

        auto name = getValueName(operand);
        if (!name) {
          complete = false;
          break;
        }
        auto found = byValueName.find(*name);
        if (found == byValueName.end() ||
            operandAxis >= static_cast<int64_t>(found->second.size()) ||
            found->second[operandAxis].empty()) {
          complete = false;
          break;
        }
        contributors.emplace_back(static_cast<unsigned>(operandIndex),
                                  found->second[operandAxis]);
      }
      if (!complete || hasStaticNonUnit || contributors.size() < 2)
        continue;
      llvm::StringRef symbol = contributors.front().second;
      if (!llvm::all_of(contributors, [&](const auto &contributor) {
            return contributor.second == symbol;
          }))
        continue;
      if (inputBindings.lookup(symbol) > 1)
        continue;

      unsigned canonical = contributors.front().first;
      for (const auto &[operandIndex, ignored] : contributors)
        if (isFunctionArgument(op->getOperand(operandIndex))) {
          canonical = operandIndex;
          break;
        }
      sources[outputAxis] = canonical;
      hasSource = true;
    }
    if (hasSource)
      op->setAttr(kBroadcastDimSourcesAttr,
                  DenseI64ArrayAttr::get(funcOp.getContext(), sources));
    return WalkResult::advance();
  });
  return annotation.wasInterrupted() ? failure() : success();
}

llvm::SmallVector<int64_t> getBroadcastDimSources(Operation *onnxOp) {
  auto sources =
      onnxOp->getAttrOfType<DenseI64ArrayAttr>(kBroadcastDimSourcesAttr);
  if (!sources)
    return {};
  return llvm::to_vector(sources.asArrayRef());
}

} // namespace mlir::hip
