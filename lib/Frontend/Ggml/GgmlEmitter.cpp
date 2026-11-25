/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Frontend/Ggml/GgmlEmitter.h"

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Frontend/Ggml/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace capture = hip::ggml::capture;
using hip::ggml::fail;

// The entry point every pass and the runtime contract look for. The capture's
// own name describes the observation, which is a different thing.
constexpr const char *kEntryPoint = "main_graph";

/// The names the runtime binds.
///
/// GGML value identities change whenever llama.cpp renumbers; these do not.
struct RuntimeAbi {
  /// By input value id, output value id, and state storage id respectively.
  std::unordered_map<int64_t, std::string> inputNames;
  std::unordered_map<int64_t, std::string> outputNames;
  std::unordered_map<int64_t, std::string> stateNames;
};

RuntimeAbi describeRuntimeAbi(const capture::CapturedGraph &graph) {
  std::unordered_map<int64_t, const capture::Value *> values;
  for (const capture::Value &value : graph.values)
    values.emplace(value.id, &value);

  RuntimeAbi abi;
  int64_t runtimeIndex = 0;
  int64_t stateIndex = 0;
  for (const capture::Input &entry : graph.inputs) {
    // A state is the one input the runtime binds once and carries forward.
    if (entry.role == "state") {
      const int64_t storageId = values.at(entry.id)->storageId;
      const std::string name = "state." + std::to_string(stateIndex++);
      abi.inputNames.emplace(entry.id, name);
      if (!abi.stateNames.emplace(storageId, name).second) {
        fail("storage ", storageId, " has more than one state ABI input");
      }
      continue;
    }
    if (entry.role != "input" && entry.role != "constant") {
      fail("input ", entry.id, " has unsupported role '" + entry.role + "'");
    }
    abi.inputNames.emplace(entry.id,
                           "runtime." + std::to_string(runtimeIndex++));
  }

  for (const capture::StateBinding &binding : graph.stateBindings) {
    if (abi.stateNames.count(binding.storageId) == 0) {
      fail("state binding for storage ", binding.storageId,
           " has no ABI input");
    }
  }

  int64_t modelIndex = 0;
  for (const capture::Output &entry : graph.outputs) {
    if (entry.kind == "state") {
      const int64_t storageId = values.at(entry.id)->storageId;
      auto name = abi.stateNames.find(storageId);
      if (name == abi.stateNames.end()) {
        fail("state output for storage ", storageId, " has no ABI input");
      }
      abi.outputNames.emplace(entry.id, name->second);
      continue;
    }
    if (entry.kind != "model") {
      fail("output ", entry.id,
           " has unsupported ABI kind '" + entry.kind + "'");
    }
    abi.outputNames.emplace(entry.id, "output." + std::to_string(modelIndex++));
  }
  return abi;
}

/// What a value's elements mean, which is the type the dialect gives it.
mlir::Type elementType(mlir::MLIRContext &context, const std::string &type) {
  if (type == "f16") {
    return mlir::Float16Type::get(&context);
  }
  if (type == "bf16") {
    return mlir::BFloat16Type::get(&context);
  }
  if (type == "f32") {
    return mlir::Float32Type::get(&context);
  }
  if (type == "f64") {
    return mlir::Float64Type::get(&context);
  }
  if (type == "i8") {
    return mlir::IntegerType::get(&context, 8);
  }
  if (type == "i16") {
    return mlir::IntegerType::get(&context, 16);
  }
  if (type == "i32") {
    return mlir::IntegerType::get(&context, 32);
  }
  if (type == "i64") {
    return mlir::IntegerType::get(&context, 64);
  }
  // Everything else is packed, and GGML decodes every packed encoding to f32
  // and to nothing else. What the bytes actually are is said in the layout,
  // because an element type can only say what a value means.
  if (!mlir::ggml::getBlockLayout(type))
    fail("unsupported GGML type: " + type);
  return mlir::Float32Type::get(&context);
}

/// The layout a captured value has: its extents and byte strides in GGML order
/// and where it begins in the buffer it occupies.
///
/// This is the type's encoding rather than an attribute beside the operation,
/// so no pass can drop it and two values with equal types hold their elements
/// the same way.
/// The encoding the bytes are in, when the element type does not already say
/// it.
mlir::StringAttr storageAttribute(mlir::OpBuilder &builder,
                                  const capture::Value &value) {
  std::optional<mlir::ggml::BlockLayout> block =
      mlir::ggml::getBlockLayout(value.ggmlType);
  if (!block || block->blockElements == 1)
    return {};
  return builder.getStringAttr(value.ggmlType);
}

mlir::ggml::LayoutAttr layoutAttribute(mlir::OpBuilder &builder,
                                       const capture::Value &value) {
  const auto extents = [&](const capture::Extents &source) {
    std::vector<mlir::Attribute> list;
    for (int64_t extent : source)
      list.push_back(builder.getI64IntegerAttr(extent));
    return list;
  };
  return mlir::ggml::LayoutAttr::get(
      builder.getContext(), extents(value.ne), extents(value.nb),
      builder.getI64IntegerAttr(value.storageOffset),
      storageAttribute(builder, value));
}

mlir::RankedTensorType tensorType(mlir::OpBuilder &builder,
                                  const capture::Value &value) {
  // GGML ne[] lists the contiguous dimension first. MLIR prints the outermost
  // dimension first, so preserve all four GGML dimensions in reverse.
  std::vector<int64_t> shape(value.ne.rbegin(), value.ne.rend());
  mlir::Type element = elementType(*builder.getContext(), value.ggmlType);
  return mlir::RankedTensorType::get(shape, element,
                                     layoutAttribute(builder, value));
}

/// The dialect operation a GGML one becomes.
///
/// Each operation declares which GGML operation it is, so the correspondence
/// is read off those declarations rather than guessed at from the spelling and
/// corrected where the guess is wrong. An operation the dialect has no
/// definition for is refused here, which is where a graph the compiler cannot
/// describe stops.
std::string dialectOperationName(mlir::MLIRContext &context,
                                 const std::string &operation) {
  const auto declared = [&]() -> std::string {
#include "hip/Dialect/Ggml/IR/GgmlOperationNames.inc"
    fail("GGML operation ", operation, " has no dialect definition");
  };
  const std::string name = declared();
  if (!context.isOperationRegistered(name))
    fail("dialect operation ", name, " is declared but not registered");
  return name;
}

/// Which operand's bytes a node's result is handed back, or nullopt when it
/// produces storage of its own.
///
/// This is the relation the dialect states on the operations themselves --
/// `ViewLikeOpInterface` for the reinterpretations,
/// `DestinationStyleOpInterface` for the write -- so that following operands
/// reaches the value that owns the bytes. It is spelled once more here to check
/// the capture against it.
std::optional<size_t> aliasedOperandSlot(const capture::Node &node) {
  if (node.operation == "TRANSPOSE")
    return 0;
#include "hip/Dialect/Ggml/IR/GgmlAliasing.inc"
  return std::nullopt;
}

/// The buffer a node writes, or nullopt when it produces one of its own.
///
/// A node that writes hands back the buffer of the operand at the slot it
/// writes through, so it has to have been given an operand there, and that
/// operand's buffer has to be the one the result names -- otherwise the result
/// would not be the buffer's new contents. What the write covers and whether
/// the buffer outlives the step are the result value and the buffer's role,
/// and are read from those.
std::optional<int64_t> writtenStorage(
    const capture::Node &node,
    const std::unordered_map<int64_t, int64_t> &operandsBySlot,
    const std::unordered_map<int64_t, const capture::Value *> &valueDescriptors,
    const capture::Value &outputValue) {
  if (!node.writeSlot)
    return std::nullopt;
  const auto refuse = [&](const auto &...text) {
    hip::ggml::fail("node ", node.id, " ", text...);
  };
  const auto operand = operandsBySlot.find(*node.writeSlot);
  if (operand == operandsBySlot.end())
    refuse("writes through GGML slot ", *node.writeSlot,
           ", which it was given no operand at");
  const auto written = valueDescriptors.find(operand->second);
  if (written == valueDescriptors.end())
    refuse("writes through undescribed value ", operand->second);
  if (written->second->storageId != outputValue.storageId)
    refuse("writes a buffer its result does not name");
  return outputValue.storageId;
}

/// The attributes one operation's configuration becomes.
///
/// GGML records an enum both by name and by ordinal; the dialect has the enum,
/// so the name is symbolized here and there is no second spelling to disagree
/// with it. The body is generated from the same declaration the capture
/// decodes by, so a value cannot be captured and miss the module.
void addDeclaredParameters(mlir::OperationState &state,
                           mlir::OpBuilder &builder,
                           const capture::NodeParameters &parameters) {
  const auto attribute = [&](const char *name, auto value) {
    using Value = decltype(value);
    if constexpr (std::is_same_v<Value, float>)
      state.addAttribute(name, builder.getF32FloatAttr(value));
    else if constexpr (std::is_same_v<Value, bool>)
      state.addAttribute(name, builder.getBoolAttr(value));
    else if constexpr (std::is_same_v<Value, int64_t>)
      state.addAttribute(name, builder.getI64IntegerAttr(value));
    else
      state.addAttribute(name, builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>(
                                   value.data(), value.size())));
  };
  const auto symbolized = [&](llvm::StringRef name, llvm::StringRef spelling) {
    mlir::Attribute symbol;
    if (name == "precision") {
      if (auto value = mlir::ggml::symbolizeComputePrecision(spelling))
        symbol =
            mlir::ggml::ComputePrecisionAttr::get(builder.getContext(), *value);
    } else if (name == "glu_kind") {
      if (auto value = mlir::ggml::symbolizeGluKind(spelling))
        symbol = mlir::ggml::GluKindAttr::get(builder.getContext(), *value);
    } else if (name == "mode") {
      if (auto value = mlir::ggml::symbolizeRopeMode(spelling))
        symbol = mlir::ggml::RopeModeAttr::get(builder.getContext(), *value);
    }
    if (!symbol)
      fail("unknown ", name, " '", spelling, "'");
    state.addAttribute(name, symbol);
  };

  // This module carries a view's offset and strides only in the result
  // #ggml.layout. Repeating them in the operation's optional `byte_offset` and
  // `byte_strides` would leave stale concrete values behind once shape
  // symbolization rewrites the layout.
  if (std::holds_alternative<capture::ViewParameters>(parameters))
    return;
#include "hip/Dialect/Ggml/IR/GgmlAttributes.inc"
}

/// The attributes that come from which operands an operation was given, rather
/// than from what it was configured with.
///
/// GGML fills a fixed-width operand array and leaves a slot empty; a dialect
/// operation has a variadic operand list and has to say how many operands each
/// argument took. Which arguments are optional is in the operation's own
/// declaration, so the body below is generated from it.
void addOperandParameters(mlir::OperationState &state, mlir::OpBuilder &builder,
                          const capture::Node &node,
                          const std::unordered_map<int64_t, int64_t> &slots,
                          const capture::Value &outputValue) {
  const auto present = [&](int64_t slot) {
    return slots.count(slot) != 0 ? 1 : 0;
  };
  const auto segments = [&](llvm::ArrayRef<int32_t> sizes) {
    state.addAttribute("operandSegmentSizes",
                       builder.getDenseI32ArrayAttr(sizes));
  };
  const auto flag = [&](llvm::StringRef name, bool value) {
    state.addAttribute(name, builder.getBoolAttr(value));
  };
  const std::string &operation = node.operation;
  [&] {
#include "hip/Dialect/Ggml/IR/GgmlOperandArguments.inc"
  }();

  if (const auto *view =
          std::get_if<capture::ViewParameters>(&node.parameters)) {
    // The offset a view starts at is the result's layout -- the capture wrote
    // it here and there, and they agreed.
    if (outputValue.storageOffset != view->byteOffset) {
      fail("view starts at byte ", view->byteOffset,
           " but its result begins at ", outputValue.storageOffset);
    }
  } else if (std::holds_alternative<capture::UnsupportedParameters>(
                 node.parameters)) {
    fail("operation ", node.operation, " has no typed semantics");
  }
}

void addValueProvenance(mlir::OperationState &state, mlir::OpBuilder &builder,
                        const capture::Value &value) {
  state.addAttribute("ggml.value_id", builder.getI64IntegerAttr(value.id));
  // The extents, the strides, and the offset are the result type's encoding,
  // and how far the buffer reaches follows from the root's.
  state.addAttribute("ggml.source_name", builder.getStringAttr(value.name));
}

mlir::ModuleOp emitModule(mlir::MLIRContext &context,
                          const capture::CapturedGraph &graph,
                          const hip::ggml::emit::Options &options) {
  const RuntimeAbi abi = describeRuntimeAbi(graph);

  std::unordered_map<int64_t, const capture::Value *> valueDescriptors;
  for (const capture::Value &value : graph.values)
    valueDescriptors.emplace(value.id, &value);
  mlir::OpBuilder typeBuilder(&context);
  const auto emittedTensorType = [&](const capture::Value &value) {
    return tensorType(typeBuilder, value);
  };

  const mlir::Location unknown = mlir::UnknownLoc::get(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(unknown);
  mlir::OpBuilder moduleBuilder(&context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  std::vector<mlir::Type> argumentTypes;
  std::vector<mlir::Type> resultTypes;
  for (const capture::Input &input : graph.inputs)
    argumentTypes.push_back(emittedTensorType(*valueDescriptors.at(input.id)));
  for (const capture::Output &output : graph.outputs)
    resultTypes.push_back(emittedTensorType(*valueDescriptors.at(output.id)));

  auto functionType = moduleBuilder.getFunctionType(argumentTypes, resultTypes);
  auto function =
      mlir::func::FuncOp::create(unknown, kEntryPoint, functionType);
  module.getBody()->push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder(entry, entry->begin());

  std::unordered_map<int64_t, mlir::Value> values;
  for (size_t index = 0; index < graph.inputs.size(); ++index) {
    const capture::Input &input = graph.inputs[index];
    values.emplace(input.id, entry->getArgument(index));
    function.setArgAttr(index, "ggml.io_name",
                        builder.getStringAttr(abi.inputNames.at(input.id)));
    function.setArgAttr(index, "ggml.source_name",
                        builder.getStringAttr(input.name));
    function.setArgAttr(index, "ggml.value_id",
                        builder.getI64IntegerAttr(input.id));
    // What the runtime does with the buffer follows from the role: a state is
    // bound once and carried forward, everything else is handed over per step.
    function.setArgAttr(
        index, "ggml.io_kind",
        builder.getStringAttr(input.role == "state" ? "state" : "runtime"));
    function.setArgAttr(index, "ggml.source_kind",
                        builder.getStringAttr(input.role));
  }
  for (size_t index = 0; index < graph.outputs.size(); ++index) {
    const capture::Output &output = graph.outputs[index];
    function.setResultAttr(
        index, "ggml.io_name",
        builder.getStringAttr(abi.outputNames.at(output.id)));
    function.setResultAttr(index, "ggml.source_name",
                           builder.getStringAttr(output.name));
    function.setResultAttr(index, "ggml.io_kind",
                           builder.getStringAttr(output.kind));
    function.setResultAttr(index, "ggml.value_id",
                           builder.getI64IntegerAttr(output.id));
  }

  std::unordered_map<int64_t, int64_t> inputIndices;
  std::unordered_map<int64_t, int64_t> outputIndices;
  for (size_t index = 0; index < graph.inputs.size(); ++index)
    inputIndices.emplace(graph.inputs[index].id, static_cast<int64_t>(index));
  for (size_t index = 0; index < graph.outputs.size(); ++index)
    outputIndices.emplace(graph.outputs[index].id, static_cast<int64_t>(index));
  std::vector<mlir::Attribute> statePairs;
  for (const capture::StateBinding &binding : graph.stateBindings) {
    // The buffer the runtime binds is the state's own value, and the state the
    // step ends with is the last node that wrote it.
    if (binding.writes.empty()) {
      fail("state ", binding.storageId,
           " is bound but nothing in the graph writes it");
    }
    statePairs.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr(
                                         abi.stateNames.at(binding.storageId))),
        builder.getNamedAttr(
            "input_index",
            builder.getI64IntegerAttr(inputIndices.at(binding.storageId))),
        builder.getNamedAttr(
            "output_index",
            builder.getI64IntegerAttr(outputIndices.at(binding.writes.back()))),
    }));
  }
  function->setAttr("ggml.state_pairs", builder.getArrayAttr(statePairs));

  // The buffer a value occupies is the one its aliased operand occupies: a view
  // hands back its source's bytes and a write hands back its destination's.
  // Those edges are the module's operands, so the recorded storage name is
  // checked against them and then left out.
  // GGML writes its KV cache in place; SSA has no such thing. Each write is
  // given a result below, and every later read of that storage is pointed at
  // it -- so the operands the module carries are the recorded ones except
  // where a write came first.
  // The nodes are in execution order, so where a node sits is when it runs.
  std::unordered_map<int64_t, const capture::Node *> nodesById;
  std::unordered_map<int64_t, int64_t> nodeOrder;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    nodesById.emplace(node.id, &node);
    nodeOrder.emplace(node.id, static_cast<int64_t>(index));
  }

  // Which buffers a node ever writes: the ones a write hands back, which are
  // the ones those nodes' results name.
  std::set<int64_t> mutableStorage;
  for (const capture::Node &node : graph.nodes) {
    if (node.writeSlot)
      mutableStorage.insert(valueDescriptors.at(node.id)->storageId);
  }

  // What each mutable storage currently holds, and when it last changed. A
  // storage starts out holding itself: the value the runtime bound.
  std::unordered_map<int64_t, int64_t> currentState;
  std::unordered_map<int64_t, int64_t> lastWrite;
  for (int64_t storageId : mutableStorage) {
    currentState[storageId] = storageId;
    lastWrite[storageId] = -1;
  }

  const auto isFullStorageView = [&](int64_t valueId, int64_t storageId) {
    const capture::Value &value = *valueDescriptors.at(valueId);
    const capture::Value &root = *valueDescriptors.at(storageId);
    return value.storageOffset == 0 && value.ne == root.ne &&
           value.nb == root.nb;
  };
  // A write that covers no elements leaves the buffer as it was.
  const auto writesNothing = [](const capture::Value &result) {
    if (result.byteCount == 0)
      return true;
    for (int64_t extent : result.ne) {
      if (extent == 0)
        return true;
    }
    return false;
  };
  // A write covers the whole buffer when it starts at its beginning and has
  // the buffer's own extents.
  // Whether a write replaces every byte of the buffer it writes through.
  //
  // Bytes and not extents: llama.cpp's V cache, when it is stored transposed,
  // is written through `ggml_reshape_2d(v, 1, ggml_nelements(v))` -- one row
  // per element, because the indices address single elements. That is the same
  // bytes as the cache and every one of them, and comparing `ne` would call it
  // a partial update because the extents are not the cache's.
  const auto coversWholeStorage = [&](const capture::Value &result) {
    const capture::Value &root = *valueDescriptors.at(result.storageId);
    return result.storageOffset == 0 && result.byteCount == root.byteCount;
  };

  // The buffer a value occupies is the one its aliased operand occupies: a
  // view hands back its source's bytes and a write hands back its
  // destination's. Those edges are the module's operands, so the recorded
  // storage name is checked against them and then left out. The check needs
  // every node, so it runs once they have all been walked.
  std::unordered_map<int64_t, int64_t> storageRoots;

  for (auto [executionIndex, node] : llvm::enumerate(graph.nodes)) {
    const int64_t outputId = node.id;

    // A read of mutable storage names the value that storage last held, not
    // the buffer the runtime bound.
    std::vector<capture::Operand> ssaOperands = node.operands;
    for (capture::Operand &operand : ssaOperands) {
      const int64_t valueId = operand.value;
      const int64_t storageId = valueDescriptors.at(valueId)->storageId;
      if (!mutableStorage.count(storageId) ||
          currentState.at(storageId) == storageId) {
        continue;
      }
      if (valueId == storageId || isFullStorageView(valueId, storageId)) {
        operand.value = currentState.at(storageId);
        continue;
      }
      // A partial alias built before the write still describes the old bytes,
      // and rebuilding it is not something this emitter can do.
      auto producer = nodesById.find(valueId);
      if (producer == nodesById.end() ||
          nodeOrder.at(producer->second->id) < lastWrite.at(storageId)) {
        fail("state alias ", valueId, " is read after storage ", storageId,
             " changed, but the alias was not rebuilt from the updated state");
      }
    }

    {
      int64_t root = outputId;
      const std::optional<size_t> slot = aliasedOperandSlot(node);
      if (slot) {
        std::optional<int64_t> source;
        for (const capture::Operand &operand : ssaOperands) {
          if (static_cast<size_t>(operand.slot) == *slot)
            source = operand.value;
        }
        // Nothing at that slot means the node was handed no buffer and owns
        // its own, which is how GGML spells a copy that allocates: it points
        // the destination slot at the node itself, and the capture does not
        // record a node as its own operand.
        if (source) {
          const auto found = storageRoots.find(*source);
          root = found == storageRoots.end() ? *source : found->second;
        }
      }
      if (root != outputId)
        storageRoots.emplace(outputId, root);
    }

    std::vector<mlir::Value> operands;
    std::vector<int64_t> operandSlots;
    std::unordered_map<int64_t, int64_t> operandsBySlot;
    for (const capture::Operand &operand : ssaOperands) {
      const auto found = values.find(operand.value);
      if (found == values.end()) {
        fail("node ", outputId, " references unavailable value ",
             operand.value);
      }
      operands.push_back(found->second);
      operandSlots.push_back(operand.slot);
      operandsBySlot[operand.slot] = operand.value;
    }

    const capture::Value &outputValue = *valueDescriptors.at(outputId);
    const mlir::OperationName registered(
        dialectOperationName(context, node.operation), &context);
    mlir::OperationState state(
        mlir::NameLoc::get(builder.getStringAttr(node.name)), registered);
    state.addOperands(operands);
    state.addTypes(emittedTensorType(outputValue));
    // The SSA operand order is the GGML slot order, and an operation writes
    // the storage its own result names. Both were recorded; both are checked
    // here and left out of the module.
    //
    // GGML leaves a gap where an optional operand was not given. An operation
    // that records how many operands each of its arguments took can carry that
    // gap; one that does not would silently shift every later operand into the
    // missing one's place, so there the slots have to be dense.
    const bool recordsOperandSegments =
        registered.hasTrait<mlir::OpTrait::AttrSizedOperandSegments>();
    int64_t previousSlot = -1;
    for (size_t index = 0; index < operandSlots.size(); ++index) {
      const int64_t slot = operandSlots[index];
      if (slot <= previousSlot) {
        fail("node ", outputId, " fills GGML slot ", slot, " from SSA operand ",
             index, ", out of GGML order");
      }
      previousSlot = slot;
      if (!recordsOperandSegments && slot != static_cast<int64_t>(index)) {
        fail("node ", outputId, " leaves GGML slot ", index, " empty, and ",
             registered.getStringRef().str(),
             " records no operand segment sizes to say so");
      }
    }
    const std::optional<int64_t> written =
        writtenStorage(node, operandsBySlot, valueDescriptors, outputValue);
    addDeclaredParameters(state, builder, node.parameters);
    addOperandParameters(state, builder, node, operandsBySlot, outputValue);
    addValueProvenance(state, builder, outputValue);
    values.emplace(outputId, builder.create(state)->getResult(0));

    // What this node leaves the buffer holding, for the reads that follow. A
    // write that covers no elements leaves it as it was, so the state keeps
    // its identity rather than gaining a result nothing produced.
    if (written && !writesNothing(outputValue)) {
      if (!coversWholeStorage(outputValue)) {
        fail("node ", node.id, " partially updates mutable storage ", *written,
             "; full-storage MemorySSA materialization is not implemented");
      }
      currentState[*written] = outputId;
      lastWrite[*written] = static_cast<int64_t>(executionIndex);
    }
  }

  // Both of these need every node, so they run once the walk is done.
  for (const capture::Value &value : graph.values) {
    const auto found = storageRoots.find(value.id);
    const int64_t root = found == storageRoots.end() ? value.id : found->second;
    if (value.storageId != root) {
      fail("value ", value.id, " (", value.name, ") names storage ",
           value.storageId, ", but its operands reach ", root);
    }
  }
  for (const capture::StateBinding &binding : graph.stateBindings) {
    if (currentState.at(binding.storageId) != binding.writes.back()) {
      fail("state ", valueDescriptors.at(binding.storageId)->name,
           " does not end the step holding its last write");
    }
  }

  std::vector<mlir::Value> outputs;
  for (const capture::Output &output : graph.outputs) {
    const auto found = values.find(output.id);
    if (found == values.end()) {
      fail("graph output references unavailable value ", output.id);
    }
    outputs.push_back(found->second);
  }
  mlir::func::ReturnOp::create(builder, unknown, outputs);

  // What the compiler's ggml-to-hip pipeline requires the module to be (and
  // selects it by).
  module->setAttr("ggml.schema",
                  moduleBuilder.getStringAttr("hip-ggml-graph-v1"));
  std::vector<mlir::Attribute> symbols;
  // The caller's symbols, where they enter: the symbolizer seeds them at
  // the inputs and the views they name.
  for (const hip::ggml::emit::CallerSymbol &symbol : options.callerSymbols) {
    std::vector<mlir::NamedAttribute> described{
        moduleBuilder.getNamedAttr("name",
                                   moduleBuilder.getStringAttr(symbol.name)),
        moduleBuilder.getNamedAttr("type",
                                   moduleBuilder.getStringAttr("index")),
        moduleBuilder.getNamedAttr(
            "minimum", moduleBuilder.getI64IntegerAttr(symbol.minimum)),
        moduleBuilder.getNamedAttr(
            "maximum", moduleBuilder.getI64IntegerAttr(symbol.maximum))};
    if (!symbol.product.empty()) {
      std::vector<llvm::StringRef> factors(symbol.product.begin(),
                                           symbol.product.end());
      described.push_back(moduleBuilder.getNamedAttr(
          "product", moduleBuilder.getStrArrayAttr(factors)));
    }
    std::vector<mlir::Attribute> places;
    for (const hip::ggml::emit::CallerSymbol::Place &place : symbol.places) {
      std::vector<mlir::NamedAttribute> where{
          moduleBuilder.getNamedAttr("input",
                                     moduleBuilder.getStringAttr(place.value)),
          moduleBuilder.getNamedAttr(
              "dimension", moduleBuilder.getI64IntegerAttr(place.dimension)),
          moduleBuilder.getNamedAttr(
              "scale", moduleBuilder.getI64IntegerAttr(place.scale))};
      if (place.offset != 0)
        where.push_back(moduleBuilder.getNamedAttr(
            "offset", moduleBuilder.getI64IntegerAttr(place.offset)));
      places.push_back(moduleBuilder.getDictionaryAttr(where));
    }
    if (!places.empty())
      described.push_back(moduleBuilder.getNamedAttr(
          "inputs", moduleBuilder.getArrayAttr(places)));
    symbols.push_back(moduleBuilder.getDictionaryAttr(described));
  }
  module->setAttr("ggml.symbols", moduleBuilder.getArrayAttr(symbols));
  return module;
}

} // namespace

namespace hip::ggml::emit {

std::string serializeModule(const capture::CapturedGraph &graph,
                            const Options &options) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ggml::GgmlDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::ModuleOp module = emitModule(context, graph, options);
  if (mlir::failed(mlir::verify(module))) {
    module.erase();
    fail("generated MLIR module failed verification");
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream << '\n';
  stream.flush();
  module.erase();

  return text;
}

} // namespace hip::ggml::emit
