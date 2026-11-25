/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlToHip.cpp - Direct GGML-to-HIP conversion pass ----------------===//

#include "GgmlToHipUtils.h"
#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTGGMLTOHIPPASS
#include "hip/Conversion/GgmlToHip/Passes.h.inc"

namespace {

static StringAttr elementTypeName(OpBuilder &builder, Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  stream.flush();
  return builder.getStringAttr(text);
}

/// The dialect can describe packed state and intermediate tensors, but HIP
/// lowering supports packed storage only as a packed runtime weight's bytes
/// (lowerPackedRuntimeWeights). Check before layouts are erased or fusions can
/// hide a packed intermediate.
static LogicalResult verifySupportedStorage(ModuleOp module) {
  const auto check = [](Type type, Operation *owner, const Twine &description,
                        bool packedWeight = false) -> LogicalResult {
    ggml::LayoutAttr layout = ggml::getGgmlLayout(type);
    if (!layout || !layout.getStorage())
      return success();
    StringAttr encoding = layout.getStorage();
    std::optional<ggml::BlockLayout> storage =
        ggml::getBlockLayout(encoding.getValue());
    if (!storage)
      return owner->emitOpError() << description << " has unsupported storage "
                                  << encoding.getValue();
    if (storage->blockElements == 1 || packedWeight)
      return success();
    return owner->emitOpError()
           << description << " is stored as " << encoding.getValue()
           << "; HIP lowering supports packed storage only for weights";
  };

  WalkResult walked = module.walk([&](Operation *op) {
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      // Declarations have no block arguments or return operations to inspect.
      for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
        if (failed(check(type, op, "function input " + Twine(index))))
          return WalkResult::interrupt();
      }
      for (auto [index, type] : llvm::enumerate(function.getResultTypes())) {
        if (failed(check(type, op, "function output " + Twine(index))))
          return WalkResult::interrupt();
      }
    }
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      // A packed runtime weight's bytes are cast to its type
      // (lowerPackedRuntimeWeights), for the packed matmul alone.
      if (failed(check(result.getType(), op, "result " + Twine(index),
                       isa<UnrealizedConversionCastOp>(op))))
        return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          if (failed(check(argument.getType(), op,
                           "block argument " + Twine(argument.getArgNumber()))))
            return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

/// Takes each value's layout out of its type and leaves it beside the operation
/// that produced it.
///
/// In the `ggml` dialect a layout is the tensor's encoding. The dialects this
/// converts to have no such encoding, and every pattern builds
/// what it produces from the type it replaces, so a layout left in a type would
/// travel into an operation whose dialect cannot read it.
/// verifySupportedStorage has checked that dense encodings match their element
/// types and that packed storage is confined to packed weights, whose
/// encoding their cast keeps.
///
/// The converted IR keeps the extents and strides anyway, for the readers that
/// still walk them once the types no longer say anything. This puts them there
/// one step early, so that the conversion has one place to read them from.
static void materializeGgmlLayouts(ModuleOp module) {
  Builder builder(module.getContext());

  // Everything is read before anything is written, because how far a buffer
  // reaches is the root's business and the walk that finds the root needs the
  // layouts still on the values it passes through.
  SmallVector<std::pair<Value, SmallVector<NamedAttribute, 4>>> recorded;
  const auto read = [&](Value value) {
    ggml::LayoutAttr layout = ggml::getGgmlLayout(value);
    if (!layout)
      return;
    SmallVector<NamedAttribute, 4> fields{
        builder.getNamedAttr("ggml.ne",
                             builder.getArrayAttr(layout.getExtents())),
        builder.getNamedAttr("ggml.nb",
                             builder.getArrayAttr(layout.getStrides())),
        builder.getNamedAttr("ggml.storage_offset", layout.getOffset())};
    if (Attribute span = ggml::deriveStorageByteLength(value))
      fields.push_back(builder.getNamedAttr("ggml.storage_byte_length", span));
    recorded.emplace_back(value, std::move(fields));
  };

  module.walk([&](func::FuncOp function) {
    if (function.getBody().empty())
      return;
    for (BlockArgument argument : function.getBody().front().getArguments())
      read(argument);
  });
  module.walk([&](Operation *op) {
    if (op->getNumResults() == 1)
      read(op->getResult(0));
  });

  for (auto &[value, fields] : recorded) {
    if (Operation *producer = value.getDefiningOp()) {
      for (NamedAttribute field : fields)
        producer->setAttr(field.getName(), field.getValue());
      continue;
    }
    auto argument = cast<BlockArgument>(value);
    auto function = cast<func::FuncOp>(argument.getOwner()->getParentOp());
    for (NamedAttribute field : fields) {
      function.setArgAttr(argument.getArgNumber(), field.getName(),
                          field.getValue());
    }
  }

  // A function result restates the layout of the value returned there, which
  // the signature is about to lose.
  module.walk([&](func::FuncOp function) {
    if (function.getBody().empty())
      return;
    Block &entry = function.getBody().front();
    auto terminator = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!terminator)
      return;
    for (auto [index, returned] : llvm::enumerate(terminator.getOperands())) {
      if (ggml::LayoutAttr layout = ggml::getGgmlLayout(returned)) {
        function.setResultAttr(index, "ggml.ne",
                               builder.getArrayAttr(layout.getExtents()));
        function.setResultAttr(index, "ggml.nb",
                               builder.getArrayAttr(layout.getStrides()));
        function.setResultAttr(index, "ggml.storage_offset",
                               layout.getOffset());
        if (Attribute span = ggml::deriveStorageByteLength(returned)) {
          function.setResultAttr(index, "ggml.storage_byte_length", span);
        }
      }
    }
  });

  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      result.setType(ggml::dropGgmlLayout(result.getType()));
  });
  module.walk([&](func::FuncOp function) {
    if (function.getBody().empty())
      return;
    Block &entry = function.getBody().front();
    for (BlockArgument argument : entry.getArguments())
      argument.setType(ggml::dropGgmlLayout(argument.getType()));
    SmallVector<Type> results;
    for (Type type : function.getResultTypes())
      results.push_back(ggml::dropGgmlLayout(type));
    function.setType(
        builder.getFunctionType(entry.getArgumentTypes(), results));
  });
}

/// Record the positional tensor ABI before GGML operations and function
/// signatures are rewritten: what hip-compiler's GenerateInterface reads (the
/// tensors' counts, shapes, element sizes and types) and, for a GGML entry
/// point, their names.
static LogicalResult generateModuleMetadata(ModuleOp module) {
  auto mainFunction = module.lookupSymbol<func::FuncOp>("main_graph");
  if (!mainFunction) {
    // The conversion pass is also used by focused helper-function tests.
    // Only the public deployment entry point needs runtime ABI metadata.
    return success();
  }

  OpBuilder builder(module.getContext());
  const bool hasGgmlAbi = module->hasAttr("ggml.schema");
  int64_t inputCount = mainFunction.getNumArguments();
  SmallVector<Attribute> inputShapes;
  SmallVector<int64_t> inputElementSizes;
  SmallVector<Attribute> inputElementTypes;
  SmallVector<Attribute> inputNames;
  for (auto [index, type] : llvm::enumerate(mainFunction.getArgumentTypes())) {
    if (isa<hip::ContextType>(type)) {
      --inputCount;
      continue;
    }
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !tensorType.getElementType().isIntOrFloat()) {
      return mainFunction.emitError(
          "GGML entry inputs must be ranked numeric tensors");
    }
    inputShapes.push_back(builder.getDenseI64ArrayAttr(tensorType.getShape()));
    inputElementSizes.push_back(
        tensorType.getElementType().getIntOrFloatBitWidth() / 8);
    inputElementTypes.push_back(
        elementTypeName(builder, tensorType.getElementType()));
    if (hasGgmlAbi) {
      auto name =
          mainFunction.getArgAttrOfType<StringAttr>(index, "ggml.io_name");
      if (!name) {
        return mainFunction.emitError()
               << "GGML entry argument " << index << " requires io_name";
      }
      inputNames.push_back(name);
    }
  }

  SmallVector<Attribute> outputShapes;
  SmallVector<int64_t> outputElementSizes;
  SmallVector<Attribute> outputElementTypes;
  SmallVector<Attribute> outputNames;
  for (auto [index, type] : llvm::enumerate(mainFunction.getResultTypes())) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !tensorType.getElementType().isIntOrFloat()) {
      return mainFunction.emitError(
          "GGML entry results must be ranked numeric tensors");
    }
    outputShapes.push_back(builder.getDenseI64ArrayAttr(tensorType.getShape()));
    outputElementSizes.push_back(
        tensorType.getElementType().getIntOrFloatBitWidth() / 8);
    outputElementTypes.push_back(
        elementTypeName(builder, tensorType.getElementType()));
    if (hasGgmlAbi) {
      auto name =
          mainFunction.getResultAttrOfType<StringAttr>(index, "ggml.io_name");
      if (!name) {
        return mainFunction.emitError()
               << "GGML entry result " << index << " requires io_name";
      }
      outputNames.push_back(name);
    }
  }

  module->setAttr("hipdnn.input_count", builder.getI64IntegerAttr(inputCount));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputElementSizes));
  module->setAttr("hipdnn.input_element_types",
                  builder.getArrayAttr(inputElementTypes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(mainFunction.getNumResults()));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputShapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputElementSizes));
  module->setAttr("hipdnn.output_element_types",
                  builder.getArrayAttr(outputElementTypes));
  if (hasGgmlAbi) {
    module->setAttr("hipdnn.input_names", builder.getArrayAttr(inputNames));
    module->setAttr("hipdnn.output_names", builder.getArrayAttr(outputNames));
  }
  return success();
}

/// What the GGML side annotated - `ggml.*` attributes on the module, its
/// functions, their arguments and results, and operations - is the
/// conversion's to read:
/// none of it is read once it is done, so the hip IR it leaves does not
/// carry it.
static void dropGgmlAnnotations(ModuleOp module) {
  const auto ours = [](NamedAttribute a) {
    StringRef name = a.getName().getValue();
    return name.starts_with("ggml.");
  };
  const auto strip = [&](DictionaryAttr attributes) -> DictionaryAttr {
    if (!attributes || llvm::none_of(attributes, ours))
      return attributes;
    SmallVector<NamedAttribute> kept;
    llvm::copy_if(attributes, std::back_inserter(kept),
                  [&](NamedAttribute a) { return !ours(a); });
    return DictionaryAttr::get(module.getContext(), kept);
  };
  module->walk([&](Operation *op) {
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      for (unsigned i = 0; i < function.getNumArguments(); ++i)
        function.setArgAttrs(i, strip(function.getArgAttrDict(i)));
      for (unsigned i = 0; i < function.getNumResults(); ++i)
        function.setResultAttrs(i, strip(function.getResultAttrDict(i)));
    }
    // An operation's own attributes: the discardable ones (a function's
    // argument and result attributes above are inherent).
    SmallVector<StringAttr> names;
    for (NamedAttribute a : op->getDiscardableAttrs())
      if (ours(a))
        names.push_back(a.getName());
    for (StringAttr name : names)
      op->removeDiscardableAttr(name);
  });
}

/// One copy of each rotation constant per function. Every rotation by the
/// same constants (`ggml.rope.*`) then reads the same ones, so CSE computes its
/// cosines and sines once a step.
static void shareConstants(ModuleOp module) {
  module.walk([](func::FuncOp function) {
    if (function.getBody().empty())
      return;
    Block &entry = function.getBody().front();
    Builder builder(function.getContext());
    // The same values as the same tensor.
    const auto key = [&](hip::ConstantOp constant) {
      return builder.getArrayAttr({constant.getValueAttr(),
                                   constant.getSourceNameAttr(),
                                   TypeAttr::get(constant.getType())});
    };
    llvm::DenseMap<Attribute, hip::ConstantOp> first;
    SmallVector<std::pair<hip::ConstantOp, hip::ConstantOp>> repeated;
    Operation *hoisted = nullptr;
    function.walk([&](hip::ConstantOp constant) {
      StringAttr name = constant.getSourceNameAttr();
      // A rotation's frequencies and amplitude are inline; every rotation by
      // the same ones shares them, and with them (CSE) its cosines and sines.
      if (!name || !name.getValue().starts_with("ggml.rope.") ||
          !constant.getValueAttr())
        return;
      const Attribute k = key(constant);
      auto [it, inserted] = first.try_emplace(k, constant);
      if (!inserted) {
        repeated.emplace_back(constant, it->second);
        return;
      }
      // At the top, in the order they were first used.
      if (hoisted)
        constant->moveAfter(hoisted);
      else
        constant->moveBefore(&entry, entry.begin());
      hoisted = constant;
    });
    for (auto [constant, shared] : repeated) {
      constant.replaceAllUsesWith(shared.getResult());
      constant.erase();
    }
  });
}

/// A packed weight the caller hands over (a ggml backend's split is handed
/// the model's tensors): an input only multiplied by. Its bytes, ui8 [1, 1, N,
/// row bytes], which the native matmul kernels read where they are; what
/// multiplies by it sees them behind a cast to its type (MatmulConversion).
/// Runs before anything reads layouts. Any other input stored as blocks (a
/// quantized KV cache) is refused.
static LogicalResult lowerPackedRuntimeWeights(ModuleOp module) {
  auto function = module.lookupSymbol<func::FuncOp>("main_graph");
  if (!function || function.getBody().empty())
    return success();
  Block &entry = function.getBody().front();
  MLIRContext *context = module.getContext();

  bool changed = false;
  for (BlockArgument argument : entry.getArguments()) {
    auto type = dyn_cast<RankedTensorType>(argument.getType());
    auto layout =
        type ? dyn_cast_or_null<ggml::LayoutAttr>(type.getEncoding()) : nullptr;
    if (!layout || !layout.getStorage())
      continue;
    std::optional<ggml::BlockLayout> blocks =
        ggml::getBlockLayout(layout.getStorage().getValue());
    if (!blocks || blocks->blockElements == 1)
      continue;
    if (!llvm::all_of(argument.getUsers(), [&](Operation *user) {
          auto product = dyn_cast<ggml::MulMatOp>(user);
          return product && product.getWeight() == argument &&
                 product.getInput() != argument;
        }))
      return function.emitError()
             << "input " << argument.getArgNumber() << " is stored as "
             << layout.getStorage().getValue()
             << " and read other than as a matmul weight";
    if (type.getRank() != 4 || !type.hasStaticShape() ||
        type.getDimSize(3) % blocks->blockElements != 0)
      return function.emitError()
             << "input " << argument.getArgNumber()
             << " is a packed weight that is not static whole blocks";
    auto ui8 = IntegerType::get(context, 8, IntegerType::Unsigned);
    auto bytesType = RankedTensorType::get(
        {type.getDimSize(0), type.getDimSize(1), type.getDimSize(2),
         type.getDimSize(3) / blocks->blockElements * blocks->typeSize},
        ui8);
    argument.setType(bytesType);
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&entry);
    auto cast = UnrealizedConversionCastOp::create(builder, argument.getLoc(),
                                                   TypeRange{type}, argument);
    // Its format, past the layouts' removal.
    cast->setAttr("ggml.source_encoding", layout.getStorage());
    argument.replaceAllUsesExcept(cast.getResult(0), cast);
    changed = true;
  }
  if (!changed)
    return success();
  SmallVector<Type> results;
  for (Value value : entry.getTerminator()->getOperands())
    results.push_back(value.getType());
  function.setType(
      FunctionType::get(context, entry.getArgumentTypes(), results));
  return success();
}

struct ConvertGgmlToHipPass
    : impl::ConvertGgmlToHipPassBase<ConvertGgmlToHipPass> {
  using ConvertGgmlToHipPassBase::ConvertGgmlToHipPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    if (failed(lowerPackedRuntimeWeights(module)) ||
        failed(verifySupportedStorage(module)) ||
        failed(generateModuleMetadata(module))) {
      signalPassFailure();
      return;
    }

    // Preserve layout evidence before any fusion rewrites the GGML graph: the
    // tensor encodings' layouts become the operations' attributes.
    materializeGgmlLayouts(module);

    RewritePatternSet fusionPatterns(&context);
    populateGgmlFusionPatterns(fusionPatterns);
    populateGgmlRmsNormScaleFusionPatterns(fusionPatterns);
    if (failed(applyPatternsGreedily(module, std::move(fusionPatterns)))) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(context);
    target.addIllegalDialect<ggml::GgmlDialect>();
    target.addLegalDialect<hip::HipDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();
    // A packed runtime weight's cast to its type, until what multiplies by
    // it takes its bytes (MatmulConversion erases it then).
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp cast) {
          return cast->hasAttr("ggml.source_encoding");
        });

    RewritePatternSet patterns(&context);
    populateGgmlBasicConversionPatterns(patterns);
    populateGgmlLayoutConversionPatterns(patterns);
    populateGgmlComputeConversionPatterns(patterns);
    populateGgmlMatmulConversionPatterns(patterns);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet hipFusionPatterns(&context);
    populateHipAddRmsNormQ8FusionPatterns(hipFusionPatterns);
    populateHipKQuantSwiGluFusionPatterns(hipFusionPatterns);
    if (failed(applyPatternsGreedily(module, std::move(hipFusionPatterns)))) {
      signalPassFailure();
      return;
    }
    shareConstants(module);
    dropGgmlAnnotations(module);
  }
};

} // namespace

} // namespace hip
} // namespace mlir
