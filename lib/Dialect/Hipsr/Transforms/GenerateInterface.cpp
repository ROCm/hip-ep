/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/artifact_abi.h"
#include "hip/flatbuffers_json.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <iterator>

#include "model_metadata_generated.h"
#include "model_metadata_schema.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRGENERATEINTERFACEPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

constexpr StringRef kInputRanksGlobal = "__hipsr_input_ranks";
constexpr StringRef kInputRanksAttr = "hipdnn.input_ranks";

/// Everything `interface::collect` reads from the module before
/// `interface::generate` starts rewriting it.
struct InterfaceInfo {
  /// The lowered `@main_graph`, still using the expanded memref ABI.
  LLVM::LLVMFuncOp graph;
  /// One rank per graph input, in argument order.
  SmallVector<int64_t> inputRanks;
  /// Model metadata as FlatBuffers, handed to the runtime at init.
  SmallVector<uint8_t> metadataBlob;
  /// The same metadata as JSON, returned by `@inference_get_metadata_json`.
  std::string metadataJson;
  /// True when the module has op-state slots and an init function to call.
  bool hasOpStates;
};

/// Returns a dummy memref for the LLVM descriptor ABI. Only rank and address
/// space matter: pointers are opaque after convert-to-llvm, so `i8` is unused.
///
/// Example: dummyMemRef(ctx, 2, 1) -> memref<?x?xi8, 1>
static MemRefType dummyMemRef(MLIRContext *context, int64_t rank,
                              unsigned addressSpace = 0) {
  return MemRefType::get(SmallVector<int64_t>(rank, ShapedType::kDynamic),
                         IntegerType::get(context, 8), AffineMap(),
                         addressSpace);
}

namespace metadata {

/// Returns metadata for one input. The shape is unknown until the session
/// runs, so every extent is `-1`.
///
/// Example: buildTensorMetadata(2) -> shape [-1, -1]
static std::unique_ptr<hip::TensorInfoT> buildTensorMetadata(int64_t rank) {
  auto tensor = std::make_unique<hip::TensorInfoT>();
  tensor->shape.assign(rank, -1);
  return tensor;
}

/// Copies the module's constant sizes and offsets into `metadata`. Does
/// nothing when the module has no constants. Fails when only one attribute is
/// present, or when the two have different lengths.
///
/// Example:
///   module attributes {hipdnn.constant_sizes = array<i64: 16, 32>,
///                      hipdnn.constant_offsets = array<i64: 0, 16>}
///   appends {size 16, offset 0} and {size 32, offset 16}
static LogicalResult addConstantMetadata(ModuleOp module,
                                         hip::HipModelMetaInfoT &metadata) {
  auto sizes =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_sizes");
  auto offsets =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_offsets");
  if (static_cast<bool>(sizes) != static_cast<bool>(offsets)) {
    return module.emitError(
        "constant sizes and offsets must be present together");
  }
  if (!sizes) {
    return success();
  }
  if (sizes.size() != offsets.size()) {
    return module.emitError(
        "constant sizes and offsets must have the same length");
  }
  for (auto [size, offset] :
       llvm::zip_equal(sizes.asArrayRef(), offsets.asArrayRef())) {
    auto constant = std::make_unique<hip::ConstantInfoT>();
    constant->size = size;
    constant->offset = offset;
    metadata.constants.push_back(std::move(constant));
  }
  return success();
}

/// Returns the model metadata: one input entry per rank plus the constants.
/// The constants file name comes from `hip.constants_file`, or defaults to
/// "constants.bin". Fails when the constant attributes disagree.
///
/// Example:
///   module attributes {hip.constants_file = "weights.bin"}
///   with inputRanks {1, 2} gives input_count 2 and filename "weights.bin"
static FailureOr<hip::HipModelMetaInfoT>
buildMetadata(ModuleOp module, ArrayRef<int64_t> inputRanks) {
  hip::HipModelMetaInfoT info;
  info.version = 1;
  if (auto file = module->getAttrOfType<StringAttr>("hip.constants_file")) {
    info.constants_filename = file.getValue().str();
  } else {
    info.constants_filename = "constants.bin";
  }

  llvm::transform(inputRanks, std::back_inserter(info.inputs),
                  buildTensorMetadata);
  info.input_count = info.inputs.size();
  if (failed(addConstantMetadata(module, info))) {
    return failure();
  }
  return info;
}

/// Writes `info` twice: as a FlatBuffers blob for the runtime, and as JSON for
/// tools. Fails when the JSON conversion reports an error.
static LogicalResult serializeMetadata(ModuleOp module,
                                       const hip::HipModelMetaInfoT &info,
                                       SmallVectorImpl<uint8_t> &blob,
                                       std::string &json) {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(hip::HipModelMetaInfo::Pack(builder, &info));
  blob.append(builder.GetBufferPointer(),
              builder.GetBufferPointer() + builder.GetSize());

  std::string error;
  if (!hip::toJson<hip::HipModelMetaInfoT>(info, hip::k_model_metadata_schema(),
                                           json, error)) {
    return module.emitError("failed to serialize metadata JSON: ") << error;
  }
  return success();
}

} // namespace metadata

// LLVM dialect ops written into the module: globals, public functions, wrappers.
namespace llvm_ir {

/// Fills the entry block of a new function and hands back the result values.
using BuildBody = llvm::function_ref<SmallVector<Value>(OpBuilder &, Block &)>;

/// Creates a constant global at the top of the module. A global has no result,
/// so call `addressOf` inside a function body where the pointer is needed.
///
/// Example: constantGlobal(module, "n", !llvm.array<1 x i64>, dense<2>)
///   llvm.mlir.global internal constant @n(dense<2> : tensor<1xi64>)
///       : !llvm.array<1 x i64>
static void constantGlobal(ModuleOp module, StringRef name, Type type,
                           Attribute value) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPoint(&module.getBody()->front());
  LLVM::GlobalOp::create(builder, module.getLoc(), type,
                         /*isConstant=*/true, LLVM::Linkage::Internal, name,
                         value);
}

/// Returns the address of global or function `name`, taken at `builder`.
///
/// Example: addressOf(builder, module, "main_graph")
///   %0 = llvm.mlir.addressof @main_graph : !llvm.ptr
static Value addressOf(OpBuilder &builder, ModuleOp module, StringRef name) {
  return LLVM::AddressOfOp::create(
             builder, module.getLoc(),
             LLVM::LLVMPointerType::get(module.getContext()), name)
      .getResult();
}

/// Creates a constant global holding `bytes`.
///
/// Example: constantBytes(module, "__metadata_json", "{}")
///   llvm.mlir.global internal constant @__metadata_json("{}")
///       : !llvm.array<2 x i8>
static void constantBytes(ModuleOp module, StringRef name, StringRef bytes) {
  MLIRContext *ctx = module.getContext();
  constantGlobal(
      module, name,
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), bytes.size()),
      StringAttr::get(ctx, bytes));
}

/// Same, for raw bytes such as the FlatBuffers metadata blob.
static void constantBytes(ModuleOp module, StringRef name,
                          ArrayRef<uint8_t> bytes) {
  constantBytes(
      module, name,
      StringRef(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
}

/// Creates a constant global holding `values`. An empty `values` gives a
/// zero-length array, which the runtime never reads.
///
/// Example: constantI64Array(module, "__hipsr_input_ranks", {1, 2})
///   llvm.mlir.global internal constant @__hipsr_input_ranks(
///       dense<[1, 2]> : tensor<2xi64>) : !llvm.array<2 x i64>
static void constantI64Array(ModuleOp module, StringRef name,
                             ArrayRef<int64_t> values) {
  Type i64Type = IntegerType::get(module.getContext(), 64);
  constantGlobal(
      module, name, LLVM::LLVMArrayType::get(i64Type, values.size()),
      DenseElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(values.size())}, i64Type),
          values));
}

/// Returns a new public function at the end of the module. `buildBody` fills
/// the entry block and hands back the values the function returns.
///
/// Example: publicFunction(module, "f", !llvm.ptr, {}, body yielding %0)
///   llvm.func @f() -> !llvm.ptr attributes {llvm.emit_c_interface} {
///     %0 = llvm.mlir.addressof @__metadata_json : !llvm.ptr
///     llvm.return %0 : !llvm.ptr
///   }
static LLVM::LLVMFuncOp publicFunction(ModuleOp module, StringRef name,
                                       Type result, ArrayRef<Type> arguments,
                                       BuildBody buildBody) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      LLVM::LLVMFuncOp::create(builder, module.getLoc(), name,
                               LLVM::LLVMFunctionType::get(result, arguments));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  function->setAttr("sym_visibility", builder.getStringAttr("public"));
  Block *entry = function.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  SmallVector<Value> results = buildBody(builder, *entry);
  LLVM::ReturnOp::create(builder, module.getLoc(), results);
  return function;
}

/// Creates a public function that forwards to a runtime entry point and
/// returns its status. `buildArguments` maps the public arguments to the call
/// operands. Fails when the runtime declaration cannot be created.
///
/// Example: runtimeWrapper<CleanupRuntime>(module, "inference_cleanup",
///                                         "hipdnn_ep_state_cleanup", ...)
///   llvm.func @inference_cleanup(%arg0: !llvm.ptr) -> i32 {
///     %0 = llvm.call @hipdnn_ep_state_cleanup(%arg0) : (!llvm.ptr) -> i32
///     llvm.return %0 : i32
///   }
template <typename Runtime>
static LogicalResult
runtimeWrapper(ModuleOp module, StringRef name, StringRef runtimeName,
               ArrayRef<Type> arguments, BuildBody buildArguments) {
  OpBuilder builder(module.getContext());
  FailureOr<LLVM::LLVMFuncOp> runtime =
      Runtime::lookupOrCreateDeclaration(builder, module, runtimeName);
  if (failed(runtime)) {
    return failure();
  }

  publicFunction(module, name, builder.getI32Type(), arguments,
                 [&](OpBuilder &body, Block &entry) {
                   SmallVector<Value> callArguments =
                       buildArguments(body, entry);
                   Value result = LLVM::CallOp::create(body, module.getLoc(),
                                                       *runtime, callArguments)
                                      .getResult();
                   return SmallVector<Value>{result};
                 });
  return success();
}

} // namespace llvm_ir

namespace interface {

/// Signatures of the runtime entry points the trampolines call. They mirror
/// the C declarations in `lib/Runtime/hipdnn_ep_runtime.h`.
using CleanupRuntime = RuntimeFunc<i32, hostPtr>;
using InitRuntime =
    RuntimeFunc<i32, hostPtr, hostPtr, hostPtr, i64, hostPtr, hostPtr>;
using ComputeRuntime =
    RuntimeFunc<i32, hostPtr, hostPtr, hostPtr, i64, hostPtr>;

/// Reads the graph, the input ranks, the metadata, and the op-state flag
/// before the pass changes any IR. Fails when `@main_graph` is missing,
/// `hipdnn.input_ranks` is missing, the metadata attributes disagree, or the
/// module declares op-state slots without `@hipdnn_ep_op_states_init_fn`.
static FailureOr<InterfaceInfo> collect(ModuleOp module) {
  auto graph = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
  if (!graph) {
    return module.emitError("expected llvm.func @main_graph");
  }

  auto ranksAttr = module->getAttrOfType<DenseI64ArrayAttr>(kInputRanksAttr);
  if (!ranksAttr) {
    return module.emitError("expected ")
           << kInputRanksAttr << "; run -hipsr-externalize-constants first";
  }
  SmallVector<int64_t> inputRanks(ranksAttr.asArrayRef());
  FailureOr<hip::HipModelMetaInfoT> info =
      metadata::buildMetadata(module, inputRanks);
  if (failed(info)) {
    return failure();
  }

  auto opStateCount =
      module->getAttrOfType<IntegerAttr>("hipdnn.num_op_state_slots");
  const bool hasOpStates = opStateCount && opStateCount.getInt() != 0;
  if (hasOpStates &&
      !module.lookupSymbol<LLVM::LLVMFuncOp>(hipdnn::abi::kOpStatesInitFn)) {
    return module.emitError("expected ") << hipdnn::abi::kOpStatesInitFn;
  }

  InterfaceInfo collected{graph, std::move(inputRanks), {}, {}, hasOpStates};
  if (failed(metadata::serializeMetadata(module, *info, collected.metadataBlob,
                                         collected.metadataJson))) {
    return failure();
  }
  return collected;
}

/// Renames the lowered graph to `@main_graph_internal` and gives it a new
/// `@main_graph` wrapper the runtime can call through a function pointer. The
/// wrapper loads one descriptor per input and unpacks it into the expanded
/// arguments. Fails when the signature does not match `inputRanks`.
///
/// Before:
///   llvm.func @main_graph(%state: !llvm.ptr,
///                         %alloc: !llvm.ptr<1>, %aligned: !llvm.ptr<1>,
///                         %offset: i64, %size: i64, %stride: i64)
///       -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
///
/// After:
///   llvm.func private @main_graph_internal(...)  // the same body, renamed
///   llvm.func private @main_graph(%state: !llvm.ptr, %inputs: !llvm.ptr)
///       -> i32 {
///     %0 = llvm.getelementptr %inputs[0] : ...
///     %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
///     %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<...>
///     llvm.call @main_graph_internal(%state, ...unpacked %2...)
///     %3 = llvm.mlir.constant(0 : i32) : i32
///     llvm.return %3 : i32
///   }
static LogicalResult wrapMainGraph(ModuleOp module, LLVM::LLVMFuncOp graph,
                                   ArrayRef<int64_t> inputRanks) {
  OpBuilder builder(module.getContext());
  LLVMTypeConverter converter(graph.getContext());
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  Type i32Type = builder.getI32Type();
  LLVM::LLVMFunctionType graphType = graph.getFunctionType();

  int64_t expectedParameters = llvm::accumulate(
      inputRanks, int64_t{1}, [&](int64_t count, int64_t rank) {
        return count + MemRefDescriptor::getNumUnpackedValues(
                           dummyMemRef(graph.getContext(), rank));
      });
  if (graphType.getNumParams() != expectedParameters) {
    return graph.emitError(
        "expanded argument count does not match input ranks");
  }

  Location loc = graph.getLoc();
  graph.setName("main_graph_internal");
  graph.setLinkage(LLVM::Linkage::Private);

  builder.setInsertionPoint(graph);
  auto wrapper = LLVM::LLVMFuncOp::create(
      builder, loc, "main_graph",
      LLVM::LLVMFunctionType::get(i32Type, {pointer, pointer}));
  wrapper.setLinkage(LLVM::Linkage::Private);
  wrapper->setAttr("passthrough",
                   builder.getArrayAttr({builder.getStringAttr("noinline")}));

  Block *entry = wrapper.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value state = entry->getArgument(0);
  Value inputs = entry->getArgument(1);
  SmallVector<Value> graphArguments{state};

  int64_t parameter = 1;
  for (auto [inputIndex, rank] : llvm::enumerate(inputRanks)) {
    auto inputPointer =
        cast<LLVM::LLVMPointerType>(graphType.getParamType(parameter));
    MemRefType memref = dummyMemRef(graph.getContext(), rank,
                                                inputPointer.getAddressSpace());
    parameter += MemRefDescriptor::getNumUnpackedValues(memref);

    Type descriptorType = converter.convertType(memref);
    if (!descriptorType) {
      return graph.emitError("failed to convert packed memref descriptor type");
    }

    Value slot = LLVM::GEPOp::create(
        builder, loc, pointer, pointer, inputs,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(inputIndex)});
    Value descriptorPointer = LLVM::LoadOp::create(builder, loc, pointer, slot);
    Value descriptor =
        LLVM::LoadOp::create(builder, loc, descriptorType, descriptorPointer);
    MemRefDescriptor::unpack(builder, loc, descriptor, memref, graphArguments);
  }

  LLVM::CallOp::create(builder, loc, graph, graphArguments);
  Value zero = LLVM::ConstantOp::create(builder, loc, i32Type,
                                        builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, zero);
  return success();
}

/// Creates the `@inference_init` entry point. It passes the metadata blob and
/// the op-state init function, which is a null pointer when the module has no
/// op states.
///
/// Output:
///   llvm.func @inference_init(%state: !llvm.ptr, %fs: !llvm.ptr,
///                             %config: !llvm.ptr) -> i32 {
///     %0 = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
///     %1 = llvm.mlir.constant(64 : i64) : i64
///     %2 = llvm.mlir.zero : !llvm.ptr
///     %3 = llvm.call @hipdnn_ep_inference_init(%state, %fs, %0, %1,
///                                              %config, %2)
///     llvm.return %3 : i32
///   }
static LogicalResult generateInferenceInit(ModuleOp module,
                                           const InterfaceInfo &info) {
  MLIRContext *context = module.getContext();
  Type pointer = LLVM::LLVMPointerType::get(context);
  SmallVector<Type> arguments(3, pointer);
  return llvm_ir::runtimeWrapper<InitRuntime>(
      module, hipdnn::abi::kInferenceInit, "hipdnn_ep_inference_init",
      arguments, [&](OpBuilder &builder, Block &entry) {
        Location loc = module.getLoc();
        llvm_ir::constantBytes(module, hipdnn::abi::kMetadataBlobGlobal,
                            info.metadataBlob);
        Value blob =
            llvm_ir::addressOf(builder, module, hipdnn::abi::kMetadataBlobGlobal);
        Value size = LLVM::ConstantOp::create(
            builder, loc, builder.getI64Type(),
            builder.getI64IntegerAttr(
                static_cast<int64_t>(info.metadataBlob.size())));
        Value opStatesInit =
            info.hasOpStates
                ? llvm_ir::addressOf(builder, module, hipdnn::abi::kOpStatesInitFn)
                : LLVM::ZeroOp::create(builder, loc, pointer).getResult();
        return SmallVector<Value>{
            entry.getArgument(0), entry.getArgument(1), blob, size,
            entry.getArgument(2), opStatesInit};
      });
}

/// Creates the `@inference_compute` entry point. The ranks are known at
/// compile time, so they travel to the runtime as a constant global plus a
/// count, next to the address of the wrapped graph.
///
/// Output, for inputRanks {1, 2}:
///   llvm.func @inference_compute(%state: !llvm.ptr,
///                                %inputs: !llvm.ptr) -> i32 {
///     %0 = llvm.mlir.addressof @__hipsr_input_ranks : !llvm.ptr
///     %1 = llvm.mlir.constant(2 : i64) : i64
///     %2 = llvm.mlir.addressof @main_graph : !llvm.ptr
///     %3 = llvm.call @hipdnn_ep_inference_compute(%state, %inputs, %0, %1, %2)
///     llvm.return %3 : i32
///   }
static LogicalResult generateInferenceCompute(ModuleOp module,
                                              ArrayRef<int64_t> inputRanks) {
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  SmallVector<Type> arguments(2, pointer);
  return llvm_ir::runtimeWrapper<ComputeRuntime>(
      module, hipdnn::abi::kInferenceCompute, "hipdnn_ep_inference_compute",
      arguments, [&](OpBuilder &builder, Block &entry) {
        Location loc = module.getLoc();
        llvm_ir::constantI64Array(module, kInputRanksGlobal, inputRanks);
        Value ranks = llvm_ir::addressOf(builder, module, kInputRanksGlobal);
        Value count = LLVM::ConstantOp::create(
            builder, loc, builder.getI64Type(),
            builder.getI64IntegerAttr(static_cast<int64_t>(inputRanks.size())));
        Value graph = llvm_ir::addressOf(builder, module, "main_graph");
        return SmallVector<Value>{entry.getArgument(0), entry.getArgument(1),
                                  ranks, count, graph};
      });
}

/// Creates the `@inference_cleanup` entry point, which hands the state back to
/// the runtime for teardown. See `llvm_ir::runtimeWrapper` for the emitted IR.
static LogicalResult generateInferenceCleanup(ModuleOp module) {
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  return llvm_ir::runtimeWrapper<CleanupRuntime>(
      module, hipdnn::abi::kInferenceCleanup, "hipdnn_ep_state_cleanup",
      {pointer}, [](OpBuilder &, Block &entry) {
        return SmallVector<Value>{entry.getArgument(0)};
      });
}

/// Creates `@inference_get_metadata_json`. The global keeps a trailing NUL so
/// the caller can read it as a C string; the module owns the storage.
///
/// Output:
///   llvm.func @inference_get_metadata_json() -> !llvm.ptr {
///     %0 = llvm.mlir.addressof @__metadata_json : !llvm.ptr
///     llvm.return %0 : !llvm.ptr
///   }
static void generateMetadataAccessor(ModuleOp module,
                                     const std::string &metadataJson) {
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  llvm_ir::publicFunction(
      module, hipdnn::abi::kInferenceGetMetadataJson, pointer, {},
      [&](OpBuilder &builder, Block &) {
        llvm_ir::constantBytes(module, hipdnn::abi::kMetadataJsonGlobal,
                            metadataJson + '\0');
        return SmallVector<Value>{
            llvm_ir::addressOf(builder, module, hipdnn::abi::kMetadataJsonGlobal)};
      });
}

/// Writes the public inference ABI into `module`: the wrapped `@main_graph`,
/// the three runtime trampolines, and the metadata accessor.
static LogicalResult generate(ModuleOp module, const InterfaceInfo &info) {
  if (failed(wrapMainGraph(module, info.graph, info.inputRanks)) ||
      failed(generateInferenceInit(module, info)) ||
      failed(generateInferenceCompute(module, info.inputRanks)) ||
      failed(generateInferenceCleanup(module))) {
    return failure();
  }
  generateMetadataAccessor(module, info.metadataJson);
  return success();
}

} // namespace interface

struct HipsrGenerateInterfacePass
    : impl::HipsrGenerateInterfacePassBase<HipsrGenerateInterfacePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<InterfaceInfo> info = interface::collect(module);
    if (failed(info) || failed(interface::generate(module, *info))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
