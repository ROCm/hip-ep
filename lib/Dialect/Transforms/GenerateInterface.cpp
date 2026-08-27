/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Generate Interface Pass - Create C-compatible interface functions
//===----------------------------------------------------------------------===//
// This pass generates four C-ABI compatible functions that wrap the internal
// @main_graph function:
// - inference_init: Allocate context, create handles, upload constants
// - inference_compute: 2-arg (state, inputs) ABI -- stage inputs, call
//   @main_graph (graph outputs are allocated in-graph via hip.alloc_output)
// - inference_cleanup: Free resources
// - inference_get_metadata_json: Return JSON metadata (input/output shapes)
//===----------------------------------------------------------------------===//

#include "compilation_options_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/artifact_abi.h"
#include "hip/debug_log.h"
#include "hip/flatbuffers_json.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "model_metadata_generated.h"
#include "model_metadata_schema.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {

/// Build a TensorInfoT native struct from module tensor attributes at index i.
static std::unique_ptr<mlir::hip::TensorInfoT>
buildTensorInfo(ArrayAttr shapes, DenseI64ArrayAttr elementSizes, size_t i) {
  auto ti = std::make_unique<mlir::hip::TensorInfoT>();
  if (shapes && i < shapes.size()) {
    if (auto shapeAttr = dyn_cast<DenseI64ArrayAttr>(shapes.getValue()[i]))
      ti->shape.assign(shapeAttr.asArrayRef().begin(),
                       shapeAttr.asArrayRef().end());
  }
  ti->element_size =
      (elementSizes && i < static_cast<size_t>(elementSizes.size()))
          ? elementSizes[i]
          : 4;
  return ti;
}

/// Build a HipModelMetaInfoT native struct from module attributes.
/// Shared by buildMetadataBlob() and buildMetadataJson() so module attributes
/// are read exactly once.
/// Each ConstantInfo carries both size and a running byte offset, enabling
/// future non-sequential or grouped constant layouts.
mlir::hip::HipModelMetaInfoT
buildMetadataNative(ModuleOp module, const std::string &constantsFile) {
  auto inputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
  auto outputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
  auto inputElementSizes =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_element_sizes");
  auto outputElementSizes =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_element_sizes");
  auto inputCountAttr =
      module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
  auto outputCountAttr =
      module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
  auto constantSizesAttr =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_sizes");
  auto constantOffsetsAttr =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_offsets");

  // Streaming-mode source descriptors. Present when constant externalization
  // emitted per-constant source info (splat / file-ref / mem). Absent in
  // full-constants-file mode (EPContext export), in which case every
  // ConstantInfo keeps source = NONE and runtime falls back to the bulk
  // constants_filename read. In hybrid mode (skipDataWrite=true with mem-addr
  // entries) some constants carry MemSource pointing at a *partial* constants
  // file that holds only mem-addr bytes.
  auto sourceKindsAttr =
      module->getAttrOfType<DenseI32ArrayAttr>("hipdnn.constant_source_kinds");
  auto splatValuesAttr = module->getAttrOfType<DenseI64ArrayAttr>(
      "hipdnn.constant_splat_elem_values");
  auto splatElemSizesAttr = module->getAttrOfType<DenseI64ArrayAttr>(
      "hipdnn.constant_splat_elem_sizes");
  auto filePathsAttr =
      module->getAttrOfType<ArrayAttr>("hipdnn.constant_file_paths");
  auto fileOffsetsAttr =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_file_offsets");
  auto memOffsetsAttr =
      module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_mem_offsets");

  mlir::hip::HipModelMetaInfoT meta;
  meta.version = 1;
  meta.constants_filename = constantsFile;

  if (constantSizesAttr) {
    auto sizes = constantSizesAttr.asArrayRef();
    auto offsets = constantOffsetsAttr ? constantOffsetsAttr.asArrayRef()
                                       : ArrayRef<int64_t>{};
    auto kinds =
        sourceKindsAttr ? sourceKindsAttr.asArrayRef() : ArrayRef<int32_t>{};
    auto splatValues =
        splatValuesAttr ? splatValuesAttr.asArrayRef() : ArrayRef<int64_t>{};
    auto splatElemSizes = splatElemSizesAttr ? splatElemSizesAttr.asArrayRef()
                                             : ArrayRef<int64_t>{};
    auto fileOffsets =
        fileOffsetsAttr ? fileOffsetsAttr.asArrayRef() : ArrayRef<int64_t>{};
    auto memOffsets =
        memOffsetsAttr ? memOffsetsAttr.asArrayRef() : ArrayRef<int64_t>{};
    for (auto i : llvm::seq<size_t>(0, sizes.size())) {
      auto ci = std::make_unique<mlir::hip::ConstantInfoT>();
      ci->size = sizes[i];
      ci->offset = (i < offsets.size()) ? offsets[i] : 0;
      auto kind = static_cast<mlir::hip::ConstantMetadataSourceKind>(
          (i < kinds.size())
              ? kinds[i]
              : static_cast<int32_t>(
                    mlir::hip::ConstantMetadataSourceKind::Bulk));
      if (kind == mlir::hip::ConstantMetadataSourceKind::Splat) {
        // Splat: extract the left-packed elem bytes out of the i64 carrier.
        auto splat = std::make_unique<mlir::hip::SplatSourceT>();
        int64_t elemSize = (i < splatElemSizes.size()) ? splatElemSizes[i] : 0;
        size_t n = static_cast<size_t>(std::min<int64_t>(elemSize, 8));
        const auto *base = reinterpret_cast<const uint8_t *>(&splatValues[i]);
        splat->elem_bytes.assign(base, base + n);
        ci->source.Set(std::move(*splat));
      } else if (kind == mlir::hip::ConstantMetadataSourceKind::File) {
        auto fref = std::make_unique<mlir::hip::FileRefSourceT>();
        if (filePathsAttr && i < filePathsAttr.size()) {
          if (auto s = dyn_cast<StringAttr>(filePathsAttr.getValue()[i]))
            fref->path = s.getValue().str();
        }
        fref->file_offset = (i < fileOffsets.size()) ? fileOffsets[i] : 0;
        ci->source.Set(std::move(*fref));
      } else if (kind == mlir::hip::ConstantMetadataSourceKind::Memory) {
        // Mem: mem-addr entry packed into
        // HipModelMetaInfo.constants_filename at mem_offset by the
        // externalizer's hybrid artifact. Runtime reads size bytes from that
        // offset through the EP FileSystem.
        auto mem = std::make_unique<mlir::hip::MemSourceT>();
        mem->mem_offset = (i < memOffsets.size()) ? memOffsets[i] : 0;
        ci->source.Set(std::move(*mem));
      }
      meta.constants.push_back(std::move(ci));
    }
  }

  if (inputShapes) {
    for (auto i : llvm::seq<size_t>(0, inputShapes.size()))
      meta.inputs.push_back(buildTensorInfo(inputShapes, inputElementSizes, i));
  }
  if (outputShapes) {
    for (auto i : llvm::seq<size_t>(0, outputShapes.size()))
      meta.outputs.push_back(
          buildTensorInfo(outputShapes, outputElementSizes, i));
  }

  meta.input_count = inputCountAttr
                         ? inputCountAttr.getInt()
                         : (int64_t)(inputShapes ? inputShapes.size() : 0);
  meta.output_count = outputCountAttr
                          ? outputCountAttr.getInt()
                          : (int64_t)(outputShapes ? outputShapes.size() : 0);

  return meta;
}

/// Build FlatBuffers binary blob from module attributes.
/// Uses generated HipModelMetaInfoT native struct (--gen-object-api) so the
/// code tracks schema changes automatically.
std::vector<uint8_t> buildMetadataBlob(ModuleOp module,
                                       const std::string &constantsFile) {
  mlir::hip::HipModelMetaInfoT meta =
      buildMetadataNative(module, constantsFile);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(mlir::hip::HipModelMetaInfo::Pack(fbb, &meta));
  const uint8_t *buf = fbb.GetBufferPointer();
  return std::vector<uint8_t>(buf, buf + fbb.GetSize());
}

/// Build JSON metadata string from module attributes.
/// Uses the embedded model_metadata schema so the output is always consistent
/// with the binary blob — no manual field mapping needed.
std::string buildMetadataJson(ModuleOp module,
                              const std::string &constantsFile) {
  mlir::hip::HipModelMetaInfoT meta =
      buildMetadataNative(module, constantsFile);
  std::string json, error;
  if (!mlir::hip::toJson<mlir::hip::HipModelMetaInfoT>(
          meta, mlir::hip::k_model_metadata_schema(), json, error)) {
    llvm::errs() << "[GenerateInterface] " << error << "\n";
    return "";
  }
  return json;
}

/// Generate global constant string for metadata JSON.
///
/// Generated IR (1 input / 1 output, shape [8], element_size 4):
///   llvm.mlir.global internal constant @__metadata_json(
///       "{\0A  \22constants_filename\22: ...}\0A\00") {addr_space = 0 : i32}
void generateMetadataGlobal(ModuleOp module, const std::string &jsonStr) {
  OpBuilder builder(module.getContext());
  auto i8Type = builder.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, jsonStr.size() + 1);

  builder.setInsertionPoint(&module.getBody()->front());
  LLVM::GlobalOp::create(builder, module.getLoc(), arrayType,
                         /*isConstant=*/true, LLVM::Linkage::Internal,
                         hipdnn::abi::kMetadataJsonGlobal,
                         builder.getStringAttr(jsonStr + '\0'));
}

/// Generate global constant for metadata FlatBuffers blob.
///
/// Generated IR (168-byte blob):
///   llvm.mlir.global internal constant @__metadata_blob(
///       "\18\00\00\00...") {addr_space = 0 : i32}
void generateMetadataBlobGlobal(ModuleOp module,
                                const std::vector<uint8_t> &blob) {
  OpBuilder builder(module.getContext());
  auto i8Type = builder.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, blob.size());

  builder.setInsertionPoint(&module.getBody()->front());
  llvm::StringRef blobRef(reinterpret_cast<const char *>(blob.data()),
                          blob.size());
  LLVM::GlobalOp::create(builder, module.getLoc(), arrayType,
                         /*isConstant=*/true, LLVM::Linkage::Internal,
                         hipdnn::abi::kMetadataBlobGlobal,
                         builder.getStringAttr(blobRef));
}

/// Generate inference_get_metadata_json() function.
///
/// Generated IR example:
///   llvm.func @inference_get_metadata_json() -> !llvm.ptr
///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
///     %0 = llvm.mlir.addressof @__metadata_json : !llvm.ptr
///     llvm.return %0 : !llvm.ptr
///   }
void generateInferenceGetMetadataJson(ModuleOp module) {
  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();

  builder.setInsertionPointToEnd(module.getBody());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
  auto funcType = LLVM::LLVMFunctionType::get(ptrType, {});

  auto funcOp = LLVM::LLVMFuncOp::create(
      builder, loc, hipdnn::abi::kInferenceGetMetadataJson, funcType);
  funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

  Block *entry = funcOp.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  Value addr = LLVM::AddressOfOp::create(builder, loc, ptrType,
                                         hipdnn::abi::kMetadataJsonGlobal);
  LLVM::ReturnOp::create(builder, loc, addr);
}

/// Build an LLVM memref descriptor struct {ptr, ptr, offset, sizes, strides}
/// from a TensorBuffer's GPU pointer and shape pointer.
static Value buildMemrefDescriptor(OpBuilder &builder, Location loc,
                                   Value gpuPtrRaw, Value shapePtr,
                                   int64_t rank, Type ptrType, Type i64Type) {
  Type gpuPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
  Value gpuPtr =
      LLVM::AddrSpaceCastOp::create(builder, loc, gpuPtrType, gpuPtrRaw);

  Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  Type memrefType = LLVM::LLVMStructType::getLiteral(
      builder.getContext(),
      {gpuPtrType, gpuPtrType, i64Type, sizeArrayType, sizeArrayType});

  Value desc = LLVM::UndefOp::create(builder, loc, memrefType);
  desc = LLVM::InsertValueOp::create(builder, loc, desc, gpuPtr,
                                     ArrayRef<int64_t>{0});
  desc = LLVM::InsertValueOp::create(builder, loc, desc, gpuPtr,
                                     ArrayRef<int64_t>{1});
  Value zero = LLVM::ConstantOp::create(builder, loc, i64Type,
                                        builder.getI64IntegerAttr(0));
  desc = LLVM::InsertValueOp::create(builder, loc, desc, zero,
                                     ArrayRef<int64_t>{2});

  Value sizes = LLVM::UndefOp::create(builder, loc, sizeArrayType);
  for (int64_t d : llvm::seq<int64_t>(0, rank)) {
    Value idx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                         builder.getI64IntegerAttr(d));
    Value dimPtr = LLVM::GEPOp::create(builder, loc, ptrType, ptrType, shapePtr,
                                       ArrayRef<LLVM::GEPArg>{idx});
    Value dimVal = LLVM::LoadOp::create(builder, loc, i64Type, dimPtr);
    sizes = LLVM::InsertValueOp::create(builder, loc, sizes, dimVal,
                                        ArrayRef<int64_t>{d});
  }
  desc = LLVM::InsertValueOp::create(builder, loc, desc, sizes,
                                     ArrayRef<int64_t>{3});

  Value strides = LLVM::UndefOp::create(builder, loc, sizeArrayType);
  Value acc = LLVM::ConstantOp::create(builder, loc, i64Type,
                                       builder.getI64IntegerAttr(1));
  for (int64_t d : llvm::reverse(llvm::seq<int64_t>(0, rank))) {
    strides = LLVM::InsertValueOp::create(builder, loc, strides, acc,
                                          ArrayRef<int64_t>{d});
    if (d > 0) {
      Value idx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                           builder.getI64IntegerAttr(d));
      Value dimPtr = LLVM::GEPOp::create(builder, loc, ptrType, ptrType,
                                         shapePtr, ArrayRef<LLVM::GEPArg>{idx});
      Value dimSize = LLVM::LoadOp::create(builder, loc, i64Type, dimPtr);
      acc = LLVM::MulOp::create(builder, loc, acc, dimSize);
    }
  }
  desc = LLVM::InsertValueOp::create(builder, loc, desc, strides,
                                     ArrayRef<int64_t>{4});

  return desc;
}

/// Emit a call to a runtime function and branch to errorBlock if it
/// returns non-zero.  Leaves the builder at the start of a new
/// continuation block.
static void emitErrorCheckedCall(OpBuilder &builder, Location loc,
                                 LLVM::LLVMFuncOp func, ValueRange args,
                                 Value errorCodePtr, Block *errorBlock,
                                 LLVM::LLVMFuncOp &parentFunc) {
  Value ret = LLVM::CallOp::create(builder, loc, func, args).getResult();
  Value zero = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                        builder.getI32IntegerAttr(0));
  Value failed =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, ret, zero);

  Block *continueBlock = parentFunc.addBlock();
  Block *storeErrorBlock = parentFunc.addBlock();
  LLVM::CondBrOp::create(builder, loc, failed, storeErrorBlock, continueBlock);

  builder.setInsertionPointToStart(storeErrorBlock);
  LLVM::StoreOp::create(builder, loc, ret, errorCodePtr);
  LLVM::BrOp::create(builder, loc, errorBlock);

  builder.setInsertionPointToStart(continueBlock);
}

// Generates the four C-ABI interface functions (inference_init,
// inference_compute, inference_cleanup, inference_get_metadata_json) for a
// lowered module. inference_compute has the (state, inputs) -> i32 signature;
// graph outputs are allocated in-graph via hip.alloc_output, not passed as
// out-params.
class GenerateInterfacePass
    : public PassWrapper<GenerateInterfacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateInterfacePass)

  explicit GenerateInterfacePass(
      const mlir::hip::CompilationOptionsT &compilationOptions)
      : compilationOptions_(compilationOptions) {}
  explicit GenerateInterfacePass(
      mlir::hip::CompilationOptionsT &&compilationOptions)
      : compilationOptions_(std::move(compilationOptions)) {}

  StringRef getArgument() const final { return "generate-interface"; }
  StringRef getDescription() const final {
    return "Generate C interface wrapper functions (inference_init, "
           "inference_compute, inference_cleanup, inference_get_metadata_json)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(verifyPrerequisites(module))) {
      signalPassFailure();
      return;
    }

    // Prefer the hip.constants_file attribute set by constant externalization;
    // compilation-option override is not yet wired end-to-end.
    std::string constantsFile = "constants.bin";
    if (auto attr =
            module->getAttrOfType<mlir::StringAttr>("hip.constants_file")) {
      constantsFile = attr.getValue().str();
    } else if (!compilationOptions_.constants_file.empty()) {
      constantsFile = compilationOptions_.constants_file;
    }

    std::vector<uint8_t> blob = buildMetadataBlob(module, constantsFile);
    generateMetadataBlobGlobal(module, blob);

    declareRuntimeFunctions(module);

    generateInferenceInit(module, blob.size());
    auto inputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
    generateInferenceCompute(module, inputShapes);
    generateInferenceCleanup(module);

    std::string json = buildMetadataJson(module, constantsFile);
    generateMetadataGlobal(module, json);
    generateInferenceGetMetadataJson(module);

    COMPILER_DEBUG_LOG("[GenerateInterface] Generated 4 interface functions\n");
  }

private:
  mlir::hip::CompilationOptionsT compilationOptions_;

  struct RuntimeFuncSpec {
    llvm::StringRef name;
    Type resultType;
    SmallVector<Type, 4> paramTypes;
  };

  SmallVector<RuntimeFuncSpec> getRuntimeFuncSpecs(OpBuilder &builder) {
    Type ptr = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32 = builder.getI32Type();
    Type i64 = builder.getI64Type();
    Type vd = LLVM::LLVMVoidType::get(builder.getContext());
    return {
        {"malloc", ptr, {i64}},
        {"free", vd, {ptr}},
        {"hipStreamCreate", i32, {ptr}},
        {"hipStreamDestroy", i32, {ptr}},
        {"hipStreamSynchronize", i32, {ptr}},
        {"miopenCreate", i32, {ptr}},
        {"miopenSetStream", i32, {ptr, ptr}},
        {"miopenDestroy", i32, {ptr}},
        {"hipblasLtCreate", i32, {ptr}},
        {"hipblasLtDestroy", i32, {ptr}},
        {"hipdnn_ep_state_cleanup", i32, {ptr}},
        {"wrap_hipMalloc", i32, {ptr, i64}},
        {"wrap_hipFree", i32, {ptr}},
        {"wrap_hipMemcpyH2D", i32, {ptr, ptr, i64, ptr}},
        {"wrap_hipMemcpyD2H", i32, {ptr, ptr, i64, ptr}},
        {"wrap_hipStreamSynchronize", i32, {ptr}},
        {"hipdnn_ep_state_get_stream", ptr, {ptr}},
        {"hipdnn_ep_pool_init", i32, {ptr, i64, ptr, i64}},
        {"hipdnn_ep_get_buffer_from_pool", ptr, {ptr, i64}},
        {"hipdnn_ep_tensor_prepare_input", i32, {ptr, ptr, i64, i64, ptr}},
        {"hipdnn_ep_tensor_free_input", vd, {ptr, ptr}},
        {"hipdnn_ep_tensor_buffer_get_gpu_ptr", ptr, {ptr}},
        {"hipdnn_ep_tensor_buffer_get_host_ptr", ptr, {ptr}},
        {"hipdnn_ep_tensor_buffer_get_shape_ptr", ptr, {ptr}},
        {"hipdnn_ep_tensor_buffer_get_rank", i64, {ptr}},
        {"hipdnn_ep_tensor_buffer_get_size_bytes", i64, {ptr}},
        {"hipdnn_ep_state_init_with_fs", i32, {ptr, ptr, ptr, i64}},
        {"hipdnn_ep_stream_sync", i32, {ptr}},
        {"hipdnn_ep_state_reset_error_flag", i32, {ptr}},
        {"hipdnn_ep_state_read_and_clear_error_flag", i32, {ptr}},
        {"hipdnn_ep_runtime_begin_compute", vd, {ptr}},
    };
  }

  void declareRuntimeFunctions(ModuleOp module) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(&module.getBody()->front());
    Location loc = module.getLoc();
    for (auto &spec : getRuntimeFuncSpecs(builder)) {
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(spec.name)) {
        auto func = LLVM::LLVMFuncOp::create(
            builder, loc, spec.name,
            LLVM::LLVMFunctionType::get(spec.resultType, spec.paramTypes));
        func.setLinkage(LLVM::Linkage::External);
      }
    }
  }

  LogicalResult verifyPrerequisites(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);

    if (module.lookupSymbol<LLVM::LLVMFuncOp>(hipdnn::abi::kInferenceInit) ||
        module.lookupSymbol<LLVM::LLVMFuncOp>(hipdnn::abi::kInferenceCompute) ||
        module.lookupSymbol<LLVM::LLVMFuncOp>(hipdnn::abi::kInferenceCleanup) ||
        module.lookupSymbol<LLVM::LLVMFuncOp>(
            hipdnn::abi::kInferenceGetMetadataJson)) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] Interface functions already exist. "
          << "Pass already ran.\n");
      return failure();
    }

    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      if (module.lookupSymbol<func::FuncOp>("main_graph")) {
        llvm::errs() << "[GenerateInterface] @main_graph is func.func, needs "
                        "llvm.func.\n"
                     << "Run --convert-hip-to-llvm first.\n";
        return failure();
      }
      llvm::errs() << "[GenerateInterface] @main_graph (llvm.func) not found\n";
      return failure();
    }

    // After convert-hip-to-llvm, @main_graph is the (ctx, inputs) -> i32
    // wrapper (no outputs param -- outputs are allocated in-graph). Both params
    // are opaque pointers.
    auto mainType = mainFunc.getFunctionType();
    constexpr unsigned kExpectedNumParams = 2; // (ctx, inputs)
    bool sigOk = mainType.getNumParams() == kExpectedNumParams &&
                 mainType.getReturnType() == i32Type;
    if (sigOk)
      for (unsigned p : llvm::seq<unsigned>(0, kExpectedNumParams))
        sigOk &= mainType.getParamType(p) == ptrType;
    if (!sigOk) {
      llvm::errs() << "[GenerateInterface] @main_graph has wrong signature.\n"
                   << "Expected: (ptr, ptr) -> i32\n";
      return failure();
    }

    for (StringRef attr : {"hipdnn.input_count", "hipdnn.input_shapes",
                           "hipdnn.output_count", "hipdnn.output_shapes"}) {
      if (!module->getAttr(attr)) {
        llvm::errs() << "[GenerateInterface] " << attr
                     << " attribute missing\n";
        return failure();
      }
    }

    return success();
  }

  /// hipdnn_ep_state_init_with_fs: alloc RuntimeState, init
  /// HIP/MIOpen/hipBLASLt, parse metadata blob, read constants via FileSystem
  /// and upload to GPU.
  ///
  /// Generated IR (no pool — from test_basic_interface.mlir):
  ///   llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32
  ///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  ///     %0 = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
  ///     %1 = llvm.mlir.constant(168 : i64) : i64
  ///     %2 = llvm.call @hipdnn_ep_state_init_with_fs(%arg0, %arg1, %0, %1)
  ///              : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  ///     llvm.return %2 : i32
  ///   }
  ///
  /// Generated IR (with pool, 2 buffers at offsets 0/4096, pool 8192):
  ///   llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32
  ///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  ///     %0 = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
  ///     %1 = llvm.mlir.constant(168 : i64) : i64
  ///     %2 = llvm.call @hipdnn_ep_state_init_with_fs(%arg0, %arg1, %0, %1)
  ///              : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  ///     %3 = llvm.mlir.constant(0 : i32) : i32
  ///     %4 = llvm.icmp "ne" %2, %3 : i32
  ///     llvm.cond_br %4, ^bb2, ^bb1
  ///   ^bb1:
  ///     %5 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
  ///     %6 = llvm.mlir.constant(2 : i64) : i64
  ///     %7 = llvm.alloca %6 x i64 : (i64) -> !llvm.ptr
  ///     // ... store offsets [0, 4096] into %7 via GEP+store ...
  ///     %14 = llvm.mlir.constant(8192 : i64) : i64
  ///     %15 = llvm.call @hipdnn_ep_pool_init(%5, %14, %7, %6)
  ///               : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32
  ///     llvm.return %15 : i32
  ///   ^bb2:
  ///     llvm.return %2 : i32
  ///   }
  void generateInferenceInit(ModuleOp module, size_t blobSize) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();

    SmallVector<Type> paramTypes = {ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp = LLVM::LLVMFuncOp::create(
        builder, loc, hipdnn::abi::kInferenceInit, funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);
    Value fsPtr = entryBlock->getArgument(1);

    Value blobPtr = LLVM::AddressOfOp::create(builder, loc, ptrType,
                                              hipdnn::abi::kMetadataBlobGlobal);
    Value blobSizeVal = LLVM::ConstantOp::create(
        builder, loc, i64Type, builder.getI64IntegerAttr((int64_t)blobSize));

    auto initFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_init_with_fs");
    LLVM::CallOp initCall = LLVM::CallOp::create(
        builder, loc, initFunc,
        ValueRange{outStatePtr, fsPtr, blobPtr, blobSizeVal});

    auto poolSizeAttr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size");
    auto bufferOffsetsAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.buffer_offsets");
    auto bufferCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.buffer_count");

    // Buffer offsets and count are optional — if absent, pool is managed at
    // runtime via hip.get_pool / hipdnn_ep_get_pool_base (no static offsets).
    // Only attempt pool_init when all three attributes are present and
    // pool_size > 0.
    bool hasPoolAttrs = poolSizeAttr && bufferOffsetsAttr && bufferCountAttr;
    bool hasPoolInit = hasPoolAttrs && poolSizeAttr.getInt() > 0;

    auto opStateCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.num_op_state_slots");
    bool hasOpState = opStateCountAttr && opStateCountAttr.getInt() > 0;

    // Op-state path: chain state-init -> (optional) pool-init -> op-state-init,
    // returning the first nonzero rc. Kept separate from the non-op-state code
    // below so models without stateful ops emit byte-identical IR. The
    // @hipdnn_ep_op_states_init_fn was emitted by --generate-op-state-init
    // before lowering and is looked up here.
    if (hasOpState) {
      Value zero_i32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                                builder.getI32IntegerAttr(0));
      Value initFailed =
          LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                               initCall.getResult(), zero_i32);
      Block *afterInitBlock = funcOp.addBlock();
      Block *returnInitErrBlock = funcOp.addBlock();
      LLVM::CondBrOp::create(builder, loc, initFailed, returnInitErrBlock,
                             afterInitBlock);

      builder.setInsertionPointToStart(returnInitErrBlock);
      LLVM::ReturnOp::create(builder, loc, initCall.getResult());

      builder.setInsertionPointToStart(afterInitBlock);
      Value statePtr = LLVM::LoadOp::create(builder, loc, ptrType, outStatePtr);

      if (hasPoolInit) {
        size_t poolSize = poolSizeAttr.getInt();
        size_t numBuffers = bufferCountAttr.getInt();
        auto offsetsAttrArray = bufferOffsetsAttr.getValue();
        Value numBuffersVal = LLVM::ConstantOp::create(
            builder, loc, i64Type, builder.getI64IntegerAttr(numBuffers));
        Value offsetsArrayPtr = LLVM::AllocaOp::create(
            builder, loc, ptrType, i64Type, numBuffersVal, 0);
        for (auto i : llvm::seq<size_t>(0, numBuffers)) {
          auto offsetAttr = dyn_cast<IntegerAttr>(offsetsAttrArray[i]);
          Value offset = LLVM::ConstantOp::create(
              builder, loc, i64Type,
              builder.getI64IntegerAttr(offsetAttr.getInt()));
          Value idx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                               builder.getI64IntegerAttr(i));
          Value elemPtr = LLVM::GEPOp::create(builder, loc, ptrType, i64Type,
                                              offsetsArrayPtr, ValueRange{idx});
          LLVM::StoreOp::create(builder, loc, offset, elemPtr);
        }
        auto poolInitFunc =
            module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init");
        Value poolSizeVal = LLVM::ConstantOp::create(
            builder, loc, i64Type, builder.getI64IntegerAttr(poolSize));
        auto poolInitCall = LLVM::CallOp::create(
            builder, loc, poolInitFunc,
            ValueRange{statePtr, poolSizeVal, offsetsArrayPtr, numBuffersVal});
        Value poolFailed =
            LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                                 poolInitCall.getResult(), zero_i32);
        Block *afterPoolBlock = funcOp.addBlock();
        Block *returnPoolErrBlock = funcOp.addBlock();
        LLVM::CondBrOp::create(builder, loc, poolFailed, returnPoolErrBlock,
                               afterPoolBlock);
        builder.setInsertionPointToStart(returnPoolErrBlock);
        LLVM::ReturnOp::create(builder, loc, poolInitCall.getResult());
        builder.setInsertionPointToStart(afterPoolBlock);
      }

      auto opStatesInitFn =
          module.lookupSymbol<LLVM::LLVMFuncOp>(hipdnn::abi::kOpStatesInitFn);
      auto opStatesCall = LLVM::CallOp::create(builder, loc, opStatesInitFn,
                                               ValueRange{statePtr});
      LLVM::ReturnOp::create(builder, loc, opStatesCall.getResult());
      return;
    }

    if (hasPoolAttrs && poolSizeAttr.getInt() > 0) {
      size_t poolSize = poolSizeAttr.getInt();
      size_t numBuffers = bufferCountAttr.getInt();
      auto offsetsAttrArray = bufferOffsetsAttr.getValue();

      Value zero_i32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                                builder.getI32IntegerAttr(0));
      Value initFailed =
          LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                               initCall.getResult(), zero_i32);

      Block *poolInitBlock = funcOp.addBlock();
      Block *returnErrorBlock = funcOp.addBlock();

      LLVM::CondBrOp::create(builder, loc, initFailed, returnErrorBlock,
                             poolInitBlock);

      builder.setInsertionPointToStart(returnErrorBlock);
      LLVM::ReturnOp::create(builder, loc, initCall.getResult());

      builder.setInsertionPointToStart(poolInitBlock);

      Value statePtr = LLVM::LoadOp::create(builder, loc, ptrType, outStatePtr);

      Value numBuffersVal = LLVM::ConstantOp::create(
          builder, loc, i64Type, builder.getI64IntegerAttr(numBuffers));

      Value offsetsArrayPtr = LLVM::AllocaOp::create(builder, loc, ptrType,
                                                     i64Type, numBuffersVal, 0);

      for (auto i : llvm::seq<size_t>(0, numBuffers)) {
        auto offsetAttr = dyn_cast<IntegerAttr>(offsetsAttrArray[i]);
        Value offset = LLVM::ConstantOp::create(
            builder, loc, i64Type,
            builder.getI64IntegerAttr(offsetAttr.getInt()));
        Value idx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                             builder.getI64IntegerAttr(i));
        Value elemPtr = LLVM::GEPOp::create(builder, loc, ptrType, i64Type,
                                            offsetsArrayPtr, ValueRange{idx});
        LLVM::StoreOp::create(builder, loc, offset, elemPtr);
      }

      auto poolInitFunc =
          module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init");
      Value poolSizeVal = LLVM::ConstantOp::create(
          builder, loc, i64Type, builder.getI64IntegerAttr(poolSize));
      auto poolInitCall = LLVM::CallOp::create(
          builder, loc, poolInitFunc,
          ValueRange{statePtr, poolSizeVal, offsetsArrayPtr, numBuffersVal});

      LLVM::ReturnOp::create(builder, loc, poolInitCall.getResult());
    } else {
      LLVM::ReturnOp::create(builder, loc, initCall.getResult());
    }
  }

  /// Generated IR overview (1 input of rank 1; outputs allocated in-graph):
  ///   llvm.func @inference_compute(%state: !llvm.ptr, %ins: !llvm.ptr) -> i32
  ///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  ///     // 1. Alloca input TensorBuffer structs on the stack
  ///     // 2. For each input:  call hipdnn_ep_tensor_prepare_input,
  ///     error-check
  ///     // 3. Build LLVM memref descriptors from TensorBuffer GPU/shape ptrs
  ///     // 4. Call @main_graph(%state, %input_memrefs) -- outputs are
  ///     //    allocated in-graph via hip.alloc_output and written directly
  ///     // 5. Call hipdnn_ep_stream_sync (GPU sync + PERF profiling output)
  ///     // 6. Read device-side error flag
  ///     // 7. Free input TensorBuffers
  ///     // On error: store error code, free inputs, return error
  ///   }
  void generateInferenceCompute(ModuleOp module, ArrayAttr inputShapes) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes{ptrType, ptrType}; // (state, inputs)
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp = LLVM::LLVMFuncOp::create(
        builder, loc, hipdnn::abi::kInferenceCompute, funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);

    // Re-publish THIS state's stream as the thread-local "current stream"
    // before any op runs. The in-tree memrefCopy (MLIR memref.copy lowering,
    // ABI-fixed signature with no `state` arg) issues its D2D copies on that
    // thread-local; it is otherwise only set by the EP's per-Compute
    // begin_compute hook. When several inference sessions (e.g. a dynamic
    // model's shape/prefill/decode specializations) are backed by ONE
    // model.dll, they share this single thread-local but each owns a DISTINCT
    // hipStream. Without this re-publish, a memref.copy in one specialization's
    // main_graph can run on a different specialization's stream than the
    // consuming kernels (which take `state`'s stream) -> unordered
    // producer/consumer -> nondeterministic data race on the copied buffer. It
    // only manifests while several specializations are concurrently live; an
    // isolated single-session run serializes coincidentally and hides it.
    // Binding it here makes every main_graph invocation's copies use its own
    // stream. begin_compute also resets the per-Compute GQA seqlens_k cache,
    // which is correct at compute entry; the EP's own begin_compute call is
    // then a harmless idempotent repeat.
    auto beginComputeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_runtime_begin_compute");
    if (beginComputeFunc)
      LLVM::CallOp::create(builder, loc, beginComputeFunc, ValueRange{state});

    Value inputsSpanPtr = entryBlock->getArgument(1);

    size_t numInputs = inputShapes ? inputShapes.size() : 0;

    Value c0_i32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                            builder.getI32IntegerAttr(0));
    Value c1_i64 = LLVM::ConstantOp::create(builder, loc, i64Type,
                                            builder.getI64IntegerAttr(1));

    auto prepareInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_prepare_input");
    auto freeInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_free_input");

    auto getGpuPtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_gpu_ptr");
    auto getShapePtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_shape_ptr");
    auto resetErrorFlagFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_state_reset_error_flag");
    auto streamSyncFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_stream_sync");
    auto readErrorFlagFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_state_read_and_clear_error_flag");

    Value errorCodePtr =
        LLVM::AllocaOp::create(builder, loc, ptrType, i32Type, c1_i64, 0);

    SmallVector<Value> inputBuffers;

    // sizeof(TensorBuffer) in the runtime (6 fields, 48 bytes on 64-bit).
    // TODO: Replace with sizeof(TensorBuffer) or a runtime query once the
    // runtime is ported into this repo.
    constexpr int64_t kTensorBufferSizeBytes = 48;

    Type i8Type = builder.getI8Type();
    Value tensorBufferSize = LLVM::ConstantOp::create(
        builder, loc, i64Type,
        builder.getI64IntegerAttr(kTensorBufferSizeBytes));

    for (auto i : llvm::seq<size_t>(0, numInputs)) {
      (void)i;
      Value bufferPtr = LLVM::AllocaOp::create(builder, loc, ptrType, i8Type,
                                               tensorBufferSize, 0);
      inputBuffers.push_back(bufferPtr);
    }

    Block *errorCleanupBlock = funcOp.addBlock();

    for (auto i : llvm::seq<size_t>(0, numInputs)) {
      Value bufferPtr = inputBuffers[i];

      Value indexVal = LLVM::ConstantOp::create(builder, loc, i64Type,
                                                builder.getI64IntegerAttr(i));

      Value rankVal = LLVM::ConstantOp::create(
          builder, loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size()));

      emitErrorCheckedCall(
          builder, loc, prepareInputFunc,
          ValueRange{state, inputsSpanPtr, indexVal, rankVal, bufferPtr},
          errorCodePtr, errorCleanupBlock, funcOp);
    }

    // Build memref structs for @main call
    Value numInputsVal = LLVM::ConstantOp::create(
        builder, loc, i64Type, builder.getI64IntegerAttr(numInputs));
    Value inputMemrefArray =
        LLVM::AllocaOp::create(builder, loc, ptrType, ptrType, numInputsVal, 0);

    for (auto i : llvm::seq<size_t>(0, numInputs)) {
      int64_t rank = cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size();

      Value bufferPtr = inputBuffers[i];
      Value gpuPtrRaw = LLVM::CallOp::create(builder, loc, getGpuPtrFunc,
                                             ValueRange{bufferPtr})
                            .getResult();
      Value shapePtr = LLVM::CallOp::create(builder, loc, getShapePtrFunc,
                                            ValueRange{bufferPtr})
                           .getResult();

      Value memref = buildMemrefDescriptor(builder, loc, gpuPtrRaw, shapePtr,
                                           rank, ptrType, i64Type);

      Value memrefPtr = LLVM::AllocaOp::create(builder, loc, ptrType,
                                               memref.getType(), c1_i64, 0);
      LLVM::StoreOp::create(builder, loc, memref, memrefPtr);

      Value indexVal = LLVM::ConstantOp::create(builder, loc, i64Type,
                                                builder.getI64IntegerAttr(i));
      Value arraySlot =
          LLVM::GEPOp::create(builder, loc, ptrType, ptrType, inputMemrefArray,
                              ArrayRef<LLVM::GEPArg>{indexVal});
      LLVM::StoreOp::create(builder, loc, memrefPtr, arraySlot);
    }

    // Reset kernel-side runtime error flag before graph execution.
    emitErrorCheckedCall(builder, loc, resetErrorFlagFunc, ValueRange{state},
                         errorCodePtr, errorCleanupBlock, funcOp);

    // Call @main_graph with arrays of pointers
    Block *mainSuccessBlock;

    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      mainSuccessBlock = funcOp.addBlock();
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] Warning: @main_graph not found\n");
      LLVM::BrOp::create(builder, loc, mainSuccessBlock);
    } else {
      mainSuccessBlock = funcOp.addBlock();
      // @main_graph(state, in_memrefs) -- outputs are allocated in-graph, so
      // there is no outputs arg.
      SmallVector<Value> mainArgs = {state, inputMemrefArray};
      emitErrorCheckedCall(builder, loc, mainFunc, mainArgs, errorCodePtr,
                           errorCleanupBlock, funcOp);
      LLVM::BrOp::create(builder, loc, mainSuccessBlock);
    }

    builder.setInsertionPointToStart(mainSuccessBlock);

    // Synchronize GPU stream after all D2H copies are queued. Also prints
    // PERF phase timing and per-op profile when HIPDNN_EP_PERF is enabled.
    emitErrorCheckedCall(builder, loc, streamSyncFunc, ValueRange{state},
                         errorCodePtr, errorCleanupBlock, funcOp);

    // Read aggregated device-side runtime error flag (no extra hot-path sync in
    // operator wrappers; check occurs at interface boundary).
    emitErrorCheckedCall(builder, loc, readErrorFlagFunc, ValueRange{state},
                         errorCodePtr, errorCleanupBlock, funcOp);

    // Free input tensors
    for (auto i : llvm::seq<size_t>(0, numInputs)) {
      Value bufferPtr = inputBuffers[i];
      LLVM::CallOp::create(builder, loc, freeInputFunc,
                           ValueRange{state, bufferPtr});
    }

    LLVM::ReturnOp::create(builder, loc, c0_i32);

    // Error cleanup block
    builder.setInsertionPointToStart(errorCleanupBlock);

    Value errorCode = LLVM::LoadOp::create(builder, loc, i32Type, errorCodePtr);

    for (auto i : llvm::seq<size_t>(0, inputBuffers.size())) {
      LLVM::CallOp::create(builder, loc, freeInputFunc,
                           ValueRange{state, inputBuffers[i]});
    }

    LLVM::ReturnOp::create(builder, loc, errorCode);
  }

  /// Generated IR example:
  ///   llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32
  ///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  ///     %rc = llvm.call @hipdnn_ep_state_cleanup(%state)
  ///     llvm.return %rc : i32
  ///   }
  void generateInferenceCleanup(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp = LLVM::LLVMFuncOp::create(
        builder, loc, hipdnn::abi::kInferenceCleanup, funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);

    auto runtimeCleanupFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_cleanup");
    auto call = LLVM::CallOp::create(builder, loc, runtimeCleanupFunc,
                                     ValueRange{state});

    LLVM::ReturnOp::create(builder, loc, call.getResult());
  }
};

} // namespace

namespace mlir {
namespace hip {

std::unique_ptr<mlir::Pass> createGenerateInterfacePass(
    const mlir::hip::CompilationOptionsT &compilationOptions) {
  return std::make_unique<GenerateInterfacePass>(compilationOptions);
}

} // namespace hip
} // namespace mlir
