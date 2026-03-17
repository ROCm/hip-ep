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
// - inference_compute: Parse inputs/outputs, call @main_graph
// - inference_cleanup: Free resources
// - inference_get_metadata_json: Return JSON metadata (input/output shapes)
//===----------------------------------------------------------------------===//

#include "compilation_options_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "hip/Compiler/Passes/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/debug_log.h"
#include "hip/flatbuffers_json.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "model_metadata_generated.h"
#include "model_metadata_schema.h"

using namespace mlir;

namespace {

/// Build a TensorInfoT native struct from module tensor attributes at index i.
static std::unique_ptr<::hip::TensorInfoT>
buildTensorInfo(ArrayAttr shapes, DenseI64ArrayAttr elementSizes, size_t i) {
  auto ti = std::make_unique<::hip::TensorInfoT>();
  if (shapes && i < shapes.size()) {
    if (auto shapeAttr = dyn_cast<DenseI64ArrayAttr>(shapes.getValue()[i]))
      ti->shape.assign(shapeAttr.asArrayRef().begin(),
                       shapeAttr.asArrayRef().end());
  }
  ti->element_size =
      (elementSizes && i < elementSizes.size()) ? elementSizes[i] : 4;
  return ti;
}

/// Build a HipModelMetaInfoT native struct from module attributes.
/// Shared by buildMetadataBlob() and buildMetadataJson() so module attributes
/// are read exactly once.
/// Each ConstantInfo carries both size and a running byte offset, enabling
/// future non-sequential or grouped constant layouts.
::hip::HipModelMetaInfoT buildMetadataNative(ModuleOp module,
                                             const std::string &constantsFile) {
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

  ::hip::HipModelMetaInfoT meta;
  meta.version = 1;
  meta.constants_filename = constantsFile;

  if (constantSizesAttr) {
    auto sizes = constantSizesAttr.asArrayRef();
    auto offsets = constantOffsetsAttr ? constantOffsetsAttr.asArrayRef()
                                       : ArrayRef<int64_t>{};
    for (size_t i = 0; i < sizes.size(); ++i) {
      auto ci = std::make_unique<::hip::ConstantInfoT>();
      ci->size = sizes[i];
      ci->offset = (i < offsets.size()) ? offsets[i] : 0;
      meta.constants.push_back(std::move(ci));
    }
  }

  if (inputShapes) {
    for (size_t i = 0; i < inputShapes.size(); i++)
      meta.inputs.push_back(buildTensorInfo(inputShapes, inputElementSizes, i));
  }
  if (outputShapes) {
    for (size_t i = 0; i < outputShapes.size(); i++)
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
  ::hip::HipModelMetaInfoT meta = buildMetadataNative(module, constantsFile);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(::hip::HipModelMetaInfo::Pack(fbb, &meta));
  const uint8_t *buf = fbb.GetBufferPointer();
  return std::vector<uint8_t>(buf, buf + fbb.GetSize());
}

/// Build JSON metadata string from module attributes.
/// Uses the embedded model_metadata schema so the output is always consistent
/// with the binary blob — no manual field mapping needed.
std::string buildMetadataJson(ModuleOp module,
                              const std::string &constantsFile) {
  ::hip::HipModelMetaInfoT meta = buildMetadataNative(module, constantsFile);
  return ::hip::toJson<::hip::HipModelMetaInfoT>(
      meta, ::hip::k_model_metadata_schema());
}

/// Generate global constant string for metadata JSON
void generateMetadataGlobal(ModuleOp module, const std::string &jsonStr) {
  OpBuilder builder(module.getContext());
  auto i8Type = builder.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, jsonStr.size() + 1);

  // Insert at beginning of module
  builder.setInsertionPoint(&module.getBody()->front());
  builder.create<LLVM::GlobalOp>(
      module.getLoc(), arrayType, /*isConstant=*/true, LLVM::Linkage::Internal,
      "__metadata_json", builder.getStringAttr(jsonStr + '\0'));
}

/// Generate global constant for metadata FlatBuffers blob
void generateMetadataBlobGlobal(ModuleOp module,
                                const std::vector<uint8_t> &blob) {
  OpBuilder builder(module.getContext());
  auto i8Type = builder.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, blob.size());

  builder.setInsertionPoint(&module.getBody()->front());
  llvm::StringRef blobRef(reinterpret_cast<const char *>(blob.data()),
                          blob.size());
  builder.create<LLVM::GlobalOp>(module.getLoc(), arrayType,
                                 /*isConstant=*/true, LLVM::Linkage::Internal,
                                 "__metadata_blob",
                                 builder.getStringAttr(blobRef));
}

/// Generate inference_get_metadata_json() function
void generateInferenceGetMetadataJson(ModuleOp module) {
  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();

  builder.setInsertionPointToEnd(module.getBody());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
  auto funcType = LLVM::LLVMFunctionType::get(ptrType, {});

  auto funcOp = builder.create<LLVM::LLVMFuncOp>(
      loc, "inference_get_metadata_json", funcType);
  funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

  Block *entry = funcOp.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  Value addr =
      builder.create<LLVM::AddressOfOp>(loc, ptrType, "__metadata_json");
  builder.create<LLVM::ReturnOp>(loc, addr);
}

class GenerateInterfacePass
    : public PassWrapper<GenerateInterfacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateInterfacePass)

  explicit GenerateInterfacePass(
      const ::hip::compiler::CompilationOptionsT &compilationOptions)
      : compilationOptions_(compilationOptions) {}

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

    // Verify prerequisites
    if (failed(verifyPrerequisites(module))) {
      signalPassFailure();
      return;
    }

    // Read metadata
    auto inputCount = module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto outputCount =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");

    // Resolve constants filename from compilation options (default:
    // constants.bin)
    const std::string constantsFile =
        !compilationOptions_.constants_file.empty()
            ? compilationOptions_.constants_file
            : "constants.bin";

    // Build FlatBuffers blob first — its size is needed inside inference_init
    std::vector<uint8_t> blob = buildMetadataBlob(module, constantsFile);
    generateMetadataBlobGlobal(module, blob);

    // Declare malloc and free at module level (before generating functions)
    declareMallocFree(module);

    // Declare all runtime library functions
    declareRuntimeFunctions(module);

    // Generate interface functions
    generateInferenceInit(module, blob.size());
    auto inputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
    auto outputShapes =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
    auto inputElementSizes =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_element_sizes");
    auto outputElementSizes =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_element_sizes");
    generateInferenceCompute(module, inputCount, inputShapes, inputElementSizes,
                             outputCount, outputShapes, outputElementSizes);
    generateInferenceCleanup(module);

    // Generate JSON metadata interface (for external tools)
    std::string json = buildMetadataJson(module, constantsFile);
    generateMetadataGlobal(module, json);
    generateInferenceGetMetadataJson(module);

    COMPILER_DEBUG_LOG("[GenerateInterface] Generated 4 interface functions\n");
  }

private:
  const ::hip::compiler::CompilationOptionsT &compilationOptions_;

  /// Returns LLVM struct type for memref: (ptr, ptr, i64, array<rank x i64>,
  /// array<rank x i64>)
  Type getMemRefStructType(OpBuilder &builder, int64_t rank,
                           unsigned addrSpace) {
    MLIRContext *ctx = builder.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, addrSpace);
    Type i64Type = builder.getI64Type();
    Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Type strideArrayType = LLVM::LLVMArrayType::get(i64Type, rank);

    return LLVM::LLVMStructType::getLiteral(
        ctx, {ptrType, ptrType, i64Type, sizeArrayType, strideArrayType});
  }

  /// TensorBuffer is now fully opaque - no struct type needed!
  /// Generated code accesses fields via runtime accessor functions:
  ///   - hipdnn_ep_tensor_buffer_get_gpu_ptr()
  ///   - hipdnn_ep_tensor_buffer_get_shape_ptr()
  /// This eliminates hardcoded field indices and struct layout coupling.

  /// Declare malloc and free functions at module level
  void declareMallocFree(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);

    // Find insertion point at start of module body (before first operation)
    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

    // Declare malloc if not already present
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("malloc")) {
      auto mallocFuncType =
          LLVM::LLVMFunctionType::get(ptrType, {builder.getI64Type()});
      auto mallocFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "malloc", mallocFuncType);
      mallocFunc.setLinkage(LLVM::Linkage::External);
    }

    // Declare free if not already present
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("free")) {
      auto freeFuncType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()), {ptrType});
      auto freeFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "free", freeFuncType);
      freeFunc.setLinkage(LLVM::Linkage::External);
    }
  }

  /// Declare runtime library functions for GPU operations
  void declareRuntimeFunctions(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    Type voidType = LLVM::LLVMVoidType::get(builder.getContext());

    // Find insertion point
    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

    // Declare HIP functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamSynchronize")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamSynchronize",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare MIOpen functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenSetStream")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenSetStream", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare hipBLASLt functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipblasLtCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipblasLtCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipblasLtDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipblasLtDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare high-level runtime state management functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_cleanup")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_cleanup", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare runtime wrapper functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("wrap_hipMalloc")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, i64Type});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "wrap_hipMalloc", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("wrap_hipFree")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "wrap_hipFree", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("wrap_hipMemcpyH2D")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, i64Type, ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "wrap_hipMemcpyH2D", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("wrap_hipMemcpyD2H")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, i64Type, ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "wrap_hipMemcpyD2H", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("wrap_hipStreamSynchronize")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "wrap_hipStreamSynchronize", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare hipdnn_ep_state_get_stream accessor
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_get_stream")) {
      // void* hipdnn_ep_state_get_stream(RuntimeState* state)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_get_stream", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare memory pooling functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init")) {
      // int hipdnn_ep_pool_init(RuntimeState* state, size_t pool_size,
      //                         const size_t* buffer_offsets, size_t
      //                         num_buffers)
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, i64Type, ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hipdnn_ep_pool_init",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_get_buffer_from_pool")) {
      // void* hipdnn_ep_get_buffer_from_pool(RuntimeState* state, size_t index)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_get_buffer_from_pool", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare tensor preparation helpers
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_prepare_input")) {
      // int hipdnn_ep_tensor_prepare_input(RuntimeState* state, span_t* inputs,
      //     size_t index, size_t expected_rank, TensorBuffer* out_buffer)
      Type sizeTType = i64Type;
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, sizeTType, sizeTType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_prepare_input", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_prepare_output")) {
      // int hipdnn_ep_tensor_prepare_output(RuntimeState* state, span_t*
      //     outputs, size_t index, size_t expected_rank,
      //     TensorBuffer* out_buffer)
      Type sizeTType = i64Type;
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, sizeTType, sizeTType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_prepare_output", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_finalize_output")) {
      // int hipdnn_ep_tensor_finalize_output(RuntimeState* state, TensorBuffer*
      // buffer)
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_finalize_output", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_free_input")) {
      // void hipdnn_ep_tensor_free_input(RuntimeState* state, TensorBuffer*
      // buffer)
      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_free_input", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare TensorBuffer field accessor functions (opaque pattern)
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_gpu_ptr")) {
      // void* hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer* buffer)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_gpu_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_host_ptr")) {
      // void* hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer* buffer)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_host_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_shape_ptr")) {
      // int64_t* hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer* buffer)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_shape_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_rank")) {
      // size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer* buffer)
      auto funcType = LLVM::LLVMFunctionType::get(i64Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_rank", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_size_bytes")) {
      // size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer* buffer)
      auto funcType = LLVM::LLVMFunctionType::get(i64Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_size_bytes", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare external-constants runtime initializer
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_state_init_with_fs")) {
      // int hipdnn_ep_state_init_with_fs(RuntimeState** out_state,
      //                                  void* fs,
      //                                  const void* metadata_blob,
      //                                  size_t blob_size)
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_init_with_fs", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }
  }

  /// Verify that module has all required prerequisites
  LogicalResult verifyPrerequisites(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);

    // 0. Check pass hasn't run before (idempotency check)
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_init") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_compute") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_cleanup") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_get_metadata_json")) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] Interface functions already exist. "
          << "Pass already ran.\n");
      return failure();
    }

    // 1. Check @main_graph exists as llvm.func with correct signature:
    // (ptr,ptr,ptr)->i32
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      // Give helpful error if it's func.func
      if (module.lookupSymbol<func::FuncOp>("main_graph")) {
        COMPILER_DEBUG_LOG(
            "[GenerateInterface] @main_graph is func.func, needs "
            "llvm.func.\n"
            << "Run --convert-hip-to-llvm first.\n");
        return failure();
      }
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] @main_graph (llvm.func) not found\n");
      return failure();
    }

    // Verify @main_graph signature
    auto mainType = mainFunc.getFunctionType();
    if (mainType.getNumParams() != 3 || mainType.getParamType(0) != ptrType ||
        mainType.getParamType(1) != ptrType ||
        mainType.getParamType(2) != ptrType ||
        mainType.getReturnType() != i32Type) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] @main_graph has wrong signature.\n"
          << "Expected: (ptr, ptr, ptr) -> i32\n");
      return failure();
    }

    // 2. Check all 4 metadata attributes exist
    if (!module->getAttr("hipdnn.input_count")) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] hipdnn.input_count attribute missing\n");
      return failure();
    }
    if (!module->getAttr("hipdnn.input_shapes")) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] hipdnn.input_shapes attribute missing\n");
      return failure();
    }
    if (!module->getAttr("hipdnn.output_count")) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] hipdnn.output_count attribute missing\n");
      return failure();
    }
    if (!module->getAttr("hipdnn.output_shapes")) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] hipdnn.output_shapes attribute missing\n");
      return failure();
    }

    return success();
  }

  /// Generate inference_init function.
  ///
  /// Signature: int inference_init(void** out_state, void* fs)
  /// Passes the address of __metadata_blob global and its compile-time size
  /// to hipdnn_ep_state_init_with_fs(out_state, fs, blob_ptr, blob_size).
  /// The runtime reads constants_filename and constant_sizes from the blob.
  void generateInferenceInit(ModuleOp module, size_t blobSize) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    // Set insertion point at end of module
    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();

    // inference_init always takes void* fs (external constants mode)
    SmallVector<Type> paramTypes = {ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_init", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);
    Value fsPtr = entryBlock->getArgument(1);

    // Pass __metadata_blob address and compile-time size to runtime init.
    // Runtime reads constants_filename and constant_sizes from the blob.
    Value blobPtr =
        builder.create<LLVM::AddressOfOp>(loc, ptrType, "__metadata_blob");
    Value blobSizeVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr((int64_t)blobSize));

    auto initFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_init_with_fs");
    LLVM::CallOp initCall = builder.create<LLVM::CallOp>(
        loc, initFunc, ValueRange{outStatePtr, fsPtr, blobPtr, blobSizeVal});

    // MemoryPoolingPass must always emit all three pool attributes (even when
    // pool_size is 0).  Missing attributes means the pass was skipped or buggy.
    auto poolSizeAttr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size");
    auto bufferOffsetsAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.buffer_offsets");
    auto bufferCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.buffer_count");

    if (!poolSizeAttr || !bufferOffsetsAttr || !bufferCountAttr) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] FATAL: memory pool attributes missing.\n"
          << "  hipdnn.pool_size: " << (poolSizeAttr ? "present" : "MISSING")
          << "\n"
          << "  hipdnn.buffer_offsets: "
          << (bufferOffsetsAttr ? "present" : "MISSING") << "\n"
          << "  hipdnn.buffer_count: "
          << (bufferCountAttr ? "present" : "MISSING") << "\n"
          << "  Ensure MemoryPoolingPass runs before GenerateInterface.\n");
      signalPassFailure();
      return;
    }

    if (poolSizeAttr.getInt() > 0) {
      // Initialize memory pool
      size_t poolSize = poolSizeAttr.getInt();
      size_t numBuffers = bufferCountAttr.getInt();
      auto offsetsAttrArray = bufferOffsetsAttr.getValue();

      // Check if init failed
      Value zero_i32 = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(0));
      Value initFailed = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ne, initCall.getResult(), zero_i32);

      // Create blocks for conditional pool init
      Block *poolInitBlock = funcOp.addBlock();
      Block *returnErrorBlock = funcOp.addBlock();

      // If state init failed, return error immediately
      builder.create<LLVM::CondBrOp>(loc, initFailed, returnErrorBlock,
                                     poolInitBlock);

      // Error return block
      builder.setInsertionPointToStart(returnErrorBlock);
      builder.create<LLVM::ReturnOp>(loc, initCall.getResult());

      // Pool initialization block
      builder.setInsertionPointToStart(poolInitBlock);

      // Load state pointer from outStatePtr
      Value statePtr = builder.create<LLVM::LoadOp>(loc, ptrType, outStatePtr);

      // Create constant array of buffer offsets
      Value numBuffersVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(numBuffers));

      // Allocate array for offsets on stack
      Value offsetsArrayPtr = builder.create<LLVM::AllocaOp>(
          loc, ptrType, i64Type, numBuffersVal, 0);

      // Fill the offsets array
      for (size_t i = 0; i < numBuffers; i++) {
        auto offsetAttr = dyn_cast<IntegerAttr>(offsetsAttrArray[i]);
        Value offset = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(offsetAttr.getInt()));
        Value idx = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(i));
        Value elemPtr = builder.create<LLVM::GEPOp>(
            loc, ptrType, i64Type, offsetsArrayPtr, ValueRange{idx});
        builder.create<LLVM::StoreOp>(loc, offset, elemPtr);
      }

      // Call hipdnn_ep_pool_init(state, pool_size, offsets, num_buffers)
      auto poolInitFunc =
          module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init");
      Value poolSizeVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(poolSize));
      auto poolInitCall = builder.create<LLVM::CallOp>(
          loc, poolInitFunc,
          ValueRange{statePtr, poolSizeVal, offsetsArrayPtr, numBuffersVal});

      // Return pool init result
      builder.create<LLVM::ReturnOp>(loc, poolInitCall.getResult());
    } else {
      // No pooling needed — return init result directly
      builder.create<LLVM::ReturnOp>(loc, initCall.getResult());
    }
  }

  /// Generate inference_compute function using tensor preparation helpers
  /// Signature: int inference_compute(void* state, span_t* inputs, span_t*
  /// outputs);
  ///
  /// Refactored to use runtime helpers for:
  /// - Tensor parsing/validation (hipdnn_ep_tensor_prepare_input/output)
  /// - GPU allocation and H2D/D2H transfers
  /// - Error handling and cleanup
  ///
  /// Generated code responsibilities:
  /// - Call helpers to prepare TensorBuffers
  /// - Build rank-specific memref structs (MLIR type system requirement)
  /// - Call @main with memref arguments
  /// - Call finalize helpers for D2H and cleanup
  ///
  /// Supports multiple input/output tensors with different ranks via
  /// array-of-pointers pattern.
  void generateInferenceCompute(ModuleOp module, IntegerAttr inputCount,
                                ArrayAttr inputShapes,
                                DenseI64ArrayAttr inputElementSizes,
                                IntegerAttr outputCount, ArrayAttr outputShapes,
                                DenseI64ArrayAttr outputElementSizes) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr, ptr, ptr) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_compute", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value inputsSpanPtr = entryBlock->getArgument(1);
    Value outputsSpanPtr = entryBlock->getArgument(2);

    // Get metadata — derive ranks from shape array sizes
    size_t numInputs = inputShapes ? inputShapes.size() : 0;
    size_t numOutputs = outputShapes ? outputShapes.size() : 0;

    // Constants
    Value c0_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0));
    Value c1_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));

    // Get helper function references (TensorBuffer is now opaque - no struct
    // type needed)
    auto prepareInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_prepare_input");
    auto prepareOutputFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_prepare_output");
    auto finalizeOutputFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_finalize_output");
    auto freeInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_free_input");

    // Get TensorBuffer accessor functions
    auto getGpuPtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_gpu_ptr");
    auto getShapePtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_shape_ptr");

    // Allocate stack variable for error code (used in error paths)
    Value errorCodePtr =
        builder.create<LLVM::AllocaOp>(loc, ptrType, i32Type, c1_i64, 0);

    // Allocate all TensorBuffer structs upfront in entry block so they dominate
    // all uses (TensorBuffer is opaque - size determined by runtime)
    SmallVector<Value> inputBuffers;
    SmallVector<Value> outputBuffers;

    // TensorBuffer is opaque, but we need to allocate storage for it.
    // Use a conservative size (48 bytes based on known layout, but this could
    // be obtained from a runtime constant in the future)
    Type i8Type = builder.getI8Type();
    Value tensorBufferSize = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(48));

    for (size_t i = 0; i < numInputs; i++) {
      Value bufferPtr = builder.create<LLVM::AllocaOp>(loc, ptrType, i8Type,
                                                       tensorBufferSize, 0);
      inputBuffers.push_back(bufferPtr);
    }
    for (size_t i = 0; i < numOutputs; i++) {
      Value bufferPtr = builder.create<LLVM::AllocaOp>(loc, ptrType, i8Type,
                                                       tensorBufferSize, 0);
      outputBuffers.push_back(bufferPtr);
    }

    // Create error cleanup block
    Block *errorCleanupBlock = funcOp.addBlock();

    // ========================================================================
    // Prepare all input tensors
    // ========================================================================
    SmallVector<Value> inputMemrefs;

    for (size_t i = 0; i < numInputs; i++) {
      // Get pre-allocated buffer
      Value bufferPtr = inputBuffers[i];

      // Create index constant
      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));

      // Create expected rank constant
      Value rankVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size()));

      // Call hipdnn_ep_tensor_prepare_input(state, inputs, index, rank, buffer)
      // element_size is read from tensor_t.element_size by the runtime
      Value retVal =
          builder
              .create<LLVM::CallOp>(loc, prepareInputFunc,
                                    ValueRange{state, inputsSpanPtr, indexVal,
                                               rankVal, bufferPtr})
              .getResult();

      // Check for error (non-zero return)
      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      // Create blocks for error path
      Block *continueBlock = funcOp.addBlock();

      // If prepare failed, store error code and jump to cleanup
      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      // Store error code before jumping to cleanup
      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      // Continue with success path
      builder.setInsertionPointToStart(continueBlock);
    }

    // ========================================================================
    // Prepare all output tensors
    // ========================================================================
    SmallVector<Value> outputMemrefs;

    for (size_t i = 0; i < numOutputs; i++) {
      // Get pre-allocated buffer
      Value bufferPtr = outputBuffers[i];

      // Create index constant
      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));

      // Create expected rank constant
      Value rankVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size()));

      // Call hipdnn_ep_tensor_prepare_output(state, outputs, index, rank,
      //                                      buffer)
      // element_size is read from tensor_t.element_size by the runtime
      Value retVal =
          builder
              .create<LLVM::CallOp>(loc, prepareOutputFunc,
                                    ValueRange{state, outputsSpanPtr, indexVal,
                                               rankVal, bufferPtr})
              .getResult();

      // Check for error
      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      // Create blocks for error path
      Block *continueBlock = funcOp.addBlock();

      // If prepare/finalize failed, store error code and jump to cleanup
      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      builder.setInsertionPointToStart(continueBlock);
    }

    // ========================================================================
    // Build memref structs for @main call
    // ========================================================================
    // Since different tensors may have different ranks, we can't use a
    // homogeneous array. Use array of pointers instead.

    // Allocate array of pointers for input memrefs
    Value numInputsVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(numInputs));
    Value inputMemrefArray =
        builder.create<LLVM::AllocaOp>(loc, ptrType, ptrType, numInputsVal, 0);

    for (size_t i = 0; i < numInputs; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size();
      Type memrefType =
          getMemRefStructType(builder, rank, 1); // GPU address space = 1

      // Load TensorBuffer fields via accessor functions (opaque pattern)
      Value bufferPtr = inputBuffers[i];

      // Call hipdnn_ep_tensor_buffer_get_gpu_ptr(buffer) to get GPU pointer
      Value gpuPtrRaw =
          builder
              .create<LLVM::CallOp>(loc, getGpuPtrFunc, ValueRange{bufferPtr})
              .getResult();

      // Cast to GPU address space (address space 1)
      Type gpuPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
      Value gpuPtr =
          builder.create<LLVM::AddrSpaceCastOp>(loc, gpuPtrType, gpuPtrRaw);

      // Call hipdnn_ep_tensor_buffer_get_shape_ptr(buffer) to get shape array
      Value shapePtr =
          builder
              .create<LLVM::CallOp>(loc, getShapePtrFunc, ValueRange{bufferPtr})
              .getResult();

      // Build memref struct using the GPU pointer we extracted
      Value memref = builder.create<LLVM::UndefOp>(loc, memrefType);

      // Set allocated pointer (field 0)
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{0});

      // Set aligned pointer (field 1)
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{1});

      // Set offset (field 2) - always 0
      Value c0_i64 = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(0));
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, c0_i64,
                                                   ArrayRef<int64_t>{2});

      // Build sizes array (field 3)
      Value sizesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, rank));
      for (int64_t dim = 0; dim < rank; dim++) {
        Value dimIndexVal = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(dim));
        Value dimPtr =
            builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, shapePtr,
                                        ArrayRef<LLVM::GEPArg>{dimIndexVal});
        Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
        sizesArray = builder.create<LLVM::InsertValueOp>(
            loc, sizesArray, dimValue, ArrayRef<int64_t>{dim});
      }
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, sizesArray,
                                                   ArrayRef<int64_t>{3});

      // Build strides array (field 4) - row-major
      Value stridesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, rank));
      Value strideAccum = c1_i64;
      for (int64_t dim = rank - 1; dim >= 0; dim--) {
        stridesArray = builder.create<LLVM::InsertValueOp>(
            loc, stridesArray, strideAccum, ArrayRef<int64_t>{dim});
        if (dim > 0) {
          Value dimIndexVal = builder.create<LLVM::ConstantOp>(
              loc, i64Type, builder.getI64IntegerAttr(dim));
          Value dimPtr =
              builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, shapePtr,
                                          ArrayRef<LLVM::GEPArg>{dimIndexVal});
          Value dimSize = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
          strideAccum = builder.create<LLVM::MulOp>(loc, strideAccum, dimSize);
        }
      }
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, stridesArray,
                                                   ArrayRef<int64_t>{4});

      // Allocate space for this memref on stack
      Value memrefPtr =
          builder.create<LLVM::AllocaOp>(loc, ptrType, memrefType, c1_i64, 0);
      builder.create<LLVM::StoreOp>(loc, memref, memrefPtr);

      // Store pointer in array
      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));
      Value arraySlot =
          builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, inputMemrefArray,
                                      ArrayRef<LLVM::GEPArg>{indexVal});
      builder.create<LLVM::StoreOp>(loc, memrefPtr, arraySlot);

      COMPILER_DEBUG_LOG("[GenerateInterface] Built input memref "
                         << i << " using opaque TensorBuffer accessors\n");
    }

    // Build output memref array similarly
    Value numOutputsVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(numOutputs));
    Value outputMemrefArray =
        builder.create<LLVM::AllocaOp>(loc, ptrType, ptrType, numOutputsVal, 0);

    for (size_t i = 0; i < numOutputs; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size();
      Type memrefType = getMemRefStructType(builder, rank, 1);

      Value bufferPtr = outputBuffers[i];

      // Call accessor functions to get fields (opaque pattern)
      Value gpuPtrRaw =
          builder
              .create<LLVM::CallOp>(loc, getGpuPtrFunc, ValueRange{bufferPtr})
              .getResult();

      // Cast to GPU address space (address space 1)
      Type gpuPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
      Value gpuPtr =
          builder.create<LLVM::AddrSpaceCastOp>(loc, gpuPtrType, gpuPtrRaw);

      // Get shape pointer via accessor
      Value shapePtr =
          builder
              .create<LLVM::CallOp>(loc, getShapePtrFunc, ValueRange{bufferPtr})
              .getResult();

      // Build memref struct (same as input logic)
      Value memref = builder.create<LLVM::UndefOp>(loc, memrefType);
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{0});
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{1});
      Value c0_i64 = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(0));
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, c0_i64,
                                                   ArrayRef<int64_t>{2});

      // Sizes
      Value sizesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, rank));
      for (int64_t dim = 0; dim < rank; dim++) {
        Value dimIndexVal = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(dim));
        Value dimPtr =
            builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, shapePtr,
                                        ArrayRef<LLVM::GEPArg>{dimIndexVal});
        Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
        sizesArray = builder.create<LLVM::InsertValueOp>(
            loc, sizesArray, dimValue, ArrayRef<int64_t>{dim});
      }
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, sizesArray,
                                                   ArrayRef<int64_t>{3});

      // Strides
      Value stridesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, rank));
      Value strideAccum = c1_i64;
      for (int64_t dim = rank - 1; dim >= 0; dim--) {
        stridesArray = builder.create<LLVM::InsertValueOp>(
            loc, stridesArray, strideAccum, ArrayRef<int64_t>{dim});
        if (dim > 0) {
          Value dimIndexVal = builder.create<LLVM::ConstantOp>(
              loc, i64Type, builder.getI64IntegerAttr(dim));
          Value dimPtr =
              builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, shapePtr,
                                          ArrayRef<LLVM::GEPArg>{dimIndexVal});
          Value dimSize = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
          strideAccum = builder.create<LLVM::MulOp>(loc, strideAccum, dimSize);
        }
      }
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, stridesArray,
                                                   ArrayRef<int64_t>{4});

      Value memrefPtr =
          builder.create<LLVM::AllocaOp>(loc, ptrType, memrefType, c1_i64, 0);
      builder.create<LLVM::StoreOp>(loc, memref, memrefPtr);

      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));
      Value arraySlot =
          builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, outputMemrefArray,
                                      ArrayRef<LLVM::GEPArg>{indexVal});
      builder.create<LLVM::StoreOp>(loc, memrefPtr, arraySlot);
    }

    // ========================================================================
    // Call @main with arrays of pointers
    // ========================================================================
    Block *mainSuccessBlock = funcOp.addBlock();

    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      COMPILER_DEBUG_LOG(
          "[GenerateInterface] Warning: @main_graph not found\n");
      builder.create<LLVM::BrOp>(loc, mainSuccessBlock);
    } else {
      Value mainRet =
          builder
              .create<LLVM::CallOp>(
                  loc, mainFunc,
                  ValueRange{state, inputMemrefArray, outputMemrefArray})
              .getResult();

      // Check for error
      Value mainFailed = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ne, mainRet, c0_i32);

      Block *storeMainErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, mainFailed, storeMainErrorBlock,
                                     mainSuccessBlock);

      builder.setInsertionPointToStart(storeMainErrorBlock);
      builder.create<LLVM::StoreOp>(loc, mainRet, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);
    }

    // ========================================================================
    // Finalize output tensors (D2H, sync, cleanup)
    // ========================================================================
    builder.setInsertionPointToStart(mainSuccessBlock);

    for (size_t i = 0; i < numOutputs; i++) {
      Value bufferPtr = outputBuffers[i];

      // Call hipdnn_ep_tensor_finalize_output(state, buffer)
      Value retVal = builder
                         .create<LLVM::CallOp>(loc, finalizeOutputFunc,
                                               ValueRange{state, bufferPtr})
                         .getResult();

      // Check for error
      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      Block *continueBlock = funcOp.addBlock();

      // If prepare/finalize failed, store error code and jump to cleanup
      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      builder.setInsertionPointToStart(continueBlock);
    }

    // ========================================================================
    // Free input tensors
    // ========================================================================
    for (size_t i = 0; i < numInputs; i++) {
      Value bufferPtr = inputBuffers[i];
      builder.create<LLVM::CallOp>(loc, freeInputFunc,
                                   ValueRange{state, bufferPtr});
    }

    // Success - return 0
    builder.create<LLVM::ReturnOp>(loc, c0_i32);

    // ========================================================================
    // Error cleanup block
    // ========================================================================
    builder.setInsertionPointToStart(errorCleanupBlock);

    // Load error code from stack variable
    Value errorCode = builder.create<LLVM::LoadOp>(loc, i32Type, errorCodePtr);

    // Best-effort cleanup: free all prepared tensors
    for (size_t i = 0; i < inputBuffers.size(); i++) {
      builder.create<LLVM::CallOp>(loc, freeInputFunc,
                                   ValueRange{state, inputBuffers[i]});
    }

    // NOTE: Do NOT call finalize_output here to avoid double-finalize bug.
    // If we reached finalization, it was already attempted there.
    // If we failed before finalization, outputs aren't ready to finalize.
    // finalize_output should only be called once per output in the success
    // path.

    // Return the error code
    builder.create<LLVM::ReturnOp>(loc, errorCode);
  }

  /// Generate inference_cleanup function with resource cleanup
  /// Signature: int inference_cleanup(void* state);
  /// Destroys GPU resources in reverse order of creation (LIFO)
  /// Uses best-effort cleanup: continues even if some operations fail
  /// Generate inference_cleanup function - simplified to call
  /// hipdnn_ep_state_cleanup() Signature: int inference_cleanup(void* state);
  ///
  /// This function is now a simple wrapper that delegates to
  /// hipdnn_ep_state_cleanup() in the runtime library. All the cleanup logic
  /// (synchronization, handle destruction, LIFO order) is in C++ code instead
  /// of LLVM IR generation.
  void generateInferenceCleanup(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_cleanup", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);

    // Call hipdnn_ep_state_cleanup(state)
    auto runtimeCleanupFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_cleanup");
    auto call = builder.create<LLVM::CallOp>(loc, runtimeCleanupFunc,
                                             ValueRange{state});

    // Return the result from hipdnn_ep_state_cleanup (always 0)
    builder.create<LLVM::ReturnOp>(loc, call.getResult());
  }
};

} // namespace

namespace hip::compiler {
namespace compiler {

std::unique_ptr<mlir::Pass> createGenerateInterfacePass(
    const ::hip::compiler::CompilationOptionsT &compilationOptions) {
  return std::make_unique<GenerateInterfacePass>(compilationOptions);
}

} // namespace compiler
} // namespace hip::compiler
