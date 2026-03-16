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
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
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
static std::unique_ptr<mlir::hip::TensorInfoT>
buildTensorInfo(ArrayAttr shapes, DenseI64ArrayAttr elementSizes, size_t i) {
  auto ti = std::make_unique<mlir::hip::TensorInfoT>();
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

  mlir::hip::HipModelMetaInfoT meta;
  meta.version = 1;
  meta.constants_filename = constantsFile;

  if (constantSizesAttr) {
    auto sizes = constantSizesAttr.asArrayRef();
    auto offsets = constantOffsetsAttr ? constantOffsetsAttr.asArrayRef()
                                       : ArrayRef<int64_t>{};
    for (size_t i = 0; i < sizes.size(); ++i) {
      auto ci = std::make_unique<mlir::hip::ConstantInfoT>();
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
  return mlir::hip::toJson<mlir::hip::HipModelMetaInfoT>(
      meta, mlir::hip::k_model_metadata_schema());
}

/// Generate global constant string for metadata JSON
void generateMetadataGlobal(ModuleOp module, const std::string &jsonStr) {
  OpBuilder builder(module.getContext());
  auto i8Type = builder.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, jsonStr.size() + 1);

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

    auto inputCount = module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto outputCount =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");

    const std::string constantsFile =
        !compilationOptions_.constants_file.empty()
            ? compilationOptions_.constants_file
            : "constants.bin";

    std::vector<uint8_t> blob = buildMetadataBlob(module, constantsFile);
    generateMetadataBlobGlobal(module, blob);

    declareMallocFree(module);

    declareRuntimeFunctions(module);

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

    std::string json = buildMetadataJson(module, constantsFile);
    generateMetadataGlobal(module, json);
    generateInferenceGetMetadataJson(module);

    llvm::errs() << "[GenerateInterface] Generated 4 interface functions\n";
  }

private:
  mlir::hip::CompilationOptionsT compilationOptions_;

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

  void declareMallocFree(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);

    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("malloc")) {
      auto mallocFuncType =
          LLVM::LLVMFunctionType::get(ptrType, {builder.getI64Type()});
      auto mallocFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "malloc", mallocFuncType);
      mallocFunc.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("free")) {
      auto freeFuncType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()), {ptrType});
      auto freeFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "free", freeFuncType);
      freeFunc.setLinkage(LLVM::Linkage::External);
    }
  }

  void declareRuntimeFunctions(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    Type voidType = LLVM::LLVMVoidType::get(builder.getContext());

    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

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

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_cleanup")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_cleanup", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

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

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_get_stream")) {
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_get_stream", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, i64Type, ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hipdnn_ep_pool_init",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_get_buffer_from_pool")) {
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_get_buffer_from_pool", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_prepare_input")) {
      Type sizeTType = i64Type;
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, sizeTType, sizeTType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_prepare_input", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_prepare_output")) {
      Type sizeTType = i64Type;
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, sizeTType, sizeTType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_prepare_output", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_finalize_output")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_finalize_output", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_free_input")) {
      auto funcType = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_free_input", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_gpu_ptr")) {
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_gpu_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_host_ptr")) {
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_host_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_shape_ptr")) {
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_shape_ptr", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_rank")) {
      auto funcType = LLVM::LLVMFunctionType::get(i64Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_rank", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_tensor_buffer_get_size_bytes")) {
      auto funcType = LLVM::LLVMFunctionType::get(i64Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_tensor_buffer_get_size_bytes", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hipdnn_ep_state_init_with_fs")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, ptrType, i64Type});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hipdnn_ep_state_init_with_fs", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }
  }

  LogicalResult verifyPrerequisites(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);

    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_init") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_compute") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_cleanup") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_get_metadata_json")) {
      llvm::errs() << "[GenerateInterface] Interface functions already exist. "
                   << "Pass already ran.\n";
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

    auto mainType = mainFunc.getFunctionType();
    if (mainType.getNumParams() != 3 || mainType.getParamType(0) != ptrType ||
        mainType.getParamType(1) != ptrType ||
        mainType.getParamType(2) != ptrType ||
        mainType.getReturnType() != i32Type) {
      llvm::errs() << "[GenerateInterface] @main_graph has wrong signature.\n"
                   << "Expected: (ptr, ptr, ptr) -> i32\n";
      return failure();
    }

    if (!module->getAttr("hipdnn.input_count")) {
      llvm::errs()
          << "[GenerateInterface] hipdnn.input_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.input_shapes")) {
      llvm::errs()
          << "[GenerateInterface] hipdnn.input_shapes attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_count")) {
      llvm::errs()
          << "[GenerateInterface] hipdnn.output_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_shapes")) {
      llvm::errs()
          << "[GenerateInterface] hipdnn.output_shapes attribute missing\n";
      return failure();
    }

    return success();
  }

  void generateInferenceInit(ModuleOp module, size_t blobSize) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();

    SmallVector<Type> paramTypes = {ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_init", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);
    Value fsPtr = entryBlock->getArgument(1);

    Value blobPtr =
        builder.create<LLVM::AddressOfOp>(loc, ptrType, "__metadata_blob");
    Value blobSizeVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr((int64_t)blobSize));

    auto initFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_init_with_fs");
    LLVM::CallOp initCall = builder.create<LLVM::CallOp>(
        loc, initFunc, ValueRange{outStatePtr, fsPtr, blobPtr, blobSizeVal});

    auto poolSizeAttr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size");
    auto bufferOffsetsAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.buffer_offsets");
    auto bufferCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.buffer_count");

    if (!poolSizeAttr || !bufferOffsetsAttr || !bufferCountAttr) {
      llvm::errs()
          << "[GenerateInterface] FATAL: memory pool attributes missing.\n"
          << "  hipdnn.pool_size: " << (poolSizeAttr ? "present" : "MISSING")
          << "\n"
          << "  hipdnn.buffer_offsets: "
          << (bufferOffsetsAttr ? "present" : "MISSING") << "\n"
          << "  hipdnn.buffer_count: "
          << (bufferCountAttr ? "present" : "MISSING") << "\n"
          << "  Ensure MemoryPoolingPass runs before GenerateInterface.\n";
      signalPassFailure();
      return;
    }

    if (poolSizeAttr.getInt() > 0) {
      size_t poolSize = poolSizeAttr.getInt();
      size_t numBuffers = bufferCountAttr.getInt();
      auto offsetsAttrArray = bufferOffsetsAttr.getValue();

      Value zero_i32 = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(0));
      Value initFailed = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ne, initCall.getResult(), zero_i32);

      Block *poolInitBlock = funcOp.addBlock();
      Block *returnErrorBlock = funcOp.addBlock();

      builder.create<LLVM::CondBrOp>(loc, initFailed, returnErrorBlock,
                                     poolInitBlock);

      builder.setInsertionPointToStart(returnErrorBlock);
      builder.create<LLVM::ReturnOp>(loc, initCall.getResult());

      builder.setInsertionPointToStart(poolInitBlock);

      Value statePtr = builder.create<LLVM::LoadOp>(loc, ptrType, outStatePtr);

      Value numBuffersVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(numBuffers));

      Value offsetsArrayPtr = builder.create<LLVM::AllocaOp>(
          loc, ptrType, i64Type, numBuffersVal, 0);

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

      auto poolInitFunc =
          module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_pool_init");
      Value poolSizeVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(poolSize));
      auto poolInitCall = builder.create<LLVM::CallOp>(
          loc, poolInitFunc,
          ValueRange{statePtr, poolSizeVal, offsetsArrayPtr, numBuffersVal});

      builder.create<LLVM::ReturnOp>(loc, poolInitCall.getResult());
    } else {
      builder.create<LLVM::ReturnOp>(loc, initCall.getResult());
    }
  }

  void generateInferenceCompute(ModuleOp module, IntegerAttr inputCount,
                                ArrayAttr inputShapes,
                                DenseI64ArrayAttr inputElementSizes,
                                IntegerAttr outputCount, ArrayAttr outputShapes,
                                DenseI64ArrayAttr outputElementSizes) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_compute", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value inputsSpanPtr = entryBlock->getArgument(1);
    Value outputsSpanPtr = entryBlock->getArgument(2);

    size_t numInputs = inputShapes ? inputShapes.size() : 0;
    size_t numOutputs = outputShapes ? outputShapes.size() : 0;

    Value c0_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0));
    Value c1_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));

    auto prepareInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_prepare_input");
    auto prepareOutputFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_prepare_output");
    auto finalizeOutputFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_finalize_output");
    auto freeInputFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_tensor_free_input");

    auto getGpuPtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_gpu_ptr");
    auto getShapePtrFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_tensor_buffer_get_shape_ptr");

    Value errorCodePtr =
        builder.create<LLVM::AllocaOp>(loc, ptrType, i32Type, c1_i64, 0);

    SmallVector<Value> inputBuffers;
    SmallVector<Value> outputBuffers;

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

    Block *errorCleanupBlock = funcOp.addBlock();

    // Prepare all input tensors
    SmallVector<Value> inputMemrefs;

    for (size_t i = 0; i < numInputs; i++) {
      Value bufferPtr = inputBuffers[i];

      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));

      Value rankVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size()));

      Value retVal =
          builder
              .create<LLVM::CallOp>(loc, prepareInputFunc,
                                    ValueRange{state, inputsSpanPtr, indexVal,
                                               rankVal, bufferPtr})
              .getResult();

      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      Block *continueBlock = funcOp.addBlock();

      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      builder.setInsertionPointToStart(continueBlock);
    }

    // Prepare all output tensors
    SmallVector<Value> outputMemrefs;

    for (size_t i = 0; i < numOutputs; i++) {
      Value bufferPtr = outputBuffers[i];

      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));

      Value rankVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size()));

      Value retVal =
          builder
              .create<LLVM::CallOp>(loc, prepareOutputFunc,
                                    ValueRange{state, outputsSpanPtr, indexVal,
                                               rankVal, bufferPtr})
              .getResult();

      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      Block *continueBlock = funcOp.addBlock();

      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      builder.setInsertionPointToStart(continueBlock);
    }

    // Build memref structs for @main call
    Value numInputsVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(numInputs));
    Value inputMemrefArray =
        builder.create<LLVM::AllocaOp>(loc, ptrType, ptrType, numInputsVal, 0);

    for (size_t i = 0; i < numInputs; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size();
      Type memrefType = getMemRefStructType(builder, rank, 1);

      Value bufferPtr = inputBuffers[i];

      Value gpuPtrRaw =
          builder
              .create<LLVM::CallOp>(loc, getGpuPtrFunc, ValueRange{bufferPtr})
              .getResult();

      Type gpuPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
      Value gpuPtr =
          builder.create<LLVM::AddrSpaceCastOp>(loc, gpuPtrType, gpuPtrRaw);

      Value shapePtr =
          builder
              .create<LLVM::CallOp>(loc, getShapePtrFunc, ValueRange{bufferPtr})
              .getResult();

      Value memref = builder.create<LLVM::UndefOp>(loc, memrefType);

      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{0});

      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{1});

      Value c0_i64 = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(0));
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, c0_i64,
                                                   ArrayRef<int64_t>{2});

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
          builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, inputMemrefArray,
                                      ArrayRef<LLVM::GEPArg>{indexVal});
      builder.create<LLVM::StoreOp>(loc, memrefPtr, arraySlot);

      llvm::errs() << "[GenerateInterface] Built input memref " << i
                   << " using opaque TensorBuffer accessors\n";
    }

    Value numOutputsVal = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(numOutputs));
    Value outputMemrefArray =
        builder.create<LLVM::AllocaOp>(loc, ptrType, ptrType, numOutputsVal, 0);

    for (size_t i = 0; i < numOutputs; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size();
      Type memrefType = getMemRefStructType(builder, rank, 1);

      Value bufferPtr = outputBuffers[i];

      Value gpuPtrRaw =
          builder
              .create<LLVM::CallOp>(loc, getGpuPtrFunc, ValueRange{bufferPtr})
              .getResult();

      Type gpuPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
      Value gpuPtr =
          builder.create<LLVM::AddrSpaceCastOp>(loc, gpuPtrType, gpuPtrRaw);

      Value shapePtr =
          builder
              .create<LLVM::CallOp>(loc, getShapePtrFunc, ValueRange{bufferPtr})
              .getResult();

      Value memref = builder.create<LLVM::UndefOp>(loc, memrefType);
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{0});
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, gpuPtr,
                                                   ArrayRef<int64_t>{1});
      Value c0_i64 = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(0));
      memref = builder.create<LLVM::InsertValueOp>(loc, memref, c0_i64,
                                                   ArrayRef<int64_t>{2});

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

    // Call @main with arrays of pointers
    Block *mainSuccessBlock = funcOp.addBlock();

    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      llvm::errs() << "[GenerateInterface] Warning: @main_graph not found\n";
      builder.create<LLVM::BrOp>(loc, mainSuccessBlock);
    } else {
      Value mainRet =
          builder
              .create<LLVM::CallOp>(
                  loc, mainFunc,
                  ValueRange{state, inputMemrefArray, outputMemrefArray})
              .getResult();

      Value mainFailed = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ne, mainRet, c0_i32);

      Block *storeMainErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, mainFailed, storeMainErrorBlock,
                                     mainSuccessBlock);

      builder.setInsertionPointToStart(storeMainErrorBlock);
      builder.create<LLVM::StoreOp>(loc, mainRet, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);
    }

    // Finalize output tensors (D2H, sync, cleanup)
    builder.setInsertionPointToStart(mainSuccessBlock);

    for (size_t i = 0; i < numOutputs; i++) {
      Value bufferPtr = outputBuffers[i];

      Value retVal = builder
                         .create<LLVM::CallOp>(loc, finalizeOutputFunc,
                                               ValueRange{state, bufferPtr})
                         .getResult();

      Value failed = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                  retVal, c0_i32);

      Block *continueBlock = funcOp.addBlock();

      Block *storeErrorBlock = funcOp.addBlock();
      builder.create<LLVM::CondBrOp>(loc, failed, storeErrorBlock,
                                     continueBlock);

      builder.setInsertionPointToStart(storeErrorBlock);
      builder.create<LLVM::StoreOp>(loc, retVal, errorCodePtr);
      builder.create<LLVM::BrOp>(loc, errorCleanupBlock);

      builder.setInsertionPointToStart(continueBlock);
    }

    // Free input tensors
    for (size_t i = 0; i < numInputs; i++) {
      Value bufferPtr = inputBuffers[i];
      builder.create<LLVM::CallOp>(loc, freeInputFunc,
                                   ValueRange{state, bufferPtr});
    }

    builder.create<LLVM::ReturnOp>(loc, c0_i32);

    // Error cleanup block
    builder.setInsertionPointToStart(errorCleanupBlock);

    Value errorCode = builder.create<LLVM::LoadOp>(loc, i32Type, errorCodePtr);

    for (size_t i = 0; i < inputBuffers.size(); i++) {
      builder.create<LLVM::CallOp>(loc, freeInputFunc,
                                   ValueRange{state, inputBuffers[i]});
    }

    builder.create<LLVM::ReturnOp>(loc, errorCode);
  }

  void generateInferenceCleanup(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_cleanup", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);

    auto runtimeCleanupFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_state_cleanup");
    auto call = builder.create<LLVM::CallOp>(loc, runtimeCleanupFunc,
                                             ValueRange{state});

    builder.create<LLVM::ReturnOp>(loc, call.getResult());
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
