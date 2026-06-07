/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Generate Interface Pass - Create C-compatible interface functions
//===----------------------------------------------------------------------===//
// This pass generates the C-ABI compatible functions that wrap the internal
// @main_graph function:
// - inference_init: Allocate context, create handles, upload constants
// - inference_compute: Parse inputs/outputs, call @main_graph
// - inference_cleanup: Free resources
// - inference_get_metadata_json: Return JSON metadata (input/output shapes)
// - inference_infer_shapes: Data-independent output-shape function (emitted
//     only when @infer_shapes is present, i.e. BuildShapeFunctionPass ran).
//===----------------------------------------------------------------------===//

#include "compilation_options_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
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

  // Streaming-mode source descriptors. Present when OnnxToHip's finalize
  // emitted per-constant source info (splat / file-ref / sidecar). Absent in
  // full-sidecar mode (EPContext export), in which case every ConstantInfo
  // keeps source = NONE and runtime falls back to the bulk constants_filename
  // read. In hybrid mode (skipDataWrite=true with mem-addr entries) some
  // constants carry SidecarSource pointing at a *partial* sidecar that holds
  // only mem-addr bytes.
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
  auto sidecarOffsetsAttr = module->getAttrOfType<DenseI64ArrayAttr>(
      "hipdnn.constant_sidecar_offsets");

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
    auto sidecarOffsets = sidecarOffsetsAttr ? sidecarOffsetsAttr.asArrayRef()
                                             : ArrayRef<int64_t>{};
    for (auto i : llvm::seq<size_t>(0, sizes.size())) {
      auto ci = std::make_unique<mlir::hip::ConstantInfoT>();
      ci->size = sizes[i];
      ci->offset = (i < offsets.size()) ? offsets[i] : 0;
      int32_t kind = (i < kinds.size()) ? kinds[i] : 0;
      if (kind == 1) {
        // Splat: extract the left-packed elem bytes out of the i64 carrier.
        auto splat = std::make_unique<mlir::hip::SplatSourceT>();
        int64_t elemSize = (i < splatElemSizes.size()) ? splatElemSizes[i] : 0;
        size_t n = static_cast<size_t>(std::min<int64_t>(elemSize, 8));
        const auto *base = reinterpret_cast<const uint8_t *>(&splatValues[i]);
        splat->elem_bytes.assign(base, base + n);
        ci->source.Set(std::move(*splat));
      } else if (kind == 2) {
        auto fref = std::make_unique<mlir::hip::FileRefSourceT>();
        if (filePathsAttr && i < filePathsAttr.size()) {
          if (auto s = dyn_cast<StringAttr>(filePathsAttr.getValue()[i]))
            fref->path = s.getValue().str();
        }
        fref->file_offset = (i < fileOffsets.size()) ? fileOffsets[i] : 0;
        ci->source.Set(std::move(*fref));
      } else if (kind == 3) {
        // Sidecar: mem-addr entry packed into
        // HipModelMetaInfo.constants_filename at sidecar_offset by the
        // OnnxToHip hybrid finalize. Runtime reads size bytes from that offset
        // through the EP FileSystem.
        auto side = std::make_unique<mlir::hip::SidecarSourceT>();
        side->sidecar_offset =
            (i < sidecarOffsets.size()) ? sidecarOffsets[i] : 0;
        ci->source.Set(std::move(*side));
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
                         "__metadata_json",
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
                         "__metadata_blob", builder.getStringAttr(blobRef));
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
      builder, loc, "inference_get_metadata_json", funcType);
  funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

  Block *entry = funcOp.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  Value addr =
      LLVM::AddressOfOp::create(builder, loc, ptrType, "__metadata_json");
  LLVM::ReturnOp::create(builder, loc, addr);
}

/// Generate inference_infer_shapes() — the data-independent output-shape
/// function the EP calls before allocating output buffers, for output dims
/// that are non-identity functions of input dims (vision patch mergers,
/// flattens) which `DimSource` cannot express.
///
/// No-op (skipped) when `@infer_shapes` is absent — e.g. hip-to-llvm-only
/// pipelines, or models whose output is not a ranked tensor so
/// BuildShapeFunctionPass bailed. Old model.dlls predating this export keep
/// working on the EP's DimSource path.
///
/// ABI (raw symbol; matches the EP-side caller):
///   int inference_infer_shapes(
///       const int64_t* const* input_shapes,  const int64_t* input_ranks,
///       int64_t input_count,
///       const void* const*    input_data,
///       int64_t* const*       output_shapes, const int64_t* output_ranks,
///       int64_t output_count);
///
/// Body: load each input dim (flattened over inputs in (tensor, dim)
/// row-major order) into the leading args of the lowered `@infer_shapes`;
/// then, for closed-form data-dependent shape fns (e.g. ONNX Range), load each
/// recorded scalar VALUE out of `input_data` (per the `hipdnn.shape_fn_data_
/// args` module attribute) into the trailing data args; call it; scatter the
/// i64 results (same flattening over outputs) into the caller-allocated
/// `output_shapes` rows. `input_ranks` / `*_count` / `output_ranks` are
/// accepted for ABI stability but unused: the per-tensor ranks are
/// compile-time constants from the module's shape metadata. `input_data` is
/// only dereferenced for the inputs the shape program actually reads (the
/// data-args descriptors); the EP passes a pointer per input regardless. The
/// flattening MUST match BuildShapeFunctionPass: @infer_shapes args = one i64
/// per (input tensor, dim) over `hipdnn.input_shapes`, then one per data arg
/// over `hipdnn.shape_fn_data_args`; results = one i64 per (output tensor,
/// dim) over `hipdnn.output_shapes`.
///
/// NOTE: adding `input_data` is an ABI break vs older model.dlls (6-arg). The
/// EP and model.dll are versioned together; clear the cached `morphizen_mlir_*`
/// DLLs after this change (same policy as any generated-interface change).
///
/// Generated IR (data-dependent Range(start,limit,delta) -> [count];
/// @infer_shapes lowered to (ptr,i64,i64,i64) -> i64, three rank-0 i64 inputs
/// so zero dim args + three data args from `hipdnn.shape_fn_data_args`):
///   llvm.func @inference_infer_shapes(%is, %ir, %ic, %id, %os, %or, %oc) ->
///   i32 {
///     %nil = llvm.mlir.zero : !llvm.ptr             // unused !hip.context
///     %r0 = llvm.load %id;        %s = llvm.load %r0   // input_data[0] ->
///     start %g1 = llvm.getelementptr %id[1]; %r1 = llvm.load %g1; %l =
///     llvm.load %r1 %g2 = llvm.getelementptr %id[2]; %r2 = llvm.load %g2; %dl
///     = llvm.load %r2 %cnt = llvm.call @infer_shapes(%nil, %s, %l, %dl)   //
///     i64 count %q0 = llvm.load %os                  // output_shapes[0]
///     llvm.store %cnt, %q0
///     llvm.return %c0_i32
///   }
void generateInferenceInferShapes(ModuleOp module) {
  // Lowered to llvm.func by --convert-hip-to-llvm (index args -> i64, multiple
  // results -> a returned struct). Absent on pipelines where
  // BuildShapeFunctionPass did not run.
  auto inferFn = module.lookupSymbol<LLVM::LLVMFuncOp>("infer_shapes");
  if (!inferFn)
    return;

  auto inputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
  auto outputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
  if (!inputShapes || !outputShapes)
    return;

  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();
  builder.setInsertionPointToEnd(module.getBody());

  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
  Type i32Type = builder.getI32Type();
  Type i64Type = builder.getI64Type();

  // ABI: in_shapes, in_ranks, in_count, in_data, out_shapes, out_ranks,
  // out_count. The in_data pointer array (arg 3) feeds the trailing data args
  // for closed-form data-dependent shape fns; shifts out_* down by one.
  SmallVector<Type> paramTypes = {ptrType, ptrType, i64Type, ptrType,
                                  ptrType, ptrType, i64Type};
  auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);
  auto funcOp = LLVM::LLVMFuncOp::create(builder, loc, "inference_infer_shapes",
                                         funcType);
  funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

  Block *entry = funcOp.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);

  Value inShapes = entry->getArgument(0);
  Value inData = entry->getArgument(3);
  Value outShapes = entry->getArgument(4);

  // @infer_shapes's arg 0 is the (unused) !hip.context, lowered to !llvm.ptr.
  // The shape program never dereferences it; pass null. The flattened input
  // dims follow.
  SmallVector<Value> callArgs;
  callArgs.push_back(LLVM::ZeroOp::create(builder, loc, ptrType));
  for (auto i : llvm::seq<size_t>(0, inputShapes.size())) {
    int64_t rank = cast<DenseI64ArrayAttr>(inputShapes.getValue()[i]).size();
    Value tIdx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                          builder.getI64IntegerAttr(i));
    Value rowSlot = LLVM::GEPOp::create(builder, loc, ptrType, ptrType,
                                        inShapes, ArrayRef<LLVM::GEPArg>{tIdx});
    Value row = LLVM::LoadOp::create(builder, loc, ptrType, rowSlot);
    for (int64_t d : llvm::seq<int64_t>(0, rank)) {
      Value dIdx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                            builder.getI64IntegerAttr(d));
      Value elt = LLVM::GEPOp::create(builder, loc, ptrType, i64Type, row,
                                      ArrayRef<LLVM::GEPArg>{dIdx});
      callArgs.push_back(LLVM::LoadOp::create(builder, loc, i64Type, elt));
    }
  }

  // Trailing data args (closed-form data-dependent shape fns, e.g. Range):
  // for each descriptor in `hipdnn.shape_fn_data_args` (in slot order, the
  // same order BuildShapeFunctionPass appended the @infer_shapes params), read
  // input_data[input_idx], byte-offset to the scalar, and load it with the
  // exact type the lowered @infer_shapes param expects.
  auto dataArgs = module->getAttrOfType<ArrayAttr>("hipdnn.shape_fn_data_args");
  if (dataArgs) {
    ArrayRef<Type> inferParams = inferFn.getFunctionType().getParams();
    Type i8Type = builder.getI8Type();
    for (Attribute a : dataArgs) {
      auto d = cast<DictionaryAttr>(a);
      int64_t inputIdx = cast<IntegerAttr>(d.get("input_idx")).getInt();
      int64_t elemOffset = cast<IntegerAttr>(d.get("elem_offset")).getInt();
      int64_t elemBits = cast<IntegerAttr>(d.get("elem_bits")).getInt();
      // Load type = the matching lowered @infer_shapes param (index->i64,
      // i32->i32, f32->f32, …). callArgs.size() is the next param position.
      Type loadTy = inferParams[callArgs.size()];
      Value tIdx = LLVM::ConstantOp::create(
          builder, loc, i64Type, builder.getI64IntegerAttr(inputIdx));
      Value rowSlot = LLVM::GEPOp::create(builder, loc, ptrType, ptrType,
                                          inData, ArrayRef<LLVM::GEPArg>{tIdx});
      Value row = LLVM::LoadOp::create(builder, loc, ptrType, rowSlot);
      int64_t byteOffset = elemOffset * (elemBits / 8);
      Value bOff = LLVM::ConstantOp::create(
          builder, loc, i64Type, builder.getI64IntegerAttr(byteOffset));
      Value elt = LLVM::GEPOp::create(builder, loc, ptrType, i8Type, row,
                                      ArrayRef<LLVM::GEPArg>{bOff});
      callArgs.push_back(LLVM::LoadOp::create(builder, loc, loadTy, elt));
    }
  }

  auto call = LLVM::CallOp::create(builder, loc, inferFn, callArgs);

  // Collect the i64 results, accounting for how func->llvm lowered the
  // multi-result function: void (0 results), a bare i64 (1 result), or a
  // struct (≥2 results).
  SmallVector<Value> outDims;
  Type retTy = inferFn.getFunctionType().getReturnType();
  if (!isa<LLVM::LLVMVoidType>(retTy)) {
    Value r = call.getResult();
    if (auto st = dyn_cast<LLVM::LLVMStructType>(retTy)) {
      for (auto i : llvm::seq<size_t>(0, st.getBody().size()))
        outDims.push_back(LLVM::ExtractValueOp::create(
            builder, loc, r, ArrayRef<int64_t>{static_cast<int64_t>(i)}));
    } else {
      outDims.push_back(r);
    }
  }

  // Scatter into the caller-allocated output rows (same flattening as the
  // @infer_shapes results).
  size_t k = 0;
  for (auto i : llvm::seq<size_t>(0, outputShapes.size())) {
    int64_t rank = cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size();
    Value tIdx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                          builder.getI64IntegerAttr(i));
    Value rowSlot =
        LLVM::GEPOp::create(builder, loc, ptrType, ptrType, outShapes,
                            ArrayRef<LLVM::GEPArg>{tIdx});
    Value row = LLVM::LoadOp::create(builder, loc, ptrType, rowSlot);
    for (int64_t d : llvm::seq<int64_t>(0, rank)) {
      if (k >= outDims.size())
        break;
      Value dIdx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                            builder.getI64IntegerAttr(d));
      Value elt = LLVM::GEPOp::create(builder, loc, ptrType, i64Type, row,
                                      ArrayRef<LLVM::GEPArg>{dIdx});
      LLVM::StoreOp::create(builder, loc, outDims[k++], elt);
    }
  }

  Value zero = LLVM::ConstantOp::create(builder, loc, i32Type,
                                        builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, zero);
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

    // Prefer the hip.constants_file attribute set by OnnxToHip pass;
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
    auto outputShapes =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
    generateInferenceCompute(module, inputShapes, outputShapes);
    generateInferenceCleanup(module);

    std::string json = buildMetadataJson(module, constantsFile);
    generateMetadataGlobal(module, json);
    generateInferenceGetMetadataJson(module);

    // Conditional: only when BuildShapeFunctionPass emitted @infer_shapes
    // (lowered to llvm.func by --convert-hip-to-llvm). No-op otherwise so
    // hip-to-llvm-only pipelines and older flows are unaffected.
    generateInferenceInferShapes(module);

    COMPILER_DEBUG_LOG("[GenerateInterface] Generated interface functions\n");
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
        {"hipdnn_ep_tensor_prepare_output", i32, {ptr, ptr, i64, i64, ptr}},
        {"hipdnn_ep_tensor_finalize_output", i32, {ptr, ptr}},
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

    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_init") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_compute") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_cleanup") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_get_metadata_json")) {
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

    auto mainType = mainFunc.getFunctionType();
    if (mainType.getNumParams() != 3 || mainType.getParamType(0) != ptrType ||
        mainType.getParamType(1) != ptrType ||
        mainType.getParamType(2) != ptrType ||
        mainType.getReturnType() != i32Type) {
      llvm::errs() << "[GenerateInterface] @main_graph has wrong signature.\n"
                   << "Expected: (ptr, ptr, ptr) -> i32\n";
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

    auto funcOp =
        LLVM::LLVMFuncOp::create(builder, loc, "inference_init", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);
    Value fsPtr = entryBlock->getArgument(1);

    Value blobPtr =
        LLVM::AddressOfOp::create(builder, loc, ptrType, "__metadata_blob");
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

  /// Generated IR overview (1 input, 1 output of rank 1):
  ///   llvm.func @inference_compute(%state: !llvm.ptr, %ins: !llvm.ptr,
  ///                                %outs: !llvm.ptr) -> i32
  ///       attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  ///     // 1. Alloca TensorBuffer structs on the stack
  ///     // 2. For each input:  call hipdnn_ep_tensor_prepare_input,
  ///     error-check
  ///     // 3. For each output: call hipdnn_ep_tensor_prepare_output,
  ///     error-check
  ///     // 4. Build LLVM memref descriptors from TensorBuffer GPU/shape ptrs
  ///     // 5. Call @main_graph(%state, %input_memrefs, %output_memrefs)
  ///     // 6. For each output: call hipdnn_ep_tensor_finalize_output,
  ///     error-check
  ///     // 7. Call hipdnn_ep_stream_sync (GPU sync + PERF profiling output)
  ///     // 8. Read device-side error flag
  ///     // 9. Free input TensorBuffers
  ///     // On error: store error code, free inputs, return error
  ///   }
  void generateInferenceCompute(ModuleOp module, ArrayAttr inputShapes,
                                ArrayAttr outputShapes) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    builder.setInsertionPointToEnd(module.getBody());

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    auto funcOp =
        LLVM::LLVMFuncOp::create(builder, loc, "inference_compute", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value inputsSpanPtr = entryBlock->getArgument(1);
    Value outputsSpanPtr = entryBlock->getArgument(2);

    size_t numInputs = inputShapes ? inputShapes.size() : 0;
    size_t numOutputs = outputShapes ? outputShapes.size() : 0;

    Value c0_i32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                            builder.getI32IntegerAttr(0));
    Value c1_i64 = LLVM::ConstantOp::create(builder, loc, i64Type,
                                            builder.getI64IntegerAttr(1));

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
    auto resetErrorFlagFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_state_reset_error_flag");
    auto streamSyncFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("hipdnn_ep_stream_sync");
    auto readErrorFlagFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "hipdnn_ep_state_read_and_clear_error_flag");

    Value errorCodePtr =
        LLVM::AllocaOp::create(builder, loc, ptrType, i32Type, c1_i64, 0);

    SmallVector<Value> inputBuffers;
    SmallVector<Value> outputBuffers;

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
    for (auto i : llvm::seq<size_t>(0, numOutputs)) {
      (void)i;
      Value bufferPtr = LLVM::AllocaOp::create(builder, loc, ptrType, i8Type,
                                               tensorBufferSize, 0);
      outputBuffers.push_back(bufferPtr);
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

    for (auto i : llvm::seq<size_t>(0, numOutputs)) {
      Value bufferPtr = outputBuffers[i];

      Value indexVal = LLVM::ConstantOp::create(builder, loc, i64Type,
                                                builder.getI64IntegerAttr(i));

      Value rankVal = LLVM::ConstantOp::create(
          builder, loc, i64Type,
          builder.getI64IntegerAttr(
              cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size()));

      emitErrorCheckedCall(
          builder, loc, prepareOutputFunc,
          ValueRange{state, outputsSpanPtr, indexVal, rankVal, bufferPtr},
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

    Value numOutputsVal = LLVM::ConstantOp::create(
        builder, loc, i64Type, builder.getI64IntegerAttr(numOutputs));
    Value outputMemrefArray = LLVM::AllocaOp::create(builder, loc, ptrType,
                                                     ptrType, numOutputsVal, 0);

    for (auto i : llvm::seq<size_t>(0, numOutputs)) {
      int64_t rank = cast<DenseI64ArrayAttr>(outputShapes.getValue()[i]).size();

      Value bufferPtr = outputBuffers[i];
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
          LLVM::GEPOp::create(builder, loc, ptrType, ptrType, outputMemrefArray,
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
      emitErrorCheckedCall(
          builder, loc, mainFunc,
          ValueRange{state, inputMemrefArray, outputMemrefArray}, errorCodePtr,
          errorCleanupBlock, funcOp);
      LLVM::BrOp::create(builder, loc, mainSuccessBlock);
    }

    // Finalize output tensors (D2H, sync, cleanup)
    builder.setInsertionPointToStart(mainSuccessBlock);

    for (auto i : llvm::seq<size_t>(0, numOutputs)) {
      Value bufferPtr = outputBuffers[i];
      emitErrorCheckedCall(builder, loc, finalizeOutputFunc,
                           ValueRange{state, bufferPtr}, errorCodePtr,
                           errorCleanupBlock, funcOp);
    }

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

    auto funcOp =
        LLVM::LLVMFuncOp::create(builder, loc, "inference_cleanup", funcType);
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
