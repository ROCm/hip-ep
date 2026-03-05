/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Generates LLVM IR wrapper functions (inference_init, inference_compute,
 * inference_cleanup) that implement the span_t/tensor_t C ABI expected by
 * the MorphiZen EP.  The wrapper handles:
 *   - H2D / D2H memory transfers via hipMemcpy
 *   - GPU buffer allocation / deallocation via hipMalloc / hipFree
 *   - Flattening tensor_t descriptors into the memref calling convention
 *     produced by MLIR's convert-func-to-llvm pass.
 */
#include "InterfaceGenerator.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include <iostream>

namespace hipdnn::compiler {

namespace {

// hipMemcpyKind enum values
constexpr int HIP_MEMCPY_H2D = 1;
constexpr int HIP_MEMCPY_D2H = 2;

// --------------------------------------------------------------------------
// Declare (or look up) an external C function in the module.
// --------------------------------------------------------------------------
llvm::FunctionCallee getOrInsert(llvm::Module &M, const char *name,
                                 llvm::FunctionType *ty) {
  return M.getOrInsertFunction(name, ty);
}

// --------------------------------------------------------------------------
// Build row-major strides from a shape array.
//   stride[rank-1] = 1
//   stride[i]      = stride[i+1] * shape[i+1]
// --------------------------------------------------------------------------
std::vector<llvm::Value *> buildStrides(llvm::IRBuilder<> &B,
                                        const std::vector<llvm::Value *> &sizes,
                                        int rank) {
  std::vector<llvm::Value *> strides(rank);
  strides[rank - 1] = B.getInt64(1);
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = B.CreateMul(strides[i + 1], sizes[i + 1]);
  return strides;
}

// --------------------------------------------------------------------------
// Compute total byte count:  product(sizes) * element_size
// --------------------------------------------------------------------------
llvm::Value *computeTotalBytes(llvm::IRBuilder<> &B,
                               const std::vector<llvm::Value *> &sizes,
                               llvm::Value *elemSize) {
  llvm::Value *n = sizes[0];
  for (size_t i = 1; i < sizes.size(); ++i)
    n = B.CreateMul(n, sizes[i]);
  return B.CreateMul(n, elemSize);
}

// --------------------------------------------------------------------------
// Load fields from a tensor_t at a given index in the tensor array.
//
//   struct tensor_t { void* data; int64_t* shape; size_t rank; size_t elem_size; };
//
// Returns {data, shape_ptr, rank, elem_size}.
// --------------------------------------------------------------------------
struct TensorFields {
  llvm::Value *data;
  llvm::Value *shapePtr;
  llvm::Value *rank;
  llvm::Value *elemSize;
};

TensorFields loadTensor(llvm::IRBuilder<> &B, llvm::Value *tensorArray,
                        int index, llvm::StructType *tensorTy) {
  auto *gep = B.CreateStructGEP(
      tensorTy,
      B.CreateGEP(tensorTy, tensorArray, B.getInt32(index)),
      0);
  TensorFields f;
  f.data = B.CreateLoad(B.getPtrTy(), gep);

  auto *shapeGep = B.CreateStructGEP(
      tensorTy,
      B.CreateGEP(tensorTy, tensorArray, B.getInt32(index)),
      1);
  f.shapePtr = B.CreateLoad(B.getPtrTy(), shapeGep);

  auto *rankGep = B.CreateStructGEP(
      tensorTy,
      B.CreateGEP(tensorTy, tensorArray, B.getInt32(index)),
      2);
  f.rank = B.CreateLoad(B.getInt64Ty(), rankGep);

  auto *elemGep = B.CreateStructGEP(
      tensorTy,
      B.CreateGEP(tensorTy, tensorArray, B.getInt32(index)),
      3);
  f.elemSize = B.CreateLoad(B.getInt64Ty(), elemGep);
  return f;
}

// --------------------------------------------------------------------------
// Load shape[i] from the tensor's shape array.
// --------------------------------------------------------------------------
llvm::Value *loadShapeDim(llvm::IRBuilder<> &B, llvm::Value *shapePtr,
                          int dim) {
  auto *gep = B.CreateGEP(B.getInt64Ty(), shapePtr, B.getInt32(dim));
  return B.CreateLoad(B.getInt64Ty(), gep);
}

} // namespace

// ============================================================================
// Public entry point
// ============================================================================
void InterfaceGenerator::addInterfaceFunctions(llvm::Module &M,
                                               const ModelMetadata &meta) {
  auto &ctx = M.getContext();
  llvm::IRBuilder<> B(ctx);

  auto *i32Ty = B.getInt32Ty();
  auto *i64Ty = B.getInt64Ty();
  auto *ptrTy = B.getPtrTy();
  auto *voidTy = B.getVoidTy();

  // struct tensor_t { ptr, ptr, i64, i64 }
  auto *tensorTy =
      llvm::StructType::create(ctx, {ptrTy, ptrTy, i64Ty, i64Ty}, "tensor_t");
  // struct span_t { ptr, i64 }
  auto *spanTy =
      llvm::StructType::create(ctx, {ptrTy, i64Ty}, "span_t");

  // External HIP runtime functions
  auto hipMallocFn = getOrInsert(
      M, "hipMalloc",
      llvm::FunctionType::get(i32Ty, {ptrTy, i64Ty}, false));
  auto hipFreeFn = getOrInsert(
      M, "hipFree",
      llvm::FunctionType::get(i32Ty, {ptrTy}, false));
  auto hipMemcpyFn = getOrInsert(
      M, "hipMemcpy",
      llvm::FunctionType::get(i32Ty, {ptrTy, ptrTy, i64Ty, i32Ty}, false));

  // malloc / free
  auto mallocFn = getOrInsert(
      M, "malloc",
      llvm::FunctionType::get(ptrTy, {i64Ty}, false));
  auto freeFn = getOrInsert(
      M, "free",
      llvm::FunctionType::get(voidTy, {ptrTy}, false));

  // Find the compiled compute function
  llvm::Function *computeFn = M.getFunction(meta.entryFunction);
  if (!computeFn) {
    std::cerr << "Warning: compute function '" << meta.entryFunction
              << "' not found in module\n";
    return;
  }

  // ========================================================================
  // inference_init(void** out_state) -> i32
  // ========================================================================
  {
    auto *fnTy = llvm::FunctionType::get(i32Ty, {ptrTy}, false);
    auto *fn = llvm::Function::Create(
        fnTy, llvm::GlobalValue::ExternalLinkage, "inference_init", M);
    fn->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);

    auto *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    B.SetInsertPoint(entry);

    llvm::Value *outState = fn->getArg(0);
    // Allocate a small state (just a marker; the compiled function manages
    // its own HIP handles internally).
    auto *state = B.CreateCall(mallocFn, {B.getInt64(8)});
    B.CreateStore(state, outState);
    B.CreateRet(B.getInt32(0));
  }

  // ========================================================================
  // inference_cleanup(void* state) -> i32
  // ========================================================================
  {
    auto *fnTy = llvm::FunctionType::get(i32Ty, {ptrTy}, false);
    auto *fn = llvm::Function::Create(
        fnTy, llvm::GlobalValue::ExternalLinkage, "inference_cleanup", M);
    fn->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);

    auto *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    B.SetInsertPoint(entry);

    B.CreateCall(freeFn, {fn->getArg(0)});
    B.CreateRet(B.getInt32(0));
  }

  // ========================================================================
  // inference_compute(void* state, span_t* inputs, span_t* outputs) -> i32
  //
  // The heavy lifter: allocates GPU buffers, copies H2D, builds flattened
  // memref descriptor arguments, calls the compiled function, copies D2H,
  // and frees GPU buffers.
  // ========================================================================
  {
    auto *fnTy =
        llvm::FunctionType::get(i32Ty, {ptrTy, ptrTy, ptrTy}, false);
    auto *fn = llvm::Function::Create(
        fnTy, llvm::GlobalValue::ExternalLinkage, "inference_compute", M);
    fn->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);

    auto *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    B.SetInsertPoint(entry);

    llvm::Value *inputsSpan = fn->getArg(1);
    llvm::Value *outputsSpan = fn->getArg(2);

    // Load span_t.data pointers (tensor_t arrays)
    auto *inDataGep = B.CreateStructGEP(spanTy, inputsSpan, 0);
    auto *inTensors = B.CreateLoad(ptrTy, inDataGep);

    auto *outDataGep = B.CreateStructGEP(spanTy, outputsSpan, 0);
    auto *outTensors = B.CreateLoad(ptrTy, outDataGep);

    int totalArgs = meta.inputCount + meta.outputCount;

    // GPU pointers and byte sizes for cleanup
    std::vector<llvm::Value *> gpuPtrs;
    // Arguments to pass to the compute function (flattened memref descriptors)
    std::vector<llvm::Value *> callArgs;

    // --- Process each input tensor ---
    for (int i = 0; i < meta.inputCount; ++i) {
      int rank = meta.inputRanks[i];
      auto tf = loadTensor(B, inTensors, i, tensorTy);

      // Load shape dimensions
      std::vector<llvm::Value *> sizes;
      for (int d = 0; d < rank; ++d)
        sizes.push_back(loadShapeDim(B, tf.shapePtr, d));

      auto *totalBytes = computeTotalBytes(B, sizes, tf.elemSize);

      // hipMalloc
      auto *gpuPtrAlloca = B.CreateAlloca(ptrTy);
      B.CreateCall(hipMallocFn, {gpuPtrAlloca, totalBytes});
      auto *gpuPtr = B.CreateLoad(ptrTy, gpuPtrAlloca);
      gpuPtrs.push_back(gpuPtr);

      // hipMemcpy H2D
      B.CreateCall(hipMemcpyFn,
                   {gpuPtr, tf.data, totalBytes, B.getInt32(HIP_MEMCPY_H2D)});

      // Build flattened memref args: (alloc_ptr, align_ptr, offset, sizes..., strides...)
      callArgs.push_back(gpuPtr);  // allocated pointer
      callArgs.push_back(gpuPtr);  // aligned pointer
      callArgs.push_back(B.getInt64(0)); // offset

      for (int d = 0; d < rank; ++d)
        callArgs.push_back(sizes[d]);

      auto strides = buildStrides(B, sizes, rank);
      for (int d = 0; d < rank; ++d)
        callArgs.push_back(strides[d]);
    }

    // --- Process each output tensor ---
    std::vector<llvm::Value *> outGpuPtrs;
    std::vector<llvm::Value *> outHostPtrs;
    std::vector<llvm::Value *> outByteSizes;

    for (int i = 0; i < meta.outputCount; ++i) {
      int rank = meta.outputRanks[i];
      auto tf = loadTensor(B, outTensors, i, tensorTy);

      std::vector<llvm::Value *> sizes;
      for (int d = 0; d < rank; ++d)
        sizes.push_back(loadShapeDim(B, tf.shapePtr, d));

      auto *totalBytes = computeTotalBytes(B, sizes, tf.elemSize);

      // hipMalloc
      auto *gpuPtrAlloca = B.CreateAlloca(ptrTy);
      B.CreateCall(hipMallocFn, {gpuPtrAlloca, totalBytes});
      auto *gpuPtr = B.CreateLoad(ptrTy, gpuPtrAlloca);
      gpuPtrs.push_back(gpuPtr);
      outGpuPtrs.push_back(gpuPtr);
      outHostPtrs.push_back(tf.data);
      outByteSizes.push_back(totalBytes);

      // Build flattened memref args
      callArgs.push_back(gpuPtr);
      callArgs.push_back(gpuPtr);
      callArgs.push_back(B.getInt64(0));

      for (int d = 0; d < rank; ++d)
        callArgs.push_back(sizes[d]);

      auto strides = buildStrides(B, sizes, rank);
      for (int d = 0; d < rank; ++d)
        callArgs.push_back(strides[d]);
    }

    // --- Call the compiled compute function ---
    B.CreateCall(computeFn, callArgs);

    // --- D2H copy for each output ---
    for (int i = 0; i < meta.outputCount; ++i) {
      B.CreateCall(
          hipMemcpyFn,
          {outHostPtrs[i], outGpuPtrs[i], outByteSizes[i],
           B.getInt32(HIP_MEMCPY_D2H)});
    }

    // --- Free all GPU buffers ---
    for (auto *gp : gpuPtrs)
      B.CreateCall(hipFreeFn, {gp});

    B.CreateRet(B.getInt32(0));
  }
}

} // namespace hipdnn::compiler
