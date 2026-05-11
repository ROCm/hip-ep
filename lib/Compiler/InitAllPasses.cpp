//===- InitAllPasses.cpp - Register all HIP dialects/passes -----*- C++ -*-===//
//
// Out-of-line definitions for the HIP `registerAllDialects` / `loadAllDialects`
// / `registerAllPasses` entry points declared in `InitAllPasses.h`.  Lives in
// its own TU so the heavy MLIR dialect transitive includes (Bufferization,
// MemRef, Tensor, LLVM, ...) compile exactly once instead of being dragged
// into every consumer that just needs the public registration prototypes.
//
//===----------------------------------------------------------------------===//

#include "hip/InitAllPasses.h"

#include "hip/Conversion/Passes.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hip {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<detail::OnnxStubDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);
}

void loadAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

void registerAllPasses() {
  mlir::registerCanonicalizerPass();
  mlir::hip::registerHipPasses();
  mlir::hip::registerHipConversionPasses();
}

} // namespace hip
} // namespace mlir
