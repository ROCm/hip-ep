/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/CompilerDriver.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/InitAllPasses.h"

#include "hip/Target/LLVM/LLVMBackend.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "hip/debug_log.h"

#include <chrono>
#include <cstdlib>
#include <sstream>

namespace hip::compiler {

bool CompilerDriver::compile(llvm::StringRef input_mlir,
                             const std::string &output_path,
                             const mlir::hip::CompilationOptionsT &options,
                             std::string &error_message) {
  hip::compiler::registerAllPasses();

  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);
  mlir::registerLLVMDialectTranslation(context);

  COMPILER_DEBUG_LOG("[CompilerDriver::compile] Input size: "
                     << input_mlir.size() << " bytes\n");

  // Wrap input in a non-owning buffer (no copy) for MLIR's parser.
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);

  // SourceMgr provides source-location tracking for parser diagnostics.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  auto t0 = timing_now();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (hipdnn_ep_timing_enabled()) {
    llvm::errs() << "[CompilerDriver] MLIR parsing: "
                 << llvm::format("%.3f", elapsed_since(t0)) << "s\n";
  }

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  return compileImpl(*module, output_path, options, error_message);
}

bool CompilerDriver::compileFromModule(
    mlir::ModuleOp module, const std::string &output_path,
    const mlir::hip::CompilationOptionsT &options, std::string &error_message) {
  hip::compiler::registerAllPasses();
  return compileImpl(module, output_path, options, error_message);
}

bool CompilerDriver::validate(llvm::StringRef input_mlir,
                              std::string &error_message) {
  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  return true;
}

bool CompilerDriver::compileImpl(mlir::ModuleOp module,
                                 const std::string &output_path,
                                 const mlir::hip::CompilationOptionsT &options,
                                 std::string &error_message) {
  const bool timing = hipdnn_ep_timing_enabled();
  auto phaseStart = timing_now();
  auto totalStart = phaseStart;

  auto logPhase = [&](const char *name) {
    if (!timing)
      return;
    llvm::errs() << "[CompilerDriver] " << name << ": "
                 << llvm::format("%.3f", record_elapsed(phaseStart)) << "s\n";
  };

  if (timing)
    llvm::errs() << "[CompilerDriver] === Compilation phases ===\n";

  if (!runMLIRPasses(module, options, error_message))
    return false;
  logPhase("runMLIRPasses");

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);
  if (!llvmModule)
    return false;
  logPhase("translateToLLVMIR");

  if (!linkRuntime(llvmModule.get(), error_message))
    return false;
  logPhase("linkRuntime");

  optimizeLLVMIR(llvmModule.get(), options.opt_level);
  logPhase("optimizeLLVMIR");

  // The per-model artifact is LLVM bitcode. It is consumed in-process by
  // ORC LLJIT (BitcodeJIT in backend-mlir-compiler/custom-op-mlir/src/)
  // and shipped inside the EPContext tar as data, not code. There is no
  // longer any object-file, LLD-link, or temp-DLL step on the runtime
  // path -- the host-side runtime has already been merged in via
  // LLVMBackend::linkRuntimeModule above, and GPU device code lives in
  // the signed EP DLL.
  if (!emitBitcode(llvmModule.get(), output_path, error_message))
    return false;
  logPhase("emitBitcode");
  if (timing) {
    llvm::errs() << "[CompilerDriver] total: "
                 << llvm::format("%.3f", elapsed_since(totalStart)) << "s\n";
  }
  return true;
}

bool CompilerDriver::runMLIRPasses(
    mlir::ModuleOp module, const mlir::hip::CompilationOptionsT &options,
    std::string &error_message) {
  mlir::PassManager pm(module.getContext());

  if (hipdnn_ep_timing_enabled())
    pm.enableTiming();

  if (options.verbose) {
    COMPILER_DEBUG_LOG("Running ONNX->HIP->LLVM->Interface passes\n");
  }

  mlir::hip::OnnxToHipPipelineOptions onnxToHipOpts;
  onnxToHipOpts.externalizeMinNumElements =
      mlir::hip::kDefaultExternalizeMinNumElements;
  onnxToHipOpts.skipConstantData = options.skip_constant_data;

  if (hipdnnHandle_) {
    compiledGraphs_ =
        std::make_shared<llvm::StringMap<mlir::hip::OwnedGraph>>();
    mlir::hip::buildOnnxToHipPipeline(
        pm, onnxToHipOpts, fileSystem_,
        static_cast<hipdnnHandle_t>(hipdnnHandle_), compiledGraphs_);
  } else {
    mlir::hip::buildOnnxToHipPipeline(pm, onnxToHipOpts, fileSystem_);
  }

  mlir::hip::HipToLLVMPipelineOptions hipToLlvmOpts;
  hipToLlvmOpts.constantsFile = options.constants_file;
  mlir::hip::buildHipToLLVMPipeline(pm, hipToLlvmOpts);

  std::unique_ptr<llvm::raw_fd_ostream> irDumpStream;
  if (const char *dumpPath = std::getenv("HIPDNN_EP_IR_DUMP_PATH")) {
    std::error_code ec;
    irDumpStream = std::make_unique<llvm::raw_fd_ostream>(dumpPath, ec);
    if (!ec) {
      module.getContext()->disableMultithreading();
      pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return true; },
                          [](mlir::Pass *, mlir::Operation *) { return true; },
                          /*printModuleScope=*/true,
                          /*printAfterOnlyOnChange=*/true,
                          /*printAfterOnlyOnFailure=*/false, *irDumpStream);
    } else {
      llvm::errs() << "[CompilerDriver] Failed to open IR dump file: "
                   << dumpPath << ": " << ec.message() << "\n";
    }
  }

  if (mlir::failed(pm.run(module))) {
    error_message = "MLIR pass pipeline failed";
    if (options.verbose) {
      llvm::errs() << "\n=== Failed Module IR ===\n";
      module.print(llvm::errs());
      llvm::errs() << "\n========================\n";
    }
    return false;
  }

  if (options.verbose)
    COMPILER_DEBUG_LOG("MLIR passes completed\n\n");

  return true;
}

std::unique_ptr<llvm::Module>
CompilerDriver::translateToLLVMIR(mlir::ModuleOp module,
                                  llvm::LLVMContext &llvmContext,
                                  std::string &error_message) {
  hipdnn::LLVMBackend backend;
  auto llvmModule = backend.translateMLIRtoLLVMIR(module, llvmContext);

  if (!llvmModule) {
    error_message = "Failed to translate MLIR to LLVM IR";
  }

  return llvmModule;
}

bool CompilerDriver::linkRuntime(llvm::Module *llvmModule,
                                 std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.linkRuntimeModule(llvmModule)) {
    error_message = "Failed to link runtime module";
    return false;
  }
  return true;
}

void CompilerDriver::optimizeLLVMIR(llvm::Module *llvmModule, int optLevel) {
  hipdnn::LLVMBackend backend;
  backend.optimizeLLVMIR(llvmModule, optLevel);
}

bool CompilerDriver::emitBitcode(llvm::Module *llvmModule,
                                 const std::string &outputPath,
                                 std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.emitBitcode(llvmModule, outputPath)) {
    error_message = "Failed to emit LLVM bitcode";
    return false;
  }
  return true;
}

} // namespace hip::compiler
