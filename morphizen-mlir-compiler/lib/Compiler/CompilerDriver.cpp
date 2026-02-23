/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-mlir-compiler/Compiler/CompilerDriver.h"
#include "compilation_options.pb.h"
#include "morphizen-mlir-compiler/Compiler/Pipeline.h"
#include "morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h"
#include "morphizen-mlir-compiler/InitAllPasses.h"

#include "morphizen-mlir-compiler/Target/LLVM/DLLLinker.h"
#include "morphizen-mlir-compiler/Target/LLVM/LLVMBackend.h"

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

#include <iostream>
#include <sstream>

namespace morphizen::mlir_compiler {

namespace {
// Helper to check if file exists
bool fileExists(const std::string& path) {
  llvm::sys::fs::file_status status;
  std::error_code EC = llvm::sys::fs::status(path, status);
  return !EC && llvm::sys::fs::exists(status);
}
} // namespace

bool CompilerDriver::compile(
    llvm::StringRef input_mlir, const std::string& output_path,
    const morphizen::mlir_compiler::CompilationOptions& options,
    std::string& error_message) {
  // Register all passes (idempotent)
  morphizen::registerAllPasses();

  // Initialize MLIR context
  mlir::MLIRContext context;
  morphizen::loadAllDialects(context);
  mlir::registerLLVMDialectTranslation(context);

  // Parse MLIR input
  // Binary-safe: do not require null terminator (bytecode may contain embedded
  // nulls)
  std::cerr << "[CompilerDriver::compile] Input size: " << input_mlir.size()
            << " bytes\n";
  if (input_mlir.size() >= 4) {
    std::cerr << "[CompilerDriver::compile] First 4 bytes: 0x" << std::hex
              << (unsigned int)(unsigned char)input_mlir[0]
              << (unsigned int)(unsigned char)input_mlir[1]
              << (unsigned int)(unsigned char)input_mlir[2]
              << (unsigned int)(unsigned char)input_mlir[3] << std::dec << "\n";
  }

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);
  std::cerr << "[CompilerDriver::compile] MemBuffer size: "
            << memBuffer->getBufferSize() << " bytes\n";

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  return compileImpl(*module, output_path, options, error_message);
}

bool CompilerDriver::compileFromModule(
    mlir::ModuleOp module, const std::string& output_path,
    const morphizen::mlir_compiler::CompilationOptions& options,
    std::string& error_message) {
  // Register all Morphizen passes (idempotent)
  morphizen::registerAllPasses();

  return compileImpl(module, output_path, options, error_message);
}

bool CompilerDriver::validate(llvm::StringRef input_mlir,
                              std::string& error_message) {
  // Initialize MLIR context
  mlir::MLIRContext context;
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<mlir::hip::HipDialect>();
  context.loadDialect<mlir::ONNXDialect>();

  // Parse MLIR input
  // Binary-safe: do not require null terminator (bytecode may contain embedded
  // nulls)
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  // Basic validation succeeded (parsing + semantic checks)
  return true;
}

bool CompilerDriver::compileImpl(
    mlir::ModuleOp module, const std::string& output_path,
    const morphizen::mlir_compiler::CompilationOptions& options,
    std::string& error_message) {
  // Step 2: Run MLIR passes
  if (!runMLIRPasses(module, options, error_message))
    return false;

  // Step 3: Translate to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);
  if (!llvmModule)
    return false;

  // Step 3.5: Link runtime module
  if (!linkRuntime(llvmModule.get(), error_message))
    return false;

  // Step 4: Optimize LLVM IR
  optimizeLLVMIR(llvmModule.get(), options.opt_level());

  // Determine intermediate file names
  std::string base_path = output_path;
  if (base_path.size() >= 4 &&
      base_path.substr(base_path.size() - 4) == ".dll") {
    base_path = base_path.substr(0, base_path.size() - 4);
  }
  std::string ll_path = base_path + ".ll";
  std::string obj_path = base_path + ".obj";

  // Step 5: Emit LLVM IR (if requested)
  if (options.output_mode() == morphizen::mlir_compiler::OUTPUT_MODE_LLVM_IR) {
    if (!emitLLVMIR(llvmModule.get(), ll_path, error_message))
      return false;
    return true; // Done - IR output requested
  }

  // Step 6: Compile to object file
  if (!compileToObject(llvmModule.get(), obj_path, error_message))
    return false;

  // Step 7: Link to DLL
  // Hardcoded internal settings (not exposed in public API)
  std::vector<std::string> export_symbols = {
      "inference_init", "inference_compute", "inference_cleanup",
      "inference_get_metadata_json", "test_hip_from_dll"};
  std::vector<std::string> libraries;
  std::vector<std::string> library_paths;

  if (!linkToDLL(obj_path, output_path, libraries, library_paths,
                 export_symbols, error_message))
    return false;

  // Step 8: Cleanup intermediates (always cleanup for simplified API)
  cleanupIntermediates(base_path);

  return true;
}

bool CompilerDriver::runMLIRPasses(
    mlir::ModuleOp module,
    const morphizen::mlir_compiler::CompilationOptions& options,
    std::string& error_message) {
  mlir::PassManager pm(module.getContext());

  if (options.verbose()) {
    std::cout << "Running ONNX→HIP→BufferDeallocation→LLVM→Interface passes\n";
  }

  // Use the SINGLE SOURCE OF TRUTH pipeline
  morphizen::compiler::PipelineOptions pipelineOpts;
  pipelineOpts.verbose = options.verbose();
  pipelineOpts.enableMemoryPooling = true;
  pipelineOpts.optLevel = options.opt_level();

  morphizen::compiler::populateMorphizenPipeline(pm, pipelineOpts);

  if (mlir::failed(pm.run(module))) {
    error_message = "MLIR pass pipeline failed";
    if (options.verbose()) {
      llvm::errs() << "\n=== Failed Module IR ===\n";
      module.print(llvm::errs());
      llvm::errs() << "\n========================\n";
    }
    return false;
  }

  if (options.verbose())
    std::cout << "✓ MLIR passes completed\n\n";

  return true;
}

std::unique_ptr<llvm::Module>
CompilerDriver::translateToLLVMIR(mlir::ModuleOp module,
                                  llvm::LLVMContext& llvmContext,
                                  std::string& error_message) {
  hipdnn::LLVMBackend backend;
  auto llvmModule = backend.translateMLIRtoLLVMIR(module, llvmContext);

  if (!llvmModule) {
    error_message = "Failed to translate MLIR to LLVM IR";
  }

  return llvmModule;
}

bool CompilerDriver::linkRuntime(llvm::Module* llvmModule,
                                 std::string& error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.linkRuntimeModule(llvmModule)) {
    error_message = "Failed to link runtime module";
    return false;
  }
  return true;
}

void CompilerDriver::optimizeLLVMIR(llvm::Module* llvmModule, int optLevel) {
  hipdnn::LLVMBackend backend;
  backend.optimizeLLVMIR(llvmModule, optLevel);
}

bool CompilerDriver::emitLLVMIR(llvm::Module* llvmModule,
                                const std::string& outputPath,
                                std::string& error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.emitLLVMIR(llvmModule, outputPath)) {
    error_message = "Failed to emit LLVM IR";
    return false;
  }
  return true;
}

bool CompilerDriver::compileToObject(llvm::Module* llvmModule,
                                     const std::string& outputPath,
                                     std::string& error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.compileToObjectFile(llvmModule, outputPath)) {
    error_message = "Failed to compile to object file";
    return false;
  }
  return true;
}

bool CompilerDriver::linkToDLL(const std::string& objPath,
                               const std::string& dllPath,
                               const std::vector<std::string>& libraries,
                               const std::vector<std::string>& library_paths,
                               const std::vector<std::string>& export_symbols,
                               std::string& error_message) {
  hipdnn::DLLLinker linker;

  if (!linker.linkDLL(objPath, dllPath, libraries, library_paths,
                      export_symbols)) {
    error_message = "Failed to link DLL";
    return false;
  }

  return true;
}

void CompilerDriver::cleanupIntermediates(const std::string& basePath) {
  std::string ll_path = basePath + ".ll";
  std::string obj_path = basePath + ".obj";

  if (fileExists(ll_path)) {
    llvm::sys::fs::remove(ll_path);
  }

  if (fileExists(obj_path)) {
    llvm::sys::fs::remove(obj_path);
  }
}

} // namespace morphizen::mlir_compiler
