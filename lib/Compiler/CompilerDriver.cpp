/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/CompilerDriver.h"
#include "hip/Compiler/Pipeline.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/InitAllPasses.h"

#include "hip/Target/LLVM/DLLLinker.h"
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

#include "hip/debug_log.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace hip::compiler {

namespace {
bool fileExists(const std::string &path) {
  llvm::sys::fs::file_status status;
  std::error_code EC = llvm::sys::fs::status(path, status);
  return !EC && llvm::sys::fs::exists(status);
}
} // namespace

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

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);

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
  if (!runMLIRPasses(module, options, error_message))
    return false;

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);
  if (!llvmModule)
    return false;

  if (!linkRuntime(llvmModule.get(), error_message))
    return false;

  optimizeLLVMIR(llvmModule.get(), options.opt_level);

  std::string base_path = output_path;
  if (base_path.size() >= 4 &&
      base_path.substr(base_path.size() - 4) == ".dll") {
    base_path = base_path.substr(0, base_path.size() - 4);
  }
  std::string ll_path = base_path + ".ll";
  std::string obj_path = base_path + ".obj";

  if (options.output_mode == mlir::hip::OutputMode::LLVM_IR) {
    if (!emitLLVMIR(llvmModule.get(), ll_path, error_message))
      return false;
    return true;
  }

  if (!compileToObject(llvmModule.get(), obj_path, error_message))
    return false;

  std::vector<std::string> export_symbols = {
      "inference_init", "inference_compute", "inference_cleanup",
      "inference_get_metadata_json", "test_hip_from_dll"};
  std::vector<std::string> libraries;
  std::vector<std::string> library_paths;

  if (const char *therock = std::getenv("THEROCK_DIST")) {
    std::string dist(therock);
    std::string lib_dir = dist + "/lib";
    library_paths.push_back(lib_dir);
    std::cout << "THEROCK_DIST detected: " << dist << "\n";
    std::cout << "  Adding library path: " << lib_dir << "\n";

    libraries.push_back("amdhip64");
    libraries.push_back("MIOpen");

    std::string hipblaslt_lib = lib_dir + "/hipblaslt.lib";
    std::string hipblaslt_dll_a = lib_dir + "/libhipblaslt.dll.a";
    if (llvm::sys::fs::exists(hipblaslt_lib))
      libraries.push_back("hipblaslt");
    else if (llvm::sys::fs::exists(hipblaslt_dll_a))
      libraries.push_back(hipblaslt_dll_a);
    else
      std::cerr << "  WARNING: hipblaslt import library not found\n";

#ifdef HIP_CUSTOM_KERNELS_LIB_PATH
    {
      std::string custom_lib = HIP_CUSTOM_KERNELS_LIB_PATH;
      if (llvm::sys::fs::exists(custom_lib)) {
        libraries.push_back(custom_lib);
        std::cout << "  Custom kernels: " << custom_lib << "\n";
      } else {
        std::cerr << "  WARNING: custom kernels lib not found at: "
                  << custom_lib << "\n";
      }
    }
#endif

    for (const auto &lib : libraries) {
      std::cout << "  Linking library: " << lib << "\n";
    }
  }

  if (!linkToDLL(obj_path, output_path, libraries, library_paths,
                 export_symbols, error_message))
    return false;

  cleanupIntermediates(base_path);

  return true;
}

bool CompilerDriver::runMLIRPasses(
    mlir::ModuleOp module, const mlir::hip::CompilationOptionsT &options,
    std::string &error_message) {
  mlir::PassManager pm(module.getContext());

  if (options.verbose) {
    std::cout << "Running ONNX->HIP->LLVM->Interface passes\n";
  }

  populateMorphizenPipeline(pm, options);

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
    std::cout << "MLIR passes completed\n\n";

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

bool CompilerDriver::emitLLVMIR(llvm::Module *llvmModule,
                                const std::string &outputPath,
                                std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.emitLLVMIR(llvmModule, outputPath)) {
    error_message = "Failed to emit LLVM IR";
    return false;
  }
  return true;
}

bool CompilerDriver::compileToObject(llvm::Module *llvmModule,
                                     const std::string &outputPath,
                                     std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.compileToObjectFile(llvmModule, outputPath)) {
    error_message = "Failed to compile to object file";
    return false;
  }
  return true;
}

bool CompilerDriver::linkToDLL(const std::string &objPath,
                               const std::string &dllPath,
                               const std::vector<std::string> &libraries,
                               const std::vector<std::string> &library_paths,
                               const std::vector<std::string> &export_symbols,
                               std::string &error_message) {
  hipdnn::DLLLinker linker;

  if (!linker.linkDLL(objPath, dllPath, libraries, library_paths,
                      export_symbols)) {
    error_message = "Failed to link DLL";
    return false;
  }

  return true;
}

void CompilerDriver::cleanupIntermediates(const std::string &basePath) {
  std::string ll_path = basePath + ".ll";
  std::string obj_path = basePath + ".obj";

  if (fileExists(ll_path)) {
    llvm::sys::fs::remove(ll_path);
  }

  if (fileExists(obj_path)) {
    llvm::sys::fs::remove(obj_path);
  }
}

} // namespace hip::compiler
