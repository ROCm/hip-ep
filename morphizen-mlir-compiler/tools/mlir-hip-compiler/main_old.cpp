/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR to HIP DLL Compiler
// Enables independent testing of the MLIR → LLVM IR → Object → DLL pipeline
//
// Usage: mlir-hip-compiler input.mlir -o output.dll [--from-onnx-mlir]
// [options]
//
// This tool links together:
// - HipDialect passes (OnnxToHip, HipToLLVM, GenerateInterface) - optional with
// --from-onnx-mlir
// - LLVM Backend (MLIR→IR translation, optimization, object compilation)
// - DLL Linker (Object→DLL linking)

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "../../lib/Backend/DLLLinker.h"
#include "../../lib/Backend/LLVMBackend.h"

// Include HIP dialect and passes
#include "../../lib/HipDialect/HipDialect.h"
#include "../../lib/HipDialect/HipPasses.h"

// Include ONNX dialect from onnx-mlir
#include "src/Dialect/ONNX/ONNXDialect.hpp"

// Include MLIR pass headers
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <string>

using namespace llvm;

// Helper function to check if file exists (LLVM 22 compatible)
static bool fileExists(const std::string& path) {
  llvm::sys::fs::file_status status;
  std::error_code EC = llvm::sys::fs::status(path, status);
  return !EC && llvm::sys::fs::exists(status);
}

// Command line options (manual parsing to avoid conflicts with LLD's
// CommandLine usage)
struct Options {
  std::string inputFilename;
  std::string outputFilename = "output.dll";
  std::string outputMode = "dll";
  int optLevel = 2;
  bool verbose = false;
  bool keepIntermediates = false;
  bool fromOnnxMlir = false;

  bool parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-o" && i + 1 < argc) {
        outputFilename = argv[++i];
      } else if (arg == "--mode" && i + 1 < argc) {
        outputMode = argv[++i];
      } else if (arg == "-O" && i + 1 < argc) {
        optLevel = std::stoi(argv[++i]);
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else if (arg == "--keep") {
        keepIntermediates = true;
      } else if (arg == "--from-onnx-mlir") {
        fromOnnxMlir = true;
      } else if (arg == "-h" || arg == "--help") {
        return false; // Trigger help
      } else if (arg[0] != '-') {
        inputFilename = arg;
      } else {
        std::cerr << "Unknown option: " << arg << "\n";
        return false;
      }
    }
    return !inputFilename.empty();
  }

  void printHelp() const {
    std::cout
        << "MLIR to HIP DLL Compiler\n\n"
        << "Usage: mlir-hip-compiler [options] <input.mlir>\n\n"
        << "Options:\n"
        << "  -o <file>          Output DLL filename (default: output.dll)\n"
        << "  --mode <mode>      Output mode: ir, object, dll (default: dll)\n"
        << "  -O <level>         Optimization level 0-3 (default: 2)\n"
        << "  -v, --verbose      Enable verbose output\n"
        << "  --keep             Keep intermediate files (.ll, .obj)\n"
        << "  --from-onnx-mlir   Process ONNX MLIR dialect\n"
        << "  -h, --help         Show this help\n";
  }
};

int main(int argc, char** argv) {
  // Parse command line options BEFORE InitLLVM to avoid CommandLine conflicts
  // with LLD
  Options opts;
  if (!opts.parse(argc, argv)) {
    opts.printHelp();
    return 1;
  }

  InitLLVM X(argc, argv);

  // Register HIP passes (MUST be done before creating PassManager)
  llvm::errs() << "[DEBUG] Registering HIP passes\n";
  mlir::hip::registerHipPasses();
  llvm::errs() << "[DEBUG] HIP passes registered\n";

  if (opts.verbose) {
    std::cout << "=== MLIR to HIP DLL Compiler ===\n";
    std::cout << "Input: " << opts.inputFilename << "\n";
    std::cout << "Output: " << opts.outputFilename << "\n";
    std::cout << "Mode: " << opts.outputMode << "\n";
    std::cout << "Optimization: O" << opts.optLevel << "\n\n";
  }

  // Initialize MLIR context and register dialects
  mlir::MLIRContext context;

  // Register all dialects (loading unused dialects is cheap and makes the tool
  // more flexible)
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<mlir::hip::HipDialect>();
  context.loadDialect<mlir::ONNXDialect>();

  mlir::registerLLVMDialectTranslation(context);

  // Parse input MLIR file
  if (opts.verbose)
    std::cout << "--- Step 1: Parsing MLIR ---\n";

  std::string errorMessage;
  auto file = mlir::openInputFile(opts.inputFilename, &errorMessage);
  if (!file) {
    std::cerr << "Error opening input file: " << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Error parsing MLIR file\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ MLIR parsed successfully\n\n";

  // Run MLIR transformation passes
  if (opts.verbose)
    std::cout << "--- Step 2: Running MLIR Passes ---\n";

  mlir::PassManager pm(&context);

  // Add our custom passes if processing ONNX-MLIR
  if (opts.fromOnnxMlir) {
    // ONNX → HIP conversion
    pm.addPass(mlir::hip::createConvertOnnxToHipPass());

    // BufferDeallocation pipeline (MLIR standard)
    // Note: API changed in newer MLIR - use individual passes instead of
    // pipeline builder These are function-level passes, so use
    // pm.nest<func::FuncOp>()
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::bufferization::createBufferLoopHoistingPass());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::bufferization::createOptimizeAllocationLivenessPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Memory pooling optimization
    // IMPORTANT: Must run AFTER BufferDeallocation, since BufferDeallocation
    // creates the final hip.alloc/hip.free ops
    llvm::errs() << "[DEBUG] About to call createMemoryPoolingPass()\n";
    pm.addPass(mlir::hip::createMemoryPoolingPass());
    llvm::errs()
        << "[DEBUG] createMemoryPoolingPass() returned, pass added to PM\n";

    // HIP → LLVM conversion
    pm.addPass(mlir::hip::createConvertHipToLLVMPass());
    pm.addPass(mlir::hip::createGenerateInterfacePass());

    if (opts.verbose) {
      std::cout
          << "Running ONNX→HIP→BufferDeallocation→LLVM→Interface passes\n";
    }
  } else {
    if (opts.verbose) {
      std::cout
          << "Skipping passes - assuming input is already in LLVM dialect\n";
    }
  }

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Error running MLIR passes\n";
    // Print the module IR to help debug
    if (opts.verbose) {
      llvm::errs() << "\n=== Failed Module IR ===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n========================\n";
    }
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ MLIR passes completed\n\n";

  // Translate MLIR to LLVM IR
  if (opts.verbose)
    std::cout << "--- Step 3: Translating to LLVM IR ---\n";

  hipdnn::LLVMBackend backend;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      backend.translateMLIRtoLLVMIR(*module, llvmContext);

  if (!llvmModule) {
    std::cerr << "Error translating MLIR to LLVM IR\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ LLVM IR generated\n\n";

  // Link Runtime module for zero-cost abstraction
  if (opts.verbose)
    std::cout << "--- Step 3.5: Linking Runtime Module ---\n";

  if (!backend.linkRuntimeModule(llvmModule.get())) {
    std::cerr << "Error linking Runtime module\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ Runtime module linked (enables cross-module inlining)\n\n";

  // Optimize LLVM IR
  if (opts.verbose)
    std::cout << "--- Step 4: Optimizing LLVM IR (O" << opts.optLevel
              << ") ---\n";
  backend.optimizeLLVMIR(llvmModule.get(), opts.optLevel);
  if (opts.verbose)
    std::cout << "✓ Optimization completed (Runtime calls inlined)\n\n";

  // Emit LLVM IR to file (if requested or keeping intermediates)
  std::string llFilename = opts.outputFilename;
  if (llFilename.size() >= 4 &&
      llFilename.substr(llFilename.size() - 4) == ".dll") {
    llFilename = llFilename.substr(0, llFilename.size() - 4) + ".ll";
  } else {
    llFilename += ".ll";
  }

  if (opts.outputMode == "ir" || opts.keepIntermediates) {
    if (opts.verbose)
      std::cout << "--- Step 5: Emitting LLVM IR ---\n";

    if (!backend.emitLLVMIR(llvmModule.get(), llFilename)) {
      std::cerr << "Error emitting LLVM IR\n";
      return 1;
    }

    if (opts.verbose)
      std::cout << "✓ LLVM IR written to: " << llFilename << "\n\n";

    if (opts.outputMode == "ir") {
      std::cout << "Output: " << llFilename << "\n";
      return 0;
    }
  }

  // Compile to object file
  std::string objFilename = opts.outputFilename;
  if (objFilename.size() >= 4 &&
      objFilename.substr(objFilename.size() - 4) == ".dll") {
    objFilename = objFilename.substr(0, objFilename.size() - 4) + ".obj";
  } else {
    objFilename += ".obj";
  }

  if (opts.verbose)
    std::cout << "--- Step 6: Compiling to Object File ---\n";

  if (!backend.compileToObjectFile(llvmModule.get(), objFilename)) {
    std::cerr << "Error compiling to object file\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ Object file created: " << objFilename << "\n\n";

  if (opts.outputMode == "object") {
    std::cout << "Output: " << objFilename << "\n";
    return 0;
  }

  // Link to DLL
  if (opts.verbose)
    std::cout << "--- Step 7: Linking to DLL ---\n";

  hipdnn::DLLLinker linker;

  // Define exported functions
  std::vector<std::string> exports = {"inference_init", "inference_compute",
                                      "inference_cleanup", "test_hip_from_dll"};

  // External libraries needed for real runtime (HIP/MIOpen/hipBLASLt)
  // Mock runtime doesn't need these as it only prints debug messages
  std::vector<std::string> libraries = {};
  std::vector<std::string> libraryPaths = {};

#ifndef BUILD_MOCK_RUNTIME
  // Real runtime requires ROCm libraries
  const char* therock_dist = std::getenv("THEROCK_DIST");
  if (therock_dist) {
    std::string therock_lib = std::string(therock_dist) + "/lib";
    libraryPaths.push_back(therock_lib);

    // Add ROCm libraries (import libraries for HIP/MIOpen/hipBLAS)
    libraries.push_back("amdhip64"); // HIP runtime
    libraries.push_back("MIOpen");   // MIOpen DNN library
    libraries.push_back("hipblas");  // hipBLAS library
    libraries.push_back(
        "hipblaslt"); // hipBLASLt library (import lib generated from DLL)
  } else {
    std::cerr << "Warning: THEROCK_DIST not set - DLL may fail to load if ROCm "
                 "libraries not in PATH\n";
  }
#endif

  if (!linker.linkDLL(objFilename, opts.outputFilename, libraries, libraryPaths,
                      exports)) {
    std::cerr << "Error linking DLL\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ DLL created: " << opts.outputFilename << "\n\n";

  // Verify DLL exports
  if (opts.verbose) {
    std::cout << "--- Step 8: Verifying DLL Exports ---\n";
    if (linker.verifyDLLExports(opts.outputFilename, exports)) {
      std::cout << "✓ All expected exports present\n\n";
    } else {
      std::cout
          << "⚠ Could not verify exports (dumpbin may not be available)\n\n";
    }
  }

  // Clean up intermediate files if not keeping
  if (!opts.keepIntermediates) {
    if (opts.verbose)
      std::cout << "--- Cleaning up intermediate files ---\n";

    if (fileExists(llFilename)) {
      std::error_code EC = llvm::sys::fs::remove(llFilename);
      if (!EC && opts.verbose)
        std::cout << "Removed: " << llFilename << "\n";
    }

    if (fileExists(objFilename)) {
      std::error_code EC = llvm::sys::fs::remove(objFilename);
      if (!EC && opts.verbose)
        std::cout << "Removed: " << objFilename << "\n";
    }

    if (opts.verbose)
      std::cout << "\n";
  }

  std::cout << "=== Compilation Successful ===\n";
  std::cout << "Output: " << opts.outputFilename << "\n";

  return 0;
}
