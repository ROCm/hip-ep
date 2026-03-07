/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR/ONNX to DLL Compiler
// Supports both .mlir and .onnx input files

#include "hip/Compiler/CompilerDriver.h"
#include "hip/InitAllPasses.h"
#include "hip/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"

#ifdef ENABLE_ONNX_FRONTEND
#include "src/Builder/FrontendDialectTransformer.hpp"
#endif

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>

using namespace udna::compiler;

struct Options {
  std::string inputFilename;
  std::string outputFilename = "output.dll";
  bool outputSpecified = false;
  // CLI-only: needed to detect "onnx-mlir" special mode that exits before
  // compilation and has no equivalent in CompilationOptionsT.
  std::string outputModeStr = "dll";
  // CLI-only: directory for DiskFileSystem; never embedded in the DLL.
  // Created automatically if it does not exist.
  std::string constantsDir;
  // All other compilation settings parsed directly into the proto struct so
  // there is no duplication or manual copy step.
  CompilationOptionsT compilerOpts;

  bool parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-o" && i + 1 < argc) {
        outputFilename = argv[++i];
        outputSpecified = true;
      } else if (arg == "--mode" && i + 1 < argc) {
        outputModeStr = argv[++i];
        if (outputModeStr == "ir") {
          compilerOpts.output_mode = OutputMode::LLVM_IR;
        } else if (outputModeStr == "dll") {
          compilerOpts.output_mode = OutputMode::DLL;
        } else if (outputModeStr != "onnx-mlir") {
          std::cerr << "Unknown mode: " << outputModeStr << "\n";
          return false;
        }
      } else if (arg == "--constants-file" && i + 1 < argc) {
        compilerOpts.constants_file = argv[++i];
      } else if (arg == "--constants-dir" && i + 1 < argc) {
        constantsDir = argv[++i];
      } else if (arg == "-O" && i + 1 < argc) {
        compilerOpts.opt_level = std::stoi(argv[++i]);
      } else if (arg == "-v" || arg == "--verbose") {
        compilerOpts.verbose = true;
      } else if (arg == "-h" || arg == "--help") {
        return false;
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
        << "MLIR/ONNX to HIP DLL Compiler\n\n"
        << "Usage: hip-compiler [options] <input.mlir|input.onnx>\n\n"
        << "Options:\n"
        << "  -o <file>                Output filename (default: output.dll)\n"
        << "  --mode <mode>            Output mode: ir, dll, onnx-mlir "
           "(default: dll)\n"
        << "  --constants-file <name>  Filename for constants data (default: "
           "constants.bin);\n"
        << "                           stored in DLL metadata so the runtime "
           "can load it\n"
        << "  --constants-dir <dir>    Directory to write the constants file "
           "into;\n"
        << "                           created if it does not exist (default: "
           "current dir)\n"
        << "  -O <level>               Optimization level 0-3 (default: 2)\n"
        << "  -v, --verbose            Enable verbose output\n"
        << "  -h, --help               Show this help\n"
        << "\nInput formats:\n"
        << "  .mlir              MLIR text (ONNX dialect)\n"
#ifdef ENABLE_ONNX_FRONTEND
        << "  .onnx              ONNX model\n"
#endif
        << "\nOutput modes:\n"
        << "  dll                Compile to DLL (default)\n"
        << "  ir                 Emit LLVM IR\n"
#ifdef ENABLE_ONNX_FRONTEND
        << "  onnx-mlir          Import ONNX and emit ONNX dialect MLIR\n"
        << "                     (prints to stdout; use -o to write to file)\n"
#endif
        ;
  }

  bool isOnnxInput() const {
    return inputFilename.size() >= 5 &&
           inputFilename.substr(inputFilename.size() - 5) == ".onnx";
  }
};

// Create constantsDir (and any missing parents) then return a DiskFileSystem
// rooted there. Falls back to "." when constantsDir is empty.
static udna::DiskFileSystem makeFileSystem(const std::string& constantsDir) {
  if (!constantsDir.empty())
    llvm::sys::fs::create_directories(constantsDir);
  return udna::DiskFileSystem(constantsDir.empty() ? "." : constantsDir.c_str());
}

int main(int argc, char** argv) {
  Options opts;
  if (!opts.parse(argc, argv)) {
    opts.printHelp();
    return 1;
  }

  llvm::InitLLVM X(argc, argv);

  if (opts.compilerOpts.verbose) {
    std::cout << "=== MLIR/ONNX to HIP DLL Compiler ===\n";
    std::cout << "Input: " << opts.inputFilename << "\n";
    std::cout << "Output: " << opts.outputFilename << "\n";
    std::cout << "Mode: " << opts.outputModeStr << "\n";
    std::cout << "Optimization: O" << opts.compilerOpts.opt_level << "\n";
    std::cout << "Constants file: " << opts.compilerOpts.constants_file << "\n";
    if (!opts.constantsDir.empty())
      std::cout << "Constants dir: " << opts.constantsDir << "\n";
    std::cout << "\n";
  }

#ifdef ENABLE_ONNX_FRONTEND
  // Handle ONNX model import
  if (opts.isOnnxInput()) {
    mlir::MLIRContext context;
    udna::compiler::loadAllDialects(context);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string errorMessage;

    onnx_mlir::ImportOptions importOpts;
    importOpts.useOnnxModelTypes = true;
    int importResult = onnx_mlir::ImportFrontendModelFile(
        llvm::StringRef(opts.inputFilename), context, module, &errorMessage,
        importOpts);
    if (importResult != 0) {
      std::cerr << "ONNX import failed: " << errorMessage << "\n";
      return 1;
    }

    if (opts.outputModeStr == "onnx-mlir") {
      if (opts.outputSpecified) {
        std::error_code ec;
        llvm::raw_fd_ostream outFile(opts.outputFilename, ec);
        if (ec) {
          std::cerr << "Error: Cannot open output file: " << opts.outputFilename
                    << ": " << ec.message() << "\n";
          return 1;
        }
        module->print(outFile);
        outFile << "\n";
        std::cout << "MLIR written to: " << opts.outputFilename << "\n";
      } else {
        module->print(llvm::outs());
        llvm::outs() << "\n";
      }
      return 0;
    }

    CompilerDriver pipeline;
    auto fs = makeFileSystem(opts.constantsDir);
    pipeline.setFileSystem(&fs);

    std::string errorMsg;
    if (!pipeline.compileFromModule(*module, opts.outputFilename,
                                   opts.compilerOpts, errorMsg)) {
      std::cerr << "Compilation failed: " << errorMsg << "\n";
      return 1;
    }

    std::cout << "=== Compilation Successful ===\n";
    std::cout << "Output: " << opts.outputFilename << "\n";
    return 0;
  }
#else
  if (opts.isOnnxInput()) {
    std::cerr << "Error: .onnx input requires ENABLE_ONNX_FRONTEND build\n";
    return 1;
  }
  if (opts.outputModeStr == "onnx-mlir") {
    std::cerr << "Error: --mode onnx-mlir requires ENABLE_ONNX_FRONTEND build\n";
    return 1;
  }
#endif

  // Standard MLIR input path — use MemoryBuffer (mmap) to avoid reading
  // the entire file into a std::string, which OOMs on large models.
  if (opts.outputModeStr == "onnx-mlir") {
    std::cerr << "Error: --mode onnx-mlir requires .onnx input file\n";
    return 1;
  }

  auto bufferOrErr = llvm::MemoryBuffer::getFile(opts.inputFilename);
  if (!bufferOrErr) {
    std::cerr << "Error: Cannot open input file: " << opts.inputFilename
              << ": " << bufferOrErr.getError().message() << "\n";
    return 1;
  }
  llvm::StringRef inputMLIR = (*bufferOrErr)->getBuffer();

  CompilerDriver pipeline;
  auto fs = makeFileSystem(opts.constantsDir);
  pipeline.setFileSystem(&fs);

  std::string errorMessage;
  if (!pipeline.compile(inputMLIR, opts.outputFilename, opts.compilerOpts,
                        errorMessage)) {
    std::cerr << "Compilation failed: " << errorMessage << "\n";
    return 1;
  }

  std::cout << "=== Compilation Successful ===\n";
  std::cout << "Output: " << opts.outputFilename << "\n";

  return 0;
}
