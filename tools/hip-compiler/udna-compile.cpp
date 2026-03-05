/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR/ONNX to DLL Compiler
// Supports both .mlir and .onnx input files

#include "udna-compiler/Compiler/CompilerDriver.h"
#include "udna-compiler/InitAllPasses.h"
#include "udna-compiler/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"

#include "src/Builder/FrontendDialectTransformer.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace udna::compiler;

struct Options {
  std::string inputFilename;
  std::string outputFilename = "output.dll";
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
        << "Usage: udna-compile [options] <input.mlir|input.onnx>\n\n"
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
        << "  .onnx              ONNX model\n"
        << "\nOutput modes:\n"
        << "  dll                Compile to DLL (default)\n"
        << "  ir                 Emit LLVM IR\n"
        << "  onnx-mlir          Import ONNX and print ONNX dialect MLIR\n";
  }

  bool isOnnxInput() const {
    return inputFilename.size() >= 5 &&
           inputFilename.substr(inputFilename.size() - 5) == ".onnx";
  }
};

static bool importOnnxModel(const std::string& filename,
                            mlir::MLIRContext& context,
                            mlir::OwningOpRef<mlir::ModuleOp>& module,
                            std::string& errorMessage) {
  onnx_mlir::ImportOptions importOpts;
  importOpts.useOnnxModelTypes = true;
  int result = onnx_mlir::ImportFrontendModelFile(
      llvm::StringRef(filename), context, module, &errorMessage, importOpts);
  return result == 0;
}

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

  // Handle ONNX model import
  if (opts.isOnnxInput()) {
    mlir::MLIRContext context;
    udna::compiler::loadAllDialects(context);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string errorMessage;

    if (!importOnnxModel(opts.inputFilename, context, module, errorMessage)) {
      std::cerr << "ONNX import failed: " << errorMessage << "\n";
      return 1;
    }

    if (opts.outputModeStr == "onnx-mlir") {
      module->print(llvm::outs());
      llvm::outs() << "\n";
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

  // Standard MLIR input path (binary mode for MLIR bytecode support)
  std::ifstream inputFile(opts.inputFilename, std::ios::binary);
  if (!inputFile) {
    std::cerr << "Error: Cannot open input file: " << opts.inputFilename
              << "\n";
    return 1;
  }

  if (opts.outputModeStr == "onnx-mlir") {
    std::cerr << "Error: --mode onnx-mlir requires .onnx input file\n";
    return 1;
  }

  std::string inputMLIR(
      (std::istreambuf_iterator<char>(inputFile)),
      std::istreambuf_iterator<char>());

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
