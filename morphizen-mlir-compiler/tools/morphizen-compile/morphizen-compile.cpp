/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR to DLL Compiler
// Thin wrapper around CompilerDriver library

#include "compilation_options.pb.h"
#include "morphizen-mlir-compiler/Compiler/CompilerDriver.h"

#include "llvm/Support/InitLLVM.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace morphizen::mlir_compiler;

// Command line options (manual parsing to avoid conflicts with LLD)
struct Options {
  std::string inputFilename;
  std::string outputFilename = "output.dll";
  std::string outputMode = "dll";
  int optLevel = 2;
  bool verbose = false;

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
        << "MLIR to HIP DLL Compiler\n\n"
        << "Usage: morphizen-compile [options] <input.mlir>\n\n"
        << "Options:\n"
        << "  -o <file>          Output DLL filename (default: output.dll)\n"
        << "  --mode <mode>      Output mode: ir, dll (default: dll)\n"
        << "  -O <level>         Optimization level 0-3 (default: 2)\n"
        << "  -v, --verbose      Enable verbose output\n"
        << "  -h, --help         Show this help\n";
  }
};

int main(int argc, char** argv) {
  // Parse command line options BEFORE InitLLVM
  Options opts;
  if (!opts.parse(argc, argv)) {
    opts.printHelp();
    return 1;
  }

  llvm::InitLLVM X(argc, argv);

  if (opts.verbose) {
    std::cout << "=== MLIR to HIP DLL Compiler ===\n";
    std::cout << "Input: " << opts.inputFilename << "\n";
    std::cout << "Output: " << opts.outputFilename << "\n";
    std::cout << "Mode: " << opts.outputMode << "\n";
    std::cout << "Optimization: O" << opts.optLevel << "\n\n";
  }

  // Read input file
  std::ifstream inputFile(opts.inputFilename);
  if (!inputFile) {
    std::cerr << "Error: Cannot open input file: " << opts.inputFilename
              << "\n";
    return 1;
  }

  std::stringstream buffer;
  buffer << inputFile.rdbuf();
  std::string inputMLIR = buffer.str();

  // Setup compiler options
  CompilationOptions compilerOpts;
  compilerOpts.set_opt_level(opts.optLevel);
  compilerOpts.set_verbose(opts.verbose);

  // Parse output mode
  if (opts.outputMode == "ir") {
    compilerOpts.set_output_mode(OUTPUT_MODE_LLVM_IR);
  } else if (opts.outputMode == "dll") {
    compilerOpts.set_output_mode(OUTPUT_MODE_DLL);
  } else {
    std::cerr << "Error: Unknown output mode: " << opts.outputMode << "\n";
    return 1;
  }

  // Compile using CompilerDriver
  CompilerDriver pipeline;
  std::string errorMessage;

  bool success = pipeline.compile(inputMLIR, opts.outputFilename, compilerOpts,
                                  errorMessage);

  if (!success) {
    std::cerr << "Compilation failed: " << errorMessage << "\n";
    return 1;
  }

  std::cout << "=== Compilation Successful ===\n";
  std::cout << "Output: " << opts.outputFilename << "\n";

  return 0;
}
