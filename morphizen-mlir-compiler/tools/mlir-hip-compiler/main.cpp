/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR to HIP DLL Compiler
// Now a thin wrapper around CompilerPipeline library

#include "../../lib/Compiler/CompilerPipeline.h"

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
  CompilerPipeline::Options compilerOpts;
  compilerOpts.from_onnx_mlir = opts.fromOnnxMlir;
  compilerOpts.opt_level = opts.optLevel;
  compilerOpts.verbose = opts.verbose;
  compilerOpts.keep_intermediates = opts.keepIntermediates;

  // Parse output mode
  if (opts.outputMode == "ir") {
    compilerOpts.output_mode = CompilerPipeline::Options::OutputMode::IR;
  } else if (opts.outputMode == "object") {
    compilerOpts.output_mode = CompilerPipeline::Options::OutputMode::Object;
  } else if (opts.outputMode == "dll") {
    compilerOpts.output_mode = CompilerPipeline::Options::OutputMode::DLL;
  } else {
    std::cerr << "Error: Unknown output mode: " << opts.outputMode << "\n";
    return 1;
  }

  // Add external libraries for real runtime (if not mock)
#ifndef USE_MOCK_RUNTIME
  const char* therock_dist = std::getenv("THEROCK_DIST");
  if (therock_dist) {
    std::string therock_lib = std::string(therock_dist) + "/lib";
    compilerOpts.library_paths.push_back(therock_lib);
    compilerOpts.libraries.push_back("amdhip64");
    compilerOpts.libraries.push_back("MIOpen");
    compilerOpts.libraries.push_back("hipblas");
    compilerOpts.libraries.push_back("hipblaslt");
  } else if (opts.verbose) {
    std::cout << "Warning: THEROCK_DIST not set - DLL may fail to load if "
                 "ROCm libraries not in PATH\n";
  }
#endif

  // Compile using CompilerPipeline
  CompilerPipeline pipeline;
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
