/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipCompiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cstring>
#include <iostream>

using namespace hipdnn::compiler;

static void printUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " [--standalone|--plugin] <input.hip.mlir> -o <output.dll>\n"
            << "\n"
            << "Modes:\n"
            << "  --standalone  (default) Export raw compute functions\n"
            << "  --plugin      Export inference_init/compute/cleanup for EP\n";
}

int main(int argc, char **argv) {
  std::string inputFilename;
  std::string outputDll;
  CompileMode mode = CompileMode::Standalone;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      outputDll = argv[++i];
    } else if (std::strcmp(argv[i], "--standalone") == 0) {
      mode = CompileMode::Standalone;
    } else if (std::strcmp(argv[i], "--plugin") == 0) {
      mode = CompileMode::Plugin;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      printUsage(argv[0]);
      return 0;
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }

  if (inputFilename.empty() || outputDll.empty()) {
    printUsage(argv[0]);
    return 1;
  }

  CompileOptions options;
  options.mode = mode;

  // Derive runtime lib directory from the executable's location
  std::string exePath =
      llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
  options.runtimeLibDir = llvm::sys::path::parent_path(exePath).str();

  std::cout << "Mode: " << (mode == CompileMode::Plugin ? "plugin" : "standalone")
            << "\n";

  auto result = HipCompiler::compileFile(inputFilename, outputDll, options);
  if (!result) {
    std::cerr << "Compilation failed\n";
    return 1;
  }

  std::cout << "DLL size: " << result->dllBytes.size() << " bytes\n";
  return 0;
}
