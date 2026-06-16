/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "compilation_options_generated.h"
#include "hip/Compiler/CompilerDriver.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "CrashHandler.h"

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-compiler");
  std::string inputFilename;
  std::string outputPath;
  std::string mode = "LLVM_IR"; // LLVM_IR (default) | NATIVE

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }

  if (inputFilename.empty() || outputPath.empty()) {
    llvm::errs() << "Usage: " << argv[0]
                 << " <input.mlir> -o <output.{bc,dll,so}> "
                    "[--mode LLVM_IR|NATIVE]\n";
    return 1;
  }

  if (mode != "LLVM_IR" && mode != "NATIVE") {
    llvm::errs() << "error: --mode must be 'LLVM_IR' or 'NATIVE'\n";
    return 1;
  }

  auto bufOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot open '" << inputFilename
                 << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }

  mlir::hip::CompilationOptionsT options;
  options.output_mode = (mode == "NATIVE") ? mlir::hip::OutputMode::NATIVE
                                           : mlir::hip::OutputMode::LLVM_IR;
  std::string errorMessage;
  hip::compiler::CompilerDriver driver;

  if (!driver.compile((*bufOrErr)->getBuffer(), outputPath, options,
                      errorMessage)) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  llvm::outs() << "Successfully generated " << outputPath << "\n";
  return 0;
}
