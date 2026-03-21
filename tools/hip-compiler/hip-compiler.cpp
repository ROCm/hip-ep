/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "compilation_options_generated.h"
#include "hip/Compiler/CompilerDriver.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  std::string inputFilename;
  std::string outputDll;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outputDll = argv[++i];
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }

  if (inputFilename.empty() || outputDll.empty()) {
    llvm::errs() << "Usage: " << argv[0] << " <input.mlir> -o <output.dll>\n";
    return 1;
  }

  auto bufOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot open '" << inputFilename
                 << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }

  mlir::hip::CompilationOptionsT options;
  std::string errorMessage;
  hip::compiler::CompilerDriver driver;

  if (!driver.compile((*bufOrErr)->getBuffer(), outputDll, options,
                      errorMessage)) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  llvm::outs() << "Successfully generated " << outputDll << "\n";
  return 0;
}
