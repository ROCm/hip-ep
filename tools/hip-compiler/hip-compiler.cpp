/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Compiler/CompilerDriver.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  std::string inputFilename;
  std::string outputDll;
  for (int argIdx = 1; argIdx < argc; ++argIdx) {
    if (std::string(argv[argIdx]) == "-o" && argIdx + 1 < argc) {
      outputDll = argv[++argIdx];
    } else if (argv[argIdx][0] != '-') {
      inputFilename = argv[argIdx];
    }
  }

  if (inputFilename.empty() || outputDll.empty()) {
    llvm::errs() << "Usage: " << argv[0] << " <input.mlir> -o <output.dll>\n";
    return 1;
  }

  auto fileOrErr = llvm::MemoryBuffer::getFile(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "error: could not open " << inputFilename << ": "
                 << fileOrErr.getError().message() << "\n";
    return 1;
  }

  mlir::hip::CompilationOptionsT options;
  std::string errorMessage;
  hip::compiler::CompilerDriver driver;

  if (!driver.compile((*fileOrErr)->getBuffer(), outputDll, options,
                      errorMessage)) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  llvm::outs() << "Successfully generated " << outputDll << "\n";
  return 0;
}
