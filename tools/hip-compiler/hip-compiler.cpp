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

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }

  if (inputFilename.empty() || outputPath.empty()) {
    // Output is LLVM bitcode (the BitcodeJIT-loaded per-model artifact).
    // .bc is the conventional extension; callers may use anything --
    // CompilerDriver writes the same bytes regardless.
    llvm::errs() << "Usage: " << argv[0] << " <input.mlir> -o <output.bc>\n";
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

  if (!driver.compile((*bufOrErr)->getBuffer(), outputPath, options,
                      errorMessage)) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  llvm::outs() << "Successfully generated " << outputPath << "\n";
  return 0;
}
