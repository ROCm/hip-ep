/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstring>

extern "C" {
typedef void* ptr;
typedef long iptr;

void Sscheme_init(void (*)(void));
void Sregister_boot_file_bytes(const char*, const unsigned char*, size_t);
const char* Skernel_version();

extern const unsigned char chez_boot_data[];
extern const size_t chez_boot_size;
}

namespace {
static bool scheme_initialized = false;
}

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime() {
  if (scheme_initialized)
    return true;

  llvm::errs() << "Initializing Chez Scheme C API...\n";
  llvm::errs() << "Boot size: " << chez_boot_size << " bytes\n";
  llvm::errs() << "Chez Scheme version: " << Skernel_version() << "\n";

  Sscheme_init(nullptr);
  Sregister_boot_file_bytes("hip-patterns.boot", chez_boot_data, chez_boot_size);

  llvm::errs() << "Chez Scheme C API linked successfully.\n";
  llvm::errs() << "\nNote: Full Scheme evaluation (Sbuild_heap) requires additional\n";
  llvm::errs() << "integration work for embedded use. Current implementation demonstrates:\n";
  llvm::errs() << "  - Boot file embedding (1.16 MB)\n";
  llvm::errs() << "  - C API linkage (libkernel.a, liblz4.a, libz.a)\n";
  llvm::errs() << "  - Scheme function calls will use C++ for now\n";
  llvm::errs() << "\nFuture work: Complete Sbuild_heap initialization and actual\n";
  llvm::errs() << "Scheme-based pattern DSL implementation.\n\n";

  scheme_initialized = true;
  return true;
}

void printOperation(const char* opName, int numOperands, int numResults, const char* genericForm) {
  if (!scheme_initialized)
    return;

  llvm::errs() << "Operation: \"" << opName << "\"\n";
  llvm::errs() << "  Operands: " << numOperands << "\n";
  llvm::errs() << "  Results: " << numResults << "\n";
  llvm::errs() << "  Generic form: " << genericForm << "\n\n";
}

} // namespace hipsr
} // namespace mlir
