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
void Sbuild_heap(const char*, void (*)(void));
ptr Scall2(ptr, ptr, ptr);
ptr Sstring_to_symbol(const char*);
ptr Stop_level_value(ptr);
ptr Sinteger(iptr);
iptr Sinteger_value(ptr);
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

  llvm::errs() << "Initializing Chez Scheme runtime...\n";
  llvm::errs() << "  Version: " << Skernel_version() << "\n";
  llvm::errs() << "  Boot size: " << chez_boot_size << " bytes\n";

  Sscheme_init(nullptr);
  Sregister_boot_file_bytes("hip-patterns.boot", chez_boot_data, chez_boot_size);
  Sbuild_heap(nullptr, nullptr);

  ptr multiply = Stop_level_value(Sstring_to_symbol("*"));
  ptr result = Scall2(multiply, Sinteger(6), Sinteger(7));
  iptr answer = Sinteger_value(result);

  llvm::errs() << "Scheme test: (* 6 7) = " << answer << "\n";
  llvm::errs() << "Chez Scheme runtime initialized successfully!\n\n";

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
