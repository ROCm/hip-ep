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
ptr Scall0(ptr);
ptr Scall1(ptr, ptr);
ptr Scall2(ptr, ptr, ptr);
ptr Scall3(ptr, ptr, ptr, ptr);
ptr Sstring_to_symbol(const char*);
ptr Stop_level_value(ptr);
ptr Sinteger(iptr);
iptr Sinteger_value(ptr);
ptr Sstring(const char*);
ptr Scons(ptr, ptr);
const char* Skernel_version();

#define Snil ((ptr)0x26)
}

#include "ChezBootPetite.h"
#include "ChezBootScheme.h"
#include "PrintOperationScm.h"

namespace {
const size_t petite_boot_size = sizeof(petite_boot_data) - 1;
const size_t scheme_boot_size = sizeof(scheme_boot_data) - 1;
const size_t print_operation_scm_size = sizeof(print_operation_scm_data) - 1;
}

namespace {
static bool scheme_initialized = false;
static ptr g_print_operation = nullptr;
}

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime() {
  if (scheme_initialized)
    return true;

  llvm::errs() << "Initializing Chez Scheme runtime...\n";
  llvm::errs() << "  Version: " << Skernel_version() << "\n";
  llvm::errs() << "  Petite boot: " << petite_boot_size << " bytes\n";
  llvm::errs() << "  Scheme boot: " << scheme_boot_size << " bytes\n";

  Sscheme_init(nullptr);
  Sregister_boot_file_bytes("petite.boot", petite_boot_data, petite_boot_size);
  Sregister_boot_file_bytes("scheme.boot", scheme_boot_data, scheme_boot_size);
  Sbuild_heap("hip-mlir-opt", nullptr);

  ptr multiply = Stop_level_value(Sstring_to_symbol("*"));
  ptr result = Scall2(multiply, Sinteger(6), Sinteger(7));
  iptr answer = Sinteger_value(result);

  llvm::errs() << "Scheme test: (* 6 7) = " << answer << "\n";

  std::string scm_code(reinterpret_cast<const char*>(print_operation_scm_data),
                       print_operation_scm_size);
  ptr eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  ptr read_sym = Stop_level_value(Sstring_to_symbol("read"));
  ptr open_string_input_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));

  ptr port = Scall1(open_string_input_port_sym, Sstring(scm_code.c_str()));
  ptr expr = Scall1(read_sym, port);
  Scall1(eval_sym, expr);

  g_print_operation = Stop_level_value(Sstring_to_symbol("print-operation"));

  llvm::errs() << "Chez Scheme runtime initialized successfully!\n\n";

  scheme_initialized = true;
  return true;
}

void printOperation(const char* opName, int numOperands, int numResults, const char* genericForm) {
  if (!scheme_initialized || !g_print_operation)
    return;

  ptr args_list = Scons(Sstring(opName),
                  Scons(Sinteger(numOperands),
                  Scons(Sinteger(numResults),
                  Scons(Sstring(genericForm), Snil))));

  ptr apply_proc = Stop_level_value(Sstring_to_symbol("apply"));
  Scall2(apply_proc, g_print_operation, args_list);
}

} // namespace hipsr
} // namespace mlir
