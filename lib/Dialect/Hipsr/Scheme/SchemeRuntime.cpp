/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstring>

// Include Chez Scheme C API header - use the ta6le machine-specific version
// where ptr is defined as void*, not the portable boot (pb) version
extern "C" {
#include "boot/ta6le/scheme.h"
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
  Sregister_boot_file_bytes("petite.boot", const_cast<void*>(static_cast<const void*>(petite_boot_data)), petite_boot_size);
  Sregister_boot_file_bytes("scheme.boot", const_cast<void*>(static_cast<const void*>(scheme_boot_data)), scheme_boot_size);
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
  ptr eof_object_p = Stop_level_value(Sstring_to_symbol("eof-object?"));

  ptr port = Scall1(open_string_input_port_sym, Sstring(scm_code.c_str()));

  while (true) {
    ptr expr = Scall1(read_sym, port);
    // Check if we hit EOF using eof-object? predicate
    if (Scall1(eof_object_p, expr) != Sfalse)
      break;
    Scall1(eval_sym, expr);
  }

  llvm::errs() << "Chez Scheme runtime initialized successfully!\n\n";

  scheme_initialized = true;
  return true;
}

std::string callSchemeFunction(const char* functionName,
                                const std::vector<void*>& args) {
  if (!scheme_initialized)
    return "";

  ptr func = Stop_level_value(Sstring_to_symbol(functionName));
  if (func == Sfalse)
    return "";

  ptr args_list = Snil;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    args_list = Scons(*it, args_list);
  }

  ptr apply_proc = Stop_level_value(Sstring_to_symbol("apply"));
  ptr result = Scall2(apply_proc, func, args_list);

  ptr string_p = Stop_level_value(Sstring_to_symbol("string?"));
  if (Scall1(string_p, result) != Sfalse) {
    // Extract string using macros - Chez strings are 32-bit chars, convert to C string
    iptr len = Sstring_length(result);
    std::string str;
    str.reserve(len);
    for (iptr i = 0; i < len; i++) {
      str.push_back(static_cast<char>(Sstring_ref(result, i)));
    }
    return str;
  }

  return "";
}

void* makeSchemeString(const char* str) {
  return Sstring(str);
}

void* makeSchemeInteger(long value) {
  return Sinteger(value);
}

} // namespace hipsr
} // namespace mlir
