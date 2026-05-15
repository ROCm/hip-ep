/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef BITCODE_JIT_H
#define BITCODE_JIT_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir_compilation::customop {

// BitcodeJIT — in-process loader for the per-model LLVM bitcode artifact
// shipped inside an EPContext tar (`model_compiled` entry).
//
// Replaces the previous "write unsigned PE to TEMP / LoadLibrary / delete"
// dance in `InferenceState::create()` with a fully-in-process ORC `LLJIT`
// pipeline. The per-model artifact is now data (a `.bc` blob) rather than
// code, so it does not need to be code-signed and does not trip WDAC /
// EDR reflective-DLL heuristics. AMD continues to sign only
// `onnxruntime_morphizen_ep.dll` and `hip-compiler.dll` (which together
// embed `runtime.bc` and the GPU fatbin).
//
// External symbols referenced by the bitcode are resolved at JIT-link
// time via ORC's `DynamicLibrarySearchGenerator::GetForCurrentProcess`,
// which walks every module already loaded in the host process. In
// practice that means:
//   * `hip_*` GPU kernel launchers (exported from the EP DLL itself --
//     each declaration in
//     `3rd-party/custom_kernels/include/hip_custom_kernels.h` is tagged with
//     `HIP_KERNEL_API` (= `__declspec(dllexport)` on Windows), and the static
//     lib is linked WHOLE_ARCHIVE into the EP DLL)
//   * `libamdhip64.dll` HIP runtime entry points
//   * `MIOpen.dll`, `hipblaslt.dll`
//   * the MSVC / GNU C runtime (`memcpy`, `memset`, exception machinery, ...)
//
// The runtime wrapper functions in `lib/Runtime/` (`hipdnn_ep_*`,
// `wrap_*`) are pre-merged into the bitcode at compile time via
// `LLVMBackend::linkRuntimeModule`, so they are internal symbols of the
// JIT module and require no host-process lookup.
class BitcodeJIT {
public:
  ~BitcodeJIT();

  BitcodeJIT(const BitcodeJIT &) = delete;
  BitcodeJIT &operator=(const BitcodeJIT &) = delete;
  BitcodeJIT(BitcodeJIT &&) = delete;
  BitcodeJIT &operator=(BitcodeJIT &&) = delete;

  // Build a JIT instance from in-memory LLVM bitcode. The bytes are
  // copied into an owned MemoryBuffer, so the caller can free `bitcode`
  // immediately. `module_name` is used purely for diagnostics
  // (`Module::setSourceFileName`).
  //
  // Returns nullptr on parse / JIT-init failure (errors logged via glog).
  static std::unique_ptr<BitcodeJIT> create(const std::vector<uint8_t> &bitcode,
                                            const std::string &module_name);

  // Resolve a symbol by name. Returns the JIT-emitted host address, or
  // nullptr if the symbol is undefined. Absent symbols are silently
  // tolerated (`llvm::consumeError` on the lookup `Error`) because
  // InferenceState probes for optional hooks such as
  // `hipdnn_ep_runtime_begin_compute`.
  void *lookup_raw(const char *name) const;

  // Typed convenience that mirrors `morphizen::Plugin::get_method<R,
  // Args...>`, so call sites inside InferenceState can switch loaders
  // with minimal churn.
  template <typename R, typename... Args>
  auto get_method(const char *name) const -> R (*)(Args...) {
    return reinterpret_cast<R (*)(Args...)>(lookup_raw(name));
  }

private:
  // Use create().
  BitcodeJIT();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mlir_compilation::customop

#endif // BITCODE_JIT_H
