/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Public sample plugin used by hip-compiler's PR-1 plugin loader test.
//
// The point of this DLL is to exercise the public ABI surface declared
// in `include/hip/Compiler/PluginAPI.h` end-to-end in CI:
//
//   1. The DLL exports `hipEpGetPluginInfo` under its unmangled C name.
//   2. The struct it returns satisfies the version + name + version
//      contract that `HipEpPluginLoader::Load` validates.
//   3. The `RegisterCallbacks` function pointer fires correctly across
//      the DLL boundary with a `HipEpPluginRegistry &`.
//
// In PR 1 the registry methods are stubs; the callback is therefore
// intentionally empty. PRs 2-4 will grow this sample to register a
// no-op pass, contribute bitcode, and contribute a sample library so
// each capability has live coverage in CI.

#include "hip/Compiler/PluginAPI.h"

namespace {

// Forward-declared in PluginAPI.h. We do not need the full definition
// in a PR-1 sample plugin because the callback below does not call any
// registry method.
using ::hip::compiler::HipEpPluginRegistry;

void registerCallbacks(HipEpPluginRegistry & /*R*/) {
  // Intentionally empty in PR 1. PR 2 fills in:
  //   R.registerPass<SamplePrintFunctionsPass>();
  //   R.requestPipelineSlot(
  //       hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
  //       "sample-print-functions");
}

} // namespace

// LLVM_ATTRIBUTE_WEAK is a no-op on Windows; the CMake target sets
// `WINDOWS_EXPORT_ALL_SYMBOLS ON` so this symbol is exported under
// its unmangled C name. On non-Windows the weak attribute lets the
// same source link statically into a tool if we ever want to.
extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo() {
  return {
      HIP_EP_PLUGIN_API_VERSION,
      "HipEpSamplePlugin",
      "0.1.0",
      &registerCallbacks,
  };
}
