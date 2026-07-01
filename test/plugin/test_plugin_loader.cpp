/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Static compiler-plugin unit test.
//
// Exercises the public plugin surface declared in
//   include/hip/Compiler/PluginAPI.h
//   include/hip/Compiler/PluginRegistry.h
// via the static registrar (`dispatchPluginRegistrationsOnce`,
// lib/Compiler/StaticPlugins.cpp), which invokes each statically-linked
// plugin's `hipEpRegisterPlugin_<id>` entry point.
//
// The sample plugin (`test/plugin/sample_plugin/`) is linked into this build
// only when `sample` is in HIPDNN_EP_COMPILER_PLUGINS. CMake tells us which
// case we are in via HIP_EP_EXPECT_SAMPLE:
//
//   * HIP_EP_EXPECT_SAMPLE == 1 (sample selected): after dispatch, the registry
//     must record the sample's contributions --
//       - the AfterConvertOnnxToHip slot request
//         "func.func(hip-ep-sample-print-functions)",
//       - the "hip_ep_sample_lib" library + a non-empty search path,
//       - the LLVM bitcode buffer (magic 'BC\xc0\xde') when the build had clang
//         to compile it (empty buffer => skipped, degraded build).
//   * HIP_EP_EXPECT_SAMPLE == 0 (no plugins): dispatch is a clean no-op and the
//     registry stays empty.
//
// Either way `dispatchPluginRegistrationsOnce()` must be safe to call and
// idempotent. Plain `main()` (no GTest) so the test needs nothing the public
// configure does not already provide.

#include "hip/Compiler/PluginRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <string>

#ifndef HIP_EP_EXPECT_SAMPLE
#define HIP_EP_EXPECT_SAMPLE 0
#endif

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
  } else {
    llvm::errs() << "[FAIL] " << what << "\n";
    ++g_failures;
  }
}

template <typename Range>
bool contains(const Range &range, llvm::StringRef needle) {
  for (const auto &e : range)
    if (llvm::StringRef(e) == needle)
      return true;
  return false;
}

} // namespace

int main() {
  using namespace hip::compiler;

  // Dispatch must be safe to call and idempotent (call twice).
  dispatchPluginRegistrationsOnce();
  dispatchPluginRegistrationsOnce();
  check(true, "dispatchPluginRegistrationsOnce() ran without crashing");

  auto slotPasses = pluginPassesForSlot(PipelineSlot::AfterConvertOnnxToHip);
  auto libs = pluginLibraries();
  auto libPaths = pluginLibraryPaths();
  auto bitcode = pluginBitcodeBuffers();

#if HIP_EP_EXPECT_SAMPLE
  llvm::outs() << "Mode: sample plugin EXPECTED (statically linked)\n";

  check(contains(slotPasses, "func.func(hip-ep-sample-print-functions)"),
        "AfterConvertOnnxToHip slot records the sample pass request");
  check(contains(libs, "hip_ep_sample_lib"),
        "pluginLibraries() records 'hip_ep_sample_lib'");
  check(!libPaths.empty(), "pluginLibraryPaths() records a search path");

  // Bitcode is present only when the build had clang to compile it; an empty
  // set is the documented degraded-build case, not a failure.
  if (bitcode.empty()) {
    llvm::outs() << "[SKIP] no plugin bitcode (degraded build without clang)\n";
  } else {
    check(bitcode.size() == 1, "exactly one plugin bitcode buffer recorded");
    const auto &buf = bitcode.front();
    const auto *bytes = static_cast<const unsigned char *>(buf.data);
    bool magicOk = buf.sizeBytes >= 4 && bytes[0] == 'B' && bytes[1] == 'C' &&
                   bytes[2] == 0xC0 && bytes[3] == 0xDE;
    check(magicOk, "plugin bitcode carries the LLVM bitcode magic");
  }
#else
  llvm::outs() << "Mode: NO plugins selected (dispatch must be a no-op)\n";

  check(slotPasses.empty(), "no plugin slot requests recorded");
  check(libs.empty(), "no plugin libraries recorded");
  check(libPaths.empty(), "no plugin library paths recorded");
  check(bitcode.empty(), "no plugin bitcode recorded");
#endif

  if (g_failures == 0) {
    llvm::outs() << "\nAll checks passed.\n";
    return 0;
  }
  llvm::errs() << "\n" << g_failures << " check(s) failed.\n";
  return 1;
}
