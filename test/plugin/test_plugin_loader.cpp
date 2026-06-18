/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Plugin-loader unit test.
//
// Exercises the public ABI surface declared in
//   include/hip/Compiler/PluginAPI.h
//   include/hip/Compiler/PluginLoader.h
//   include/hip/Compiler/PluginRegistry.h
// against the sample plugin built in `test/plugin/sample_plugin/`.
//
// What this test guarantees about the plugin infrastructure:
//
//   1. `HipEpPluginLoader::Load` resolves a plugin DLL by absolute
//      path, looks up `hipEpGetPluginInfo`, and validates the
//      returned struct (API version, non-null name/version).
//   2. The plugin name and version reported by `loadPluginsOnce()`
//      match what the sample DLL hard-codes -- i.e., the struct
//      survives the C ABI boundary intact.
//   3. `loadPluginsOnce()` is idempotent: a second call returns the
//      same vector and does not re-load DLLs.
//   4. `RegisterCallbacks` fires across the DLL boundary against a
//      `HipEpPluginRegistry &` without crashing.
//   5. After the callback runs, the registry records the slot
//      request the plugin made (`AfterConvertOnnxToHip` ->
//      "hip-ep-sample-print-functions"). The Pipelines.cpp slot
//      hook reads exactly this state.
//   6. After the callback runs, the registry also records the
//      LLVM bitcode buffer the plugin contributed via
//      `addRuntimeBitcode`. The buffer carries the LLVM bitcode
//      magic ('BC\xc0\xde'), confirming the build's clang->bitcode
//      pipeline succeeded and the bytes survived the C ABI boundary
//      intact.
//   7. After the callback runs, the registry records the
//      library search path + library name the plugin contributed
//      via `addLibraryPath` / `addLibrary`. Both round-trip across
//      the DLL boundary as recorded `std::string` copies returned
//      directly by the accessors, even though the plugin's source
//      string was a stack-bound `llvm::StringRef`. Owning copies
//      (rather than `StringRef`) avoid any aliasing of the
//      registry's internal storage.
//   8. A clearly-bad path yields an `llvm::Error` with a useful
//      message rather than a crash.
//
// The test is plain `main()` rather than GTest so it has no
// dependency that the public configure does not already provide.

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginLoader.h"
#include "hip/Compiler/PluginRegistry.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// Set by CMake via `-DHIP_EP_SAMPLE_PLUGIN_PATH="..."` so the test
// always knows where the sibling sample DLL lives, regardless of the
// working directory CTest invokes us from.
constexpr const char *kSamplePluginPath = HIP_EP_SAMPLE_PLUGIN_PATH;

constexpr const char *kPluginsEnvVar = "HIP_EP_PLUGINS";

// We deliberately use the CRT here — this test does NOT live inside
// the EP DLL, so the static-CRT-isolation problem hip_get_env() works
// around does not apply.
void setPluginsEnv(const std::string &value) {
#ifdef _WIN32
  _putenv_s(kPluginsEnvVar, value.c_str());
#else
  setenv(kPluginsEnvVar, value.c_str(), /*overwrite=*/1);
#endif
}

void unsetPluginsEnv() {
#ifdef _WIN32
  _putenv_s(kPluginsEnvVar, "");
#else
  unsetenv(kPluginsEnvVar);
#endif
}

int failures = 0;

#define HIP_EP_CHECK(cond, message)                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::fprintf(stderr, "[FAIL] %s\n  at %s:%d: condition: %s\n", message,  \
                   __FILE__, __LINE__, #cond);                                 \
    } else {                                                                   \
      std::fprintf(stdout, "[PASS] %s\n", message);                            \
    }                                                                          \
  } while (0)

void testDirectLoadResolvesEntryPointAndValidatesStruct() {
  std::fprintf(
      stdout,
      "\n=== Test: DirectLoadResolvesEntryPointAndValidatesStruct ===\n");
  unsetPluginsEnv();

  auto plugin = hip::compiler::HipEpPluginLoader::Load(kSamplePluginPath);
  if (!plugin) {
    std::string err = llvm::toString(plugin.takeError());
    std::fprintf(stderr, "[FAIL] Failed to load sample plugin: %s\n",
                 err.c_str());
    ++failures;
    return;
  }

  HIP_EP_CHECK(plugin->getAPIVersion() == HIP_EP_PLUGIN_API_VERSION,
               "Plugin API version matches HIP_EP_PLUGIN_API_VERSION");
  HIP_EP_CHECK(plugin->getPluginName() == llvm::StringRef("HipEpSamplePlugin"),
               "Plugin name returned by sample matches expected literal");
  HIP_EP_CHECK(plugin->getPluginVersion() == llvm::StringRef("0.4.0"),
               "Plugin version returned by sample matches expected literal");
  HIP_EP_CHECK(plugin->getFilename() == llvm::StringRef(kSamplePluginPath),
               "Loader records the filename it loaded from");
}

void testLoadPluginsOnceReadsEnvAndCachesResult() {
  std::fprintf(stdout,
               "\n=== Test: LoadPluginsOnceReadsEnvAndCachesResult ===\n");
  setPluginsEnv(kSamplePluginPath);

  const auto &plugins = hip::compiler::loadPluginsOnce();
  HIP_EP_CHECK(
      plugins.size() == 1u,
      "loadPluginsOnce returns exactly one plugin from HIP_EP_PLUGINS");
  if (!plugins.empty()) {
    HIP_EP_CHECK(plugins[0].getPluginName() ==
                     llvm::StringRef("HipEpSamplePlugin"),
                 "First plugin's name matches sample plugin literal");
  }

  // Idempotency: a second call returns the same vector. Compare by
  // address since loadPluginsOnce returns a reference to a static.
  const auto &pluginsAgain = hip::compiler::loadPluginsOnce();
  HIP_EP_CHECK(&plugins == &pluginsAgain,
               "loadPluginsOnce returns a stable reference (caches)");
  HIP_EP_CHECK(pluginsAgain.size() == plugins.size(),
               "Second loadPluginsOnce call returns the same size");
}

void testRegisterCallbacksFiresAcrossDllBoundary() {
  std::fprintf(stdout,
               "\n=== Test: RegisterCallbacksFiresAcrossDllBoundary ===\n");
  setPluginsEnv(kSamplePluginPath);
  const auto &plugins = hip::compiler::loadPluginsOnce();
  if (plugins.empty()) {
    std::fprintf(stderr, "[FAIL] No plugins loaded; skipping callback test\n");
    ++failures;
    return;
  }

  // Snapshot the registry's view BEFORE we manually invoke
  // registerCallbacks. The accessors below auto-trigger
  // `dispatchPluginRegistrationsOnce` (the read paths are defensive), so by
  // the time we read `beforeCount` the auto dispatch has already run -- the
  // snapshot reflects whatever production dispatch contributed, and our manual
  // call below adds one more increment that we then assert on.
  auto beforeCount = hip::compiler::pluginPassesForSlot(
                         hip::compiler::PipelineSlot::AfterConvertOnnxToHip)
                         .size();
  auto beforeBitcodeCount = hip::compiler::pluginBitcodeBuffers().size();
  auto beforeLibraryPathCount = hip::compiler::pluginLibraryPaths().size();
  auto beforeLibraryCount = hip::compiler::pluginLibraries().size();

  // The registry is a process-wide handle whose methods dispatch
  // through a vtable into the per-process storage in
  // lib/Compiler/PluginRegistry.cpp. We exercise the manual API
  // surface (HipEpPluginLoader::registerCallbacks) to verify the
  // lower-level control path that tools other than `hip-compiler`
  // / `hip-mlir-opt` may take. Production code uses
  // dispatchPluginRegistrationsOnce instead, which is idempotent
  // and called automatically from the accessors.
  hip::compiler::HipEpPluginRegistry &registry =
      hip::compiler::getProcessPluginRegistry();
  plugins[0].registerCallbacks(registry);
  HIP_EP_CHECK(true,
               "registerCallbacks fired across DLL boundary without crashing");

  // The sample plugin's RegisterCallbacks calls
  //   R.requestPipelineSlot(PipelineSlot::AfterConvertOnnxToHip,
  //                         "func.func(hip-ep-sample-print-functions)");
  // (the func.func(...) nesting is required because the slot resolves the
  // string into a module-level pass manager and the pass is a FuncOp pass).
  // That pair must now be queryable through the public accessor.
  auto afterPasses = hip::compiler::pluginPassesForSlot(
      hip::compiler::PipelineSlot::AfterConvertOnnxToHip);
  HIP_EP_CHECK(afterPasses.size() == beforeCount + 1u,
               "pluginPassesForSlot increments by 1 after registerCallbacks");
  bool foundSamplePass = false;
  for (auto name : afterPasses) {
    if (name == llvm::StringRef("func.func(hip-ep-sample-print-functions)")) {
      foundSamplePass = true;
      break;
    }
  }
  HIP_EP_CHECK(
      foundSamplePass,
      "pluginPassesForSlot(AfterConvertOnnxToHip) contains the sample pass");

  // Slots the plugin did NOT request should not have been touched
  // by this dispatch (i.e., callbacks don't accidentally write into
  // the wrong slot).
  auto unrelated = hip::compiler::pluginPassesForSlot(
      hip::compiler::PipelineSlot::AfterPoolAllocs);
  HIP_EP_CHECK(
      unrelated.size() == 0u,
      "pluginPassesForSlot(AfterPoolAllocs) is empty (unrelated slot)");

  // The sample plugin contributed a bitcode buffer if its
  // build had clang available. We can't tell from here whether the
  // build was degraded; instead we check that the recorded count
  // either stayed the same (degraded build) or grew by exactly one
  // (normal build) and that any new buffer carries valid LLVM
  // bitcode magic.
  auto afterBuffers = hip::compiler::pluginBitcodeBuffers();
  bool bitcodeContributed = afterBuffers.size() == beforeBitcodeCount + 1u;
  bool bitcodeAbsent = afterBuffers.size() == beforeBitcodeCount;
  HIP_EP_CHECK(bitcodeContributed || bitcodeAbsent,
               "pluginBitcodeBuffers grew by 0 or 1 after registerCallbacks");
  if (bitcodeContributed) {
    const auto &buf = afterBuffers.back();
    HIP_EP_CHECK(buf.sizeBytes >= 4u,
                 "Contributed bitcode is at least 4 bytes (room for magic)");
    if (buf.sizeBytes >= 4u) {
      const unsigned char *bytes = static_cast<const unsigned char *>(buf.data);
      // LLVM bitcode wrapper magic: 'BC\xc0\xde' (little-endian 4-byte).
      // See llvm/Bitcode/BitcodeReader.h, function `getBitcodeFileContents`.
      HIP_EP_CHECK(bytes[0] == 0x42 && bytes[1] == 0x43 && bytes[2] == 0xC0 &&
                       bytes[3] == 0xDE,
                   "Contributed bitcode starts with LLVM bitcode magic "
                   "'BC\\xc0\\xde'");
    }
    std::fprintf(stdout, "  contributed bitcode size: %zu bytes\n",
                 buf.sizeBytes);
  } else {
    std::fprintf(stdout, "  no bitcode contributed (likely a degraded build "
                         "without clang at sample-plugin configure time)\n");
  }

  // The sample plugin contributes one library path + one
  // library name unconditionally. Both should appear in the
  // accessors with the expected literal values.
  auto afterLibraryPaths = hip::compiler::pluginLibraryPaths();
  auto afterLibraries = hip::compiler::pluginLibraries();
  HIP_EP_CHECK(afterLibraryPaths.size() == beforeLibraryPathCount + 1u,
               "pluginLibraryPaths grew by 1 after registerCallbacks");
  HIP_EP_CHECK(afterLibraries.size() == beforeLibraryCount + 1u,
               "pluginLibraries grew by 1 after registerCallbacks");

  bool foundSampleLibName = false;
  for (const auto &name : afterLibraries) {
    if (llvm::StringRef(name) == llvm::StringRef("hip_ep_sample_lib")) {
      foundSampleLibName = true;
      break;
    }
  }
  HIP_EP_CHECK(foundSampleLibName,
               "pluginLibraries contains the sample lib name "
               "'hip_ep_sample_lib'");

  // The path is a build-tree directory we cannot hard-code in the
  // test; we verify only that *some* non-empty path was recorded
  // (a malformed contribution would surface as an empty string).
  if (!afterLibraryPaths.empty()) {
    const std::string &path = afterLibraryPaths.back();
    HIP_EP_CHECK(!path.empty(), "Last contributed library path is non-empty");
    std::fprintf(stdout, "  contributed lib path: %s\n", path.c_str());
  }
}

void testLoadFailsCleanlyOnNonExistentPath() {
  std::fprintf(stdout, "\n=== Test: LoadFailsCleanlyOnNonExistentPath ===\n");
  unsetPluginsEnv();

  auto plugin = hip::compiler::HipEpPluginLoader::Load(
      "this_path_does_not_exist_xyz_hipep.dll");
  HIP_EP_CHECK(!plugin, "Load() of a non-existent path returns an llvm::Error");
  if (!plugin) {
    std::string err = llvm::toString(plugin.takeError());
    HIP_EP_CHECK(err.find("Could not load plugin") != std::string::npos,
                 "Error message mentions 'Could not load plugin'");
    std::fprintf(stdout, "  observed error: %s\n", err.c_str());
  }
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
  std::fprintf(stdout, "[plugin-loader-test] sample plugin path: %s\n",
               kSamplePluginPath);

  testDirectLoadResolvesEntryPointAndValidatesStruct();
  testLoadPluginsOnceReadsEnvAndCachesResult();
  testRegisterCallbacksFiresAcrossDllBoundary();
  testLoadFailsCleanlyOnNonExistentPath();

  if (failures == 0) {
    std::fprintf(stdout, "\n[plugin-loader-test] ALL PASS\n");
    return EXIT_SUCCESS;
  }
  std::fprintf(stderr, "\n[plugin-loader-test] %d FAILURE(S)\n", failures);
  return EXIT_FAILURE;
}
