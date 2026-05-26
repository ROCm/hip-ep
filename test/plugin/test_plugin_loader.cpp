/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// PR-1 plugin-loader unit test.
//
// Exercises the public ABI surface declared in
//   include/hip/Compiler/PluginAPI.h
//   include/hip/Compiler/PluginLoader.h
// against the sample plugin built in `test/plugin/sample_plugin/`.
//
// What this test guarantees about the PR-1 plugin infrastructure:
//
//   1. `HipEpPluginLoader::Load` resolves a plugin DLL by absolute
//      path, looks up `hipEpGetPluginInfo`, and validates the
//      returned struct (API version, non-null name/version).
//   2. The plugin name and version reported by `loadPluginsOnce()`
//      match what the sample DLL hard-codes — i.e., the struct
//      survives the C ABI boundary intact.
//   3. `loadPluginsOnce()` is idempotent: a second call returns the
//      same vector and does not re-load DLLs.
//   4. `RegisterCallbacks` fires across the DLL boundary against a
//      `HipEpPluginRegistry &` without crashing.
//   5. A clearly-bad path yields an `llvm::Error` with a useful
//      message rather than a crash.
//
// The test is plain `main()` rather than GTest so it has no
// dependency that the public configure does not already provide.
// PR 2 will likely promote this to GTest once the plugin loader
// has more methods worth checking.

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
  HIP_EP_CHECK(plugin->getPluginVersion() == llvm::StringRef("0.1.0"),
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

  hip::compiler::HipEpPluginRegistry registry;
  // Sample callback is empty in PR 1 but non-null. The pass condition
  // here is the absence of a crash. PR 2 layers real assertions on
  // top once the registry has observable methods.
  plugins[0].registerCallbacks(registry);
  HIP_EP_CHECK(true,
               "registerCallbacks fired across DLL boundary without crashing");
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
