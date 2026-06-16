/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "LoadedArtifact.h"

#include "../../common/temp_path.hpp"
#include "LlvmIrJit.h"
// morphizen.hpp must precede plugin.hpp (morphizen/_sanity_check.hpp enforces
// this include order).
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>

namespace mlir_compilation::customop {

namespace {

std::string nativeExtension() {
#ifdef _WIN32
  return ".dll";
#else
  return ".so";
#endif
}

// morphizen::Plugin::create takes a base path (no extension) and appends the
// platform suffix itself.
std::string stripExtension(const std::string &path) {
  return std::filesystem::path(path).replace_extension("").string();
}

std::vector<uint8_t> readFileBytes(const std::string &path,
                                   std::string *error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error)
      *error = "failed to open artifact '" + path + "'";
    return {};
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

} // namespace

LoadedArtifact::LoadedArtifact() = default;

LoadedArtifact::~LoadedArtifact() {
  // Unload the backend first (Plugin = FreeLibrary/dlclose; LlvmIrJit tears
  // down the ORC JIT), then remove any temp file we spilled for Native.
  std::visit([](auto &backend) { backend.reset(); }, backend_);
  if (!temp_native_path_.empty())
    std::remove(temp_native_path_.c_str());
}

std::unique_ptr<LoadedArtifact> LoadedArtifact::createInMemory(
    const std::vector<uint8_t> &bytes, ArtifactKind kind,
    const std::string &module_name, std::string *error) {
  std::unique_ptr<LoadedArtifact> art(new LoadedArtifact());
  art->kind_ = kind;

  if (kind == ArtifactKind::LLVM_IR) {
    auto jit = LlvmIrJit::create(bytes, module_name);
    if (!jit) {
      if (error)
        *error = "LlvmIrJit::create failed for '" + module_name + "'";
      return nullptr;
    }
    art->backend_ = std::move(jit);
    return art;
  }

  // Native: morphizen::Plugin loads from a path, so spill the bytes to a temp
  // file. Record the path before the load so the destructor cleans it up even
  // if Plugin::create fails below.
  const std::string dll_path =
      mlir_compiler_utils::generateTempPath(nativeExtension());
  {
    std::ofstream out(dll_path, std::ios::binary);
    if (!out) {
      if (error)
        *error = "failed to create temp artifact file '" + dll_path + "'";
      return nullptr;
    }
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  art->temp_native_path_ = dll_path;

  auto plugin = morphizen::Plugin::create(stripExtension(dll_path).c_str());
  if (!plugin) {
    if (error)
      *error = "morphizen::Plugin::create failed for '" + dll_path +
               "' (check that dependencies -- amdhip64, custom_kernels_<arch>, "
               "... -- are resolvable)";
    return nullptr; // destructor removes the temp file
  }
  art->backend_ = std::move(plugin);
  return art;
}

std::unique_ptr<LoadedArtifact>
LoadedArtifact::createFromFile(const std::string &path, std::string *error) {
  std::vector<uint8_t> bytes = readFileBytes(path, error);
  if (bytes.empty()) {
    if (error && error->empty())
      *error = "artifact '" + path + "' is empty";
    return nullptr;
  }

  std::unique_ptr<LoadedArtifact> art(new LoadedArtifact());
  if (!artifactKindFromPath(path, art->kind_)) {
    if (error)
      *error = "cannot determine artifact format from extension of '" + path +
               "' (expected .bc, .dll, or .so)";
    return nullptr;
  }
  const bool native = (art->kind_ == ArtifactKind::NATIVE);

  if (!native) {
    auto jit = LlvmIrJit::create(bytes, path);
    if (!jit) {
      if (error)
        *error = "LlvmIrJit::create failed for '" + path + "'";
      return nullptr;
    }
    art->backend_ = std::move(jit);
    return art;
  }

  // Native artifact already on disk: load directly from `path` (no temp file).
  auto plugin = morphizen::Plugin::create(stripExtension(path).c_str());
  if (!plugin) {
    if (error)
      *error = "morphizen::Plugin::create failed for '" + path +
               "' (check that dependencies like custom_kernels_<arch> are "
               "co-located)";
    return nullptr;
  }
  art->backend_ = std::move(plugin);
  return art;
}

void *LoadedArtifact::lookup_raw(const char *name) const {
  if (!name)
    return nullptr;
  if (const auto *jit = std::get_if<std::unique_ptr<LlvmIrJit>>(&backend_)) {
    return *jit ? (*jit)->lookup_raw(name) : nullptr;
  }
  if (const auto *plugin =
          std::get_if<std::unique_ptr<morphizen::Plugin>>(&backend_)) {
    if (!*plugin)
      return nullptr;
    // Plugin exposes no raw-symbol accessor; fetch a bare function pointer and
    // reinterpret to void* (the dlsym/GetProcAddress idiom -- function and
    // object pointers are interchangeable on the x86-64 targets we ship).
    return reinterpret_cast<void *>((*plugin)->get_method<void>(name));
  }
  return nullptr;
}

} // namespace mlir_compilation::customop
