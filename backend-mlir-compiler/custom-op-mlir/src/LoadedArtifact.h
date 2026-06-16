/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef LOADED_ARTIFACT_H
#define LOADED_ARTIFACT_H

#include "artifact_format.h" // ArtifactKind
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace morphizen {
struct Plugin; // native .dll/.so loader (LoadLibrary/dlopen)
} // namespace morphizen

namespace mlir_compilation::customop {

class LlvmIrJit;

// One loader for a per-model artifact, independent of format:
//   * LLVM IR (.bc) -> in-process ORC JIT (LlvmIrJit). No temp file.
//   * Native (.dll/.so) -> morphizen::Plugin (LoadLibrary/dlopen).
//
// Both backends expose the same `get_method<R, Args...>` form, so callers
// resolve the per-model C ABI (`inference_*`, optional runtime hooks)
// identically regardless of format. Shared by the EP (in-memory EPContext
// model) and the standalone tools (artifact on disk), so the two never drift.
//
// The factories never LOG(FATAL): they return nullptr and (optionally) a
// reason string, leaving error policy to the caller -- the EP escalates to
// FATAL, the tools return an exit code.
class LoadedArtifact {
public:
  ~LoadedArtifact();

  LoadedArtifact(const LoadedArtifact &) = delete;
  LoadedArtifact &operator=(const LoadedArtifact &) = delete;
  LoadedArtifact(LoadedArtifact &&) = delete;
  LoadedArtifact &operator=(LoadedArtifact &&) = delete;

  // In-memory artifact (EPContext model). `kind` is chosen by the caller
  // (metadata + magic-byte cross-check). For Native the bytes are spilled to a
  // temp file -- morphizen::Plugin loads from a path -- which this object
  // removes on destruction. `module_name` is a diagnostic tag for the JIT.
  static std::unique_ptr<LoadedArtifact>
  createInMemory(const std::vector<uint8_t> &bytes, ArtifactKind kind,
                 const std::string &module_name, std::string *error = nullptr);

  // On-disk artifact (standalone tools). Format is sniffed from the file's
  // magic bytes. Native is loaded directly from `path` (no temp file, so
  // co-located deps such as custom_kernels_<arch> resolve as usual); LLVM IR
  // is read and JITed in-process.
  static std::unique_ptr<LoadedArtifact>
  createFromFile(const std::string &path, std::string *error = nullptr);

  ArtifactKind kind() const { return kind_; }

  // Resolve a C-ABI symbol; nullptr when absent (callers probe optional hooks
  // this way).
  void *lookup_raw(const char *name) const;

  template <typename R, typename... Args>
  auto get_method(const char *name) const -> R (*)(Args...) {
    return reinterpret_cast<R (*)(Args...)>(lookup_raw(name));
  }

private:
  LoadedArtifact();

  // Exactly one backend owns the loaded artifact; the variant makes that a
  // type-level one-of. Alternatives are unique_ptrs over forward-declared
  // types -- complete enough here; the variant is only ever destroyed/reset in
  // the .cpp where both types are complete.
  using Backend = std::variant<std::unique_ptr<LlvmIrJit>,
                               std::unique_ptr<morphizen::Plugin>>;
  Backend backend_;
  ArtifactKind kind_ = ArtifactKind::LLVM_IR;
  std::string temp_native_path_; // non-empty -> remove on destruction
};

} // namespace mlir_compilation::customop

#endif // LOADED_ARTIFACT_H
