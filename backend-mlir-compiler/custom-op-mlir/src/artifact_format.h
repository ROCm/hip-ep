/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef ARTIFACT_FORMAT_H
#define ARTIFACT_FORMAT_H

#include <string>

namespace mlir_compilation::customop {

// Per-model artifact format. Mirrors CompilationOptions.output_mode and is
// recorded in the EPContext metadata (mlir_metadata::Metadata).
//   LLVM_IR -> OS-portable LLVM IR (serialized as .bc), JIT-loaded in-process
//              via LlvmIrJit.
//   NATIVE  -> per-OS .dll/.so loaded via morphizen::Plugin (LoadLibrary/
//              dlopen).
enum class ArtifactKind { LLVM_IR, NATIVE };

// EPContext metadata artifact_format values (producer<->EP contract).
inline constexpr const char *kArtifactFormatNative = "NATIVE";
inline constexpr const char *kArtifactFormatLlvmIr = "LLVM_IR";

// Map the recorded artifact_format string to a kind. Returns false for an
// empty/unknown value -- the producer always records one (see pass_main.cpp),
// so the EP treats a missing value as fatal. Dependency-free (no logging) so
// the EP and the standalone tools can share it.
inline bool artifactKindFromFormat(const std::string &format_str,
                                   ArtifactKind &out) {
  if (format_str == kArtifactFormatNative) {
    out = ArtifactKind::NATIVE;
    return true;
  }
  if (format_str == kArtifactFormatLlvmIr) {
    out = ArtifactKind::LLVM_IR;
    return true;
  }
  return false;
}

// Map a bare artifact file path to a kind by extension, for the standalone
// tools (which load a file with no accompanying EPContext metadata): .dll/.so
// -> NATIVE, .bc -> LLVM_IR. Returns false for any other extension.
inline bool artifactKindFromPath(const std::string &path, ArtifactKind &out) {
  auto ends_with = [&](const char *suffix) {
    const std::string s(suffix);
    return path.size() >= s.size() &&
           path.compare(path.size() - s.size(), s.size(), s) == 0;
  };
  if (ends_with(".dll") || ends_with(".so")) {
    out = ArtifactKind::NATIVE;
    return true;
  }
  if (ends_with(".bc")) {
    out = ArtifactKind::LLVM_IR;
    return true;
  }
  return false;
}

} // namespace mlir_compilation::customop

#endif // ARTIFACT_FORMAT_H
