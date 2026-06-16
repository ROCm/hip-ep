/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef ARTIFACT_FORMAT_H
#define ARTIFACT_FORMAT_H

#include <cstddef>
#include <cstdint>
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

// True if the leading bytes are LLVM bitcode magic: 'BC' 0xC0 0xDE, or the
// 0x0B17C0DE bitcode-wrapper magic.
inline bool hasBitcodeMagic(const uint8_t *data, size_t size) {
  return (size >= 4 && data[0] == 'B' && data[1] == 'C' && data[2] == 0xC0 &&
          data[3] == 0xDE) ||
         (size >= 4 && data[0] == 0xDE && data[1] == 0xC0 && data[2] == 0x17 &&
          data[3] == 0x0B);
}

// True if the leading bytes are a native image: PE 'MZ' (Windows) or ELF
// 0x7F 'ELF' (Linux).
inline bool hasNativeMagic(const uint8_t *data, size_t size) {
  return (size >= 2 && data[0] == 'M' && data[1] == 'Z') ||
         (size >= 4 && data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' &&
          data[3] == 'F');
}

// Decide the artifact kind from the recorded format string, cross-checked
// against the artifact's magic bytes. Returns the kind; when the metadata and
// magic bytes clearly disagree and `mismatch` is non-null, sets `*mismatch` to
// a human-readable reason (the caller decides how to report it). Dependency-
// free (no logging) so both the EP and the hip-test / hip-inspect tools can
// share it.
inline ArtifactKind artifactKindFromMetadata(const std::string &format_str,
                                             const uint8_t *data, size_t size,
                                             std::string *mismatch = nullptr) {
  const bool llvm_ir_magic = hasBitcodeMagic(data, size);
  const bool native_magic = hasNativeMagic(data, size);

  // EPContext models produced before this field existed (e.g. the native-only
  // EP) carry an empty artifact_format. Trust the magic bytes there instead of
  // assuming a format, so a cached native artifact still loads rather than
  // tripping the mismatch check below.
  if (format_str.empty())
    return native_magic ? ArtifactKind::NATIVE : ArtifactKind::LLVM_IR;

  const ArtifactKind kind = (format_str == kArtifactFormatNative)
                                ? ArtifactKind::NATIVE
                                : ArtifactKind::LLVM_IR;
  if (mismatch) {
    if (kind == ArtifactKind::LLVM_IR && !llvm_ir_magic && native_magic) {
      *mismatch = "metadata artifact_format='" + format_str +
                  "' (LLVM_IR) but the artifact's magic bytes are native "
                  "(PE/ELF)";
    } else if (kind == ArtifactKind::NATIVE && !native_magic && llvm_ir_magic) {
      *mismatch = "metadata artifact_format='NATIVE' but the artifact's magic "
                  "bytes are LLVM bitcode";
    }
  }
  return kind;
}

} // namespace mlir_compilation::customop

#endif // ARTIFACT_FORMAT_H
