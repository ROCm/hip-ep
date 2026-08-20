/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef ARTIFACT_ABI_VALIDATION_H
#define ARTIFACT_ABI_VALIDATION_H

#include "hip/artifact_abi.h"

#include <cstdint>
#include <string>

namespace mlir_compilation::customop {

enum class ArtifactAbiError {
  None,
  Missing,
  Malformed,
  Mismatch,
};

struct ArtifactAbiValidation {
  ArtifactAbiError error = ArtifactAbiError::None;
  uint64_t token = hipdnn::abi::kArtifactAbiToken;

  explicit operator bool() const { return error == ArtifactAbiError::None; }
};

inline ArtifactAbiValidation validateArtifactAbiToken(uint64_t token) {
  if (token == 0)
    return {ArtifactAbiError::Missing, token};
  if (hipdnn::abi::artifactAbiMagic(token) != hipdnn::abi::kArtifactAbiMagic)
    return {ArtifactAbiError::Malformed, token};
  if (hipdnn::abi::artifactAbiVersion(token) !=
      hipdnn::abi::kArtifactAbiVersion)
    return {ArtifactAbiError::Mismatch, token};
  return {ArtifactAbiError::None, token};
}

inline std::string artifactAbiErrorMessage(const ArtifactAbiValidation &result,
                                           const char *source) {
  const std::string prefix =
      std::string(source ? source : "artifact") + " ABI handshake";
  switch (result.error) {
  case ArtifactAbiError::None:
    return {};
  case ArtifactAbiError::Missing:
    return prefix + " is missing; regenerate the cached artifact/EPContext";
  case ArtifactAbiError::Malformed:
    return prefix + " is malformed (token=" + std::to_string(result.token) +
           "); regenerate the cached artifact/EPContext";
  case ArtifactAbiError::Mismatch:
    return prefix + " version mismatch: artifact=" +
           std::to_string(hipdnn::abi::artifactAbiVersion(result.token)) +
           ", runtime=" + std::to_string(hipdnn::abi::kArtifactAbiVersion) +
           "; regenerate the cached artifact/EPContext";
  }
  return prefix + " validation failed";
}

using ArtifactAbiQueryFn = uint64_t (*)();

inline ArtifactAbiValidation validateArtifactAbiSymbol(void *symbol) {
  if (!symbol)
    return {ArtifactAbiError::Missing, 0};
  auto query = reinterpret_cast<ArtifactAbiQueryFn>(symbol);
  return validateArtifactAbiToken(query());
}

} // namespace mlir_compilation::customop

#endif // ARTIFACT_ABI_VALIDATION_H
