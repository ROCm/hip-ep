/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef ARTIFACT_METADATA_H
#define ARTIFACT_METADATA_H

#include "artifact_abi_validation.h"
#include "google/protobuf/util/json_util.h"
#include "metadata.pb.h"

#include <string>

namespace mlir_compilation::customop {

// Parse the metadata carried by a fresh/custom-op or restored EPContext node
// and reject stale/missing ABI tokens before the artifact is opened. The
// artifact's own exported handshake is validated again by LoadedArtifact.
inline bool parseAndValidateArtifactMetadata(const std::string &json,
                                             mlir_metadata::Metadata &metadata,
                                             std::string &error) {
  metadata.Clear();
  error.clear();
  auto status = google::protobuf::util::JsonStringToMessage(json, &metadata);
  if (!status.ok()) {
    error = "failed to parse MLIR artifact metadata: " + status.ToString();
    return false;
  }

  ArtifactAbiValidation validation =
      validateArtifactAbiToken(metadata.artifact_abi());
  if (!validation) {
    error = artifactAbiErrorMessage(validation, "EPContext metadata");
    return false;
  }
  return true;
}

} // namespace mlir_compilation::customop

#endif // ARTIFACT_METADATA_H
