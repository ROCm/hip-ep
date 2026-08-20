/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free tests for the generated-artifact/runtime ABI handshake and the
// EPContext metadata seam that selects the LLVM-IR/native loader.

#include "artifact_abi_validation.h"
#include "artifact_format.h"
#include "artifact_metadata.h"
#include "google/protobuf/util/json_util.h"
#include "hip/artifact_abi.h"
#include "metadata.pb.h"

#include <cstdio>
#include <string>

namespace {

int gFailures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++gFailures;                                                             \
    }                                                                          \
  } while (0)

uint64_t currentAbi() { return hipdnn::abi::kArtifactAbiToken; }
uint64_t staleAbi() {
  return hipdnn::abi::artifactAbiToken(hipdnn::abi::kArtifactAbiVersion + 1);
}
uint64_t malformedAbi() { return 17; }

void testGeneratedSymbolValidation() {
  using namespace mlir_compilation::customop;

  CHECK(validateArtifactAbiSymbol(reinterpret_cast<void *>(&currentAbi)));

  ArtifactAbiValidation missing = validateArtifactAbiSymbol(nullptr);
  CHECK(missing.error == ArtifactAbiError::Missing);
  CHECK(
      artifactAbiErrorMessage(missing, "generated artifact").find("missing") !=
      std::string::npos);

  ArtifactAbiValidation stale =
      validateArtifactAbiSymbol(reinterpret_cast<void *>(&staleAbi));
  CHECK(stale.error == ArtifactAbiError::Mismatch);
  CHECK(artifactAbiErrorMessage(stale, "generated artifact")
            .find("version mismatch") != std::string::npos);

  ArtifactAbiValidation malformed =
      validateArtifactAbiSymbol(reinterpret_cast<void *>(&malformedAbi));
  CHECK(malformed.error == ArtifactAbiError::Malformed);
  CHECK(artifactAbiErrorMessage(malformed, "generated artifact")
            .find("malformed") != std::string::npos);
}

std::string metadataJson(uint64_t abi, const char *format) {
  mlir_metadata::Metadata metadata;
  metadata.set_artifact_filename("model_compiled");
  metadata.set_artifact_format(format);
  if (abi != 0)
    metadata.set_artifact_abi(abi);
  std::string json;
  auto status = google::protobuf::util::MessageToJsonString(metadata, &json);
  CHECK(status.ok());
  return json;
}

void testEpContextMetadataFormat(
    const char *format, mlir_compilation::customop::ArtifactKind kind) {
  using namespace mlir_compilation::customop;

  mlir_metadata::Metadata parsed;
  std::string error;
  CHECK(parseAndValidateArtifactMetadata(
      metadataJson(hipdnn::abi::kArtifactAbiToken, format), parsed, error));
  CHECK(error.empty());
  CHECK(parsed.artifact_filename() == "model_compiled");

  ArtifactKind parsedKind;
  CHECK(artifactKindFromFormat(parsed.artifact_format(), parsedKind));
  CHECK(parsedKind == kind);
}

void testEpContextMetadataValidation() {
  using namespace mlir_compilation::customop;

  testEpContextMetadataFormat(kArtifactFormatLlvmIr, ArtifactKind::LLVM_IR);
  testEpContextMetadataFormat(kArtifactFormatNative, ArtifactKind::NATIVE);

  mlir_metadata::Metadata parsed;
  std::string error;
  CHECK(!parseAndValidateArtifactMetadata(
      metadataJson(
          hipdnn::abi::artifactAbiToken(hipdnn::abi::kArtifactAbiVersion + 1),
          kArtifactFormatLlvmIr),
      parsed, error));
  CHECK(error.find("version mismatch") != std::string::npos);

  error.clear();
  CHECK(!parseAndValidateArtifactMetadata(
      metadataJson(0, kArtifactFormatLlvmIr), parsed, error));
  CHECK(error.find("missing") != std::string::npos);

  error.clear();
  CHECK(!parseAndValidateArtifactMetadata(
      metadataJson(17, kArtifactFormatLlvmIr), parsed, error));
  CHECK(error.find("malformed") != std::string::npos);

  error.clear();
  CHECK(!parseAndValidateArtifactMetadata(
      R"({"artifactFilename":"model_compiled","artifactFormat":"LLVM_IR",)"
      R"("artifactAbi":"not-a-version"})",
      parsed, error));
  CHECK(error.find("failed to parse") != std::string::npos);
}

} // namespace

int main() {
  testGeneratedSymbolValidation();
  testEpContextMetadataValidation();

  if (gFailures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", gFailures);
    return 1;
  }
  std::puts("Artifact ABI handshake tests passed");
  return 0;
}
