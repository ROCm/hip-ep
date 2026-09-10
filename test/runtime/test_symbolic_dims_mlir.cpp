/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir-graph.hpp"
#include "mlir-model.hpp"
#include "morphizen/symbolic_dims.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAILED: " << message << "\n";
  return condition;
}

} // namespace

int main() {
  using morphizen::mlir_impl::MLIRModel;

  const std::string metadata = morphizen::encode_symbolic_dim_records(
      {{"main_graph", "input", {"N", "D"}},
       {"main_graph", "output", {"N", "D"}}});
  auto model = MLIRModel::create_empty();
  model->set_metadata_prop(std::string(morphizen::kOnnxDimParamsMetadataKey),
                           metadata);
  model->set_metadata_prop(
      std::string(morphizen::kInitializerDataDigestMetadataKey),
      std::string(64, 'a'));
  bool ok = check(
      model->has_metadata_prop(morphizen::kCompilerGraphDigestMetadataKey),
      "initializer digest must finalize compiler graph identity");
  ok &=
      check(model->get_metadata_prop(morphizen::kCompilerGraphDigestMetadataKey)
                    .size() == 64,
            "compiler graph identity must be SHA-256");

  const std::string first = model->main_graph().save_string();
  ok &= check(!first.empty(), "first serialization must produce bytecode");
  ok &= check(model->main_graph().save_string() == first,
              "repeated serialization must restore temporary module metadata");

  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     "hipdnn_symbolic_dims_roundtrip.mlirbc";
  {
    std::ofstream output(path, std::ios::binary);
    output.write(first.data(), static_cast<std::streamsize>(first.size()));
    ok &= check(static_cast<bool>(output), "bytecode write must succeed");
  }

  auto loaded = MLIRModel::load(path.string());
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  ok &= check(loaded != nullptr, "serialized bytecode must load");
  if (!loaded)
    return 1;
  ok &= check(loaded->has_metadata_prop(morphizen::kOnnxDimParamsMetadataKey),
              "loaded model must import symbolic metadata");
  ok &= check(loaded->get_metadata_prop(morphizen::kOnnxDimParamsMetadataKey) ==
                  metadata,
              "loaded model must preserve canonical metadata bytes");
  ok &= check(!loaded->has_metadata_prop(
                  morphizen::kInitializerDataDigestMetadataKey) &&
                  !loaded->has_metadata_prop(
                      morphizen::kCompilerGraphDigestMetadataKey),
              "cache-only digests must not become serialized model authority");
  ok &= check(loaded->main_graph().save_string() == first,
              "load/re-serialize must be byte-identical");

  loaded->getModule()->setAttr(morphizen::kOnnxDimParamsModuleAttr,
                               loaded->get_symbolic_dim_attr());
  try {
    (void)loaded->main_graph().save_string();
    ok &= check(false, "conflicting live module metadata must be rejected");
  } catch (const std::runtime_error &) {
  }

  return ok ? 0 : 1;
}
