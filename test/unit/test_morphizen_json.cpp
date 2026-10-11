/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// JSON round-trip for the C++ replacements of MorphiZen's protobuf messages.
// Header-only: this is part of the default GPU-free unit suite.

#include "metadata.hpp"
#include "morphizen/messages.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void require(bool condition, const char *expression, int line) {
  if (!condition) {
    std::cerr << "FAIL line " << line << ": " << expression << "\n";
    ++failures;
  }
}

#define REQUIRE(condition)                                                     \
  require(static_cast<bool>(condition), #condition, __LINE__)

} // namespace

int main() {
  const char *config = R"json({
    "passes": [
      {"name": "init", "plugin": "morphizen-pass-init"},
      {"name": "mlir-pass", "plugin": "morphizen-level1-pass-mlir-compiler"}
    ],
    "target": "mlir-target",
    "targets": [{
      "name": "mlir-target",
      "pass": ["init", "mlir-pass"],
      "provider_options": {"pass.init.enable_dump": "0"}
    }],
    "provider_options": {"log_level": "info"},
    "ignored_unknown": true
  })json";

  auto parsed = morphizen::ConfigProto::FromJsonString(config);
  REQUIRE(parsed.passes().size() == 2);
  REQUIRE(parsed.passes()[0].name() == "init");
  REQUIRE(parsed.passes()[1].plugin() == "morphizen-level1-pass-mlir-compiler");
  REQUIRE(parsed.target() == "mlir-target");
  REQUIRE(parsed.targets().size() == 1);
  REQUIRE(parsed.provider_options().at("log_level") == "info");

  auto round_trip = morphizen::ConfigProto::FromJsonString(
      morphizen::json::dump(parsed.ToJson()));
  REQUIRE(round_trip.passes().size() == 2);
  REQUIRE(round_trip.target() == "mlir-target");
  REQUIRE(round_trip.provider_options().at("log_level") == "info");
  REQUIRE(round_trip.targets()[0].provider_options().at(
              "pass.init.enable_dump") == "0");

  const char *overlay = R"json({
    "passes": [{"name": "extra", "plugin": "extra-plugin"}],
    "target": "other-target",
    "provider_options": {"log_level": "verbose", "new_key": "1"}
  })json";
  parsed.MergeFrom(morphizen::ConfigProto::FromJsonString(overlay));
  REQUIRE(parsed.passes().size() == 3);
  REQUIRE(parsed.passes()[2].name() == "extra");
  REQUIRE(parsed.target() == "other-target");
  REQUIRE(parsed.provider_options().at("log_level") == "verbose");
  REQUIRE(parsed.provider_options().at("new_key") == "1");
  REQUIRE(parsed.targets().size() == 1);

  morphizen::ConfigProto with_target;
  with_target.set_target("kept");
  with_target.MergeFrom(morphizen::ConfigProto{});
  REQUIRE(with_target.target().empty());

  mlir_metadata::Metadata metadata;
  metadata.set_artifact_filename("model.bc");
  metadata.set_artifact_format("bc");
  auto *input = metadata.add_inputs();
  input->set_name("x");
  input->set_rank(2);
  input->add_shape(-1);
  input->add_shape(16);
  auto *output = metadata.add_outputs();
  output->set_name("y");
  output->add_shape(4);
  auto again = mlir_metadata::Metadata::FromJsonString(metadata.ToJsonString());
  REQUIRE(again.artifact_filename() == "model.bc");
  REQUIRE(again.artifact_format() == "bc");
  REQUIRE(again.inputs().size() == 1);
  REQUIRE(again.inputs()[0].name() == "x");
  REQUIRE(again.inputs()[0].rank() == 2);
  REQUIRE(again.inputs()[0].shape().size() == 2);
  REQUIRE(again.inputs()[0].shape()[0] == -1);
  REQUIRE(again.inputs()[0].shape()[1] == 16);
  REQUIRE(again.outputs().size() == 1);
  REQUIRE(again.outputs()[0].name() == "y");
  REQUIRE(again.outputs()[0].shape()[0] == 4);

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
