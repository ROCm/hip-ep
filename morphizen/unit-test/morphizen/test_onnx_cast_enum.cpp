/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test_environment.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>

// Helper function to check if MLIR backend is being used at runtime. Mirrors
// test_onnx_custom.cpp -- the importer code under test here (mlir-graph.cpp's
// add_node()) only runs when the MLIR backend is active.
static bool isMLIRBackendActive() {
  auto *api = morphizen::get_the_global_api_unsafe();
  if (!api || !api->get_lib_name) {
    return false;
  }
  auto lib_name = api->get_lib_name();
  if (!lib_name.get()) {
    return false;
  }
  return std::string(lib_name->c_str()) == "morphizen-mlir-imp";
}

class OnnxCastEnumTest : public ::testing::Test {};

// Regression guard: UINT8=2 and BOOL=9 both import as MLIR `ui8`, so the
// importer must stash the original ONNX enum in onnx_to_enum for CastToHip
// to recover later. Uses node.get_attr_int() (not MLIR text diffing) since
// it reads the live in-memory attribute regardless of serialization format.
TEST_F(OnnxCastEnumTest, CastToBoolPreservesOriginalEnum) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path =
      CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("onnx_cast_bool.onnx");
  auto data_path = std::filesystem::path("onnx_cast_bool.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("mask", {1, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("mask_bool", {1, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_BOOL)};

  // to = 9 (BOOL).
  graph.add_node(
      "cast_to_bool", "", "Cast", "", {input[0]}, {output[0]},
      morphizen::NodeAttributesBuilder()
          .add("to", int64_t(ONNX_NAMESPACE::TensorProto_DataType_BOOL))
          .build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  auto nodes = graph.nodes();
  ASSERT_EQ(nodes.size(), 1u);
  auto node = nodes[0];

  // The original BOOL(9) enum must survive the importer's conversion to a
  // TypeAttr (which alone cannot distinguish BOOL from UINT8).
  ASSERT_TRUE(node.has_attr("onnx_to_enum"))
      << "onnx.Cast should carry the preserved onnx_to_enum attribute";
  EXPECT_EQ(node.get_attr_int("onnx_to_enum"),
            ONNX_NAMESPACE::TensorProto_DataType_BOOL)
      << "onnx_to_enum should preserve the original BOOL(9) enum";

  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Regression guard: onnx_to_enum is MLIR-internal bookkeeping and must not
// leak as a user-visible ONNX node attribute. Exercises the same accessor
// path real EP code uses to republish node attributes externally
// (MorphiZenEP::convert_morphizen_node_to_ort_node in
// ort-bridge/src/morphizen-ep.cpp).
TEST_F(OnnxCastEnumTest, OnnxToEnumNotUserVisible) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_cast_bool_keys.onnx");
  auto data_path = std::filesystem::path("onnx_cast_bool_keys.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("mask", {1, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("mask_bool", {1, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_BOOL)};

  graph.add_node(
      "cast_to_bool", "", "Cast", "", {input[0]}, {output[0]},
      morphizen::NodeAttributesBuilder()
          .add("to", int64_t(ONNX_NAMESPACE::TensorProto_DataType_BOOL))
          .build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  auto nodes = graph.nodes();
  ASSERT_EQ(nodes.size(), 1u);
  auto node = nodes[0];

  ASSERT_TRUE(node.has_attr("onnx_to_enum"))
      << "sanity check: attribute must actually be present for this test to "
         "be meaningful";

  // Same accessor the real ORT-EP bridge uses to republish node attributes
  // externally (NodeConstRef converts implicitly to const morphizen::Node&).
  const morphizen::Node &raw_node = node;
  auto &attrs = morphizen::node_get_attributes_ref(raw_node);
  auto keys = MORPHIZEN_ORT_API(node_attributes_get_keys)(
      const_cast<morphizen::NodeAttributes &>(attrs));
  ASSERT_TRUE(keys.get() != nullptr);

  for (const auto &key : *keys) {
    EXPECT_NE(key, "onnx_to_enum")
        << "onnx_to_enum must not appear in the user-visible attribute keys";
  }

  // "to" is a real, legitimate user-visible ONNX attribute and should still
  // be present -- confirms the skip only filters onnx_to_enum, not Cast's
  // attributes wholesale.
  bool found_to = false;
  for (const auto &key : *keys) {
    if (key == "to") {
      found_to = true;
      break;
    }
  }
  EXPECT_TRUE(found_to) << "'to' should remain a user-visible attribute";

  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}
