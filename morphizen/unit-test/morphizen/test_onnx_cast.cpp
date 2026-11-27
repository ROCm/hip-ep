/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test_environment.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <gtest/gtest.h>

namespace {

bool isMLIRBackendActive() {
  auto *api = morphizen::get_the_global_api_unsafe();
  if (!api || !api->get_lib_name)
    return false;
  auto lib_name = api->get_lib_name();
  return lib_name.get() &&
         std::string(lib_name->c_str()) == "morphizen-mlir-imp";
}

// MLIR bytecode starts with the magic bytes 'M' 'L' 0xEF 'R'.
bool isMLIRTextFormat(const std::string &mlir) {
  return !(mlir.size() >= 4 && mlir[0] == 'M' && mlir[1] == 'L' &&
           static_cast<unsigned char>(mlir[2]) == 0xEF && mlir[3] == 'R');
}

// Imports a one-node Cast graph and returns the serialized MLIR.
std::string importCast(const char *file, int64_t onnx_to,
                       ONNX_NAMESPACE::TensorProto_DataType output_type) {
  auto path = CMAKE_CURRENT_BINARY_PATH / std::filesystem::path(file);
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};
  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("x", {4}, ONNX_NAMESPACE::TensorProto_DataType_INT64)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("y", {4}, output_type)};
  graph.add_node("cast_0", "", "Cast", "", input, output,
                 morphizen::NodeAttributesBuilder().add("to", onnx_to).build());
  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  auto mlir = graph.save_string();
  return mlir.get() ? *mlir : std::string();
}

} // namespace

class OnnxCastTest : public ::testing::Test {};

// BOOL tensors are stored as ui8, so the importer marks a cast to BOOL with
// `to = i1`; otherwise the compiler could not tell it from Cast(to=UINT8).
TEST_F(OnnxCastTest, CastToBoolKeepsBoolInToAttribute) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive())
    GTEST_SKIP() << "MLIR backend not active at runtime";

  std::string mlir = importCast("onnx_cast_bool.onnx",
                                ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  ASSERT_FALSE(mlir.empty());
  if (!isMLIRTextFormat(mlir))
    GTEST_SKIP() << "MLIR output is bytecode";
  EXPECT_NE(mlir.find("to = i1"), std::string::npos) << mlir;
}

TEST_F(OnnxCastTest, CastToUint8StaysUint8InToAttribute) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive())
    GTEST_SKIP() << "MLIR backend not active at runtime";

  std::string mlir = importCast("onnx_cast_uint8.onnx",
                                ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                                ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  ASSERT_FALSE(mlir.empty());
  if (!isMLIRTextFormat(mlir))
    GTEST_SKIP() << "MLIR output is bytecode";
  EXPECT_NE(mlir.find("to = ui8"), std::string::npos) << mlir;
  EXPECT_EQ(mlir.find("to = i1"), std::string::npos) << mlir;
}
