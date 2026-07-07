/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test_environment.hpp"
#include "morphizen/morphizen.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>

class ModelTest : public ::testing::Test {};

TEST_F(ModelTest, Load) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  LOG(INFO) << "model: " << model->ref().name() << " is loaded";
}

TEST_F(ModelTest, Clone) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto cloned_model = model->ref().clone();
  LOG(INFO) << "cloned model: " << cloned_model->ref().name() << " is cloned";
  cloned_model->main_graph().save(CMAKE_CURRENT_BINARY_PATH /
                                  "resnet50_cloned.onnx");
}

TEST_F(ModelTest, MainGraph) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  LOG(INFO) << "main graph: " << graph.name() << " is loaded";
}

TEST_F(ModelTest, SetAndGetMetadata) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);

  // Set metadata
  std::string key = "author";
  std::string value = "John Doe";
  model->set_metadata(key, value);

  // Get metadata
  std::string retrievedValue = model->ref().get_metadata(key);
  EXPECT_EQ(retrievedValue, value);

  // Has metadata
  EXPECT_TRUE(model->ref().has_metadata(key));
  EXPECT_FALSE(model->ref().has_metadata("non-existing-key"));
}

TEST_F(ModelTest, ImplicitConversion) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);

  // Implicit conversion to onnxruntime::Model reference
  onnxruntime::Model &ortModel = *model;

  // Implicit conversion to const onnxruntime::Model reference
  const onnxruntime::Model &constOrtModel = *model;

  // Perform assertions or further operations on the converted models
  // ...
  auto name = MORPHIZEN_ORT_API(graph_get_name)(
      MORPHIZEN_ORT_API(model_main_graph)(*model));
  LOG(INFO) << "model: " << name << " is loaded. ptr=" << (void *)&constOrtModel
            << " " << (void *)&ortModel;
}

TEST_F(ModelTest, ModelCreationTest) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #034): MLIR backend dyn_cast assertion failure in Model::create
  GTEST_SKIP() << "MLIR backend: dyn_cast assertion failure (Issue #034)";
#endif
  auto path = CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("new.onnx");
  auto data_path = std::filesystem::path("new.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {
      {"appple", 10}, {"banana", 20}, {"cherry", 30}};
  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();
  std::vector<int64_t> i_shape = {8};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> i = {
      graph.new_node_arg("my_input", i_shape,
                         ONNX_NAMESPACE::TensorProto_DataType_INT64)};
  std::vector<int64_t> o_shape = {8};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> o = {
      graph.new_node_arg("my_output", i_shape,
                         ONNX_NAMESPACE::TensorProto_DataType_INT64)};

  auto newly_created_node =
      graph.add_node("My_ReLu_unique_name", "", "Relu", "test", i, o,
                     morphizen::NodeAttributesBuilder().build());

  LOG(INFO) << "happy, a new node is created " << newly_created_node;
  graph.set_inputs({i[0].value()});
  graph.set_outputs({o[0].value()});
  EXPECT_EQ(path, graph.model_path());
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  // ASSERT_TRUE(std::filesystem::exists(CMAKE_CURRENT_BINARY_PATH /
  // data_path));
}
