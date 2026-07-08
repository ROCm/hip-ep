/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/morphizen.hpp"
#include "test_environment.hpp"
#ifdef MORPHIZEN_ENABLE_BOOST
#include <boost/process.hpp>
#endif
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
class GraphTest : public ::testing::Test {};

TEST_F(GraphTest, LoadAndSave) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.set_name("resent50_by_morphizen");
  graph.resolve();
  LOG(INFO) << "model: " << graph.name() << " is loaded" << std::endl;
  auto inputs = graph.inputs();
  LOG(INFO) << "graph inputs:";
  for (auto &input : inputs) {
    LOG(INFO) << "  " << input.name() << " => " << input;
  }
  auto outputs = graph.outputs();
  LOG(INFO) << "graph outputs:";
  for (auto &output : outputs) {
    LOG(INFO) << "  " << output.name() << " => " << output;
  }
  auto c = 0;
  LOG(INFO) << "first 10 initializers:";
  for (auto &init : graph.constant_initializers()) {
    LOG(INFO) << " " << c++ << " " << init << std::endl;
  }
  for (auto &init : graph.constant_initializers()) {
    // NOTE: constant_initializers is not ordered.
    LOG(INFO) << "  " << init.name() << " => " << init;
    EXPECT_EQ(init.is_constant(), true);
    if (init.name() == "1261") {
      EXPECT_EQ(init.is_unknown_shape(), false) << init;
      EXPECT_EQ(init.is_scalar(), true) << init;
      EXPECT_EQ(init.is_zero_shape(), false) << init;
      EXPECT_EQ(init.element_type(), 3) << init;
      auto data = init.const_data_as_i8();
      EXPECT_EQ(data, 0);
    }
    if (init.name() == "module_10.weight") {
      EXPECT_EQ(init.is_unknown_shape(), false) << init;
      EXPECT_EQ(init.is_scalar(), false) << init;
      EXPECT_EQ(init.is_zero_shape(), false) << init;
      EXPECT_EQ(init.element_type(), 1) << init;
      auto data = init.const_data_as_f32_span();
      EXPECT_EQ(data.size(), 16384u);
    }
  }
  auto output_dir = CMAKE_CURRENT_BINARY_PATH;

  auto resnet50_file = output_dir / "resnet50.onnx";
  std::filesystem::remove(resnet50_file);
  std::filesystem::remove(output_dir / "resnet50.dat");
  LOG(INFO) << "Saving file to " << resnet50_file.u8string();
  graph.save(resnet50_file.u8string(), "resnet50.dat", 128u);
  // graph.save_without(resnet50_file.u8string(), "resnet50.dat", 128u);
  EXPECT_TRUE(std::filesystem::exists(resnet50_file));
  // TODO: fix this bug:
  // 1. the resnet50.onnx is too big, external data is still not supported
  // 2. resnet50.dat is stored next to "pt_resnet50.onnx", i.e. the original
  // model, we need to save it to next to "resnet50.onnx"
  // 3. make sure "exteran_data.location" is relative path.
  // EXPECT_TRUE(std::filesystem::exists("resnet50.dat"));
  {
    auto model_1 = morphizen_cxx::Model::load(resnet50_file);
    auto graph_1 = model_1->main_graph();
    if (morphizen::MorphizenOrtApi2::has_graph_set_name) {
      EXPECT_EQ(graph_1.name(), "resent50_by_morphizen");
    }
    std::ostringstream python_code;
    python_code << "import onnx"
                << "\n";
    python_code << "m=onnx.load(r\"" << resnet50_file.u8string() << "\")"
                << "\n";
    python_code << "onnx.checker.check_model(m)"
                << "\n";
#ifdef MORPHIZEN_ENABLE_BOOST
    auto exit_code =
        boost::process::system(PYTHON_EXE.u8string(), "-c", python_code.str());
    EXPECT_EQ(exit_code, 0) << "onnx.checker.check_model failed";
#else
    GTEST_SKIP()
        << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
  }
}
TEST_F(GraphTest, CloneAndSave) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto cloned_model = model->ref().clone();
  auto main_graph = cloned_model->main_graph();
  auto output_dir = CMAKE_CURRENT_BINARY_PATH;
  auto resnet50_file = output_dir / "resnet50_for_mut_save.onnx";
  LOG(INFO) << " save graph to " << resnet50_file;
  auto filter_out_special_tensor = true;
  main_graph.mut_save(resnet50_file.u8string(), "resnet50_for_mut_save.dat",
                      128u, filter_out_special_tensor);
}
TEST_F(GraphTest, FindNodeArgGraphInput) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto optionalNodeArg = graph.find_node_arg("blob.1");
  EXPECT_TRUE(optionalNodeArg.has_value());
  auto nodeArg = optionalNodeArg.value();
  LOG(INFO) << "nodeArg: " << nodeArg.name() << " => " << nodeArg;
  EXPECT_EQ(nodeArg.name(), "blob.1");
  EXPECT_EQ(nodeArg.is_graph_input(), true);
  EXPECT_EQ(nodeArg.is_graph_output(), false);
  EXPECT_EQ(nodeArg.is_constant(), false);
  EXPECT_EQ(nodeArg.is_unknown_shape(), false);
  EXPECT_EQ(nodeArg.is_scalar(), false);
  EXPECT_EQ(nodeArg.is_zero_shape(), false);
  EXPECT_EQ(nodeArg.element_type(), 1);
  auto shape = nodeArg.shape();
  EXPECT_TRUE(shape != nullptr);
  EXPECT_EQ(*shape, std::vector<int64_t>({1, 3, 224, 224}));
}
TEST_F(GraphTest, FindNodeArgGraphOutput) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto optionalNodeArg = graph.find_node_arg("1327");
  EXPECT_TRUE(optionalNodeArg.has_value());
  auto nodeArg = optionalNodeArg.value();
  LOG(INFO) << "nodeArg: " << nodeArg.name() << " => " << nodeArg;
  EXPECT_EQ(nodeArg.name(), "1327");
  EXPECT_EQ(nodeArg.is_graph_input(), false);
  EXPECT_EQ(nodeArg.is_graph_output(), true);
  EXPECT_EQ(nodeArg.is_constant(), false);
  EXPECT_EQ(nodeArg.is_unknown_shape(), false);
  EXPECT_EQ(nodeArg.is_scalar(), false);
  EXPECT_EQ(nodeArg.is_zero_shape(), false);
  EXPECT_EQ(nodeArg.element_type(), 1);
  auto shape = nodeArg.shape();
  EXPECT_TRUE(shape != nullptr);
  EXPECT_EQ(*shape, std::vector<int64_t>({1, 1000}));
}

TEST_F(GraphTest, NodesInTopologicalOrder) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  LOG(INFO) << "model: " << graph.name() << " is loaded" << std::endl;
  auto nodes = graph.nodes_in_topological_order();
  for (auto &node : nodes) {
    LOG(INFO) << node;
    auto intputs = node.inputs();
    auto outputs = node.outputs();
    for (auto input : intputs) {
      if (input.has_value()) {
        LOG(INFO) << "  input: " << input.value().name() << " => "
                  << input.value();
      } else {
        LOG(INFO) << "  input: nullptr";
      }
      for (auto output : outputs) {
        if (output.has_value()) {
          LOG(INFO) << "  output: " << output.value().name() << " => "
                    << output.value();
        } else {
          LOG(INFO) << "  output: nullptr";
        }
      }
    }
  }
}

TEST_F(GraphTest, NodeIndex) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node = graph.find_node("1321");
  EXPECT_TRUE(node.has_value());
  auto node_index = node.value().index();
  EXPECT_EQ(node_index, 487); // ? ORT can make node index stable?
  auto op_type = node.value().op_type();
  EXPECT_EQ(op_type, "Gemm");
  auto op_domain = node.value().op_domain();
  EXPECT_EQ(op_domain, "");
}
TEST_F(GraphTest, FindConsumers) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto optionalNodeArg = graph.find_node_arg("1321");
  EXPECT_TRUE(optionalNodeArg.has_value());
  auto nodeArg = optionalNodeArg.value();
  LOG(INFO) << "nodeArg: " << nodeArg.name() << " => " << nodeArg;
  auto nodes = nodeArg.find_consumers();
  EXPECT_EQ(nodes.size(), 1u);
  LOG(INFO) << "consumers is " << nodes[0];
}

TEST_F(GraphTest, NodeArgFindProducer) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  LOG(INFO) << "model: " << graph.name() << " is loaded" << std::endl;
  auto node_arg = graph.find_node_arg("1327");
  EXPECT_EQ(node_arg.has_value(), true) << "cannot find node_arg 1327";
  auto node = node_arg.value().find_producer();
  ASSERT_TRUE(node.has_value());
  LOG(INFO) << "found node's producer: " << node.value();
}
TEST_F(GraphTest, Fuse) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto meta_def = morphizen::MetaDefProto();
  meta_def.add_inputs("111");
  meta_def.add_outputs("138");
  meta_def.add_nodes("138");
  meta_def.add_nodes("135");
  meta_def.add_nodes("134");
  meta_def.add_nodes("131");
  meta_def.add_nodes("128");
  meta_def.add_nodes("127");
  meta_def.add_nodes("126");
  meta_def.add_nodes("123");
  meta_def.add_nodes("120");
  meta_def.add_nodes("117");
  meta_def.add_nodes("114");
  meta_def.add_constant_initializers("112");
  meta_def.add_constant_initializers("113");
  meta_def.add_constant_initializers("115");
  meta_def.add_constant_initializers("116");
  meta_def.add_constant_initializers("118");
  meta_def.add_constant_initializers("119");
  meta_def.add_constant_initializers("121");
  meta_def.add_constant_initializers("122");
  meta_def.add_constant_initializers("124");
  meta_def.add_constant_initializers("125");
  meta_def.add_constant_initializers("129");
  meta_def.add_constant_initializers("130");
  meta_def.add_constant_initializers("132");
  meta_def.add_constant_initializers("133");
  meta_def.add_constant_initializers("136");
  meta_def.add_constant_initializers("137");
  meta_def.add_constant_initializers("module_2.bias");
  meta_def.add_constant_initializers("module_2.weight");
  auto node = morphizen_cxx::graph_fuse(graph, meta_def);
  LOG(INFO) << " fused_node=" << node;
  // FIXME: support save subgrahp
  // the saved graph cannot be read by Netron.
  // graph.save("C:\\temp\\a.onnx", "C:\\temp\\a.dat", 128u);
  LOG(INFO) << "fused node: " << node << std::endl;
  LOG(INFO) << "graph after fuse: " << graph << std::endl;
}

TEST_F(GraphTest, TryFuse) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto [meta_def, error] = morphizen_cxx::graph_try_fuse(
      graph, "a_name", {"111"}, {"138"}, {}, "CUSTOM");
  ASSERT_TRUE(meta_def != nullptr) << error.comments;
  LOG(INFO) << " fused_node=" << meta_def->DebugString();
}

TEST_F(GraphTest, NewConstantInitializer) {
#ifdef MORPHIZEN_ENABLE_BOOST
  auto SAMPLE_ONNX = CMAKE_CURRENT_BINARY_PATH / "sample.onnx";
  auto exit_code = boost::process::system(
      PYTHON_EXE.u8string(),
      (TEST_SRC_DIR / "morphizen" / "create_sample_onnx_model.py").u8string(),
      SAMPLE_ONNX.u8string());
  ASSERT_TRUE(exit_code == 0) << "Failed to generate test file";
  LOG(INFO) << "LOADING " << (SAMPLE_ONNX) << std::endl;
  auto model = morphizen_cxx::Model::load(SAMPLE_ONNX);
  auto graph = morphizen_cxx::GraphRef(model->main_graph());
  graph.resolve();
  // Test for new_constant_initializer_i8

  auto new_i8 = graph.new_constant_initializer_i8(100);
  LOG(INFO) << "new_i8.name() = " << new_i8.name();
  EXPECT_EQ(new_i8.const_data_as_i8(), 100);

  // Test for new_constant_initializer_u8

  auto new_u8 = graph.new_constant_initializer_u8(101);
  LOG(INFO) << "new_u8.name() = " << new_u8.name();
  EXPECT_EQ(new_u8.const_data_as_u8(), 101);

  // Test for new_constant_initializer_i16

  auto new_i16 = graph.new_constant_initializer_i16(1000);
  LOG(INFO) << "new_i16.name() = " << new_i16.name();
  EXPECT_EQ(new_i16.const_data_as_i16(), 1000);

  // Test for new_constant_initializer_u16

  auto new_u16 = graph.new_constant_initializer_u16(1001);
  LOG(INFO) << "new_u16.name() = " << new_u16.name();
  EXPECT_EQ(new_u16.const_data_as_u16(), 1001);

  // Test for new_constant_initializer_i32

  auto new_i32 = graph.new_constant_initializer_i32(100000);
  LOG(INFO) << "new_i32.name() = " << new_i32.name();
  EXPECT_EQ(new_i32.const_data_as_i32(), 100000);

  // Test for new_constant_initializer_u32

  auto new_u32 = graph.new_constant_initializer_u32(100001);
  LOG(INFO) << "new_u32.name() = " << new_u32.name();
  EXPECT_EQ(new_u32.const_data_as_u32(), 100001);

  // Test for new_constant_initializer_i64

  auto new_i64 = graph.new_constant_initializer_i64(10000000);
  LOG(INFO) << "new_i64.name() = " << new_i64.name();
  EXPECT_EQ(new_i64.const_data_as_i64(), 10000000);

  // Test for new_constant_initializer_u64

  auto new_u64 = graph.new_constant_initializer_u64(100000001);
  LOG(INFO) << "new_u64.name() = " << new_u64.name();
  EXPECT_EQ(new_u64.const_data_as_u64(), 100000001);

  // Test for new_constant_initializer_f32
  auto new_f32 = graph.new_constant_initializer_f32(1024.0f);
  LOG(INFO) << "new_f32.name() = " << new_f32.name();
  EXPECT_EQ(new_f32.const_data_as_f32(), 1024.0f);

  auto new_f64 = graph.new_constant_initializer_f64(2048.0);
  LOG(INFO) << "new_f64.name() = " << new_f64.name();
  EXPECT_EQ(new_f64.const_data_as_f64(), 2048.0f);

  auto new_bf16 = graph.new_constant_initializer_bf16(16);
  LOG(INFO) << "new_bf16.name() = " << new_bf16.name();
  EXPECT_EQ(new_bf16.const_data_as_bf16(), 16);

  auto new_fp16 = graph.new_constant_initializer_fp16(17);
  LOG(INFO) << "new_fp16.name() = " << new_fp16.name();
  EXPECT_EQ(new_fp16.const_data_as_fp16(), 17);

  // Test for new_constant_initializer_u8_span

  auto new_u8_span = graph.new_constant_initializer_u8_span(
      std::vector<uint8_t>{100, 101, 102}, {1, 3});
  LOG(INFO) << "new_u8_span.name() = " << new_u8_span.name();
  {
    auto data = new_u8_span.const_data_as_u8_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 100);
    EXPECT_EQ(data[1], 101);
    EXPECT_EQ(data[2], 102);
  }

  // Test for new_constant_initializer_i16_span

  auto new_i16_span = graph.new_constant_initializer_i16_span(
      std::vector<int16_t>{1000, 1001, 1002}, {1, 3});
  LOG(INFO) << "new_i16_span.name() = " << new_i16_span.name();
  {
    auto data = new_i16_span.const_data_as_i16_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 1000);
    EXPECT_EQ(data[1], 1001);
    EXPECT_EQ(data[2], 1002);
  }

  // Test for new_constant_initializer_u16_span

  auto new_u16_span = graph.new_constant_initializer_u16_span(
      std::vector<uint16_t>{10000, 10001, 10002}, {1, 3});
  LOG(INFO) << "new_u16_span.name() = " << new_u16_span.name();
  {
    auto data = new_u16_span.const_data_as_u16_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 10000);
    EXPECT_EQ(data[1], 10001);
    EXPECT_EQ(data[2], 10002);
  }

  // Test for new_constant_initializer_i32_span

  auto new_i32_span = graph.new_constant_initializer_i32_span(
      std::vector<int32_t>{100000, 100001, 100002}, {1, 3});
  LOG(INFO) << "new_i32_span.name() = " << new_i32_span.name();
  {
    auto data = new_i32_span.const_data_as_i32_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 100000);
    EXPECT_EQ(data[1], 100001);
    EXPECT_EQ(data[2], 100002);
  }

  // Test for new_constant_initializer_u32_span

  auto new_u32_span = graph.new_constant_initializer_u32_span(
      std::vector<uint32_t>{1000000, 1000001, 1000002}, {1, 3});
  LOG(INFO) << "new_u32_span.name() = " << new_u32_span.name();
  {
    auto data = new_u32_span.const_data_as_u32_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 1000000);
    EXPECT_EQ(data[1], 1000001);
    EXPECT_EQ(data[2], 1000002);
  }

  // Test for new_constant_initializer_i64_span

  auto new_i64_span = graph.new_constant_initializer_i64_span(
      std::vector<int64_t>{10000000, 10000001, 10000002}, {1, 3});
  LOG(INFO) << "new_i64_span.name() = " << new_i64_span.name();
  {
    auto data = new_i64_span.const_data_as_i64_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 10000000);
    EXPECT_EQ(data[1], 10000001);
    EXPECT_EQ(data[2], 10000002);
  }

  // Test for new_constant_initializer_u64_span

  auto new_u64_span = graph.new_constant_initializer_u64_span(
      std::vector<uint64_t>{100000000, 100000001, 100000002}, {1, 3});
  LOG(INFO) << "new_u64_span.name() = " << new_u64_span.name();
  {
    auto data = new_u64_span.const_data_as_u64_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 100000000);
    EXPECT_EQ(data[1], 100000001);
    EXPECT_EQ(data[2], 100000002);
  }

  // Test for new_constant_initializer_u64_span

  auto new_f32_span = graph.new_constant_initializer_f32_span(
      std::vector<float>{32.0f, 64.0f, 128.0f}, {1, 3});
  LOG(INFO) << "new_f32_span.name() = " << new_f32_span.name();
  {
    auto data = new_f32_span.const_data_as_f32_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 32.0f);
    EXPECT_EQ(data[1], 64.0f);
    EXPECT_EQ(data[2], 128.0f);
  }

  auto new_f64_span = graph.new_constant_initializer_f64_span(
      std::vector<double>{2.0, 4.0, 8.0}, {1, 3});
  LOG(INFO) << "new_f64_span.name() = " << new_f64_span.name();
  {
    auto data = new_f64_span.const_data_as_f64_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 2.0);
    EXPECT_EQ(data[1], 4.0);
    EXPECT_EQ(data[2], 8.0);
  }
  auto new_bf16_span = graph.new_constant_initializer_bf16_span(
      std::vector<morphizen_cxx::bf16_t>{2, 4, 8}, {1, 3});
  LOG(INFO) << "new_bf16_span.name() = " << new_bf16_span.name();
  {
    auto data = new_bf16_span.const_data_as_bf16_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], 8);
  }

  auto new_fp16_span = graph.new_constant_initializer_fp16_span(
      std::vector<morphizen_cxx::fp16_t>{2, 4, 8}, {1, 3});
  LOG(INFO) << "new_fp16_span.name() = " << new_fp16_span.name();
  {
    auto data = new_fp16_span.const_data_as_fp16_span();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], 8);
  }

  auto output_node = graph.find_node("output");
  ASSERT_TRUE(output_node.has_value());
  auto input_node_arg = graph.find_node_arg("input");
  ASSERT_TRUE(input_node_arg.has_value());
  // try to use these constant intializers;
  std::shared_ptr<morphizen::PassContext> context =
      morphizen::PassContext::create();
  if (1) {
    auto pass_proto = std::make_unique<morphizen::PassProto>();
    pass_proto->set_plugin("morphizen-pass_init");
    pass_proto->set_name("GraphTest.NewConstantInitializer");
    auto pass = morphizen::IPass::create_pass(context, *pass_proto);
    auto newly_added_node = morphizen_cxx::graph_node_builder(graph, *pass)
                                .set_input_node_args_ex({
                                    input_node_arg.value(),
                                    new_i8,
                                    new_u8,
                                    new_i16,
                                    new_u16,
                                    new_i32,
                                    new_u32,
                                    new_i64,
                                    new_u64,
                                    new_f32,
                                    new_f64,
                                    new_bf16,
                                    new_fp16,
                                    new_u8_span,
                                    new_i16_span,
                                    new_u16_span,
                                    new_i32_span,
                                    new_u32_span,
                                    new_i64_span,
                                    new_u64_span,
                                    new_f32_span,
                                    new_f64_span,
                                    new_bf16_span,
                                    new_fp16_span,
                                })
                                .set_op_type("conv2d")
                                .set_data_type("int8")
                                .set_shape(std::vector<int64_t>{1, 224, 224, 3})
                                .set_anchor_point1(output_node.value())
                                .build_ex();
    LOG(INFO) << "newly added node is " << newly_added_node;
    graph.save(CMAKE_CURRENT_BINARY_PATH / "new_constant_initializer.onnx",
               "new_constant_initializer.dat", 128u);
  }
#else
  GTEST_SKIP()
      << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
}

TEST_F(GraphTest, VirtualFuse) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto [meta_def, error] = morphizen_cxx::graph_try_fuse(
      graph, "a_name", {"111"}, {"138"}, {}, "CUSTOM");
  ASSERT_TRUE(meta_def != nullptr) << error.comments;
  LOG(INFO) << " fused_node=" << meta_def->DebugString();
  auto subgraph = morphizen_cxx::graph_virtual_fuse(graph, *meta_def);
  for (auto node : subgraph.nodes()) {
    LOG(INFO) << " node = " << node;
  }
  EXPECT_TRUE(true);
}
