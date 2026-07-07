/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "test_environment.hpp"
#ifdef MORPHIZEN_ENABLE_BOOST
#include <boost/process.hpp>
#endif
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "morphizen/morphizen.hpp"
const static std::filesystem::path TEST_CONSTANT_INITIALIZER_ONNX =
    TEST_CWD / "test_constant_initializer.onnx";
class ConstDataTest : public ::testing::Test {
public:
  template <typename F> void run_test(int line, F check) {
#ifdef MORPHIZEN_ENABLE_BOOST
    auto test_constant_initializer_onnx =
        TEST_CWD / (std::string("test_constant_initializer_") +
                    std::to_string(line) + ".onnx")
                       .c_str();
    auto exit_code = boost::process::system(
        PYTHON_EXE.u8string(),
        (TEST_SRC_DIR / "morphizen" / "test_constant_initializer.py")
            .u8string(),
        test_constant_initializer_onnx.u8string());
    ASSERT_TRUE(exit_code == 0) << "Failed to generate test file";
    ASSERT_TRUE(std::filesystem::exists(test_constant_initializer_onnx));
    LOG(INFO) << "LOADING " << test_constant_initializer_onnx << std::endl;
    auto model = morphizen_cxx::Model::load(test_constant_initializer_onnx);
    auto cloned_model = model->ref().clone();
    graph =
        std::make_unique<morphizen_cxx::GraphRef>(cloned_model->main_graph());
    graph->resolve();
    check();
#else
    GTEST_SKIP()
        << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
  }

public:
  std::unique_ptr<morphizen_cxx::GraphRef> graph;
};

TEST_F(ConstDataTest, int8_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int8_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT8);
    auto const_data = const_value_opt.value().const_data_as_i8();
    EXPECT_EQ(const_data, -1);
  });
}

TEST_F(ConstDataTest, int8) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int8");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT8);
    auto const_data = const_value_opt.value().const_data_as_i8_span();
    auto const_data_value =
        std::vector<int8_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<int8_t>({-1, -2}));
  });
}

TEST_F(ConstDataTest, uint8_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint8_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    auto const_data = const_value_opt.value().const_data_as_u8();
    EXPECT_EQ(const_data, 2);
  });
}

TEST_F(ConstDataTest, uint8) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint8");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    auto const_data = const_value_opt.value().const_data_as_u8_span();
    auto const_data_value =
        std::vector<uint8_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<uint8_t>({3, 4}));
  });
}

TEST_F(ConstDataTest, int16_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int16_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT16);
    auto const_data = const_value_opt.value().const_data_as_i16();
    EXPECT_EQ(const_data, -3);
  });
}

TEST_F(ConstDataTest, int16) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int16");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT16);
    auto const_data = const_value_opt.value().const_data_as_i16_span();
    auto const_data_value =
        std::vector<int16_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<int16_t>({-3, -4}));
  });
}

TEST_F(ConstDataTest, uint16_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint16_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    auto const_data = const_value_opt.value().const_data_as_u16();
    EXPECT_EQ(const_data, 4);
  });
}

TEST_F(ConstDataTest, uint16) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint16");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    auto const_data = const_value_opt.value().const_data_as_u16_span();
    auto const_data_value =
        std::vector<uint16_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<uint16_t>({5, 6}));
  });
}

TEST_F(ConstDataTest, int32_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int32_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto const_data = const_value_opt.value().const_data_as_i32();
    EXPECT_EQ(const_data, -5);
  });
}

TEST_F(ConstDataTest, int32) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int32");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto const_data = const_value_opt.value().const_data_as_i32_span();
    auto const_data_value =
        std::vector<int32_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<int32_t>({-5, -6}));
  });
}

TEST_F(ConstDataTest, uint32_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint32_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    auto const_data = const_value_opt.value().const_data_as_u32();
    EXPECT_EQ(const_data, 6);
  });
}

TEST_F(ConstDataTest, uint32) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint32");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    auto const_data = const_value_opt.value().const_data_as_u32_span();
    auto const_data_value =
        std::vector<uint32_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<uint32_t>({7, 8}));
  });
}

TEST_F(ConstDataTest, int64_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int64_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto const_data = const_value_opt.value().const_data_as_i64();
    EXPECT_EQ(const_data, -7);
  });
}

TEST_F(ConstDataTest, int64) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_int64");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto const_data = const_value_opt.value().const_data_as_i64_span();
    auto const_data_value =
        std::vector<int64_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<int64_t>({-7, -8}));
  });
}

TEST_F(ConstDataTest, uint64_scalar) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint64_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT64);
    auto const_data = const_value_opt.value().const_data_as_u64();
    EXPECT_EQ(const_data, 8);
  });
}

TEST_F(ConstDataTest, uint64) {
  run_test(__LINE__, [this]() {
    auto const_value_opt = graph->find_node_arg("const_uint64");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT64);
    auto const_data = const_value_opt.value().const_data_as_u64_span();
    auto const_data_value =
        std::vector<uint64_t>(const_data.begin(), const_data.end());
    EXPECT_EQ(const_data_value, std::vector<uint64_t>({9, 10}));
  });
}
