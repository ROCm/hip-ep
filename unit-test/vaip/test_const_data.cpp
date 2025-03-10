/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "debug_logger.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "unit_test_env_params.hpp"
#include "vaip/vaip.hpp"

class ConstDataTest : public DebugLogger {
public:
  template <typename F> void run_test(F check) {
    LOG(INFO) << "LOADING " << ENV_PARAM(TEST_CONSTANT_INITIALIZER_ONNX)
              << std::endl;
    auto model =
        vaip_cxx::Model::load(ENV_PARAM(TEST_CONSTANT_INITIALIZER_ONNX));
    auto cloned_model = model->clone();
    graph = std::make_unique<vaip_cxx::GraphRef>(cloned_model->main_graph());
    graph->resolve();
    check();
  }

public:
  std::unique_ptr<vaip_cxx::GraphRef> graph;
};

TEST_F(ConstDataTest, int8_scalar) {
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_int8_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT8);
    auto const_data = const_value_opt.value().const_data_as_i8();
    EXPECT_EQ(const_data, -1);
  });
}

TEST_F(ConstDataTest, int8) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_uint8_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    auto const_data = const_value_opt.value().const_data_as_u8();
    EXPECT_EQ(const_data, 2);
  });
}

TEST_F(ConstDataTest, uint8) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_int16_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT16);
    auto const_data = const_value_opt.value().const_data_as_i16();
    EXPECT_EQ(const_data, -3);
  });
}

TEST_F(ConstDataTest, int16) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_uint16_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    auto const_data = const_value_opt.value().const_data_as_u16();
    EXPECT_EQ(const_data, 4);
  });
}

TEST_F(ConstDataTest, uint16) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_int32_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto const_data = const_value_opt.value().const_data_as_i32();
    EXPECT_EQ(const_data, -5);
  });
}

TEST_F(ConstDataTest, int32) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_uint32_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    auto const_data = const_value_opt.value().const_data_as_u32();
    EXPECT_EQ(const_data, 6);
  });
}

TEST_F(ConstDataTest, uint32) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_int64_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto const_data = const_value_opt.value().const_data_as_i64();
    EXPECT_EQ(const_data, -7);
  });
}

TEST_F(ConstDataTest, int64) {
  run_test([this]() {
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
  run_test([this]() {
    auto const_value_opt = graph->find_node_arg("const_uint64_scalar");
    ASSERT_TRUE(const_value_opt);
    EXPECT_EQ(const_value_opt.value().element_type(),
              ONNX_NAMESPACE::TensorProto_DataType_UINT64);
    auto const_data = const_value_opt.value().const_data_as_u64();
    EXPECT_EQ(const_data, 8);
  });
}

TEST_F(ConstDataTest, uint64) {
  run_test([this]() {
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
