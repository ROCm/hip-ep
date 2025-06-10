/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/config_reader.hpp" // NOLINT
#include "morphizen/vaip.hpp"          // NOLINT
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
// clang-format off
// NOLINTBEGIN
#include "morphizen/vaip.hpp"
#include "./test_environment.hpp"
#include "../src/pass_context_imp.hpp"
// clang-format on
namespace vaip_core {
std::shared_ptr<PassContextImp>
initialize_context(const std::string& model_path,
                   const onnxruntime::Graph& onnx_graph,
                   const std::vector<vaip_cxx::NodeConstRef>& ep_context_nodes,
                   const onnxruntime::ProviderOptions& options);
}

// Test fixture for PassContext
class PassContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary resources before each test
    passContext = vaip_core::PassContext::create();
    dynamic_cast<vaip_core::PassContextImp*>(passContext.get())
        ->pass_context_log_dir_ = CMAKE_CURRENT_BINARY_PATH;
  }

  void TearDown() override {
    // Clean up any resources after each test
    passContext.reset();
  }

  // Pointer to the PassContext object
  std::unique_ptr<vaip_core::PassContext> passContext;
};

// Test case for read_file function
TEST_F(PassContextTest, ReadFileTest) {
  for (auto i = 0; i < 3; ++i) {
    // Test file name
    std::string filename =
        std::string("test_file_") + std::to_string(i) + ".txt";

    // Create a test file with some content
    std::string fileContent = "This is a test file for " + std::to_string(i);
    bool writeResult =
        passContext->write_file(filename, gsl::make_span(fileContent));

    // Assert that the file was successfully written
    ASSERT_TRUE(writeResult);

    // Read the file using the read_file function
    auto readResult = passContext->read_file_c8(filename);

    // Assert that the file was read successfully
    ASSERT_TRUE(readResult.has_value());

    // Assert that the content of the file matches the expected content
    ASSERT_EQ(std::string(readResult->data(), readResult->size()), fileContent);
  }
  auto tar_file_stream = vaip_core::IStreamWriter::from_path(
      CMAKE_CURRENT_BINARY_PATH / "ReadFileTest.tar");
  passContext->cache_files_to_tar_file(*tar_file_stream);
  // { // read it back from another pass context object.
  //   passContext->tar_file_to_cache_files(CMAKE_CURRENT_BINARY_PATH /
  //                                        "ReadFileTest.tar");
  //   for (auto i = 0; i < 3; ++i) {
  //     std::string filename =
  //         std::string("test_file_") + std::to_string(i) + ".txt";
  //     // Read the file using the read_file function
  //     auto readResult = passContext->read_file_c8(filename);

  //     // Assert that the file was read successfully
  //     ASSERT_TRUE(readResult.has_value());
  //     // Assert that the content of the file matches the expected content
  //     std::string fileContent = "This is a test file for " +
  //     std::to_string(i); ASSERT_EQ(std::string(readResult->data(),
  //     readResult->size()),
  //               fileContent);
  //   }
  // }
  // { // read it back from a chunk of memory.
  //   std::ifstream tar_ball(CMAKE_CURRENT_BINARY_PATH / "ReadFileTest.tar",
  //                          std::ios::binary);
  //   // read the whole file into `content` from tar_ball.
  //   std::vector<char> content((std::istreambuf_iterator<char>(tar_ball)),
  //                             std::istreambuf_iterator<char>());
  //   passContext->tar_mem_to_cache_files(content.data(), content.size());
  //   for (auto i = 0; i < 3; ++i) {
  //     std::string filename =
  //         std::string("test_file_") + std::to_string(i) + ".txt";
  //     // Read the file using the read_file function
  //     auto readResult = passContext->read_file_c8(filename);

  //     // Assert that the file was read successfully
  //     ASSERT_TRUE(readResult.has_value());
  //     // Assert that the content of the file matches the expected content
  //     std::string fileContent = "This is a test file for " +
  //     std::to_string(i); ASSERT_EQ(std::string(readResult->data(),
  //     readResult->size()),
  //               fileContent);
  //   }
  // }
}
TEST_F(PassContextTest, UntarCacheTest) {
  for (auto i = 0; i < 3; ++i) {
    // Test file name
    std::string filename =
        std::string("UntarCacheTest.test_file_") + std::to_string(i) + ".txt";

    // Create a test file with some content
    std::string fileContent = "This is a test file for " + std::to_string(i);
    bool writeResult =
        passContext->write_file(filename, gsl::make_span(fileContent));

    ASSERT_TRUE(writeResult);
  }
}

TEST_F(PassContextTest, TestEmptyFiles) {
  auto buffer = std::vector<char>{};
  for (auto i = 0; i < 3; ++i) {
    // Test file name
    std::string filename =
        std::string("TestEmptyFiles.test_file_") + std::to_string(i) + ".txt";

    // Create a test file with some content
    std::string fileContent = "This is a test file for " + std::to_string(i);
    if (i == 1) {
      fileContent = "";
    }
    bool writeResult =
        passContext->write_file(filename, gsl::make_span(fileContent));

    ASSERT_TRUE(writeResult);
    auto tar_file_stream = vaip_core::IStreamWriter::from_path(
        CMAKE_CURRENT_BINARY_PATH / "TestEmptyFiles.tar");
    passContext->cache_files_to_tar_file(*tar_file_stream);
    buffer = passContext->cache_files_to_tar_mem();
  }
  {
    // passContext->tar_mem_to_cache_files(&buffer[0], buffer.size());
    for (auto i = 0; i < 3; ++i) {
      // Test file name
      std::string filename =
          std::string("TestEmptyFiles.test_file_") + std::to_string(i) + ".txt";

      // Create a test file with some content
      std::string fileContent = "This is a test file for " + std::to_string(i);
      if (i == 1) {
        fileContent = "";
      }
      // Read the file using the read_file function
      auto readResult = passContext->read_file_c8(filename);

      // Assert that the file was read successfully
      ASSERT_TRUE(readResult.has_value());
      ASSERT_EQ(std::string(readResult->data(), readResult->size()),
                fileContent);
    }
  }
}

// TEST_F(PassContextTest, OnDiskTarTest) {
//   auto ctx1 = vaip_core::PassContext::create();
//   auto dir = std::filesystem::current_path() / "OnDiskTarTest";
//   if (std::filesystem::exists(dir)) {
//     std::filesystem::remove_all(dir);
//   }
//   std::filesystem::create_directory(dir);

//   auto res1_dir = dir / "res1_dir";
//   auto res2_dir = dir / "res2_dir";
//   auto res3_dir = dir / "res3_dir";
//   std::filesystem::create_directory(res1_dir);
//   std::filesystem::create_directory(res2_dir);
//   std::filesystem::create_directory(res3_dir);

//   auto empty_file_path = dir / "empty_file.txt";
//   std::ofstream empty_file(empty_file_path);
//   empty_file.close();
//   auto simple_file_path = dir / "simple_file.txt";
//   std::ofstream simple_file(simple_file_path);
//   simple_file << "12345\n23456\n";
//   simple_file.close();
//   auto pad_file_path = dir / "pad_file.txt";
//   std::ofstream pad_file(pad_file_path, std::ios::binary);
//   for (int i = 0; i < 512; i++) {
//     pad_file << static_cast<char>(i % 256);
//   }
//   pad_file.close();

//   auto recursive_path = dir / "deep";
//   std::filesystem::create_directory(recursive_path);
//   recursive_path = recursive_path / "deeper";
//   std::filesystem::create_directory(recursive_path);
//   auto deep_file_path = recursive_path / "deep.bin";
//   std::ofstream deep_file(deep_file_path, std::ios::binary);
//   std::vector<uint64_t> long_bytes(1013 * 1023 * 671);
//   std::fill(long_bytes.begin(), long_bytes.end(), 9);
//   deep_file.write(reinterpret_cast<const char*>(long_bytes.data()),
//                   long_bytes.size());
//   deep_file.close();

//   auto tar_file_path = dir / "x.tar";
//   auto tar_file_stream = vaip_core::IStreamWriter::from_path(tar_file_path);
//   std::ignore = ctx1->cache_files_to_tar_file(*tar_file_stream);

//   auto ctx2 = vaip_core::PassContext::create();
//   ctx2->tar_file_to_cache_files(tar_file_path);

//   auto bytes = ctx1->cache_files_to_tar_mem();
//   auto ctx3 = vaip_core::PassContext::create();
//   ctx3->tar_mem_to_cache_files(bytes.data(), bytes.size());

//   for (const auto& f :
//        std::filesystem::recursive_directory_iterator(res1_dir)) {
//     if (std::filesystem::is_regular_file(f.path())) {
//       std::ifstream file1(f.path(), std::ios::binary);
//       std::vector<char> buffer1(std::istreambuf_iterator<char>(file1), {});
//       auto relative_path = std::filesystem::relative(f, res1_dir);

//       auto f2_path = res2_dir / relative_path;
//       std::ifstream file2(f2_path, std::ios::binary);
//       std::vector<char> buffer2(std::istreambuf_iterator<char>(file2), {});
//       CHECK_EQ(buffer1.size(), buffer2.size());
//       ASSERT_TRUE(std::equal(buffer1.begin(), buffer1.end(),
//       buffer2.begin()));

//       auto f3_path = res3_dir / relative_path;
//       std::ifstream file3(f3_path, std::ios::binary);
//       std::vector<char> buffer3(std::istreambuf_iterator<char>(file3), {});
//       CHECK_EQ(buffer1.size(), buffer3.size());
//       ASSERT_TRUE(std::equal(buffer1.begin(), buffer1.end(),
//       buffer3.begin()));
//     }
//   }
// }

TEST_F(PassContextTest, TestCompress) {
  constexpr auto test_data_size = 128u * 1024u;
  auto test_data = std::vector<uint8_t>();
  test_data.resize(test_data_size);
  for (auto i = 0u; i < test_data_size; ++i) {
    test_data[i] = static_cast<uint8_t>(i % 256u);
  }
  auto compressed_data = vaip_core::compress(test_data, 9);
  auto decompressed_data = std::vector<uint8_t>();
  decompressed_data.resize(test_data_size);
  auto uncompressed_data = vaip_core::uncompress(compressed_data);
  ASSERT_EQ(uncompressed_data.size(), test_data.size());
  ASSERT_EQ(uncompressed_data, test_data);
}

TEST_F(PassContextTest, TestGzTar) {
  {
    for (auto i = 0; i < 3; ++i) {
      // Test file name
      std::string filename =
          std::string("TestGzTar.test_file_") + std::to_string(i) + ".txt";

      // Create a test file with some content
      std::string fileContent = "This is a test file for " + std::to_string(i);
      bool writeResult =
          passContext->write_file(filename, gsl::make_span(fileContent));

      // Assert that the file was successfully written
      ASSERT_TRUE(writeResult);

      // Read the file using the read_file function
      auto readResult = passContext->read_file_c8(filename);

      // Assert that the file was read successfully
      ASSERT_TRUE(readResult.has_value());

      // Assert that the content of the file matches the expected content
      ASSERT_EQ(std::string(readResult->data(), readResult->size()),
                fileContent);
    }
    auto tar_mem = passContext->cache_files_to_tar_mem();
    auto gz_tar_mem = vaip_core::compress(tar_mem);
    vaip_core::dump_binary(CMAKE_CURRENT_BINARY_PATH / "TestGzTar.tar.gz",
                           gz_tar_mem);
  }
  { // read it back from another pass context object.
    auto gz_tar_mem = vaip_core::slurp_binary_c8(CMAKE_CURRENT_BINARY_PATH /
                                                 "TestGzTar.tar.gz");
    auto tar_mem = vaip_core::uncompress(gz_tar_mem);
    // passContext->tar_mem_to_cache_files(tar_mem.data(), tar_mem.size());
    for (auto i = 0; i < 3; ++i) {
      std::string filename =
          std::string("TestGzTar.test_file_") + std::to_string(i) + ".txt";
      // Read the file using the read_file TestGzTar
      auto readResult = passContext->read_file_c8(filename);

      // Assert that the file was read successfully
      ASSERT_TRUE(readResult.has_value());
      // Assert that the content of the file matches the expected content
      std::string fileContent = "This is a test file for " + std::to_string(i);
      ASSERT_EQ(std::string(readResult->data(), readResult->size()),
                fileContent);
    }
  }
}

// TEST_F(PassContextTest, TestLongFilenames) {
//   auto buffer = std::vector<char>{};
//   std::string long_name(101u, 'x');
//   auto tar_mem = std::vector<char>();
//   for (auto i = 0; i < 101; ++i) {
//     long_name[i] = (char)('0' + (i % 10));
//   }
//   for (auto i = 0; i < 3; ++i) {
//     // Test file name
//     std::string filename = std::to_string(i) + long_name + std::to_string(i);
//     // Create a test file with some content
//     std::string fileContent = "This is a test file for " + std::to_string(i);
//     if (i == 1) {
//       fileContent = "";
//     }
//     bool writeResult =
//         passContext->write_file(filename, gsl::make_span(fileContent));

//     ASSERT_TRUE(writeResult);
//     auto stream = vaip_core::IStreamWriter::from_path(
//         CMAKE_CURRENT_BINARY_PATH / "TestLongFileName.tar");
//     passContext->cache_files_to_tar_file(*stream);
//     tar_mem = passContext->cache_files_to_tar_mem();
//   }
//   {
//     passContext->tar_file_to_cache_files(CMAKE_CURRENT_BINARY_PATH /
//                                          "TestLongFileName.tar");
//     for (auto i = 0; i < 3; ++i) {
//       // Test file name
//       std::string filename = std::to_string(i) + long_name +
//       std::to_string(i);
//       // Create a test file with some content
//       std::string fileContent = "This is a test file for " +
//       std::to_string(i); if (i == 1) {
//         fileContent = "";
//       }
//       // Read the file using the read_file function
//       auto readResult = passContext->read_file_c8(filename);

//       // Assert that the file was read successfully
//       ASSERT_TRUE(readResult.has_value());
//       ASSERT_EQ(std::string(readResult->data(), readResult->size()),
//                 fileContent);
//     }
//   }
//   {
//     passContext->tar_mem_to_cache_files(tar_mem.data(), tar_mem.size());
//     for (auto i = 0; i < 3; ++i) {
//       // Test file name
//       std::string filename = std::to_string(i) + long_name +
//       std::to_string(i);
//       // Create a test file with some content
//       std::string fileContent = "This is a test file for " +
//       std::to_string(i); if (i == 1) {
//         fileContent = "";
//       }
//       // Read the file using the read_file function
//       auto readResult = passContext->read_file_c8(filename);

//       // Assert that the file was read successfully
//       ASSERT_TRUE(readResult.has_value());
//       ASSERT_EQ(std::string(readResult->data(), readResult->size()),
//                 fileContent);
//     }
//   }
//   // ASSERT_TRUE(false);
// }
// Test fixture for PassContext
namespace vaip_core {
// put it to namespace vaip_core, so that friend class in PassContextImp works
// easily.
class PassContextConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = vaip_cxx::Model::load(RESNET_50_PATH);
    // Set up any necessary resources before each test
  }
  void CreateContext(onnxruntime::ProviderOptions provider_options) {
    passContext_ = vaip_core::initialize_context(
        model_->ref().model_path().u8string(), model_->ref().main_graph(), {},
        provider_options);
  }
  void TearDown() override {
    // Clean up any resources after each test
    passContext_.reset();
  }
  void load_context_json(const std::filesystem::path& context_json_path) {
    LOG(INFO) << "================== update context.json from "
              << context_json_path << " =======";
    auto stream = std::make_unique<std::ifstream>(context_json_path);
    auto dst = passContext_->open_file_for_write("context.json");
    if (!stream->is_open()) {
      LOG(ERROR) << "Failed to open context json file: "
                 << context_json_path.u8string();
      return;
    }
    if (!dst) {
      LOG(ERROR) << "Failed to open context json file for write: "
                 << context_json_path.u8string();
      return;
    }
    std::string line;
    while (std::getline(*stream, line)) {
      if (!line.empty()) {
        dst->fwrite(line.c_str(), line.size());
        dst->fwrite("\n", 1);
      }
    }
    passContext_->update_pass_context_from_context_json_in_cache();
  }
  // Path to the ResNet-50 model
  std::unique_ptr<vaip_cxx::Model> model_ = nullptr;
  // Pointer to the PassContext object
  std::shared_ptr<vaip_core::PassContextImp> passContext_ = nullptr;
};
} // namespace vaip_core
using namespace vaip_core;
TEST_F(PassContextConfigTest, Config) {
  auto cache_dir = CMAKE_CURRENT_BINARY_PATH / "c1";
  std::string cache_key = "d41d8cd98f00b204e9800998ecf8427e";
  auto log_dir = cache_dir / cache_key;
  CreateContext(onnxruntime::ProviderOptions{
      {"cacheDir", cache_dir.u8string()},
  });
  ASSERT_EQ(passContext_->get_log_dir(), log_dir);
  CreateContext(onnxruntime::ProviderOptions{
      {"cache_dir", cache_dir.u8string()},
  });
  ASSERT_EQ(passContext_->get_log_dir(), log_dir);
}

TEST_F(PassContextConfigTest, Target) {
  std::string cache_key = "d41d8cd98f00b204e9800998ecf8427e";
  auto config_file = CMAKE_CURRENT_SOURCE_PATH / "vaip" /
                     "test_pass_context.data" / "sample_config_1.json";
  auto context_json = CMAKE_CURRENT_SOURCE_PATH / "vaip" /
                      "test_pass_context.data" / "sample_context_1.json";
  CreateContext(onnxruntime::ProviderOptions{
      {"k0", "value0_in_provider_option"},
      {"cache_dir", "cache_dir_in_provider_option"},
      {"cache_key", "cache_key_in_provider_option"},
      {"config_file", config_file.u8string()},
  });

  load_context_json(context_json);
  EXPECT_EQ(passContext_->get_provider_option("k_not_exists", "value_in_code"),
            "value_in_code");
  EXPECT_EQ(passContext_->get_provider_option("k0", "value0_in_code"),
            "value0_in_provider_option");
  EXPECT_EQ(passContext_->get_provider_option("k1", "value1_in_code"),
            "value1_in_context.json");
  EXPECT_EQ(passContext_->get_provider_option("k2", "value2_in_code"),
            "value2_in_config");
  EXPECT_EQ(passContext_->get_provider_option("k3", "value3_in_code"),
            "value3_in_mep_table");
  EXPECT_EQ(passContext_->get_provider_option("k4", "value4_in_code"),
            "value4_in_target_proto");
  EXPECT_EQ(passContext_->get_log_dir(),
            std::filesystem::path("cache_dir_in_provider_option") /
                "cache_key_in_provider_option");
  auto all_provider_options = passContext_->get_all_provider_options();
  EXPECT_EQ(all_provider_options["k0"], "value0_in_provider_option");
  EXPECT_EQ(all_provider_options["k1"], "value1_in_context.json");
  EXPECT_EQ(all_provider_options["k2"], "value2_in_config");
  EXPECT_EQ(all_provider_options["k3"], "value3_in_mep_table");
  EXPECT_EQ(all_provider_options["k4"], "value4_in_target_proto");
  for (auto& kv : all_provider_options) {
    auto& [k, v] = kv;
    std::cout << "      all PO " << k << " = " << v << std::endl;
  }
  std::cout << "DONE" << std::endl;
}
