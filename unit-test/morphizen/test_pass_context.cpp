/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/config_reader.hpp" // NOLINT
#include "morphizen/morphizen.hpp"     // NOLINT
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
// clang-format off
// NOLINTBEGIN
#include "morphizen/morphizen.hpp"
#include "./test_environment.hpp"
#include "../src/pass_context_imp.hpp"
// clang-format on
namespace morphizen {
std::shared_ptr<PassContextImp> initialize_context(
    const std::string& model_path, const onnxruntime::Graph& onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef>& ep_context_nodes,
    const onnxruntime::ProviderOptions& options,
    std::unique_ptr<LoggerAdapter> logger_adapter);
}

// Test fixture for PassContext
class PassContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary resources before each test
    passContext = morphizen::PassContext::create();
    dynamic_cast<morphizen::PassContextImp*>(passContext.get())
        ->pass_context_log_dir_ = CMAKE_CURRENT_BINARY_PATH;
  }

  void TearDown() override {
    // Clean up any resources after each test
    passContext.reset();
  }

  // Pointer to the PassContext object
  std::unique_ptr<morphizen::PassContext> passContext;
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
  std::ofstream tar_file_stream(CMAKE_CURRENT_BINARY_PATH / "ReadFileTest.tar",
                                std::ios::binary);
  passContext->cache_files_to_tar_file(tar_file_stream);
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
    std::ofstream tar_file_stream(
        CMAKE_CURRENT_BINARY_PATH / "TestEmptyFiles.tar", std::ios::binary);
    passContext->cache_files_to_tar_file(tar_file_stream);
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
//   auto ctx1 = morphizen::PassContext::create();
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
//   auto tar_file_stream = morphizen::IStreamWriter::from_path(tar_file_path);
//   std::ignore = ctx1->cache_files_to_tar_file(*tar_file_stream);

//   auto ctx2 = morphizen::PassContext::create();
//   ctx2->tar_file_to_cache_files(tar_file_path);

//   auto bytes = ctx1->cache_files_to_tar_mem();
//   auto ctx3 = morphizen::PassContext::create();
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

// TODO: move it to another test file.
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
//     auto stream = morphizen::IStreamWriter::from_path(
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
namespace morphizen {
// put it to namespace morphizen, so that friend class in PassContextImp works
// easily.
class PassContextConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = morphizen_cxx::Model::load(RESNET_50_PATH);
    // Set up any necessary resources before each test
  }
  void CreateContext(onnxruntime::ProviderOptions provider_options) {
    passContext_ = morphizen::initialize_context(
        model_->ref().model_path().u8string(), model_->ref().main_graph(), {},
        provider_options, nullptr);
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
  std::unique_ptr<morphizen_cxx::Model> model_ = nullptr;
  // Pointer to the PassContext object
  std::shared_ptr<morphizen::PassContextImp> passContext_ = nullptr;
};
} // namespace morphizen
using namespace morphizen;
TEST_F(PassContextConfigTest, Config) {
  auto cache_dir = CMAKE_CURRENT_BINARY_PATH / "c1";
  std::string cache_key =
      "33ad2fe7c4a7b71e55f5cbd9c0569bb4"; // use graph io based memory md5value.
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

TEST_F(PassContextConfigTest, ProviderOptions) {
  std::string cache_key = "33ad2fe7c4a7b71e55f5cbd9c0569bb4";
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

TEST_F(PassContextConfigTest, TargetSpecifiedByEndUserNotValid) {

  try {
    CreateContext(onnxruntime::ProviderOptions{
        {"target", "target-not-exists"},
    });
    ASSERT_TRUE(false) << "Should throw exception when target is not exists";
  } catch (const std::invalid_argument& e) {
    std::string error_message = e.what();
    ASSERT_TRUE(error_message.find("not a valid target") != std::string::npos)
        << " Expected error message to contain 'not a valid target', but got: "
        << error_message;
  }
}

TEST_F(PassContextConfigTest, TargetSpecifiedByEndUserValid) {
  CreateContext(onnxruntime::ProviderOptions{
      {"target", "dummy-target"},
  });
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "bingo");
}

TEST_F(PassContextConfigTest, TargetInConfigFileNotValidTarget) {

  try {
    CreateContext(onnxruntime::ProviderOptions{
        {"config_file",
         (CMAKE_CURRENT_SOURCE_PATH / "vaip" / "test_pass_context.data" /
          "sample_config_for_target_disovery_not_valid_target.json")
             .u8string()},
    });
    ASSERT_TRUE(false) << "Should throw exception when target is not exists";
  } catch (const std::invalid_argument& e) {
    std::string error_message = e.what();
    ASSERT_TRUE(error_message.find("not a valid target") != std::string::npos)
        << " Expected error message to contain 'not a valid target', but got: "
        << error_message;
  }
}

TEST_F(PassContextConfigTest, TargetInConfigFileValidTarget) {
  CreateContext(onnxruntime::ProviderOptions{
      {"config_file",
       (CMAKE_CURRENT_SOURCE_PATH / "vaip" / "test_pass_context.data" /
        "sample_config_for_target_disovery_valid_target.json")
           .u8string()},
  });
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "bingo");
}

TEST_F(PassContextConfigTest, TargetInMepTableValidTarget) {
  CreateContext(onnxruntime::ProviderOptions{
      {"config_file",
       (CMAKE_CURRENT_SOURCE_PATH / "vaip" / "test_pass_context.data" /
        "sample_config_for_target_disovery_valid_in_mep_table.json")
           .u8string()},
  });
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "mep-target-hit-in-sampel-config");
}

TEST_F(PassContextConfigTest, TargetInMepTableValidTarget_builtin_config) {
  CreateContext(onnxruntime::ProviderOptions{});
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "mep-target-hit");
}

TEST_F(PassContextConfigTest,
       TargetInMepTableValidTarget_builtin_config_auto_disvoery) {
  // first we need to modify model inputs and outputs to make mep table miss.
  auto graph = model_->main_graph();

  auto rename_inputs = [this](morphizen_cxx::NodeArgConstRef old)
      -> morphizen_cxx::NodeArgConstRef {
    auto graph = model_->main_graph();
    return graph.new_node_arg(
        old.name() + "_new_name", *old.shape(),
        (ONNX_NAMESPACE::TensorProto_DataType)old.element_type());
  };
  auto old_inputs = model_->main_graph().inputs();

  auto inputs = std::vector<morphizen_cxx::NodeArgConstRef>{};
  for (auto& old_input : old_inputs) {
    auto new_input = rename_inputs(old_input);
    inputs.push_back(new_input);
  }
  model_->main_graph().set_inputs(inputs);
  model_->set_metadata("relu_dq_target_name",
                       "relu_dq_target_by_auto_discovery");
  CreateContext(onnxruntime::ProviderOptions{});
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "auto-discovery-hit");
}
// "99_vaip_centralized_target_discovery", the plugin is ordered alphabetically
// by name so it is probably the laster resort.
//
// we must register this along with a pass or custom op, morphizen::core is not
// build with WHOLE_ARCHIVE enabled. it would be removed by linker if not used.
static std::string get_meta(const onnxruntime::Model& model,
                            const std::string& key) {
  if (MORPHIZEN_ORT_API(model_has_meta_data)(model, key))
    return *(MORPHIZEN_ORT_API(model_get_meta_data)(model, key));
  return "";
}
static std::optional<std::string>
relu_dq_centralized_target_discovery(const morphizen::ConfigProto& config,
                                     const onnxruntime::Model& model) {
  auto graph = morphizen_cxx::GraphConstRef(MORPHIZEN_ORT_API(model_main_graph)(
      const_cast<onnxruntime::Model&>(model)));
  auto ret = std::optional<std::string>();
  auto relu_dq_target_name = get_meta(model, "relu_dq_target_name");
  if (!relu_dq_target_name.empty()) {
    ret = relu_dq_target_name;
  }
  return ret;
}
static ::morphizen::StaticPluginRegister
    __plugin_register("99_relu_dq_centralized_target_discovery",
                      "morphizen_target_discovery",
                      (void*)&relu_dq_centralized_target_discovery);
