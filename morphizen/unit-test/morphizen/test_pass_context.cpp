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
    const std::string &model_path, const onnxruntime::Graph &onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs,
    std::unique_ptr<LoggerAdapter> logger_adapter);
}

// Test fixture for PassContext
class PassContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary resources before each test
    passContext = morphizen::PassContext::create();
    // pass_context_log_dir_ removed - dump directory now accessed via
    // get_dump_directory()
  }

  void TearDown() override {
    // Clean up any resources after each test
    passContext.reset();
  }

  // Pointer to the PassContext object
  std::unique_ptr<morphizen::PassContext> passContext;
};

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
    std::map<std::string, std::string> empty_session_configs;
    passContext_ = morphizen::initialize_context(
        model_->ref().model_path().u8string(), model_->ref().main_graph(), {},
        provider_options, empty_session_configs, nullptr);
  }
  void TearDown() override {
    // Clean up any resources after each test
    passContext_.reset();
  }
  void load_context_json(const std::filesystem::path &context_json_path) {
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

// The accessor is per-thread process state, so clear it before returning to
// keep the remaining tests on this thread unaffected.
TEST_F(PassContextTest, GetRunOption) {
  const std::map<std::string, std::string> run_options{
      {"qnn.htp_perf_mode", "burst"}};
  auto get_entry = [](const void *state,
                      const char *name) -> morphizen::DllSafe<std::string> {
    const auto &options =
        *static_cast<const std::map<std::string, std::string> *>(state);
    auto it = options.find(name);
    if (it == options.end()) {
      return morphizen::DllSafe<std::string>();
    }
    return morphizen::DllSafe<std::string>(it->second);
  };

  // No accessor installed yet: every lookup falls back to the default.
  EXPECT_EQ(passContext->get_run_option("qnn.htp_perf_mode", "default"),
            "default");

  morphizen::set_run_option_accessor(&run_options, get_entry);
  EXPECT_EQ(passContext->get_run_option("qnn.htp_perf_mode", "default"),
            "burst");
  EXPECT_EQ(passContext->get_run_option("absent.key", "default"), "default");

  morphizen::set_run_option_accessor(nullptr, nullptr);
  EXPECT_EQ(passContext->get_run_option("qnn.htp_perf_mode", "default"),
            "default");
}

TEST_F(PassContextConfigTest, Config) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto cache_dir = CMAKE_CURRENT_BINARY_PATH / "c1";
  std::string cache_key =
      "33ad2fe7c4a7b71e55f5cbd9c0569bb4"; // use graph io based memory md5value.
  auto dump_dir = cache_dir / cache_key;
  CreateContext(onnxruntime::ProviderOptions{
      {"dump_dir", cache_dir.u8string()},
  });
  ASSERT_EQ(passContext_->get_dump_directory(), dump_dir);
  CreateContext(onnxruntime::ProviderOptions{
      {"dump_dir", cache_dir.u8string()},
  });
  ASSERT_EQ(passContext_->get_dump_directory(), dump_dir);
}

TEST_F(PassContextConfigTest, ProviderOptions) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  std::string cache_key = "33ad2fe7c4a7b71e55f5cbd9c0569bb4";
  auto config_file = CMAKE_CURRENT_SOURCE_PATH / "morphizen" /
                     "test_pass_context.data" / "sample_config_1.json";
  auto context_json = CMAKE_CURRENT_SOURCE_PATH / "morphizen" /
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
            "value3_in_target_proto");
  EXPECT_EQ(passContext_->get_provider_option("k4", "value4_in_code"),
            "value4_in_target_proto");
  EXPECT_EQ(passContext_->get_dump_directory(),
            std::filesystem::path("cache_dir_in_provider_option") /
                "cache_key_in_provider_option");
  auto all_provider_options = passContext_->get_all_provider_options();
  EXPECT_EQ(all_provider_options["k0"], "value0_in_provider_option");
  EXPECT_EQ(all_provider_options["k1"], "value1_in_context.json");
  EXPECT_EQ(all_provider_options["k2"], "value2_in_config");
  EXPECT_EQ(all_provider_options["k3"], "value3_in_target_proto");
  EXPECT_EQ(all_provider_options["k4"], "value4_in_target_proto");
  for (auto &kv : all_provider_options) {
    auto &[k, v] = kv;
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
  } catch (const std::invalid_argument &e) {
    std::string error_message = e.what();
    ASSERT_TRUE(error_message.find("not a valid target") != std::string::npos)
        << " Expected error message to contain 'not a valid target', but got: "
        << error_message;
  }
}

TEST_F(PassContextConfigTest, TargetSpecifiedByEndUserValid) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
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
         (CMAKE_CURRENT_SOURCE_PATH / "morphizen" / "test_pass_context.data" /
          "sample_config_for_target_disovery_not_valid_target.json")
             .u8string()},
    });
    ASSERT_TRUE(false) << "Should throw exception when target is not exists";
  } catch (const std::invalid_argument &e) {
    std::string error_message = e.what();
    ASSERT_TRUE(error_message.find("not a valid target") != std::string::npos)
        << " Expected error message to contain 'not a valid target', but got: "
        << error_message;
  }
}

TEST_F(PassContextConfigTest, TargetInConfigFileValidTarget) {
  CreateContext(onnxruntime::ProviderOptions{
      {"config_file",
       (CMAKE_CURRENT_SOURCE_PATH / "morphizen" / "test_pass_context.data" /
        "sample_config_for_target_disovery_valid_target.json")
           .u8string()},
  });
  auto dummy_option =
      passContext_->get_provider_option("dummy_provier_option_for_test");
  ASSERT_EQ(dummy_option, "bingo");
}

// "99_morphizen_centralized_target_discovery", the plugin is ordered
// alphabetically by name so it is probably the laster resort.
//
// we must register this along with a pass or custom op, morphizen::core is not
// build with WHOLE_ARCHIVE enabled. it would be removed by linker if not used.
static std::string get_meta(const onnxruntime::Model &model,
                            const std::string &key) {
  if (MORPHIZEN_ORT_API(model_has_meta_data)(model, key))
    return *(MORPHIZEN_ORT_API(model_get_meta_data)(model, key));
  return "";
}
static std::optional<std::string>
relu_dq_centralized_target_discovery(const morphizen::ConfigProto &config,
                                     const onnxruntime::Model &model) {
  auto graph = morphizen_cxx::GraphConstRef(MORPHIZEN_ORT_API(model_main_graph)(
      const_cast<onnxruntime::Model &>(model)));
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
                      (void *)&relu_dq_centralized_target_discovery);
