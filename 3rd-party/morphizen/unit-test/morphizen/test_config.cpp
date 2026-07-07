/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../morphizen-core/src/config.hpp"
#include "../morphizen-core/src/pass_context_imp.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
// disable this test
static const char config[] =
    R"json(
{
   "sessionOptions": {
     "cacheDir" : "cacheDir",
     "cache_dir" : "cache_dir",
     "enable_cache_file_io_in_mem":"1"
   }
}
)json";
TEST(ConfigTest, Simple) {
  auto config_proto = morphizen::Config::parse_from_string(config);
  LOG(INFO) << "config: " << config_proto.DebugString();
  // add_custom_field is removed, so that we need passcontext.set_config_prot();
  // cache_dir has been removed - dump_dir is accessed via get_dump_directory()
  //
  auto pass_context =
      morphizen::PassContextImp::create_pass_context(config_proto);
  auto &config_proto_in_context = pass_context->get_config_proto();
  // Verify pass context created successfully
  EXPECT_TRUE(pass_context != nullptr);
  // root field enable_cache_file_io_in_mem is obsoleted.
}

TEST(ConfigTest, EmptyProviderOption) {
  auto options = onnxruntime::ProviderOptions{{"log_level", "info"}};
  auto json_config = morphizen::get_config_json_str(options);
  LOG(INFO) << "json_config: " << json_config;
  auto config_proto = morphizen::Config::parse_from_string(json_config.c_str());
  LOG(INFO) << "config: " << config_proto.DebugString();
}

TEST(ConfigTest, ProviderOptionCacheDir) {
  auto options = onnxruntime::ProviderOptions{
      {"log_level", "info"},
      {"dump_dir", "hello1"},
  };
  std::map<std::string, std::string> empty_session_configs;
  auto pass_context = morphizen::PassContextImp::create_pass_context(
      options, empty_session_configs);
  // cache_dir removed - dump_dir accessed via get_dump_directory()
  auto dump_dir = pass_context->get_dump_directory();
  EXPECT_EQ("hello1", dump_dir.string());
  LOG(INFO) << "config: " << pass_context->get_config_proto().DebugString();
}

TEST(ConfigTest, SessionConfigs) {
  auto session_configs = std::map<std::string, std::string>{
      {"ep.context.enable", "1"},
      {"ep.shared_context", "1"},
  };
  // dirty hack for testing
  auto api = const_cast<morphizen::OrtApiForMorphizen *>(morphizen::api());
  auto old_session_option_configuration = api->session_option_configuration;
  api->session_option_configuration =
      [](void *mmap, void *session_option,
         void (*push)(void *mmap, const char *name, const char *value)) {
        auto self = reinterpret_cast<std::map<std::string, std::string> *>(
            session_option);
        for (auto &[key, value] : *self) {
          push(mmap, key.c_str(), value.c_str());
        }
      };

  auto options = onnxruntime::ProviderOptions{
      {"log_level", "info"},
      {"dump_dir", "hello1"},
      {"session_options",
       std::to_string((uintptr_t)(static_cast<void *>(&session_configs)))},
  };
  auto pass_context =
      morphizen::PassContextImp::create_pass_context(options, session_configs);
  auto &config_proto = pass_context->get_config_proto();
  // cache_dir removed - dump_dir accessed via get_dump_directory()
  auto dump_dir = pass_context->get_dump_directory();
  EXPECT_EQ("hello1", dump_dir.string());
  LOG(INFO) << "config: " << config_proto.DebugString();
  // Session configs now passed separately and accessed via get_session_config()
  for (auto &[key, value] : session_configs) {
    LOG(INFO) << "session_configs: " << key << " = " << value;
    auto sc_value = pass_context->get_session_config(key);
    ASSERT_TRUE(sc_value.has_value());
    EXPECT_EQ(value, sc_value.value());
  }
  api->session_option_configuration = old_session_option_configuration;
}
