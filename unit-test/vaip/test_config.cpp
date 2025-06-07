/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../vaip-core/src/config.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/vaip.hpp"
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
     "cacheDir" : "hello1",
     "cache_key" : "key",
     "enable_cache_file_io_in_mem":"1"
   }
}
)json";
TEST(ConfigTest, Simple) {
  auto config_proto = vaip_core::Config::parse_from_string(config);
  LOG(INFO) << "config: " << config_proto.DebugString();
  // when both cache_dir and cacheDir are set, cache_dir should be used
  EXPECT_EQ("hello1", config_proto.cache_dir());
  EXPECT_TRUE(config_proto.enable_cache_file_io_in_mem());
}

TEST(ConfigTest, EmptyProviderOption) {
  auto options = onnxruntime::ProviderOptions{{"log_level", "info"}};
  auto json_config = vaip_core::get_config_json_str(options);
  LOG(INFO) << "json_config: " << json_config;
  auto config_proto = vaip_core::Config::parse_from_string(json_config.c_str());
  LOG(INFO) << "config: " << config_proto.DebugString();
}

TEST(ConfigTest, ProviderOptionCacheDir) {
  auto options = onnxruntime::ProviderOptions{
      {"log_level", "info"},
      {"cache_dir", "hello1"},
  };
  auto json_config = vaip_core::get_config_json_str(options);
  LOG(INFO) << "json_config: " << json_config;
  auto config_proto = vaip_core::Config::parse_from_string(json_config.c_str());
  EXPECT_EQ("hello1", config_proto.cache_dir());
  LOG(INFO) << "config: " << config_proto.DebugString();
}

TEST(ConfigTest, SessionConfigs) {
  auto session_configs = std::map<std::string, std::string>{
      {"ep.context.enable", "1"},
      {"ep.shared_context", "1"},
  };
  // dirty hack for testing
  auto api = const_cast<vaip_core::OrtApiForVaip*>(vaip_core::api());
  auto old_session_option_configuration = api->session_option_configuration;
  api->session_option_configuration =
      [](void* mmap, void* session_option,
         void (*push)(void* mmap, const char* name, const char* value)) {
        auto self = reinterpret_cast<std::map<std::string, std::string>*>(
            session_option);
        for (auto& [key, value] : *self) {
          push(mmap, key.c_str(), value.c_str());
        }
      };

  auto options = onnxruntime::ProviderOptions{
      {"log_level", "info"},
      {"cache_dir", "hello1"},
      {"session_options",
       std::to_string((uintptr_t)(static_cast<void*>(&session_configs)))},
  };
  auto json_config = vaip_core::get_config_json_str(options);
  LOG(INFO) << "json_config: " << json_config;
  auto config_proto = vaip_core::Config::parse_from_string(json_config.c_str());
  EXPECT_EQ("hello1", config_proto.cache_dir());
  LOG(INFO) << "config: " << config_proto.DebugString();
  auto& sc = config_proto.session_configs();
  for (auto& [key, value] : session_configs) {
    LOG(INFO) << "session_configs: " << key << " = " << value;
    auto it = sc.find(key);
    ASSERT_NE(it, sc.end());
    EXPECT_EQ(value, it->second);
  }
  api->session_option_configuration = old_session_option_configuration;
}
