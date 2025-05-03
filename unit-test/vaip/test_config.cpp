/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../vaip-core/src/config.hpp"
#include "debug_logger.hpp"
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