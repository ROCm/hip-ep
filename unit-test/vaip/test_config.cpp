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