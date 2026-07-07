/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/morphizen.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(PluginTest, StaticHelloPlugin) {
  {
    const char *(*func)() = []() -> const char * {
      LOG(INFO) << "Hello, world!";
      return "hello, world!";
    };
    ::morphizen::StaticPluginRegister RegisterPlugin("hello_plugin",
                                                     "say_hello", (void *)func);
    auto plugin = morphizen::Plugin::get("hello_plugin");
    ASSERT_TRUE(plugin != nullptr);
    auto fp = plugin->get_method<const char *>("say_hello");
    ASSERT_TRUE(fp != nullptr) << "no such function";
    auto result = fp();
    ASSERT_STREQ(result, "hello, world!");
  }
  if (0) {
    // disable this test, because we need to unload the
    // onnxruntime_vitisai_ep.dll to cover this test.
    auto plugin = morphizen::Plugin::get("hello_plugin");
    ASSERT_TRUE(plugin == nullptr);
  }
}
TEST(PluginTest, DynamicHelloPlugin) {
  {
    auto plugin = morphizen::Plugin::get("hello_plugin_dll");
    ASSERT_TRUE(plugin != nullptr);
    auto fp = plugin->get_method<const char *>("say_hello");
    ASSERT_TRUE(fp != nullptr) << "no such function";
    auto result = fp();
    ASSERT_STREQ(result, "hello, world!");
  }
}
// todo long filename test
