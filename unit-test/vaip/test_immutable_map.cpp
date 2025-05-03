/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

//
#include "debug_logger.hpp"
//
#include "../vaip-core/src/pattern/immutable_map.hpp"

using namespace vaip_core::immutable_map;
class ImmutableMapTest : public DebugLogger {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ImmutableMapTest, InsertSingleNode) {
  open_logger_file("ImmutableMapTest.InsertSingleNode.log");
  using Map = ImmutableMap<int, std::string>;
  auto m1 = Map();
  auto result = m1.insert({1, "one"});
  MY_LOG() << "result = " << result << std::endl;
}

TEST_F(ImmutableMapTest, InsertMultipleNodes) {
  open_logger_file("ImmutableMapTest.InsertMultipleNodes.log");
  using Map = ImmutableMap<int, std::string>;
  auto m0 = Map();
  EXPECT_EQ(m0.size(), 0);
  auto c = 1;
  auto maps = std::vector<Map>();
  maps.push_back(m0);
  for (auto x : {"one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                 "fifteen", "sixteen"}) {
    maps.push_back(maps.back().insert({c++, x}));
  }
  c = 0;
  for (auto& m : maps) {
    MY_LOG() << "m[" << c << "]"
             << " = " << m << std::endl;
  }
  MY_LOG() << "maps.back().size() = " << maps.back().size() << std::endl;
  EXPECT_EQ(maps.back().size(), 16);
  auto& m3 = maps[3];
  auto v3 = m3.find(3);
  EXPECT_EQ(m3.size(), 3);
  ASSERT_TRUE(v3 != nullptr);
  EXPECT_EQ(*v3, "three");
  auto v4 = m3.find(4);
  EXPECT_EQ(v4, nullptr);
  for (auto elt : maps.back()) {
    MY_LOG() << "   " << elt.first << " ---> " << elt.second << std::endl;
  }
}
