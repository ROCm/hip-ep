/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/env_config.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "../../morphizen-pattern/src/immutable_map.hpp"

using namespace morphizen::immutable_map;
class ImmutableMapTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ImmutableMapTest, InsertSingleNode) {
  using Map = ImmutableMap<int, std::string>;
  auto m1 = Map();
  auto result = m1.insert({1, "one"});
  LOG(INFO) << "result = " << result << std::endl;
}

TEST_F(ImmutableMapTest, InsertMultipleNodes) {
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
    LOG(INFO) << "m[" << c << "]"
              << " = " << m << std::endl;
  }
  LOG(INFO) << "maps.back().size() = " << maps.back().size() << std::endl;
  EXPECT_EQ(maps.back().size(), 16);
  auto& m3 = maps[3];
  auto v3 = m3.find(3);
  EXPECT_EQ(m3.size(), 3);
  ASSERT_TRUE(v3 != nullptr);
  EXPECT_EQ(*v3, "three");
  auto v4 = m3.find(4);
  EXPECT_EQ(v4, nullptr);
  for (auto elt : maps.back()) {
    LOG(INFO) << "   " << elt.first << " ---> " << elt.second << std::endl;
  }
}
