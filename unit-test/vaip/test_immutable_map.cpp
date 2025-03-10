/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 2023 Advanced Micro Devices, Inc. All rights
 * reserved.
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

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

//
#include "debug_logger.hpp"
//
#include "../vaip/src/pattern/immutable_map.hpp"

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
