/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <exception>
// must include glog/logging before morphizen.hpp
#include <glog/logging.h>
//
#include "./test_environment.hpp"
#include "morphizen/morphizen.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <memory>
using namespace morphizen;

static std::string slurp_txt(const std::filesystem::path &filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  std::stringstream sstr;
  auto buf = std::string();
  while (getline(in, buf).good()) {
    // exclude line starts with '#'
    if (buf[0] == '#') {
      continue;
    }
    sstr << buf << "\n";
  }
  in.close();
  return sstr.str();
}

static std::unique_ptr<AnchorPoint>
create(const std::filesystem::path &filename) {

  auto text = slurp_txt(filename);
  auto anchor_point = morphizen::AnchorPoint::create_proto();
  try {
    *anchor_point = AnchorPointProto::FromJsonString(text);
  } catch (const json::ParseError &) {
    CHECK(false) << "parse error: " << text;
  }
  return AnchorPoint::create(*anchor_point);
}

static void test_optimize1(const IPass &pass,
                           const std::filesystem::path &filename) {
  auto input = create(filename);
  LOG(INFO) << "input:\n"                               //
            << input->get_proto().DebugString() << "\n" //
            << "output:\n"                              //
            << input->optimize(pass)->get_proto().DebugString();
}

static void test_append(const IPass &pass, const std::string &f1,
                        const std::string &f2) {
  auto cwd =
      std::filesystem::path(__FILE__).parent_path() / "test_anchor_point.data";
  auto i1 = create(cwd / f1);
  auto i2 = create(cwd / f2);
  LOG(INFO) << "input:\n"                            //
            << i1->get_proto().DebugString() << "\n" //
            << i2->get_proto().DebugString() << "\n";
  LOG(INFO) << "output\n" //
            << i1->append(pass, *i2)->get_proto().DebugString() << "\n";
}

class TestAnchorPoint : public ::testing::Test {
public:
  void test_optimize(const std::filesystem::path &file) {
    auto cwd = std::filesystem::path(__FILE__).parent_path() /
               "test_anchor_point.data";
    std::shared_ptr<PassContext> context = PassContext::create();
    auto pass_proto = PassProto();
    pass_proto.set_name("test");
    pass_proto.set_plugin("morphizen-pass_init");
    auto pass = IPass::create_pass(context, pass_proto);
    test_optimize1(*pass, cwd / file);
  }
};

TEST_F(TestAnchorPoint, Case0) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  test_optimize("case0.prototxt");
}
TEST_F(TestAnchorPoint, Case1) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  test_optimize("case1.prototxt");
}
TEST_F(TestAnchorPoint, Case2) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  test_optimize("case2.prototxt");
}
TEST_F(TestAnchorPoint, Case3) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  test_optimize("case3.prototxt");
}
TEST_F(TestAnchorPoint, Case4) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  test_optimize("case4.prototxt");
}
TEST_F(TestAnchorPoint, Append) {
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  auto cwd =
      std::filesystem::path(__FILE__).parent_path() / "test_anchor_point.data";
  std::shared_ptr<PassContext> context = PassContext::create();
  auto pass_proto = PassProto();
  pass_proto.set_name("test");
  pass_proto.set_plugin("morphizen-pass_init");
  auto pass = IPass::create_pass(context, pass_proto);
  test_append(*pass, "case0.prototxt", "case1.prototxt");
}
