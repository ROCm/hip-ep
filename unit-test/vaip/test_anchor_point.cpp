/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
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
#include "debug_logger.hpp"
//
#include <exception>
// must include glog/logging before vaip.hpp
#include <glog/logging.h>
//
#include "vaip/vaip.hpp"

#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <memory>
using namespace vaip_core;

static std::string slurp_txt(const std::filesystem::path& filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

static std::unique_ptr<AnchorPoint>
create(const std::filesystem::path& filename) {

  auto text = slurp_txt(filename);
  auto anchor_point = AnchorPointProto();
  CHECK(google::protobuf::TextFormat::ParseFromString(text, &anchor_point))
      << "parse error: " << text;
  return AnchorPoint::create(anchor_point);
}

static void test_optimize1(const IPass& pass,
                           const std::filesystem::path& filename) {
  auto input = create(filename);
  LOG(INFO) << "input:\n"                               //
            << input->get_proto().DebugString() << "\n" //
            << "output:\n"                              //
            << input->optimize(pass)->get_proto().DebugString();
}

static void test_append(const IPass& pass, const std::string& f1,
                        const std::string& f2) {
  auto cwd =
      std::filesystem::path(__FILE__).parent_path() / "test_anchor_point.data";
  auto i1 = create(cwd / f1);
  auto i2 = create(cwd / f2);
  LOG(INFO) << "input:\n"                            //
            << i1->get_proto().DebugString() << "\n" //
            << i2->get_proto().DebugString() << "\n";
  LOG(INFO) << "output\n"                            //
            << i1->append(pass, *i2)->get_proto().DebugString() << "\n";
}

class TestAnchorPoint : public DebugLogger {
public:
  void test_optimize(const std::filesystem::path& file) {
    auto cwd = std::filesystem::path(__FILE__).parent_path() /
               "test_anchor_point.data";
    std::shared_ptr<PassContext> context = PassContext::create();
    auto pass_proto = PassProto();
    pass_proto.set_name("test");
    pass_proto.set_plugin("vaip-pass_init");
    auto pass = IPass::create_pass(context, pass_proto);
    test_optimize1(*pass, cwd / file);
  }
};

TEST_F(TestAnchorPoint, Case0) { test_optimize("case0.prototxt"); }
TEST_F(TestAnchorPoint, Case1) { test_optimize("case1.prototxt"); }
TEST_F(TestAnchorPoint, Case2) { test_optimize("case2.prototxt"); }
TEST_F(TestAnchorPoint, Case3) { test_optimize("case3.prototxt"); }
TEST_F(TestAnchorPoint, Case4) { test_optimize("case4.prototxt"); }
TEST_F(TestAnchorPoint, Append) {
  auto cwd =
      std::filesystem::path(__FILE__).parent_path() / "test_anchor_point.data";
  std::shared_ptr<PassContext> context = PassContext::create();
  auto pass_proto = PassProto();
  pass_proto.set_name("test");
  pass_proto.set_plugin("vaip-pass_init");
  auto pass = IPass::create_pass(context, pass_proto);
  test_append(*pass, "case0.prototxt", "case1.prototxt");
}
