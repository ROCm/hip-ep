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

#include "./pattern_or.hpp"

#include "./pattern_log.hpp"
#include "vaip/util.hpp"
namespace vaip_core {
PatternOr::PatternOr(int id, std::vector<std::shared_ptr<Pattern>> args)
    : Pattern(id), or_patterns_(std::move(args)) {}
PatternOr::~PatternOr() {}
std::string PatternOr::debug_string() const {
  auto ret =
      std::string("#") + std::to_string(this->get_id()) + std::string("(");
  if (!or_patterns_.empty()) {
    ret += "OR ";
    ret += or_patterns_[0]->debug_string();
    for (auto i = 1u; i < or_patterns_.size(); i++) {
      ret += ", " + or_patterns_[i]->debug_string();
    }
  }
  ret += std::string(")");
  return ret;
}

BinderBuilderPtr PatternOr::match_uncached(const onnxruntime::Graph& graph,
                                           const NodeInput& node_input,
                                           const BinderBuilder& binder) const {

  auto ret = BinderBuilderPtr();
  auto index = 0u;
  auto size = or_patterns_.size();
  for (auto& p : or_patterns_) {
    ret = p->match_cached(graph, node_input, binder);
    if (ret) {
      MY_LOG(1) << "MATCH OK. ID=" << get_id() << " " << index << "/" << size
                << " OK "
                << ", node=" << node_input_as_string(node_input);
      return ret->add(this->get_id(), node_input);
    } else {
      MATCH_FAILED << " " << index << "/" << size
                   << ", node=" << node_input_as_string(node_input);
    }
    index = index + 1;
  }
  MATCH_FAILED << " ALL FAIL "
               << ", node=" << node_input_as_string(node_input);
  return nullptr;
}

} // namespace vaip_core
