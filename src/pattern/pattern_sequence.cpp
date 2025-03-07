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

#include "./pattern_sequence.hpp"
#include "./pattern_log.hpp"
#include "vaip/node.hpp"
#include "vaip/node_arg.hpp"
namespace vaip_core {
PatternSequence::PatternSequence(
    int id, gsl::span<const std::shared_ptr<Pattern>> patterns)
    : Pattern(id), patterns_{patterns.begin(), patterns.end()} {
  CHECK(!patterns.empty()) << "wronge number of patterns.";
}
PatternSequence::~PatternSequence() {}
std::string PatternSequence::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("*");
  ret += std::string(")");
  return ret;
}

BinderBuilderPtr
PatternSequence::match_uncached(const onnxruntime::Graph& graph1,
                                const NodeInput& node_input,
                                const BinderBuilder& binder) const {
  auto graph = vaip_cxx::GraphConstRef(graph1);
  auto ret = patterns_.front()->match_cached(graph, node_input, binder);
  if (ret == nullptr) {
    MY_LOG(1) << "MATCH FAIL. ID=" << get_id()
              << ", the first pattern does not matched: "
              << (node_input.node != nullptr
                      ? node_as_string(*node_input.node)
                      : node_arg_as_string(*node_input.node_arg));
    return nullptr;
  }
  auto nodes = graph.nodes();
  auto patter_size = patterns_.size();
  for (size_t i = 1; i < patter_size; ++i) {
    auto result_i = vaip_core::BinderBuilderPtr();
    MY_LOG(1) << "PatternSequence try to match pattern"    //
              << "[" << i << " of " << patter_size << "] " //
              << "in all other nodes.\n";
    auto mached_node = std::optional<vaip_cxx::NodeConstRef>();
    for (auto j = 0u; j < nodes.size(); ++j) {
      auto node = nodes[j];
      for (auto output_node_arg : node.outputs()) {
        if (output_node_arg == std::nullopt) {
          continue;
        }
        auto node_input =
            vaip_core::NodeInput{node.ptr(), output_node_arg->ptr()};
        result_i = patterns_[i]->match_cached(graph, node_input, *ret);
        if (result_i != nullptr) {
          mached_node = node;
          break;
        }
      }
      if (result_i != nullptr) {
        break;
      }
      MY_LOG(1) << "PatternSequence continue to try other nodes ignore above "
                   "errors"
                << std::endl;
    }
    if (result_i != nullptr) {
      MY_LOG(1) << "MATCH OK. sub-pattern"                   //
                << "[" << i << " of " << patter_size << "] " //
                << " match " << *mached_node;
      ret = std::move(result_i);
    } else {
      MY_LOG(1) << "MATCH FAIL. PatternSequence ID=" << get_id()
                << " sub-pattern"
                << "[" << i << " of " << patter_size
                << "] does not match any node: ";
      ret = nullptr;
    }
    if (ret == nullptr) {
      return nullptr;
    }
  }
  MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", sequence matched: "
            << (node_input.node != nullptr
                    ? node_as_string(*node_input.node)
                    : node_arg_as_string(*node_input.node_arg));
  return ret;
}

} // namespace vaip_core
