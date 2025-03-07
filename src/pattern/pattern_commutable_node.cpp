/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc.
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

#include "./pattern_commutable_node.hpp"
#include "./pattern_log.hpp"
#include "vaip/graph.hpp"

namespace vaip_core {
static std::string get_op_type(const std::string& op_type) {
  auto pos = op_type.find(':');
  if (pos == std::string::npos) {
    return std::string("onnx") + ":" + op_type;
  }
  return op_type;
}
PatternCommutableNode::PatternCommutableNode(
    int id, const std::string& op_type, const std::shared_ptr<Pattern>& arg1,
    const std::shared_ptr<Pattern>& arg2)
    : Pattern(id), op_type_(get_op_type(op_type)), arg1_(arg1), arg2_(arg2) {
  CHECK(arg1_ != nullptr);
  CHECK(arg2_ != nullptr);
}

PatternCommutableNode::~PatternCommutableNode() {}
static std::string get_full_op_type(const onnxruntime::Node& node) {
  auto domain = VAIP_ORT_API(node_op_domain)(node);
  if ("" == domain) {
    domain = "onnx";
  };
  return domain + ":" + VAIP_ORT_API(node_op_type)(node);
}

BinderBuilderPtr
PatternCommutableNode::match_uncached(const onnxruntime::Graph& graph,
                                      const NodeInput& node_input,
                                      const BinderBuilder& binder) const {
  if (node_input.node == nullptr) {
    MATCH_FAILED << " not a node: " << node_arg_as_string(*node_input.node_arg);
    return nullptr;
  }
  const auto& node = *node_input.node;
  auto full_op_type = get_full_op_type(node);
  if (full_op_type != this->op_type_) {
    MATCH_FAILED << " expect node_type is " << this->op_type_
                 << " actually node type is " << full_op_type
                 << node_as_string(node);
    return nullptr;
  }
  auto inputs = node_get_inputs(node);
  if (inputs.size() != 2) {
    MATCH_FAILED << " expect 2 inputs, actually " << inputs.size()
                 << node_as_string(node);
    return nullptr;
  }
  const auto ret0 = binder.add(this->get_id(), node_input);
  auto match = [&](const std::shared_ptr<Pattern>& arg1,
                   const std::shared_ptr<Pattern>& arg2) -> BinderBuilderPtr {
    MY_LOG(1) << " ID=" << get_id() << " try pattern arg1=" << arg1->get_id()
              << " arg2=" << arg2->get_id();
    auto match_ret = arg1->match_cached(graph, inputs[0], *ret0);
    if (match_ret) {
      match_ret = arg2->match_cached(graph, inputs[1], *match_ret);
    }
    if (match_ret == nullptr) {
      MY_LOG(1) << " ID=" << get_id() << " MATCH FAILED. "
                << "[p1=" << arg1->get_id() << " ,p2=" << arg2->get_id() << "]";
    }
    return match_ret;
  };
  auto ret = BinderBuilderPtr();

  auto ret12 = match(arg1_, arg2_);
  if (ret12 != nullptr) {
    MY_LOG(1) << " ID=" << get_id()         //
              << " match "                  //
              << " ["                       //
              << "arg1=" << arg1_->get_id() //
              << ","                        //
              << "arg2=" << arg2_->get_id() //
              << "] "
              << " OK";
    ret = std::move(ret12);
  } else {
    MY_LOG(1) << " ID=" << get_id()         //
              << " match "                  //
              << " ["                       //
              << "arg1=" << arg1_->get_id() //
              << ","                        //
              << "arg2=" << arg2_->get_id() //
              << "] "
              << " failed. try "
              << " ["                       //
              << "arg2=" << arg2_->get_id() //
              << ","                        //
              << "arg1=" << arg1_->get_id() //
              << "] ";
    auto ret21 = match(arg2_, arg1_);
    ret = std::move(ret21);
  }
  if (ret == nullptr) {
    MATCH_FAILED << " both arg match failed."
                 << " arg1=" << arg1_->get_id() << " arg2=" << arg2_->get_id();
  } else {
    MY_LOG(1) << "MATCH OK. ID=" << get_id()
              << ", node=" << node_as_string(node);
  }
  return ret;
}
std::string PatternCommutableNode::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += this->op_type_;
  ret += std::string("(");
  ret += arg1_->debug_string() + ", " + arg2_->debug_string();
  ret += std::string(")");
  return ret;
}
void PatternCommutableNode::fill_ops_name(
    std::vector<std::string>& list_of_ops_name) const {
  arg1_->fill_ops_name(list_of_ops_name);
  list_of_ops_name.emplace_back("CommutableNode -> " + this->op_type_);
  arg2_->fill_ops_name(list_of_ops_name);
  list_of_ops_name.emplace_back("CommutableNode -> " + this->op_type_);
}
} // namespace vaip_core
