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

#include "./anchor_point_imp.hpp"
#include "vaip/anchor_point.hpp"
#include <glog/logging.h>
#include <initializer_list>
#include <ios>
#include <sstream>
#include <vaip/vaip_ort_api.h>
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#endif
#include <google/protobuf/text_format.h>
#ifdef _WIN32
#  pragma warning(pop)
#endif

#include "./pass_imp.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_ANCHOR_POINT, "0")

namespace vaip_core_imp {

AnchorPointImp::AnchorPointImp(const AnchorPointProto& proto) : proto_{proto} {}

AnchorPointImp::~AnchorPointImp() {}

const AnchorPointProto& AnchorPointImp::get_proto() const { return proto_; }

} // namespace vaip_core_imp

namespace vaip_core {

using namespace vaip_core_imp;

std::unique_ptr<AnchorPoint> AnchorPoint::identity(const IPass& pass,
                                                   const NodeArg& node_arg) {
  return AnchorPoint::identity(pass, node_arg_get_name(node_arg));
}
std::unique_ptr<AnchorPoint>
AnchorPoint::identity(const IPass& pass, const std::string& node_arg_name) {
  auto next = find_anchor_point(pass, node_arg_name);
  auto proto = AnchorPointProto();
  proto.set_op_type(AnchorPoint::IDENTITY_OP);
  proto.set_pass(pass.name());
  if (next == nullptr) {
    proto.set_origin_node(node_arg_name);
    proto.set_name(node_arg_name);
  } else {
    proto.set_name(next->get_proto().name());
    *proto.mutable_next() = next->get_proto();
  }
  return std::make_unique<AnchorPointImp>(proto);
}

std::unique_ptr<AnchorPoint>
AnchorPoint::find_anchor_point(const IPass& pass, const std::string& name) {
  auto& context = dynamic_cast<const PassContextImp&>(*pass.get_context());
  const auto& origin_nodes = context.context_proto.origin_nodes();
  auto it = origin_nodes.find(name);
  auto ret = std::unique_ptr<AnchorPoint>{};
  if (it != origin_nodes.end()) {
    ret = std::make_unique<AnchorPointImp>(it->second);
  }
  return ret;
}

std::unique_ptr<AnchorPoint>
AnchorPoint::find_anchor_point(IPass& pass, const Graph& graph,
                               const std::string& name) {
  auto ret = find_anchor_point(pass, name);
  if (ret == nullptr) {
    auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, name);
    if (node_arg != nullptr) {
      auto proto = AnchorPointProto();
      proto.set_op_type(AnchorPoint::IDENTITY_OP);
      proto.set_origin_node(name);
      proto.set_name(name);
      ret = std::make_unique<AnchorPointImp>(proto);
      ret->insert_into_context(pass);
    }
  }
  return ret;
}
} // namespace vaip_core
