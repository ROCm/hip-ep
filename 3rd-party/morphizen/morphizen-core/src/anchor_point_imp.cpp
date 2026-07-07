/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./anchor_point_imp.hpp"
#include "morphizen/anchor_point.hpp"
#include <glog/logging.h>
#include <initializer_list>
#include <ios>
#include <morphizen/morphizen_ort_api.h>
#include <sstream>
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#include <google/protobuf/text_format.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "./pass_imp.hpp"
#include "morphizen/env_config.hpp"
DEF_ENV_PARAM(DEBUG_ANCHOR_POINT, "0")

namespace morphizen_imp {

AnchorPointImp::AnchorPointImp(const AnchorPointProto &proto) : proto_{proto} {}

AnchorPointImp::~AnchorPointImp() {}

const AnchorPointProto &AnchorPointImp::get_proto() const { return proto_; }

} // namespace morphizen_imp

namespace morphizen {

using namespace morphizen_imp;

std::unique_ptr<AnchorPoint> AnchorPoint::identity(const IPass &pass,
                                                   const NodeArg &node_arg) {
  return AnchorPoint::identity(pass, node_arg_get_name(node_arg));
}
std::unique_ptr<AnchorPoint>
AnchorPoint::identity(const IPass &pass, const std::string &node_arg_name) {
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
AnchorPoint::find_anchor_point(const IPass &pass, const std::string &name) {
  auto &context = dynamic_cast<const PassContextImp &>(*pass.get_context());
  const auto &origin_nodes = context.context_proto.origin_nodes();
  auto it = origin_nodes.find(name);
  auto ret = std::unique_ptr<AnchorPoint>{};
  if (it != origin_nodes.end()) {
    ret = std::make_unique<AnchorPointImp>(it->second);
  }
  return ret;
}

std::unique_ptr<AnchorPoint>
AnchorPoint::find_anchor_point(IPass &pass, const Graph &graph,
                               const std::string &name) {
  auto ret = find_anchor_point(pass, name);
  if (ret == nullptr) {
    auto graph_ref = morphizen_cxx::GraphConstRef(graph);
    auto node_arg_opt = graph_ref.find_node_arg(name);
    if (node_arg_opt.has_value()) {
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
} // namespace morphizen
