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

#include "vaip/anchor_point.hpp"
#include "./anchor_point_imp.hpp"
#include "./pass_imp.hpp"
#include "vaip/node.hpp"
#include "vaip/util.hpp"
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <iterator>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_ANCHOR_POINT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR_POINT) >= n)

namespace vaip_core {
using namespace vaip_core_imp;

static AnchorPointProto new_anchor_point_proto(const IPass& pass,
                                               const std::string& name,
                                               const std::string& op_type) {
  auto ret = AnchorPointProto();
  ret.set_op_type(op_type);
  ret.set_name(name);
  ret.set_pass(pass.name());
  return ret;
}

template <typename T>
T fold(
    const T init, const AnchorPointProto& proto,
    std::function<T(const T value, const AnchorPointProto& anchor_point_proto)>
        f,
    bool process_last_anchor_point = true) {
  auto ret = init;
  auto* anchor_point_proto = &proto;
  for (;
       !anchor_point_proto->has_origin_node() && anchor_point_proto->has_next();
       anchor_point_proto = &anchor_point_proto->next()) {
    CHECK(anchor_point_proto->has_next())
        << " " << anchor_point_proto->DebugString();
    ret = f(ret, *anchor_point_proto);
  }
  if (process_last_anchor_point) {
    CHECK(anchor_point_proto->has_origin_node());
    CHECK(!anchor_point_proto->has_next());
    ret = f(ret, *anchor_point_proto);
  }
  return ret;
}

static void copy_anchor_point_proto_payload(const AnchorPointProto& from,
                                            AnchorPointProto& to) {
  to.set_op_type(from.op_type());
  to.set_name(from.name());
  to.set_pass(from.pass());
  *to.mutable_attribute() = from.attribute();
}

static std::pair<std::vector<AnchorPointProto>, std::string>
split_anchor_point(const AnchorPointProto& anchor_point_proto) {
  auto ret0 = std::vector<AnchorPointProto>();
  auto ret1 = std::string();
  ret0.reserve(5);
  AnchorPoint::create(anchor_point_proto)
      ->for_each([&ret0, &ret1](const AnchorPointProto& anchor_point_proto) {
        auto elt = AnchorPointProto();
        copy_anchor_point_proto_payload(anchor_point_proto, elt);
        ret0.push_back(std::move(elt));
        if (anchor_point_proto.has_origin_node()) {
          ret1 = anchor_point_proto.origin_node();
        }
      });
  return std::make_pair(ret0, ret1);
}

static std::string get_name_suffix(int suffix) {
  auto ret = std::string("");
  if (suffix != 0) {
    ret = ret + "_vaip_" + std::to_string(suffix);
  }
  return ret;
}
static AnchorPointProto combine_anchor_point(
    const IPass& pass,
    const std::pair<std::vector<AnchorPointProto>, std::string>& pair) {
  auto ret = AnchorPointProto();
  auto p = &ret;
  auto q = &ret; // q->next
  for (auto x = pair.first.begin(); x != pair.first.end(); ++x) {
    q = p;
    copy_anchor_point_proto_payload(*x, *p);
    p = p->mutable_next();
  }
  if (q->has_next()) {
    q->clear_next();
  } else {
    CHECK(pair.first.empty());
    q->set_op_type(AnchorPoint::IDENTITY_OP);
    q->set_pass("combine_empty");
    auto& context = dynamic_cast<const PassContextImp&>(*pass.get_context());
    q->set_name(pair.second + get_name_suffix(context.allocate_suffix()));
  }
  q->set_origin_node(pair.second);
  return ret;
}

std::unique_ptr<AnchorPoint>
AnchorPoint::create(const IPass& pass, const NodeArg& node_arg,
                    const Description& desciption) {
  return AnchorPoint::create(pass, node_arg_get_name(node_arg), desciption);
}

std::unique_ptr<AnchorPoint>
AnchorPoint::create(const IPass& pass, const std::string& node_arg_name,
                    const Description& desciption) {
  auto& context = dynamic_cast<const PassContextImp&>(*pass.get_context());
  const auto& origin_nodes = context.context_proto.origin_nodes();
  auto it = origin_nodes.find(node_arg_name);
  auto proto = desciption.proto_;
  CHECK(!proto.op_type().empty());
  auto previous_name = std::string();
  if (it != origin_nodes.end()) {
    previous_name = it->second.name();
    *proto.mutable_next() = it->second;
  } else {
    previous_name = node_arg_name;
    proto.set_origin_node(node_arg_name);
  }
  proto.set_pass(pass.name());
  auto name = proto.name();
  if (name.empty()) {
    if (proto.op_type() == AnchorPoint::IDENTITY_OP) {
      proto.set_name(previous_name);
    } else {
      auto origin_node_name = AnchorPointImp(proto).origin_node_arg_name();
      proto.set_name(origin_node_name +
                     get_name_suffix(context.allocate_suffix()));
    }
  }
  return std::make_unique<AnchorPointImp>(proto);
}

std::unique_ptr<AnchorPoint> AnchorPoint::alias1(const IPass& pass,
                                                 const Graph& graph,
                                                 const std::string& origin_name,
                                                 const std::string& new_name) {
  auto ret = find_anchor_point(const_cast<IPass&>(pass), graph, origin_name);
  // possible cause: tensor_name_remove_xir_suffix returned an invalid name that
  // is not ended with _vaip_\d+
  CHECK(ret != nullptr) << "origin_name = " << origin_name;
  const_cast<AnchorPointProto&>(ret->get_proto()).set_name(new_name);
  return ret;
}

static int get_fix_point(const Graph& graph, const Node& node) {
  // node type  only support  `DequantizeLinear` and `QuantizeLinear`
  auto op_type = VAIP_ORT_API(node_op_type)(node);
  CHECK(op_type == "DequantizeLinear" || op_type == "QuantizeLinear");
  auto inputs = node_get_input_node_args(node);
  CHECK_GE(inputs.size(), 2);
  auto scale = node_arg_get_const_data_as_float(graph, *inputs[1]);
  auto retp = scale_to_fix_point(scale);
  CHECK_NOTNULL(retp);
  return *retp;
}

static float get_scale(const Graph& graph, const Node& node) {
  // node type  only support  `VitisDequantizeLinear` and `VitisQuantizeLinear`
  auto op_type = VAIP_ORT_API(node_op_type)(node);
  CHECK(op_type == "VitisDequantizeLinear" || op_type == "VitisQuantizeLinear");
  auto inputs = node_get_input_node_args(node);
  CHECK_GE(inputs.size(), 2);
  auto scale = node_arg_get_const_data_as_float(graph, *inputs[1]);
  return scale;
}

static int get_zero_point(const Graph& graph, const Node& node) {
  // node type  only support  `VitisDequantizeLinear` and `VitisQuantizeLinear`
  auto op_type = VAIP_ORT_API(node_op_type)(node);
  CHECK(op_type == "VitisDequantizeLinear" || op_type == "VitisQuantizeLinear");
  auto inputs = node_get_input_node_args(node);
  CHECK_GE(inputs.size(), 3);
  auto data_type = VAIP_ORT_API(node_arg_get_element_type)(*inputs[2]);
  int zero_point = 0;
  switch (data_type) {
  case onnx::TensorProto_DataType_BFLOAT16:
    zero_point = (int)(node_arg_get_const_data_as_bf16(graph, *inputs[2]));
    break;
  default:
    // TODO
    LOG(FATAL) << "datatype is not supported: " << data_type;
  }
  return zero_point;
}

std::vector<std::int64_t> inverse_order(gsl::span<const int64_t> orders) {
  std::vector<std::int64_t> ret(orders.size());
  for (size_t i = 0; i < orders.size(); i++) {
    ret[orders[i]] = static_cast<std::int64_t>(i);
  }
  return ret;
}

std::unique_ptr<AnchorPoint>
AnchorPoint::create_from_siso_path(const IPass& pass, const Graph& graph,
                                   const std::vector<const Node*>& path1) {
  auto context = pass.get_context();
  CHECK(!path1.empty());
  auto path = std::vector<const Node*>(path1.rbegin(), path1.rend());
  auto origin_node_name = node_get_output_name(*path1.front());
  auto part = std::vector<AnchorPointProto>();
  for (auto i = 1u; i < path.size(); ++i) {
    auto& n = *path[i];
    auto op_type = VAIP_ORT_API(node_op_type)(n);
    if (op_type == "DequantizeLinear" || op_type == "QuantizeLinear") {
      // NOTE: float2fix is the reverse op of "DequantizeLinear"
      // NOTE: fix2float is the reverse op of "QuantizeLinear"
      auto anchor_point_name = node_get_output_name(*path[i - 1]);
      std::string reverse_op =
          op_type == "DequantizeLinear" ? "float2fix" : "fix2float";
      auto proto = new_anchor_point_proto(pass, anchor_point_name, reverse_op);
      proto.mutable_attribute()->mutable_fix_attr()->set_fix_point(
          get_fix_point(graph, n));
      part.emplace_back(std::move(proto));
    } else if (op_type == "Transpose") {
      // test case: edgenext_small_rw 9df0a329,see vaip#1231
      auto perm = node_get_attr_ints(n, "perm");
      auto anchor_point_name = node_get_output_name(*path[i - 1]);
      auto proto = new_anchor_point_proto(pass, anchor_point_name, "transpose");
      for (auto order : inverse_order(perm)) {
        proto.mutable_attribute()->mutable_transpose_attr()->add_order(
            order); // todo
      }
      part.emplace_back(std::move(proto));
    } else if (op_type == "VitisDequantizeLinear" ||
               op_type == "VitisQuantizeLinear") {
      // NOTE: quantize_linear is the reverse op of "VitisDequantizeLinear"
      // NOTE: dequantize_linear is the reverse op of "VitisQuantizeLinear"
      auto anchor_point_name = node_get_output_name(*path[i - 1]);
      std::string reverse_op = op_type == "VitisDequantizeLinear"
                                   ? "quantize_linear"
                                   : "dequantize_linear";
      auto proto = new_anchor_point_proto(pass, anchor_point_name, reverse_op);
      proto.mutable_attribute()->mutable_qdq_attr()->set_scale(
          get_scale(graph, n));
      proto.mutable_attribute()->mutable_qdq_attr()->set_zero_point(
          get_zero_point(graph, n));
      part.emplace_back(std::move(proto));
    } else {
      LOG(FATAL) << "Not supported: " << node_as_string(n);
    }
  }
  return create(combine_anchor_point(pass, {part, origin_node_name}));
}

std::unique_ptr<AnchorPoint>
AnchorPoint::create(const AnchorPointProto& proto) {
  return std::make_unique<AnchorPointImp>(proto);
}

std::unique_ptr<AnchorPoint>
AnchorPoint::create(const IPass& pass, const AnchorPointProto& next_proto,
                    const std::string& name, const Description& desciption) {
  auto proto = desciption.proto_;
  CHECK(!proto.op_type().empty());
  *proto.mutable_next() = next_proto;
  proto.set_pass(pass.name());
  proto.set_name(name);
  return std::make_unique<AnchorPointImp>(proto);
}

AnchorPoint::AnchorPoint() {}
AnchorPoint ::~AnchorPoint() {}

AnchorPoint::Description::Description(const std::string& op) {
  proto_.set_op_type(op);
}

AnchorPoint::Description::Description(const AnchorPointProto& anchor_point)
    : proto_{anchor_point} {}
AnchorPoint::Description::Description(const std::string& op,
                                      const AnchorPointTransposeOpAttr& attr) {
  CHECK((op == "transpose") || (op == "Transpose"));
  proto_.set_op_type(op);
  *proto_.mutable_attribute()->mutable_transpose_attr() = attr;
}
AnchorPoint::Description::Description(const std::string& op,
                                      const AnchorPointFixAttr& attr) {
  proto_.set_op_type(op);
  if (op == "fix" || op == "float2fix" || op == "fix2float") {
    *proto_.mutable_attribute()->mutable_fix_attr() = attr;
  } else {
    LOG(FATAL) << "unsupported";
  }
}
AnchorPoint::Description::Description(const std::string& op,
                                      const AnchorPointPadOpAttr& attr) {
  CHECK_EQ(op, "pad");
  proto_.set_op_type(op);
  *proto_.mutable_attribute()->mutable_pad_attr() = attr;
}
AnchorPoint::Description
AnchorPoint::Description::create_by_json(const std::string& anchor_point_json) {
  AnchorPointProto anchor_point_proto;
  auto status = google::protobuf::util::JsonStringToMessage(
      anchor_point_json, &anchor_point_proto);
  if (!status.ok()) {
    LOG(FATAL) << "cannot parse json string :" << anchor_point_json;
  }
  return Description{anchor_point_proto};
}

std::string AnchorPoint::op_debug_string() const {
  std::ostringstream str;
  const AnchorPointProto* p = nullptr;
  int c = 0;
  str << "\n";
  for (p = &this->get_proto(); p->has_next(); p = &p->next()) {
    if (c != 0) {
      str << " --\n";
    }
    str << p->name() << " <-- " << p->op_type() << "@" << p->pass();
    c = c + 1;
  }
  if (c != 0) {
    str << " --\n";
  }
  str << p->name() << " <-- " << p->op_type() << "@" << p->pass() << " --\n"
      << p->origin_node();
  return str.str();
}

std::string AnchorPoint::origin_node_arg_name() const {
  std::string ret = "";
  ret = fold<std::string>(
      ret, this->get_proto(),
      [](const std::string& origin_node_arg_name,
         const AnchorPointProto& anchor_point_proto) -> std::string {
        auto ret = origin_node_arg_name;
        if (anchor_point_proto.has_origin_node()) {
          CHECK(ret.empty()); // only last one has origin node.
          ret = anchor_point_proto.origin_node();
        }
        return ret;
      });
  return ret;
}

bool AnchorPoint::is_identity(bool test_all) const {
  auto ret = false;
  if (test_all) {
    ret = fold<bool>(
        true, this->get_proto(),
        [](bool value, const AnchorPointProto& anchor_point_proto) -> bool {
          return value &&
                 anchor_point_proto.op_type() == AnchorPoint::IDENTITY_OP;
        });
  } else {
    ret = this->get_proto().op_type() == AnchorPoint::IDENTITY_OP;
  }
  return ret;
}

const AnchorPointProto* AnchorPoint::find_op(const std::string& op) const {
  const AnchorPointProto* ret = nullptr;
  return fold<const AnchorPointProto*>(
      ret, this->get_proto(),
      [&op](const AnchorPointProto* state,
            const AnchorPointProto& anchor_point_proto)
          -> const AnchorPointProto* {
        auto ret = state;
        if (ret == nullptr) {
          if (anchor_point_proto.op_type() == op) {
            ret = &anchor_point_proto;
          }
        }
        return ret;
      });
}

static bool is_identity_transpose_ap(const AnchorPointProto& ap) {
  auto ret = false;
  if (ap.attribute().has_transpose_attr()) {
    auto expected = 0;
    ret = ap.op_type() == "transpose";
    for (auto i : ap.attribute().transpose_attr().order()) {
      ret = ret && (i == expected);
      expected = expected + 1;
    }
  }
  return ret;
}

static std::pair<std::vector<AnchorPointProto>, std::string>
remove_identity(std::pair<std::vector<AnchorPointProto>, std::string>&& pair) {
  pair.first.erase(std::remove_if(pair.first.begin(), pair.first.end(),
                                  [](const AnchorPointProto& ap) {
                                    return ap.op_type() ==
                                               AnchorPoint::IDENTITY_OP ||
                                           is_identity_transpose_ap(ap);
                                  }),
                   pair.first.end());
  return std::move(pair);
}

static std::pair<std::vector<AnchorPointProto>, std::string>
merge_fix2float_and_float2fix(
    std::pair<std::vector<AnchorPointProto>, std::string>&& pair) {
  auto to_be_removed =
      std::set<std::vector<AnchorPointProto>::difference_type>{};
  auto find_next = [](std::vector<AnchorPointProto>::iterator from,
                      std::vector<AnchorPointProto>::iterator end,
                      const char* op_type) {
    return find_if(from, end, [op_type](const AnchorPointProto& proto) {
      return proto.op_type() == op_type;
    });
  };
  for (auto it = pair.first.begin(); it != pair.first.end();) {
    auto end = pair.first.end();
    auto next = end;
    if (it->op_type() == "fix2float") {
      next = find_next(it, end, "float2fix");
    } else if (it->op_type() == "float2fix") {
      next = find_next(it, end, "fix2float");
    } else if (it->op_type() == "quantize_linear") {
      next = find_next(it, end, "dequantize_linear");
    } else if (it->op_type() == "dequantize_linear") {
      next = find_next(it, end, "quantize_linear");
    }
    if (next != end) {
      to_be_removed.insert(it - pair.first.begin());
      to_be_removed.insert(next - pair.first.begin());
      it = next + 1;
    } else {
      it = it + 1;
    }
  }
  auto new_first = std::vector<AnchorPointProto>{};
  for (std::vector<AnchorPointProto>::difference_type i = 0;
       i < (std::vector<AnchorPointProto>::difference_type)pair.first.size();
       ++i) {
    if (to_be_removed.find(i) == to_be_removed.end()) {
      new_first.push_back(pair.first[i]);
    }
  }
  return std::make_pair(new_first, pair.second);
}

static AnchorPointProto optimize_internal(
    const IPass& pass,
    std::pair<std::vector<AnchorPointProto>, std::string>&& pair) {
  return combine_anchor_point(
      pass, merge_fix2float_and_float2fix(remove_identity(std::move(pair))));
}

std::unique_ptr<AnchorPoint> AnchorPoint::optimize(const IPass& pass) const {
  auto ret =
      create(optimize_internal(pass, split_anchor_point(this->get_proto())));
  MY_LOG(1) << "before optimization:\n"
            << this->op_debug_string() << "\nafter optimization:\n"
            << ret->op_debug_string();
  return ret;
}

void AnchorPoint::insert_into_context(IPass& pass) const {
  auto& context = dynamic_cast<PassContextImp&>(*pass.get_context());
  auto origin_nodes = context.context_proto.mutable_origin_nodes();
  const auto& name_with_suffix = this->get_proto().name();
  auto insert_it = origin_nodes->insert(
      google::protobuf::MapPair{name_with_suffix, this->get_proto()});
  CHECK(insert_it.second)
      << "duplicated node arg name: " << name_with_suffix
      << "original anchor point:\n"
      << AnchorPoint::create(insert_it.first->second)->op_debug_string()
      << "new anchor point:\n"
      << this->op_debug_string();
}

// std::unique_ptr<AnchorPoint>
// AnchorPoint::append(const std::string& node_arg,
//                     const std::string& origin_node_name,
//                     const Description& description) {
//   auto anchor_point = find_anchor_point(node_arg);
//   CHECK(anchor_point != nullptr) << "cannot find anchor point: " << node_arg;
//   return anchor_point->append(origin_node_name, description);
// }

static std::unique_ptr<AnchorPoint>
append_internal(const IPass& pass, const AnchorPointProto& a,
                const AnchorPointProto& b,
                int use_origin_node_a_or_b /* 0 for 1, 1 for b */) {
  auto context = pass.get_context();
  auto p1 = split_anchor_point(a);
  auto p2 = split_anchor_point(b);
  auto origin_node_name = use_origin_node_a_or_b == 0 ? p1.second : p2.second;
  p1.first.reserve(p1.first.size() + p2.first.size());
  std::copy(p2.first.begin(), p2.first.end(), std::back_inserter(p1.first));
  p1.second = origin_node_name;
  return AnchorPoint::create(combine_anchor_point(pass, std::move(p1)));
}

std::unique_ptr<AnchorPoint>
AnchorPoint::append(const IPass& pass, const std::string& origin_node_name,
                    const Description& description) const {
  return append_internal(
      pass, this->get_proto(),
      create(pass, origin_node_name, description)->get_proto(), 1);
}

std::unique_ptr<AnchorPoint>
AnchorPoint::append(const IPass& pass, const AnchorPoint& rest) const {
  return append_internal(pass, this->get_proto(), rest.get_proto(), 1);
}

void AnchorPoint::for_each(
    const std::function<void(const AnchorPointProto&)>& func) const {
  fold<void*>(nullptr, this->get_proto(),
              [&func](void*, const AnchorPointProto& anchor_point_proto) {
                func(anchor_point_proto);
                return nullptr;
              });
}

} // namespace vaip_core
