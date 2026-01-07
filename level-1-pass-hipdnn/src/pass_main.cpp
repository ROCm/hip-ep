/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "hipdnn.pb.h"
#include "hipdnn_pattern_json.hpp"
#include <filesystem>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)
namespace {
using namespace vaip_core;
using namespace vaip_cxx;

struct Level1HipDnn {
  Level1HipDnn(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    std::shared_ptr<Pattern> pattern_ =
        vaip_core::PatternBuilder().create_by_json(
            std::string((const char*)hipdnn_json));
    CHECK(pattern_ != nullptr) << "Pattern hipdnn not found";
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input = vaip_cxx::NodeArgConstRef::from_node_arg(
              *graph, *binder["input"].node_arg);
          auto output = vaip_cxx::NodeArgConstRef::from_node_arg(
              *graph, *binder["output"].node_arg);
          auto conv_node = binder["hipdnn_op"].node;
          auto unique_id = output.name();
          auto [meta_def, fuse_error] =
              self_.try_fuse(*graph, unique_id, {input.name()}, {output.name()},
                             {}, "HIPDNN");
          if (meta_def == nullptr) {
            MY_LOG(1) << "fuse error: " << fuse_error.comments;
            return false;
          } else {
            MY_LOG(1) << "merge hipdnn operation";
            auto hipdnn_param = hipdnn::HipdnnParamProto();
            
            // Set device and kernel type
            hipdnn_param.set_device_id("0");
            hipdnn_param.set_kernel_type("conv");
            
            // Extract Conv attributes from the node
            hipdnn_param.set_op_type(node_get_op_type(*conv_node));
            auto pads_attr = node_get_attr_ints(*conv_node, "pads");
            if (pads_attr.has_value()) {
              for (auto pad : pads_attr.value()) {
                hipdnn_param.add_pads(pad);
              }
            }
            auto strides_attr = node_get_attr_ints(*conv_node, "strides");
            if (strides_attr.has_value()) {
              for (auto stride : strides_attr.value()) {
                hipdnn_param.add_strides(stride);
              }
            }
            auto dilations_attr = node_get_attr_ints(*conv_node, "dilations");
            if (dilations_attr.has_value()) {
              for (auto dilation : dilations_attr.value()) {
                hipdnn_param.add_dilations(dilation);
              }
            }
            auto group_attr = node_get_attr_int(*conv_node, "group");
            if (group_attr.has_value()) {
              hipdnn_param.set_group(group_attr.value());
            }
            
            auto hipdnn_json_str = std::string();
            auto status = google::protobuf::util::MessageToJsonString(
                hipdnn_param, &hipdnn_json_str);
            self->attach_meta_def_param(*meta_def, hipdnn_json_str.c_str());
            self->fuse(*graph, std::move(*meta_def));
          }
          return true; // return true if graph is modified.
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Level1HipDnn, vaip_pass_level1_hipdnn)
