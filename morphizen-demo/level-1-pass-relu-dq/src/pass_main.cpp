/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "relu_dq.pb.h"
#include "relu_dq_pattern_json.hpp"
#include <filesystem>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_RELU_DQ, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_RELU_DQ) >= n)
namespace {
using namespace morphizen;
using namespace morphizen_cxx;

struct Level1Dummy {
  Level1Dummy(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    std::shared_ptr<Pattern> pattern_ =
        morphizen::PatternBuilder().create_by_json(
            std::string((const char*)relu_dq_json));
    CHECK(pattern_ != nullptr) << "Pattern merge_conv_relu_maxpool not found";
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // CHECK(*binder["127"].node_arg != nullptr) << "input node is null";
          // CHECK(*binder["131"].node_arg != nullptr) << "output node is null";
          auto input = morphizen_cxx::NodeArgConstRef::from_node_arg(
              *graph, *binder["127"].node_arg);
          auto output = morphizen_cxx::NodeArgConstRef::from_node_arg(
              *graph, *binder["131"].node_arg);
          auto unique_id = output.name();
          auto [meta_def, fuse_error] =
              self_.try_fuse(*graph, unique_id, {input.name()}, {output.name()},
                             {}, "RELU_DQ");
          if (meta_def == nullptr) {
            MY_LOG(1) << "fuse error: " << fuse_error.comments;
            return false;
          } else {
            MY_LOG(1) << "merge relu-dq";
            auto relu_dq_param = relu_dq::ReluDqParamProto();
            *relu_dq_param.mutable_sample_string() = "sample_string_value";
            relu_dq_param.set_sample_int(1234);
            relu_dq_param.add_sample_ints(1);
            relu_dq_param.add_sample_ints(2);
            relu_dq_param.add_sample_strings(input.name());
            relu_dq_param.add_sample_strings(output.name());
            // demo for writing a file into ep context model
            auto ep_context_file_name = output.name() + ".dat";
            relu_dq_param.set_ep_context_file_name(ep_context_file_name);
            auto stream =
                self_.get_context()->open_file_for_write(ep_context_file_name);
            if (stream == nullptr) {
              MY_LOG(1) << "open file error " << ep_context_file_name;
              return false;
            }
            std::string sample_content =
                "this is a sample binary data for tensor " + output.name();
            CHECK_EQ(
                stream->fwrite(sample_content.c_str(), sample_content.size()),
                sample_content.size())
                << "write file error " << ep_context_file_name;
            stream.reset(); // close file
            relu_dq_param.set_ep_context_file_size(sample_content.size());
            auto relu_dq_json_str = std::string();
            auto status = google::protobuf::util::MessageToJsonString(
                relu_dq_param, &relu_dq_json_str);
            self->attach_meta_def_param(*meta_def, relu_dq_json_str.c_str());
            self->fuse(*graph, std::move(*meta_def));
          }
          return true; // return true if graph is modified.
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_MORPHIZEN_PASS(Level1Dummy, morphizen_pass_level1_dummy)
