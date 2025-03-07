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

#include "vaip/pass.hpp"
#define VAIP_USE_DEPRECATED_API 1
#include "vaip/anchor_point.hpp"
#include "vaip/graph.hpp"
#include "vaip/node.hpp"
#include "vaip/node_arg.hpp"
#include "vaip/node_attr.hpp"
#include "vaip/tensor_proto.hpp"
#include "vaip/util.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <vaip/my_ort.h>
#include <vaip/vaip_core.hpp>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_NODE_BUILDER, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_NODE_BUILDER) >= n)
namespace vaip_core {

VAIP_DLL_SPEC Node&
graph_add_node(Graph& graph, const std::string& name,
               const std::string& op_type, const std::string& description,
               const std::vector<const NodeArg*>& input_args,
               const std::vector<const NodeArg*>& output_args,
               NodeAttributesPtr attributes, const std::string& domain) {
  auto& ret = VAIP_ORT_API(graph_add_node)(graph, name, op_type, description,
                                           input_args, output_args,
                                           *attributes.get(), domain);
  return ret;
}

VAIP_DLL_SPEC std::vector<const NodeArg*>
node_inputs_2_node_args(const std::vector<NodeInput>& inputs) {
  auto input_args = std::vector<const NodeArg*>();
  input_args.resize(inputs.size());
  for (auto i = 0u; i < inputs.size(); ++i) {
    input_args[i] = inputs[i].node_arg;
  }
  return input_args;
}

void graph_set_name(Graph& graph, const std::string& name) {
  VaipOrtApi2::graph_set_name(graph, name.c_str());
}
NodeBuilder::NodeBuilder(Graph& graph, IPass& pass)
    : graph_{graph}, pass_{&pass} {}

static int data_type_2_element_type(const std::string& data_type) {
  auto ret = 0;
  if (data_type == "float32") {
    ret = onnx::TensorProto_DataType_FLOAT;
  } else if (data_type == "int8") {
    ret = onnx::TensorProto_DataType_INT8;
  } else if (data_type == "int32") {
    ret = onnx::TensorProto_DataType_INT32;
  } else if (data_type == "int64") {
    ret = onnx::TensorProto_DataType_INT64;
  } else if (data_type == "uint8") {
    ret = onnx::TensorProto_DataType_UINT8;
  } else if (data_type == "float16") {
    // It seems that FP16 is float16, and BFLOAT16 is bf16. but xir don't
    // support it or don't care now. test case: adobe fp16 model.
    ret = onnx::TensorProto_DataType_FLOAT16;
  } else if (data_type == "bfloat16") {
    ret = onnx::TensorProto_DataType_BFLOAT16;
  } else if (data_type == "uint16") {
    ret = onnx::TensorProto_DataType_UINT16;
  } else if (data_type == "int16") {
    ret = onnx::TensorProto_DataType_INT16;
  } else {
    LOG(FATAL) << "data_type " << data_type << " " //
        ;
  }
  return ret;
}
const Node& NodeBuilder::build() {
  CHECK_GE(num_of_outputs_, 1u);
  CHECK_EQ(num_of_outputs_, anchor_node_arg_.size())
      << "must invoke set_anchor_point1/2/3/4 before ::build()";
  CHECK(!op_type_.empty())
      << "must invoke clone_op_type/set_op_type before ::build()";
  if (domain_ == "com.xilinx") {
    // clang-format off
    // TODO: check only for com.xilinx ops?
    CHECK_EQ(num_of_outputs_, data_type_.size())
      << "must invoke set_anchor_point1/2/3/4 with same number of call to set_data_type";
    CHECK_EQ(num_of_outputs_, shape_.size())
      << "must invoke set_anchor_point1/2/3/4 with same number of call to set_shape";
    // clang-format on
  }

  auto output_args = std::vector<const NodeArg*>(num_of_outputs_);
  for (auto i = 0u; i < num_of_outputs_; ++i) {
    if (!anchor_node_arg_[i].has_value()) {
      CHECK(shape_[i].empty());
      CHECK(data_type_[i].empty());
      CHECK(anchor_point_[i] == nullptr);
      // ORT does not allow nullptr when create a new node, try empty string();
      auto name = std::string();
      auto empty_node_arg = VAIP_ORT_API(graph_get_node_arg)(graph_, name);
      if (empty_node_arg == nullptr) {
        // ususually it won't go here. but if it goes here, it means the there
        // is no the empty node arg;
        empty_node_arg = &VAIP_ORT_API(node_arg_new)(graph_, name, nullptr, 0);
      }
      output_args[i] = empty_node_arg;
    } else {
      CHECK(!data_type_[i].empty());
      CHECK(anchor_point_[i] != nullptr);
      auto reuse_existing_node_arg = anchor_point_[i]->get_proto().name() ==
                                     anchor_node_arg_[i].value().name();
      if (reuse_existing_node_arg) {
        output_args[i] = anchor_node_arg_[i].value().ptr();
      } else {
        // create a new node arg;
        auto name = anchor_point_[i]->get_proto().name();
        CHECK(!name.empty())
            << anchor_point_[i]->op_debug_string() << " i = " << i;
        output_args[i] = &VAIP_ORT_API(node_arg_new)(
            graph_, name, &shape_[i], data_type_2_element_type(data_type_[i]));
      }
    }
  }
  auto name_with_suffix = anchor_point_[0]->get_proto().name();
  auto& newly_added_node = graph_add_node(
      graph_, std::string("vaip_node_") + name_with_suffix, op_type_,
      description_, input_args_, output_args, std::move(attrs_), domain_);

  if (domain_ == "com.xilinx") {
    if (num_of_outputs_ == 1u) {
      attrs_builder_.add("data_type", data_type_[0]);
      if (!shape_[0].empty()) {
        // now, all xilinx.com op support scalar.
        // to support scalar, from ort there is no different between non
        // existant attr and empty vector.
        attrs_builder_.add("shape", shape_[0]);
      } else {
        // xilinx op shape exist but value is empty to support scalar
        attrs_builder_.add("shape", std::vector<int64_t>{});
      }
    } else {
      for (auto i = 0u; i < num_of_outputs_; i++) {
        if (!shape_[i].empty()) {
          attrs_builder_.add(std::string("data_type_") + std::to_string(i),
                             data_type_[i]);
          attrs_builder_.add(std::string("shape_") + std::to_string(i),
                             shape_[i]);
        }
      }
    }
  }
  attrs_builder_.merge_into(newly_added_node);
  for (auto i = 0u; i < num_of_outputs_; ++i) {
    if (!anchor_node_arg_[i].has_value()) {
      continue;
    }
    auto existing_node = anchor_node_arg_[i].value().find_producer();
    if (anchor_point_[i]->is_identity(/*test_all=*/false)) {
      auto origin_node_arg_name = anchor_point_[i]->origin_node_arg_name();
      MY_LOG(1) << " try to replace node: "
                << "origin_node_arg_name " << origin_node_arg_name << " "    //
                << "anchor_node_arg_ " << anchor_node_arg_[i].value() << " " //
                << "new_node " << node_as_string(newly_added_node) << " "    //
                << "name_suffix " << name_with_suffix << " "                 //
                << "anchor_point_ " << anchor_point_[i]->op_debug_string()
                << " "                                                       //
          ;
      if (existing_node.has_value()) {
        auto existing_node_args = existing_node.value().outputs();
        CHECK(existing_node_args[0].has_value()) << existing_node.value();
        MY_LOG(1) << " node is deleted: " << existing_node.value();
        VAIP_ORT_API(graph_remove_node)
        (graph_,
         {existing_node.value().ptr(), existing_node_args[0].value().ptr()});
      } else {
        MY_LOG(1) << " cannot find the node, might be deleted already.";
      }
    } else {
      auto origin_node_arg_name = anchor_point_[i]->origin_node_arg_name();
      MY_LOG(1) << " update anchor point: num_of_outputs_ = "
                << num_of_outputs_;
      MY_LOG(1) << " add node: "
                << "origin_node_arg_name " << origin_node_arg_name << " "    //
                << "anchor_node_arg_ " << anchor_node_arg_[i].value() << " " //
                << "new_node " << node_as_string(newly_added_node) << " "    //
                << "name_suffix " << name_with_suffix << " "                 //
                << "anchor_point_ " << anchor_point_[i]->op_debug_string()
                << " "                                                       //
          ;
      anchor_point_[i]->insert_into_context(*pass_);
    }
  }
  return newly_added_node;
}
vaip_cxx::NodeConstRef NodeBuilder::build_ex() {
  return vaip_cxx::NodeConstRef::from_node(graph_, build());
}
NodeBuilder& NodeBuilder::clone_node(const Node& node) {
  clone_inputs(node);
  clone_op_type(node);
  clone_attrs(node);
  clone_shape(node);
  clone_data_type(node);
  return *this;
}

NodeBuilder& NodeBuilder::clone_inputs(const Node& node) {
  auto inputs = node_get_inputs(node);
  input_args_ = node_inputs_2_node_args(inputs);
  return *this;
}

NodeBuilder& NodeBuilder::append_input(const Node& node) {
  input_args_.push_back(&node_get_output_node_arg(node));
  return *this;
}

NodeBuilder& NodeBuilder::clone_op_type(const Node& node) {
  op_type_ = VAIP_ORT_API(node_op_type)(node);
  domain_ = VAIP_ORT_API(node_op_domain)(node);
  return *this;
}

NodeBuilder& NodeBuilder::clone_attrs(const Node& node) {
  attrs_ = node_clone_attributes(node);
  return *this;
}

NodeAttributesBuilder& NodeBuilder::get_attrs_builder() {
  return attrs_builder_;
}

NodeBuilder& NodeBuilder::clone_shape(const Node& node) {
  auto& node_arg = node_get_output_node_arg(node);
  return clone_shape(node_arg);
}
NodeBuilder& NodeBuilder::clone_shape(const NodeArg& node_arg) {
  auto shape = node_arg_get_shape_i64(node_arg);
  CHECK(shape != nullptr) << "does not support dynamice shape";
  set_shape(*shape);
  return *this;
}
NodeBuilder& NodeBuilder::set_shape(const gsl::span<const int64_t>& shape) {
  if (shape_.size() < num_of_outputs_) {
    shape_.emplace_back();
  }
  CHECK_EQ(shape_.size(), num_of_outputs_);
  shape_.back().assign(shape.begin(), shape.end());
  return *this;
}
NodeBuilder& NodeBuilder::clone_data_type(const Node& node) {
  auto args = node_get_output_node_args(node);
  CHECK_EQ(args.size(), 1u) << "TODO: support multiple outputs";
  return clone_data_type(*args[0]);
}

NodeBuilder& NodeBuilder::clone_data_type(const NodeArg& node_arg) {
  auto data_type =
      data_type_to_string(VAIP_ORT_API(node_arg_get_element_type)(node_arg));
  return set_data_type(data_type);
}

NodeBuilder& NodeBuilder::set_data_type(const std::string& data_type) {
  if (data_type_.size() < num_of_outputs_) {
    data_type_.emplace_back();
  }
  CHECK_EQ(data_type_.size(), num_of_outputs_);
  data_type_.back() = data_type;
  return *this;
}

NodeBuilder& NodeBuilder::set_anchor_point1(const Node& node1) {
  auto node = vaip_cxx::NodeConstRef::from_node(graph_, node1);
  auto args = node.outputs();
  for (auto i = 0u; i < args.size(); ++i) {
    if (args[i].has_value()) {
      set_anchor_point1(args[i].value());
    } else {
      skip_optional_output();
    }
  }
  return *this;
}

NodeBuilder& NodeBuilder::set_anchor_point1(const NodeArg& node_arg1) {
  anchor_node_arg_.emplace_back(
      vaip_cxx::NodeArgConstRef::from_node_arg(graph_, node_arg1));
  CHECK_EQ(anchor_node_arg_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point1 more than once";

  anchor_point_.emplace_back(AnchorPoint::identity(*pass_, node_arg1));
  CHECK_EQ(anchor_point_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point1 more than once";

  clone_shape(node_arg1);
  clone_data_type(node_arg1);
  return *this;
}

NodeBuilder&
NodeBuilder::set_anchor_point2(const NodeArg& node_arg,
                               const AnchorPoint::Description& description) {
  CHECK_EQ(shape_.size(), num_of_outputs_)
      << "must call set_shape() before call set_anchor_point2";
  return set_anchor_point3(node_arg, description, std::move(shape_.back()));
}
NodeBuilder&
NodeBuilder::set_anchor_point3(const NodeArg& node_arg,
                               const AnchorPoint::Description& description,
                               const std::vector<int64_t>& shape) {
  CHECK_EQ(data_type_.size(), num_of_outputs_)
      << "must call set_data_type() before call set_anchor_point2/3";
  return set_anchor_point4(node_arg, description, shape, data_type_.back());
}

NodeBuilder& NodeBuilder::set_anchor_point4(
    const NodeArg& node_arg, const AnchorPoint::Description& description,
    const std::vector<int64_t>& shape, const std::string& data_type) {
  set_shape(shape);
  set_data_type(data_type);
  anchor_node_arg_.emplace_back(
      vaip_cxx::NodeArgConstRef::from_node_arg(graph_, node_arg));
  CHECK_EQ(anchor_node_arg_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point2/3/4 more than once";

  anchor_point_.emplace_back(
      AnchorPoint::create(*pass_, node_arg, description));
  CHECK_EQ(anchor_point_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point2/3/4 more than once";
  return *this;
}
NodeBuilder& NodeBuilder::add_output() {
  CHECK_EQ(anchor_node_arg_.size(), num_of_outputs_)
      << "must call set_anchor_point1/2/3/4 before add_output";
  CHECK_EQ(anchor_point_.size(), num_of_outputs_)
      << "must call set_anchor_point1/2/3/4 before add_output";
  // TODO: do we need check if (op_domain == "com.xilinx") ?
  CHECK_EQ(shape_.size(), num_of_outputs_)
      << "must call set_shape or clone_shape before add_output";
  CHECK_EQ(data_type_.size(), num_of_outputs_)
      << "must call set_shape or clone_shape before add_output";
  num_of_outputs_ = num_of_outputs_ + 1u;
  return *this;
}
NodeBuilder& NodeBuilder ::skip_optional_output() {
  shape_.emplace_back();
  data_type_.emplace_back();
  anchor_node_arg_.emplace_back(std::nullopt);
  anchor_point_.emplace_back(nullptr);
  return *this;
}

NodeBuilder& NodeBuilder::set_op_type(const std::string& op_type,
                                      const std::string& domain) {
  op_type_ = op_type;
  domain_ = domain;
  return *this;
}

NodeBuilder& NodeBuilder::set_input_node_args(
    const std::vector<const NodeArg*>& input_args) {
  input_args_ = input_args;
  return *this;
}
NodeBuilder& NodeBuilder::set_input_node_args_ex(
    const std::vector<vaip_cxx::NodeArgConstRef>& input_args) {
  input_args_.resize(input_args.size());
  for (auto i = 0u; i < input_args.size(); ++i) {
    input_args_[i] = input_args[i].ptr();
  }
  return *this;
}
NodeBuilder&
NodeBuilder::set_input_nodes(const std::vector<const Node*>& input_nodes) {
  input_args_.resize(input_nodes.size());
  for (auto i = 0u; i < input_args_.size(); ++i) {
    auto outputs = node_get_output_node_args(*input_nodes[i]);
    CHECK_EQ(outputs.size(), 1u);
    input_args_[i] = outputs[0];
  }
  return *this;
}

VAIP_DLL_SPEC std::vector<const Node*>
graph_get_output_nodes(const Graph& graph) {
  auto graph_outputs = graph_get_outputs(graph);
  auto leaf_nodes = std::vector<const Node*>();
  leaf_nodes.reserve(graph_outputs.size());
  for (auto& o : graph_outputs) {
    if (o) {
      auto n = VAIP_ORT_API(graph_producer_node)(graph, node_arg_get_name(*o));
      if (n != nullptr) {
        leaf_nodes.push_back(n);
      }
    }
  }
  return leaf_nodes;
}

void graph_gc(Graph& graph) {
  std::vector<const Node*> leaf_nodes;
  auto all_nodes = graph_nodes(graph);
  auto graph_outputs = graph_get_outputs(graph);
  leaf_nodes.reserve(graph_outputs.size());
  for (auto n : all_nodes) {
    CHECK(n != nullptr);
    auto node_outputs = node_get_output_node_args(*n);
    auto found = std::any_of(node_outputs.begin(), node_outputs.end(),
                             [&graph_outputs](const NodeArg* x) {
                               return std::find(graph_outputs.begin(),
                                                graph_outputs.end(),
                                                x) != graph_outputs.end();
                             });
    if (found) {
      leaf_nodes.push_back(n);
    }
  }
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      graph,      //
      leaf_nodes, //
      nullptr,    //
      [&all_nodes](const Node* n) mutable {
        all_nodes.erase(std::remove(all_nodes.begin(), all_nodes.end(), n),
                        all_nodes.end());
      }, //
      nullptr);
  MY_LOG(1) << "prepare to remove " << all_nodes.size() << " nodes";
  for (auto n : all_nodes) {
    MY_LOG(1) << "\tremove " << node_as_string(*n);
    VAIP_ORT_API(graph_remove_node)(graph, {n, nullptr});
  }
}

VAIP_DLL_SPEC void graph_resolve(Graph& graph, bool force) {
  auto status = VAIP_ORT_API(graph_resolve)(graph, force);
  CHECK(status == 0) << " resolve error: " << status;
  return;
}

const Model& graph_get_model(const Graph& graph) {
  return VAIP_ORT_API(graph_get_model)(graph);
}

std::vector<const Node*> graph_nodes(const Graph& graph) {
  return *VAIP_ORT_API(graph_nodes_unsafe)(graph);
}

std::vector<const NodeArg*> graph_get_inputs(const Graph& graph) {
  return *VAIP_ORT_API(graph_get_inputs_unsafe)(graph);
}

VAIP_DLL_SPEC std::vector<const NodeArg*>
graph_get_outputs(const Graph& graph) {
  return *VAIP_ORT_API(graph_get_outputs_unsafe)(graph);
}

VAIP_DLL_SPEC std::vector<size_t>
graph_get_node_in_topoligical_order(const Graph& graph) {
  std::vector<size_t> ret;
  auto output_nodes = graph_get_output_nodes(graph);
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      graph,        //
      output_nodes, // leaf nodes, output
      nullptr,      // enter
      [&ret](const Node* n) mutable {
        ret.push_back(VAIP_ORT_API(node_get_index)(*n));
      }, //
      nullptr);
  return ret;
}

static std::string indent(int level) {
  return std::string((size_t)(level * 4), ' ');
}

extern std::string node_args_as_string(
    const std::vector<const NodeArg*>& args); /* defined in node.cpp */
static std::string graph_as_string_subgraph(const Graph& graph, int level) {
  std::ostringstream str;
  str << indent(level) << "graph[name=" << VAIP_ORT_API(graph_get_name)(graph)
      << "] = {";
  str << "\n"
      << indent(level + 1)
      << "inputs = " << node_args_as_string(graph_get_inputs(graph)) << "\n"
      << indent(level + 1)
      << "outputs=" << node_args_as_string(graph_get_outputs(graph));
  auto nodes = graph_get_node_in_topoligical_order(graph);
  for (auto node_idx : nodes) {
    auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
    if (node == nullptr) { // should never goes here
      str << "\n" << indent(level + 1) << "null";

    } else {
      str << "\n"
          << indent(level + 1) << " [" << node_idx << "]"
          << node_as_string(*node);
      auto is_fused = VAIP_ORT_API(node_type_is_fused)(*node);
      if (is_fused) {
        str << "\n"
            << graph_as_string_subgraph(
                   VAIP_ORT_API(node_get_function_body)(*node), level + 1);
      }
    }
  }
  str << "\n}\n";
  return str.str();
}

std::string graph_as_string(const Graph& graph) {
  return graph_as_string_subgraph(graph, 0);
}

std::vector<const Node*>
graph_get_consumer_nodes(const Graph& graph, const std::string& node_arg_name) {
  return *VAIP_ORT_API(graph_get_consumer_nodes_unsafe)(graph, node_arg_name);
}

void graph_replace_node_arg(const Graph& graph, const IPass& pass,
                            const NodeArg& from, const NodeArg& to) {
  CHECK(*node_arg_get_shape_i64(from) == *node_arg_get_shape_i64(to))
      << "mismatch shape between from and to nodeargs";
  CHECK(VAIP_ORT_API(node_arg_get_element_type)(from) ==
        VAIP_ORT_API(node_arg_get_element_type)(to))
      << "mismatch data type between from and to nodeargs";

  auto from_nodearg_name = node_arg_get_name(from);
  auto consumers = graph_get_consumer_nodes(graph, from_nodearg_name);

  for (auto consumer : consumers) {
    auto inputs = node_get_inputs(*consumer);
    auto input_nodeargs = std::vector<const NodeArg*>();
    input_nodeargs.reserve(inputs.size());
    for (auto input : inputs) {
      CHECK(input.node_arg != nullptr);
      if (node_arg_exists(*input.node_arg)) {
        if (node_arg_get_name(*input.node_arg) == node_arg_get_name(from)) {
          input_nodeargs.emplace_back(&to);
        } else {
          input_nodeargs.emplace_back(input.node_arg);
        }
      } else {
        input_nodeargs.emplace_back(input.node_arg);
      }
    }
    NodeBuilder(const_cast<Graph&>(graph), const_cast<IPass&>(pass))
        .clone_node(*consumer)
        .set_input_node_args(input_nodeargs)
        .set_anchor_point1(*consumer)
        .build();
  }
  graph_resolve(const_cast<Graph&>(graph));
  return;
}

} // namespace vaip_core

namespace vaip_cxx {
const std::string& GraphConstRef::name() const {
  return VAIP_ORT_API(graph_get_name)(*this);
}
GraphConstRef::~GraphConstRef() {}

const std::filesystem::path& GraphConstRef::model_path() const {
  return VAIP_ORT_API(get_model_path)(*this);
}

std::vector<NodeArgConstRef> GraphConstRef::inputs() const {
  auto inputs = VAIP_ORT_API(graph_get_inputs_unsafe)(*this);
  auto ret = std::vector<NodeArgConstRef>();
  ret.reserve(inputs->size());
  for (auto i = 0u; i < inputs->size(); ++i) {
    auto ptr = (*inputs)[i];
    CHECK(ptr != nullptr);
    auto node_arg = NodeArgConstRef(graph_, *ptr);
    ret.push_back(node_arg);
  }
  return ret;
}
std::vector<NodeArgConstRef> GraphConstRef::outputs() const {
  auto outputs = VAIP_ORT_API(graph_get_outputs_unsafe)(*this);
  auto ret = std::vector<NodeArgConstRef>();
  ret.reserve(outputs->size());
  for (auto i = 0u; i < outputs->size(); ++i) {
    auto ptr = (*outputs)[i];
    CHECK(ptr != nullptr);
    auto node_arg = NodeArgConstRef(graph_, *ptr);
    ret.push_back(node_arg);
  }
  return ret;
}
std::vector<NodeArgConstRef> GraphConstRef::constant_initializers() const {
  auto& initializers = VAIP_ORT_API(graph_get_all_initialized_tensors)(*this);
  auto ret = std::vector<NodeArgConstRef>();
  ret.reserve(initializers.size());
  for (auto constant : initializers) {
    auto& name = constant.first;
    auto node_arg = VAIP_ORT_API(graph_get_node_arg)(*this, name);
    CHECK(node_arg != nullptr) << "cannot get node arg: name=" << name;
    auto node_arg_2 =
        NodeArgConstRef(graph_, const_cast<vaip_core::NodeArg&>(*node_arg));
    ret.push_back(node_arg_2);
  }
  return ret;
}

std::vector<NodeConstRef> GraphConstRef::nodes() const {
  auto nodes = vaip_core::graph_nodes(*this);
  auto ret = std::vector<NodeConstRef>();
  ret.reserve(nodes.size());
  for (auto i = 0u; i < nodes.size(); ++i) {
    auto ptr = nodes[i];
    CHECK(ptr != nullptr);
    auto node = NodeConstRef(graph_, *ptr);
    ret.push_back(node);
  }
  return ret;
}

void GraphConstRef::save(const std::filesystem::path& file_path) const {
  VAIP_ORT_API(graph_save)
  (*this, file_path.u8string(), "", std::numeric_limits<size_t>::max());
}

void GraphConstRef::save(const std::filesystem::path& file_path,
                         const std::filesystem::path& external_data_file,
                         size_t threshold) const {
  VAIP_ORT_API(graph_save)
  (*this, file_path.u8string(), external_data_file.u8string(), threshold);
}

std::optional<NodeArgConstRef>
GraphConstRef::find_node_arg(const std::string& name) const {
  auto node_arg = VAIP_ORT_API(graph_get_node_arg)(*this, name);
  if (node_arg != nullptr) {
    return NodeArgConstRef(graph_, *node_arg);
  } else {
    return std::nullopt;
  }
}

std::vector<NodeConstRef>
GraphConstRef::find_consumers(const std::string& name) const {
  auto consumers = vaip_core::graph_get_consumer_nodes(*this, name);
  auto ret = std::vector<NodeConstRef>();
  ret.reserve(consumers.size());
  for (auto i = 0u; i < consumers.size(); ++i) {
    auto ptr = consumers[i];
    if (ptr != nullptr) {
      auto node = NodeConstRef(graph_, *ptr);
      ret.push_back(node);
    } else {
      LOG(WARNING) << " one of consumers is nullptr, name=" << name;
    }
  }
  return ret;
}
std::optional<NodeConstRef>
GraphConstRef::find_node(const std::string& name) const {
  auto node = VAIP_ORT_API(graph_producer_node)(graph_, name);
  if (node == nullptr) {
    return std::nullopt;
  }
  return NodeConstRef(graph_, *node);
}

std::pair<std::unique_ptr<vaip_core::MetaDefProto>, vaip_core::TryFuseError>
GraphConstRef::try_fuse(const std::string& name,
                        const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs,
                        const std::vector<std::string>& constant_initializers,
                        const std::string& device) const {
  return vaip_core::IPass_try_fuse(*this, name, inputs, outputs,
                                   constant_initializers, device);
}
Subgraph
GraphConstRef::virtual_fuse(const vaip_core::MetaDefProto& meta_def) const {
  auto inputs = std::vector<NodeArgConstRef>();
  auto outputs = std::vector<NodeArgConstRef>();
  auto nodes = std::vector<NodeConstRef>();
  auto constant_initializers = std::vector<NodeArgConstRef>();
  inputs.reserve(meta_def.inputs_size());
  outputs.reserve(meta_def.outputs_size());
  nodes.reserve(meta_def.nodes_size());
  constant_initializers.reserve(meta_def.constant_initializers_size());
  for (auto& input : meta_def.inputs()) {
    auto node_arg = find_node_arg(input);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << input;
    inputs.push_back(node_arg.value());
  }
  for (auto& output : meta_def.outputs()) {
    auto node_arg = find_node_arg(output);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << output;
    outputs.push_back(node_arg.value());
  }
  std::set<size_t> node_indice;
  for (auto it = meta_def.nodes().begin(), end = meta_def.nodes().end();
       it != end; ++it) {
    // it is important to keep nodes in topological order
    auto node = find_node(*it);
    CHECK(node.has_value()) << "cannot find node: " << *it;
    node_indice.insert(node.value().index());
  }
  nodes.reserve(node_indice.size());
  std::vector<size_t> ret;
  auto output_nodes = std::vector<const onnxruntime::Node*>();
  output_nodes.reserve(outputs.size());
  for (auto& output : outputs) {
    output_nodes.push_back(output.find_producer().value().ptr());
  }
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      *this,        //
      output_nodes, // leaf nodes, output
      nullptr,      // enter
      [&ret, this, &nodes](const onnxruntime::Node* n) mutable {
        nodes.push_back(NodeConstRef::from_node(*this, *n));
      }, //
      [&node_indice, this](const onnxruntime::Node* from,
                           const onnxruntime::Node* to) -> bool {
        auto in_body =
            node_indice.find(NodeConstRef::from_node(*this, *to).index()) !=
            node_indice.end();
        bool stop = !in_body;
        return stop;
      });
  CHECK_EQ(nodes.size(), node_indice.size());
  for (auto& initializer_name : meta_def.constant_initializers()) {
    auto node_arg = find_node_arg(initializer_name);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << initializer_name;
    constant_initializers.push_back(node_arg.value());
  }
  return Subgraph(inputs, outputs, nodes, constant_initializers);
}
NodeConstRef GraphConstRef::node(size_t index) const {
  auto node = VAIP_ORT_API(graph_get_node)(*this, index);
  CHECK(node != nullptr) << "cannot get node: index=" << index;
  return NodeConstRef(graph_, *node);
}

std::vector<NodeConstRef> GraphConstRef::nodes_in_topological_order() const {
  auto node_indices = vaip_core::graph_get_node_in_topoligical_order(*this);
  auto ret = std::vector<NodeConstRef>();
  ret.reserve(node_indices.size());
  for (auto i : node_indices) {
    auto node = VAIP_ORT_API(graph_get_node)(*this, i);
    CHECK(node != nullptr);
    ret.push_back(NodeConstRef(graph_, *node));
  }
  return ret;
}

std::string GraphConstRef::to_string() const {
  return vaip_core::graph_as_string(*this);
}
std::ostream& operator<<(std::ostream& os, const GraphConstRef& graph) {
  return os << graph.to_string();
}
GraphRef::GraphRef(vaip_core::Graph& graph) : GraphConstRef(graph) {}

GraphRef::~GraphRef() {}
void GraphRef::set_name(const std::string& name) {
  vaip_core::graph_set_name(*this, name);
}
bool GraphRef::resolve(bool force) {
  return VAIP_ORT_API(graph_resolve)(*this, force) == 0;
}
NodeRef GraphRef::fuse(const vaip_core::MetaDefProto& meta_def) {
  auto name = meta_def.id();
  // TODO, op_type and domain is hard coded at ORT side.
  // com.xilinx::super_layer.
  auto op_type = std::string("not_used_op");
  auto inputs = std::vector<std::string>{meta_def.inputs().begin(),
                                         meta_def.inputs().end()};
  auto outputs = std::vector<std::string>{meta_def.outputs().begin(),
                                          meta_def.outputs().end()};
  auto constant_initializers =
      std::vector<std::string>{meta_def.constant_initializers().begin(),
                               meta_def.constant_initializers().end()};
  auto nodes = std::vector<size_t>();
  nodes.reserve(meta_def.nodes_size());
  for (auto& first_node_arg_name : meta_def.nodes()) {
    auto node = find_node(first_node_arg_name);
    CHECK(node.has_value()) << "cannot find node: " << first_node_arg_name;
    nodes.push_back(node.value().index());
  }
  vaip_core::Node& fused_node = VAIP_ORT_API(graph_fuse)(
      *this, name, op_type, nodes, inputs, outputs, constant_initializers);
  resolve();
  return NodeRef(*this, fused_node);
}
vaip_core::NodeBuilder GraphRef::node_builder(vaip_core::IPass& pass) {
  return vaip_core::NodeBuilder(*this, pass);
}
void GraphRef::gc() { vaip_core::graph_gc(*this); }

static std::string
graph_ref_generate_unique_constant_initializer_name(const GraphConstRef& graph,
                                                    const std::string& prefix) {
  auto name = std::string();
  if (prefix.empty()) {
    name = std::string("vaip_constant_initializer_") +
           std::to_string(graph.constant_initializers().size());
  } else {
    name = prefix;
  }
  return name;
}
#define VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(type, cxx_type,               \
                                                 tensor_data_type)             \
  NodeArgRef GraphRef::new_constant_initializer_##type(                        \
      cxx_type value, const std::string& name_hint) {                          \
    const std::vector<int64_t> shape = {};                                     \
    const std::vector<cxx_type> values = {value};                              \
    auto name =                                                                \
        graph_ref_generate_unique_constant_initializer_name(*this, name_hint); \
    auto tensor = vaip_core::tensor_proto_new_##type(name, shape, values);     \
    VAIP_ORT_API(graph_add_initialized_tensor)(*this, *tensor);                \
    auto& newly_create_node_arg = VAIP_ORT_API(node_arg_new)(                  \
        *this, name, &shape,                                                   \
        ONNX_NAMESPACE::TensorProto_DataType_##tensor_data_type);              \
    return NodeArgRef(*this, newly_create_node_arg);                           \
  }

VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(i8, int8_t, INT8)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(u8, uint8_t, UINT8)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(i16, int16_t, INT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(u16, uint16_t, UINT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(i32, int32_t, INT32)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(u32, uint32_t, UINT32)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(i64, int64_t, INT64)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(u64, uint64_t, UINT64)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(f32, float, FLOAT)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(f64, double, DOUBLE)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(bf16, bf16_t, BFLOAT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER(fp16, fp16_t, FLOAT16)

#define VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(type, cxx_type,          \
                                                      tensor_data_type)        \
  NodeArgRef GraphRef::new_constant_initializer_##type##_span(                 \
      gsl::span<const cxx_type> values_span,                                   \
      const std::vector<int64_t>& shape, const std::string& name_hint) {       \
    std::vector<cxx_type> values(values_span.begin(), values_span.end());      \
    auto name =                                                                \
        graph_ref_generate_unique_constant_initializer_name(*this, name_hint); \
    auto tensor = vaip_core::tensor_proto_new_##type(name, shape, values);     \
    VAIP_ORT_API(graph_add_initialized_tensor)(*this, *tensor);                \
    auto& newly_create_node_arg = VAIP_ORT_API(node_arg_new)(                  \
        *this, name, &shape,                                                   \
        ONNX_NAMESPACE::TensorProto_DataType_##tensor_data_type);              \
    return NodeArgRef(*this, newly_create_node_arg);                           \
  }

VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(i8, int8_t, INT8)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(u8, uint8_t, UINT8)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(i16, int16_t, INT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(u16, uint16_t, UINT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(i32, int32_t, INT32)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(u32, uint32_t, UINT32)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(i64, int64_t, INT64)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(u64, uint64_t, UINT64)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(f32, float, FLOAT)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(f64, double, DOUBLE)

VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(bf16, bf16_t, BFLOAT16)
VAIP_CXX_DEFINE_NEW_CONSTANT_INITIALIZER_SPAN(fp16, fp16_t, FLOAT16)

void GraphRef::set_inputs(const std::vector<NodeArgConstRef>& inputs) {
  auto inputs_ptr = std::vector<const vaip_core::NodeArg*>();
  inputs_ptr.reserve(inputs.size());
  for (auto& input : inputs) {
    inputs_ptr.push_back(input.ptr());
  }
  VAIP_ORT_API(graph_set_inputs)(*this, inputs_ptr);
}
void GraphRef::set_outputs(const std::vector<NodeArgConstRef>& outputs) {
  auto outputs_ptr = std::vector<const vaip_core::NodeArg*>();
  outputs_ptr.reserve(outputs.size());
  for (auto& output : outputs) {
    outputs_ptr.push_back(output.ptr());
  }
  VAIP_ORT_API(graph_set_outputs)(*this, outputs_ptr);
}
NodeArgConstRef
GraphRef::new_node_arg(const std::string& name,
                       const std::vector<int64_t>& shape,
                       ONNX_NAMESPACE::TensorProto_DataType data_type) {
  return NodeArgConstRef::from_node_arg(
      self(), VAIP_ORT_API(node_arg_new)(*this, name, &shape, data_type));
}
NodeRef
GraphRef::add_node(const std::string& name, const std::string& op_domain,
                   const std::string& op_type, const std::string& description,
                   const std::vector<std::optional<NodeArgConstRef>>& inputs,
                   const std::vector<std::optional<NodeArgConstRef>>& outputs,
                   vaip_core::NodeAttributesPtr attributes) {
  auto inputs_ptr = std::vector<const vaip_core::NodeArg*>();
  inputs_ptr.reserve(inputs.size());
  // in onnxruntime, the node_arg with an empty name is used to
  // represent the optional node arg.
  auto the_optional_arg = VAIP_ORT_API(graph_get_node_arg)(*this, "");
  if (the_optional_arg == nullptr) {
    auto shape = std::vector<int64_t>{};
    the_optional_arg = &VAIP_ORT_API(node_arg_new)(
        *this, "", &shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  }
  for (auto& input : inputs) { // namespace vaip_cxx
    if (input) {
      inputs_ptr.push_back(input->ptr());
    } else {
      inputs_ptr.push_back(the_optional_arg);
    }
  }

  auto outputs_ptr = std::vector<const vaip_core::NodeArg*>();
  outputs_ptr.reserve(outputs.size());
  for (auto& output : outputs) {
    if (output) {
      outputs_ptr.push_back(output->ptr());
    } else {
      outputs_ptr.push_back(the_optional_arg);
    }
  }

  auto& new_node = VAIP_ORT_API(graph_add_node)(
      *this, name, op_type, description, inputs_ptr, outputs_ptr,
      *attributes.get(), op_domain);
  return NodeRef(*this, new_node);
}
} // namespace vaip_cxx
