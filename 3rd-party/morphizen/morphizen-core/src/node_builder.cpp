/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/node_builder.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/tensor_proto.hpp"
#include "morphizen/util.hpp"
#include <glog/logging.h>
#include <morphizen/my_ort.h>

DEF_ENV_PARAM(DEBUG_NODE_BUILDER, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_NODE_BUILDER) >= n)

namespace morphizen {

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
    // support it or don't care now. test case: ado be fp16 model.
    ret = onnx::TensorProto_DataType_FLOAT16;
  } else if (data_type == "bfloat16") {
    ret = onnx::TensorProto_DataType_BFLOAT16;
  } else if (data_type == "uint16") {
    ret = onnx::TensorProto_DataType_UINT16;
  } else if (data_type == "int16") {
    ret = onnx::TensorProto_DataType_INT16;
  } else if (data_type == "int1") {
    ret = onnx::TensorProto_DataType_BOOL;
  } else {
    LOG(FATAL) << "data_type " << data_type << " " //
        ;
  }
  return ret;
}

NodeBuilder::NodeBuilder(Graph& graph, IPass& pass)
    : graph_{graph}, pass_{&pass} {}

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
      auto graph_ref = morphizen_cxx::GraphConstRef(graph_);
      auto empty_node_arg_opt = graph_ref.find_node_arg(name);
      const NodeArg* empty_node_arg = nullptr;
      if (!empty_node_arg_opt.has_value()) {
        // ususually it won't go here. but if it goes here, it means the there
        // is no the empty node arg;
        auto graph_mut = morphizen_cxx::GraphRef(graph_);
        empty_node_arg =
            graph_mut
                .new_node_arg(
                    name, static_cast<ONNX_NAMESPACE::TensorProto_DataType>(0))
                .ptr();
      } else {
        empty_node_arg = empty_node_arg_opt.value().ptr();
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
        auto graph_mut = morphizen_cxx::GraphRef(graph_);
        output_args[i] =
            graph_mut
                .new_node_arg(name, shape_[i],
                              static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
                                  data_type_2_element_type(data_type_[i])))
                .ptr();
      }
    }
  }
  auto name_with_suffix = anchor_point_[0]->get_proto().name();
  // Suppress deprecation warning: graph_add_node is marked deprecated but is
  // intentionally used here. The NodeBuilder class provides the recommended
  // interface, but internally it still needs to call graph_add_node.
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4996) // deprecated declaration
#endif
  auto& newly_added_node = graph_add_node(
      graph_, std::string("morphizen_node_") + name_with_suffix, op_type_,
      description_, input_args_, output_args, std::move(attrs_), domain_);
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

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
    // After graph modifications (e.g., after calling graph add_node/remove_node
    // but before graph resolve), the graph is in a temporarily invalid state.
    // Avoid calling get_producer/get_consumer functions as their return values
    // are unstable and may reference either pre-modification or
    // post-modification nodes, depending on the specific scenario and timing
    // requirements.
    auto existing_node = anchor_producer_node_[i];
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
        // Note: existing_node is cached, so has_value() always returns true.
        // However, when node has multiple outputs, different outputs may
        // reference the same node through different anchor_points. During
        // fusion operations, the node might be deleted when processing one
        // output, but the cached existing_node still exists when processing
        // another output. We need to filter out this case.
        //
        // Example testcase: [PSU1] dd_merge_dqd_gqa pass
        // pattern : *->Q (a) -> GQA(output0, output1, output2) -> DQ (b)
        //        set_anchor_point1(b)
        //        add_output()
        //        set_anchor_point1(output1)
        //        add_output()
        //        set_anchor_point1(output2)
        // result:  * -> FLATMHA (b, output1, output2)
        //
        // the existing_node_args is empty means the node has been deleted
        // already.
        if (existing_node_args.size() == 0) {
          MY_LOG(1) << " node has no outputs, might be deleted already.";
          continue;
        }
        CHECK(existing_node_args[0].has_value()) << existing_node.value();
        MY_LOG(1) << " node is deleted: " << existing_node.value();
        MORPHIZEN_ORT_API(graph_remove_node)
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

morphizen_cxx::NodeConstRef NodeBuilder::build_ex() {
  return morphizen_cxx::NodeConstRef::from_node(graph_, build());
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
  auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph_, node);
  op_type_ = node_ref.op_type();
  domain_ = node_ref.op_domain();
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
  auto arg_ref =
      morphizen_cxx::NodeArgConstRef::from_node_arg(graph_, node_arg);
  auto data_type = data_type_to_string(arg_ref.element_type());
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
  auto node = morphizen_cxx::NodeConstRef::from_node(graph_, node1);
  auto args = node.outputs();

  for (auto i = 0u; i < args.size(); ++i) {
    if (i > 0u) {
      add_output();
    }

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
      morphizen_cxx::NodeArgConstRef::from_node_arg(graph_, node_arg1));
  CHECK_EQ(anchor_node_arg_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point1 more than once";
  anchor_producer_node_.emplace_back(
      anchor_node_arg_.rbegin()->value().find_producer());

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
      morphizen_cxx::NodeArgConstRef::from_node_arg(graph_, node_arg));
  CHECK_EQ(anchor_node_arg_.size(), num_of_outputs_)
      << "cannot invoke set_anchor_point2/3/4 more than once";
  anchor_producer_node_.emplace_back(
      anchor_node_arg_.rbegin()->value().find_producer());

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
  anchor_producer_node_.emplace_back(std::nullopt);
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
    const std::vector<morphizen_cxx::NodeArgConstRef>& input_args) {
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

// Graph utility functions that depend on NodeBuilder

void graph_replace_node_arg(const Graph& graph, const IPass& pass,
                            const NodeArg& from, const NodeArg& to) {
  CHECK(*node_arg_get_shape_i64(from) == *node_arg_get_shape_i64(to))
      << "mismatch shape between from and to nodeargs";
  auto from_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(graph, from);
  auto to_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(graph, to);
  CHECK(from_ref.element_type() == to_ref.element_type())
      << "mismatch data type between from and to nodeargs";

  auto from_nodearg_name = node_arg_get_name(from);
  auto consumers_cxx =
      morphizen_cxx::GraphConstRef(graph).find_consumers(from_nodearg_name);
  // Convert to raw pointers for the loop
  auto consumers = std::vector<const Node*>();
  consumers.reserve(consumers_cxx.size());
  for (auto& c : consumers_cxx) {
    consumers.push_back(c.ptr());
  }

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
  morphizen_cxx::GraphRef(const_cast<Graph&>(graph)).resolve();
  return;
}

} // namespace morphizen

namespace morphizen_cxx {

// Free functions that extend graph functionality with morphizen-core types
// Declarations are in morphizen-core/include/morphizen/graph_extensions.hpp

morphizen::NodeBuilder graph_node_builder(morphizen_cxx::GraphRef& graph,
                                          morphizen::IPass& pass) {
  return morphizen::NodeBuilder(graph, pass);
}

std::pair<std::unique_ptr<morphizen::MetaDefProto>, morphizen::TryFuseError>
graph_try_fuse(const morphizen_cxx::GraphConstRef& graph,
               const std::string& name, const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs,
               const std::vector<std::string>& constant_initializers,
               const std::string& device) {
  return morphizen::IPass_try_fuse(graph, name, inputs, outputs,
                                   constant_initializers, device);
}

morphizen_cxx::Subgraph
graph_virtual_fuse(const morphizen_cxx::GraphConstRef& graph,
                   const morphizen::MetaDefProto& meta_def) {
  auto inputs = std::vector<morphizen_cxx::NodeArgConstRef>();
  auto outputs = std::vector<morphizen_cxx::NodeArgConstRef>();
  auto nodes = std::vector<morphizen_cxx::NodeConstRef>();
  auto constant_initializers = std::vector<morphizen_cxx::NodeArgConstRef>();
  inputs.reserve(meta_def.inputs_size());
  outputs.reserve(meta_def.outputs_size());
  nodes.reserve(meta_def.nodes_size());
  constant_initializers.reserve(meta_def.constant_initializers_size());
  for (auto& input : meta_def.inputs()) {
    auto node_arg = graph.find_node_arg(input);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << input;
    inputs.push_back(node_arg.value());
  }
  for (auto& output : meta_def.outputs()) {
    auto node_arg = graph.find_node_arg(output);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << output;
    outputs.push_back(node_arg.value());
  }
  std::set<size_t> node_indice;
  for (auto it = meta_def.nodes().begin(), end = meta_def.nodes().end();
       it != end; ++it) {
    // it is important to keep nodes in topological order
    auto node = graph.find_node(*it);
    CHECK(node.has_value()) << "cannot find node: " << *it;
    node_indice.insert(node.value().index());
  }
  nodes.reserve(node_indice.size());
  std::vector<size_t> ret;
  auto output_nodes = std::vector<NodeConstRef>();
  output_nodes.reserve(outputs.size());
  for (auto& output : outputs) {
    output_nodes.push_back(output.find_producer().value());
  }
  graph.reverse_dfs_from_multi(
      gsl::make_span(output_nodes), // leaf nodes, output
      nullptr,                      // enter
      [&nodes](NodeConstRef n) mutable {
        nodes.push_back(n);
        return false; // leave callback return value
      },              //
      nullptr,        // comp
      [&node_indice](NodeConstRef /*from*/, NodeConstRef to) -> bool {
        auto in_body = node_indice.find(to.index()) != node_indice.end();
        bool stop = !in_body;
        return stop;
      });
  CHECK_EQ(nodes.size(), node_indice.size());
  for (auto& initializer_name : meta_def.constant_initializers()) {
    auto node_arg = graph.find_node_arg(initializer_name);
    CHECK(node_arg.has_value()) << "cannot find node arg: " << initializer_name;
    constant_initializers.push_back(node_arg.value());
  }
  return Subgraph(inputs, outputs, nodes, constant_initializers);
}

NodeRef graph_fuse(GraphRef& graph, const morphizen::MetaDefProto& meta_def) {
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
    auto node = graph.find_node(first_node_arg_name);
    CHECK(node.has_value()) << "cannot find node: " << first_node_arg_name;
    nodes.push_back(node.value().index());
  }
  morphizen::Node& fused_node = morphizen::graph_fuse(
      graph, name, op_type, nodes, inputs, outputs, constant_initializers);
  graph.resolve();
  return NodeRef::from_node(graph, fused_node);
}

} // namespace morphizen_cxx
