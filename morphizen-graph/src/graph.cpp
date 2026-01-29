/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// NOTE: NodeBuilder has been moved to vaip-core (node_builder.cpp)
// This file now contains only low-level graph wrapper functions.

#include "morphizen/graph.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include "morphizen/node_attr.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <limits>
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>

// Simple logging macro for graph operations
#define MY_LOG(n) LOG(INFO)

// Wrapper functions for tensor_proto_new calls
// These forward to VAIP_ORT_API and handle f32/f64 naming differences
namespace {
#define DEFINE_TENSOR_PROTO_NEW_WRAPPER(type, cxx_type)                        \
  inline vaip_core::TensorProto* tensor_proto_new_##type(                      \
      const std::string& name, const std::vector<int64_t>& shape,              \
      const std::vector<cxx_type>& data) {                                     \
    return VAIP_ORT_API(tensor_proto_new_##type)(name, shape, data);           \
  }

DEFINE_TENSOR_PROTO_NEW_WRAPPER(i8, int8_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(u8, uint8_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(i16, int16_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(u16, uint16_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(i32, int32_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(u32, uint32_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(i64, int64_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(u64, uint64_t)

// bf16/fp16 use int16_t in VAIP_ORT_API
inline vaip_core::TensorProto*
tensor_proto_new_bf16(const std::string& name,
                      const std::vector<int64_t>& shape,
                      const std::vector<int16_t>& data) {
  return VAIP_ORT_API(tensor_proto_new_bf16)(name, shape, data);
}

inline vaip_core::TensorProto*
tensor_proto_new_fp16(const std::string& name,
                      const std::vector<int64_t>& shape,
                      const std::vector<int16_t>& data) {
  return VAIP_ORT_API(tensor_proto_new_fp16)(name, shape, data);
}

// Special cases: VAIP_ORT_API uses "floats"/"doubles", not "f32"/"f64"
inline vaip_core::TensorProto*
tensor_proto_new_f32(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<float>& data) {
  return VAIP_ORT_API(tensor_proto_new_floats)(name, shape, data);
}

inline vaip_core::TensorProto*
tensor_proto_new_f64(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<double>& data) {
  return VAIP_ORT_API(tensor_proto_new_doubles)(name, shape, data);
}

// Additional types used in REVERT_MEM_TAG_DEF
#if VAIP_ORT_API_MAJOR >= 13
DEFINE_TENSOR_PROTO_NEW_WRAPPER(i4, int8_t)
DEFINE_TENSOR_PROTO_NEW_WRAPPER(u4, uint8_t)
#endif

#if VAIP_ORT_API_MAJOR >= 19
DEFINE_TENSOR_PROTO_NEW_WRAPPER(bool, uint8_t)
#endif

#undef DEFINE_TENSOR_PROTO_NEW_WRAPPER
} // namespace

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
  VAIP_ORT_API(graph_set_name)(graph, name.c_str());
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

  // Remove nodes in reverse order to handle dependencies correctly
  for (auto it = all_nodes.rbegin(); it != all_nodes.rend(); ++it) {
    auto n = *it;
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

// graph_replace_node_arg has been moved to vaip-core
// It depends on NodeBuilder which is now in vaip-core

const Node* graph_producer_node(const Graph& graph,
                                const std::string& node_arg_name) {
  return VAIP_ORT_API(graph_producer_node)(graph, node_arg_name);
}

const NodeArg* graph_get_node_arg(const Graph& graph, const std::string& name) {
  return VAIP_ORT_API(graph_get_node_arg)(graph, name);
}

const std::string& graph_get_name(const Graph& graph) {
  return VAIP_ORT_API(graph_get_name)(graph);
}

void graph_reverse_dfs_from(
    const Graph& graph, size_t node_index,
    const std::function<bool(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp,
    bool subgraph_sensitive) {
  auto node = VAIP_ORT_API(graph_get_node)(graph, node_index);
  std::vector<const Node*> nodes = {node};
  auto stop = [](const Node*, const Node*) { return false; };
  // Wrap leave callback to convert void return to bool
  auto leave_wrapper = [&leave](const Node* n) -> bool {
    leave(n);
    return false; // Return value doesn't matter for leave
  };
  // Use graph_reverse_dfs_from_preemp which has bool returns for enter/leave
  VAIP_ORT_API(graph_reverse_dfs_from_preemp)
  (graph, gsl::make_span(nodes), enter, leave_wrapper, comp, stop);
}

void graph_reverse_dfs_from_multi(
    const Graph& graph, gsl::span<const Node* const> from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& stop) {
  // Wrap callbacks to match VAIP_ORT_API signature (which expects bool returns)
  auto enter_wrapper =
      enter ? std::function<bool(const Node*)>([&enter](const Node* n) -> bool {
        enter(n);
        return false; // Continue traversal
      })
            : std::function<bool(const Node*)>(nullptr);

  auto leave_wrapper =
      leave ? std::function<bool(const Node*)>([&leave](const Node* n) -> bool {
        leave(n);
        return false; // Continue traversal
      })
            : std::function<bool(const Node*)>(nullptr);

  VAIP_ORT_API(graph_reverse_dfs_from_preemp)
  (graph, from, enter_wrapper, leave_wrapper, nullptr, stop);
}

void graph_fuse(Graph& graph, const std::string& name,
                const std::string& op_type,
                const std::vector<const Node*>& nodes,
                const std::vector<std::string>& inputs,
                const std::vector<std::string>& outputs,
                const std::vector<std::string>& constant_initializers) {
  // Convert node pointers to node indices
  std::vector<size_t> node_indices;
  node_indices.reserve(nodes.size());
  for (const auto* node : nodes) {
    node_indices.push_back(VAIP_ORT_API(node_get_index)(*node));
  }
  VAIP_ORT_API(graph_fuse)
  (graph, name, op_type, node_indices, inputs, outputs, constant_initializers);
}

Node& graph_fuse(Graph& graph, const std::string& name,
                 const std::string& op_type, const std::vector<size_t>& nodes,
                 const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs,
                 const std::vector<std::string>& constant_initializers) {
  return VAIP_ORT_API(graph_fuse)(graph, name, op_type, nodes, inputs, outputs,
                                  constant_initializers);
}

// Model operations
Graph& model_main_graph(Model& model) {
  return VAIP_ORT_API(model_main_graph)(model);
}

const std::string& model_get_meta_data(const Model& model,
                                       const std::string& key) {
  return *VAIP_ORT_API(model_get_meta_data)(model, key);
}

bool model_has_meta_data(const Model& model, const std::string& key) {
  return VAIP_ORT_API(model_has_meta_data)(model, key);
}

Model* model_clone(const Model& model) {
  // Use max int64_t as threshold to keep all data inline (no external file)
  return VAIP_ORT_API(model_clone)(model, std::numeric_limits<int64_t>::max());
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

vaip_core::DllSafe<std::string> GraphConstRef::save_string() const {
#if VAIP_ORT_API_MAJOR >= 18
  return VAIP_ORT_API(graph_save_string)(*this);
#else
  LOG(FATAL) << "graph_save_string is not supported in this version of ORT";
#endif
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

// try_fuse and virtual_fuse have been moved to vaip-core
// They depend on MetaDefProto and TryFuseError which are vaip-core types
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

void GraphConstRef::reverse_dfs_from(
    size_t node_index, const std::function<bool(const vaip_core::Node*)>& enter,
    const std::function<void(const vaip_core::Node*)>& leave,
    const std::function<bool(const vaip_core::Node*, const vaip_core::Node*)>&
        comp,
    bool subgraph_sensitive) const {
  vaip_core::graph_reverse_dfs_from(*this, node_index, enter, leave, comp,
                                    subgraph_sensitive);
}

GraphRef::GraphRef(vaip_core::Graph& graph) : GraphConstRef(graph) {}

GraphRef::~GraphRef() {}
void GraphRef::set_name(const std::string& name) {
  vaip_core::graph_set_name(*this, name);
}
bool GraphRef::resolve(bool force) {
  return VAIP_ORT_API(graph_resolve)(*this, force) == 0;
}
// GraphRef::fuse and GraphRef::node_builder have been moved to vaip-core
// They depend on MetaDefProto and NodeBuilder which are vaip-core types
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
    auto tensor = tensor_proto_new_##type(name, shape, values);                \
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
    auto tensor = tensor_proto_new_##type(name, shape, values);                \
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
NodeArgConstRef
GraphRef::new_node_arg(const std::string& name,
                       ONNX_NAMESPACE::TensorProto_DataType data_type) {
  return NodeArgConstRef::from_node_arg(
      self(), VAIP_ORT_API(node_arg_new)(*this, name, nullptr, data_type));
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
void GraphRef::mut_save(const std::filesystem::path& file_path,
                        const std::filesystem::path& external_data_file,
                        size_t threshold, bool filter_out_special_tensor) {
  if (filter_out_special_tensor) {
    prune_special_tensor_proto();
  }
  if (name().empty()) {
    set_name("no_name");
  }
  save(file_path, external_data_file, threshold);
}

vaip_core::DllSafe<std::string>
GraphRef::mut_save_string(bool filter_out_special_tensor) {
  if (filter_out_special_tensor) {
    prune_special_tensor_proto();
  }
  if (name().empty()) {
    set_name("no_name");
  }
  return save_string();
}

#if VAIP_ORT_API_MAJOR >= 7
static GraphConstRef get_original_graph(const std::string& ptr) {
  uintptr_t ptr1 = std::stoull(ptr);
  return *reinterpret_cast<vaip_core::Graph*>(ptr1);
}
#endif
void revert_mem_tag_tensor(GraphRef target_graph, const std::string& name,
                           const int element_type,
                           const std::unique_ptr<std::vector<int64_t>>& shape,
                           size_t ptr, size_t size) {
  auto base = reinterpret_cast<void*>(ptr);
  std::vector<int64_t> empty = {};
#define REVERT_MEM_TAG_DEF(type, cxx_type, tensor_data_type)                   \
  case ONNX_NAMESPACE::TensorProto_DataType_##tensor_data_type: {              \
    auto beg = reinterpret_cast<cxx_type*>(base);                              \
    auto end = beg + (size / sizeof(cxx_type));                                \
    auto values = std::vector<cxx_type>(beg, end);                             \
    auto tensor =                                                              \
        tensor_proto_new_##type(name, shape.get() ? *shape : empty, values);   \
    VAIP_ORT_API(graph_add_initialized_tensor)(target_graph, *tensor);         \
    break;                                                                     \
  }
  switch (element_type) {
    REVERT_MEM_TAG_DEF(i8, int8_t, INT8)
    REVERT_MEM_TAG_DEF(u8, uint8_t, UINT8)
    REVERT_MEM_TAG_DEF(i16, int16_t, INT16)
    REVERT_MEM_TAG_DEF(u16, uint16_t, UINT16)
    REVERT_MEM_TAG_DEF(i32, int32_t, INT32)
    REVERT_MEM_TAG_DEF(u32, uint32_t, UINT32)
    REVERT_MEM_TAG_DEF(i64, int64_t, INT64)
    REVERT_MEM_TAG_DEF(u64, uint64_t, UINT64)
#if VAIP_ORT_API_MAJOR >= 13
    REVERT_MEM_TAG_DEF(i4, int8_t, INT4)
    REVERT_MEM_TAG_DEF(u4, uint8_t, UINT4)
#endif
#if VAIP_ORT_API_MAJOR >= 19
    REVERT_MEM_TAG_DEF(bool, uint8_t, BOOL)
#endif
    REVERT_MEM_TAG_DEF(f32, float, FLOAT)
    REVERT_MEM_TAG_DEF(f64, double, DOUBLE)
    REVERT_MEM_TAG_DEF(bf16, bf16_t, BFLOAT16)
    REVERT_MEM_TAG_DEF(fp16, fp16_t, FLOAT16)
  default:
    LOG(FATAL) << "unknown type " << element_type << " name=" << name;
  }
}
static void prune_special_tensor_proto_for_graph(GraphRef target_graph,
                                                 GraphConstRef original_graph,
                                                 NodeArgConstRef node_arg) {
  bool is_first_call = target_graph == original_graph;
  std::string location = "";
  location.reserve(1024);
  size_t size = 0;
  size_t offset = 0;
  size_t checksum = 0;
  int external_data = VAIP_ORT_API(node_arg_external_location)(
      original_graph, node_arg, location, offset, size, checksum);
  CHECK_LE(location.size(), 1024)
      << "External data location is too long: " << location.size();
  if (external_data && !location.empty() && (location.front() == '<')) {
    auto new_original_graph = get_original_graph(location.substr(1));
    prune_special_tensor_proto_for_graph(target_graph, new_original_graph,
                                         node_arg);
  } else if (!is_first_call) {
    auto& original_tensor = vaip_core::node_arg_get_const_data_as_tensor(
        original_graph, node_arg /*hopefully, only node arg name is used.*/);
    if (external_data && !location.empty() && (location.front() == '*')) {
      VAIP_ORT_API(graph_remove_initialized_tensor)
      (target_graph, node_arg.name());
      revert_mem_tag_tensor(target_graph, node_arg.name(),
                            node_arg.element_type(), node_arg.shape(), offset,
                            size);
    } else {
      VAIP_ORT_API(graph_remove_initialized_tensor)
      (target_graph, node_arg.name());
      VAIP_ORT_API(graph_add_initialized_tensor)
      (target_graph, original_tensor);
    }
  }
}
void GraphRef::prune_special_tensor_proto() {
#if VAIP_ORT_API_MAJOR >= 7

  auto all = constant_initializers();
  for (auto node_arg : all) {
    prune_special_tensor_proto_for_graph(*this, *this, node_arg);
  }
#endif
}
} // namespace vaip_cxx
