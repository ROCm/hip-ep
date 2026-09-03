/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/pass.hpp"
#include "morphizen/graph.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <unordered_set>
#include <utility>
namespace morphizen {

IPass::action_t PassInfo::get_action(size_t index) const {
  CHECK_LT(index, this->size);
  auto ret = IPass::action_t();
  auto type = this->processes[index].type;
  switch (type) {
  case 0:
    ret = this->processes[index].proc.process_graph;
    break;
  case 1:
    ret = create_action_from_node_action(
        this->processes[index].proc.process_node);
    break;
  default:
    LOG(FATAL) << "unknown type:" << type;
    ;
  }
  return ret;
}

static bool node_arg_is_graph_input(const Graph &graph,
                                    const std::string &node_arg_name) {
  auto graph_inputs = morphizen_cxx::GraphConstRef(graph).inputs();
  bool ret = std::any_of(graph_inputs.begin(), graph_inputs.end(),
                         [&node_arg_name](const auto &node_arg) -> bool {
                           return node_arg.name() == node_arg_name;
                         });
  return ret;
}
static bool node_arg_is_initializer(const Graph &graph,
                                    const std::string &node_arg_name) {
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  auto all_initializer = graph_ref.constant_initializers();
  return std::any_of(all_initializer.begin(), all_initializer.end(),
                     [&node_arg_name](const auto &init) {
                       return init.name() == node_arg_name;
                     });
}

// return values are Node found by node_arg name and the node_arg names
// that cannot find the node through node_arg
// Cannot find the node are three scenarios where the node cannot be found using
// the node_arg_name.
// 1) node_arg is graph_input  --  This is normal
// 2) node_arg is initizlizer  -- This is normal
// 3) node_arg's produce node has been fuse to other subgraph. And this is
// within the subgraph's body, not the output.  -- This node is contended by two
// subgraphs, and we must relinquish the fusion of the second subgraph.
static std::pair<std::vector<const Node *>, std::string>
node_arg_names_to_nodes(const Graph &graph,
                        const std::vector<std::string> &node_arg_names,
                        bool allow_node_not_found) {
  std::stringstream ss;
  auto ret = std::vector<const Node *>();
  ret.reserve(node_arg_names.size());
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  for (auto &onnx_node_arg_name : node_arg_names) {
    auto node_arg_opt = graph_ref.find_node_arg(onnx_node_arg_name);
    const Node *deq = nullptr;
    if (node_arg_opt.has_value()) {
      auto producer_opt = node_arg_opt.value().find_producer();
      if (producer_opt.has_value()) {
        deq = producer_opt.value().ptr();
      }
    }
    // NOTE: todo: potentially deq could be nullptr if node_arg is a
    // constant initializer or graph inputs.
    if (deq == nullptr) {
      // If nodearg does not exist, maybe the node is fused and an exception
      // should be thrown. The reason why `graph_get_node_arg` is not used
      // is because ort does not maintain the consistency of nodearg well,
      // this nodearg not exist but graph_get_node_arg won't return nullptr.
      // testcase:#1304
      bool node_arg_is_node_output =
          (!node_arg_is_graph_input(graph, onnx_node_arg_name)) &&
          (!node_arg_is_initializer(graph, onnx_node_arg_name));
      // A producerless graph-input or constant-initializer output is legal (ORT
      // const folding can leave a constant as a graph output). Only a genuine
      // node-output with no producer is a fusion error, so record and FATAL
      // only in that case; otherwise the fuse is given up gracefully.
      if (node_arg_is_node_output) {
        ss << onnx_node_arg_name << ",";
        if (!allow_node_not_found) {
          LOG(FATAL) << "cannot find producer. onnx_node_arg_name="
                     << onnx_node_arg_name;
        }
      }
    } else {
      auto found = std::find(ret.begin(), ret.end(), deq) != ret.end();
      // to support multiple outputs, the node might already be inserted.
      if (!found) {
        ret.push_back(deq);
      }
    }
  }
  return std::make_pair(ret, std::string(ss.str()));
}

static std::vector<morphizen_cxx::NodeArgConstRef> node_args_names_to_node_arg(
    const Graph &graph,
    const std::vector<morphizen_cxx::NodeArgConstRef> &graph_inputs,
    const std::vector<std::string> &input_names) {
  std::vector<morphizen_cxx::NodeArgConstRef> ret;
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  for (const auto &input_name : input_names) {
    auto node_arg_opt = graph_ref.find_node_arg(input_name);
    if (!node_arg_opt.has_value()) {
      continue;
    }
    auto &node_arg_ref = node_arg_opt.value();
    // Check if this node arg is in graph_inputs
    bool intersection = false;
    for (const auto &graph_input : graph_inputs) {
      intersection = intersection || graph_input.ptr() == node_arg_ref.ptr();
    }
    if (intersection) {
      ret.push_back(node_arg_ref);
    }
  }
  return ret;
}

static std::vector<std::string>
calculate_return_values(const Graph &graph, const Node &output_node,
                        const std::vector<const Node *> &body_nodes) {
  auto ret = std::vector<std::string>();
  auto args = node_get_output_node_args(output_node);
  auto graph_outputs = morphizen_cxx::GraphConstRef(graph).outputs();
  for (auto arg : args) {
    if (arg == nullptr) {
      // for optional outputs
      continue;
    }
    auto &arg_name = node_arg_get_name(*arg);
    auto consumers =
        morphizen_cxx::GraphConstRef(graph).find_consumers(arg_name);
    auto num_of_external_out_edges = 0;
    // Check if this arg is a graph output
    auto is_graph_output =
        std::any_of(graph_outputs.begin(), graph_outputs.end(),
                    [arg](const auto &out) { return out.ptr() == arg; });
    if (is_graph_output) {
      num_of_external_out_edges = num_of_external_out_edges + 1;
    }
    // Check if consumers are outside body_nodes
    for (auto &c : consumers) {
      auto found = std::find(body_nodes.begin(), body_nodes.end(), c.ptr()) !=
                   body_nodes.end();
      if (!found) {
        num_of_external_out_edges = num_of_external_out_edges + 1;
      }
    }
    if (num_of_external_out_edges != 0) {
      ret.push_back(arg_name);
    }
  }
  return ret;
}

static std::vector<std::string>
calculate_arguments(const Graph &graph, const Node &input_node,
                    const std::vector<const Node *> &body_nodes,
                    const std::set<std::string> &initializers) {
  auto ret = std::vector<std::string>();
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  auto graph_inputs = graph_ref.inputs();

  // node.all_inputs() unions explicit operands and implicit captures so
  // the fused-subgraph MetaDef::inputs sees boundary tensors captured by
  // Loop / If / Scan bodies.
  auto input_node_ref =
      morphizen_cxx::NodeConstRef::from_node(graph, input_node);
  for (auto &opt_arg : input_node_ref.all_inputs()) {
    if (!opt_arg.has_value()) {
      // testcase : hrnet_w18_small, optional node input
      continue;
    }
    const auto *arg = opt_arg.value().ptr();
    auto &arg_name = node_arg_get_name(*arg);
    auto node_arg_opt = graph_ref.find_node_arg(arg_name);
    const Node *producer = nullptr;
    if (node_arg_opt.has_value()) {
      auto producer_opt = node_arg_opt.value().find_producer();
      if (producer_opt.has_value()) {
        producer = producer_opt.value().ptr();
      }
    }
    auto num_of_external_in_edges = 0;
    auto is_graph_input =
        std::any_of(graph_inputs.begin(), graph_inputs.end(),
                    [arg](const auto &inp) { return inp.ptr() == arg; });
    if (is_graph_input) {
      num_of_external_in_edges = num_of_external_in_edges + 1;
    }
    auto found = std::find(body_nodes.begin(), body_nodes.end(), producer) !=
                 body_nodes.end();
    if (!found) {
      num_of_external_in_edges = num_of_external_in_edges + 1;
    }
    auto is_initializer = std::find(initializers.begin(), initializers.end(),
                                    arg_name) != initializers.end();
    if (num_of_external_in_edges != 0 && !is_initializer) {
      ret.push_back(arg_name);
    }
  }
  return ret;
}

static std::vector<std::string>
calculate_return_values(const Graph &graph,
                        const std::vector<const Node *> &body_nodes) {
  auto ret = std::vector<std::string>();
  ret.reserve(body_nodes.size());
  for (auto i = 0u; i < body_nodes.size(); ++i) {
    CHECK(body_nodes[i] != nullptr);
    auto r = calculate_return_values(graph, *body_nodes[i], body_nodes);
    std::copy(r.begin(), r.end(), std::back_inserter(ret));
  }
  return ret;
}

static std::vector<std::string>
calculate_arguments(const Graph &graph,
                    const std::vector<const Node *> &body_nodes,
                    const std::set<std::string> &initializers) {
  auto ret = std::vector<std::string>();
  ret.reserve(body_nodes.size());
  auto arguments =
      std::unordered_set<std::string>(); // does not guarantee any order
  arguments.reserve(body_nodes.size());
  for (auto i = 0u; i < body_nodes.size(); ++i) {
    CHECK(body_nodes[i] != nullptr);
    auto calc_res =
        calculate_arguments(graph, *body_nodes[i], body_nodes, initializers);
    for (auto r : calc_res) {
      if (arguments.find(r) == arguments.end()) {
        arguments.insert(r);
        ret.push_back(r);
      }
    }
  }
  return ret;
}

static std::vector<std::string>
check_loop(const Graph &graph, const std::vector<const Node *> &input_nodes,
           const std::vector<const Node *> &output_nodes) {
  auto ret = false;
  std::vector<std::string> maybe_loop_path;
  // key : node pointer
  // value : path from one of `input_nodes` to the `key`
  std::unordered_map<const Node *, std::vector<const Node *>> map_route;
  for (auto input_node : input_nodes) {
    map_route[input_node] = {input_node};
  }

  // Convert raw pointers to NodeConstRef
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  auto input_nodes_cxx = std::vector<morphizen_cxx::NodeConstRef>();
  input_nodes_cxx.reserve(input_nodes.size());
  for (auto *n : input_nodes) {
    input_nodes_cxx.push_back(
        morphizen_cxx::NodeConstRef::from_node(graph, *n));
  }

  graph_ref.reverse_dfs_from_multi(
      // we start traval from inputs_nodes, and if we found any one of
      // input nodes depends on one of output nodes topolocially, then
      // a loop is detected.
      gsl::make_span(input_nodes_cxx),
      nullptr, // enter
      [&output_nodes, &ret, &map_route, &graph,
       &maybe_loop_path](morphizen_cxx::NodeConstRef current_node) mutable {
        auto hit_output = std::find(output_nodes.begin(), output_nodes.end(),
                                    current_node.ptr()) != output_nodes.end();
        if (hit_output) {
          ret = true;
          auto route = map_route[current_node.ptr()];
          for (size_t i = 0; i < route.size(); i++) {
            auto node = route[i];
            auto node_ref =
                morphizen_cxx::NodeConstRef::from_node(graph, *node);
            maybe_loop_path.push_back(
                morphizen::node_arg_get_name(node_ref.first_output_node_arg()));
          }
        }
        return false; // leave callback return value
      },
      nullptr, // comp
      [&ret, &map_route](morphizen_cxx::NodeConstRef from,
                         morphizen_cxx::NodeConstRef to) -> bool {
        // There may exist multiple paths and we only find one of them
        // ,because we only care if has connection
        CHECK(map_route.find(from.ptr()) != map_route.end())
            << "path not exists:" << from.to_string();
        if (map_route.find(to.ptr()) == map_route.end()) {
          auto route = map_route[from.ptr()];
          route.push_back(to.ptr());
          map_route[to.ptr()] = route;
        }
        // break if loop is detected. return true,
        // graph_reverse_dfs_from should terminate travel.
        return ret;
      });
  return maybe_loop_path;
}

// prefer the order of try_fuse argument instead of topological order if
// possible
static const std::vector<std::string>
get_combined_inputs(const std::vector<std::string> &inputs,
                    const std::vector<std::string> &return_values) {
  std::vector<std::string> ret;

  std::map<size_t, std::string> idx_input;
  std::unordered_map<std::string, size_t> input_idx;

  for (size_t i = 0; i < return_values.size(); ++i) {
    idx_input.insert({i, return_values[i]});
    input_idx.insert({return_values[i], i});
  }
  for (auto i : inputs) {
    auto iter = input_idx.find(i);
    if (iter != input_idx.end()) {
      ret.push_back(i);
      idx_input.erase(idx_input.find(iter->second));
    }
  }

  for (auto it = idx_input.begin(); it != idx_input.end(); ++it) {
    ret.push_back(it->second);
  }
  return ret;
}

static std::vector<std::string> get_edge_node_arg_names(const Node *from,
                                                        const Node *to) {
  auto ret = std::vector<std::string>();
  auto from_input_args = node_get_input_node_args(*from);
  auto to_output_args = node_get_output_node_args(*to);
  for (auto &arg : to_output_args) {
    // Skip nullptr or non-existent args from optional ONNX slots.
    // Without the nullptr guard, std::find matches nullptr==nullptr
    // between from_input_args and to_output_args, creating a false edge.
    if (arg == nullptr || !node_arg_exists(*arg)) {
      continue;
    }
    if (std::find(from_input_args.begin(), from_input_args.end(), arg) !=
        from_input_args.end()) {
      ret.push_back(node_arg_get_name(*arg));
    }
  }
  CHECK(!ret.empty()) << "[try fuse failed] not exist a edge between "
                      << morphizen::node_as_string(*from) << " and "
                      << morphizen::node_as_string(*to);
  return ret;
}

static bool is_subset(const std::vector<std::string> &subset,
                      const std::vector<std::string> &superset) {
  std::unordered_set<std::string> s(superset.begin(), superset.end());
  for (const auto &item : subset) {
    if (s.find(item) == s.end()) {
      return false;
    }
  }
  return true;
}
static std::vector<morphizen_cxx::NodeConstRef>
graph_get_isolated_nodes(const Graph &graph) {
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  auto all_nodes = graph_ref.nodes();
  auto graph_outputs = graph_ref.outputs();

  // Find leaf nodes (nodes that produce graph outputs)
  std::vector<const Node *> leaf_nodes_raw;
  leaf_nodes_raw.reserve(graph_outputs.size());
  for (auto &n : all_nodes) {
    auto node_outputs = node_get_output_node_args(*n.ptr());
    auto found = std::any_of(
        node_outputs.begin(), node_outputs.end(),
        [&graph_outputs](const NodeArg *x) {
          return std::any_of(graph_outputs.begin(), graph_outputs.end(),
                             [x](const auto &out) { return out.ptr() == x; });
        });
    if (found) {
      leaf_nodes_raw.push_back(n.ptr());
    }
  }

  // Convert leaf_nodes to NodeConstRef for DFS
  std::vector<morphizen_cxx::NodeConstRef> leaf_nodes_cxx;
  leaf_nodes_cxx.reserve(leaf_nodes_raw.size());
  for (auto *n : leaf_nodes_raw) {
    leaf_nodes_cxx.push_back(morphizen_cxx::NodeConstRef::from_node(graph, *n));
  }

  // Remove reachable nodes via DFS
  std::vector<morphizen_cxx::NodeConstRef> result =
      all_nodes; // Start with all nodes
  morphizen_cxx::GraphConstRef(graph).reverse_dfs_from_multi(
      gsl::make_span(leaf_nodes_cxx), //
      nullptr,                        //
      [&result](morphizen_cxx::NodeConstRef n) mutable {
        result.erase(
            std::remove_if(result.begin(), result.end(),
                           [&n](const morphizen_cxx::NodeConstRef &node) {
                             return node.index() == n.index();
                           }),
            result.end());
        return false; // leave callback return value
      },              //
      nullptr,        // comp
      nullptr);       // stop

  return result;
}

MORPHIZEN_DLL_SPEC
std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
IPass::try_fuse(const Graph &graph, const std::string &name,
                const std::vector<std::string> &inputs,
                const std::vector<std::string> &outputs,
                const std::vector<std::string> &constant_initializers1,
                const std::string &device) const {
  return IPass_try_fuse(graph, name, inputs, outputs, constant_initializers1,
                        device);
}

std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
IPass_try_fuse(const Graph &graph, const std::string &name,
               const std::vector<std::string> &inputs,
               const std::vector<std::string> &outputs,
               const std::vector<std::string> &constant_initializers1,
               const std::string &device) {
  auto constant_initializers = std::set<std::string>(
      constant_initializers1.begin(), constant_initializers1.end());
  auto bodies = std::vector<std::string>();
  auto body_nodes = std::vector<const Node *>();
  // The input can be the graph input as well, see issue 1043 for model and
  // pattern
  auto [input_nodes, find_input_nodes_msg] =
      node_arg_names_to_nodes(graph, inputs, true /* allow node not found*/);
  auto [output_nodes, find_output_nodes_msg] =
      node_arg_names_to_nodes(graph, outputs, false /* node must be found */);
  auto graph_inputs = morphizen_cxx::GraphConstRef(graph).inputs();
  auto node_inputs = node_args_names_to_node_arg(graph, graph_inputs, inputs);
  auto trasverse_out_of_bound = [&graph_inputs,
                                 &node_inputs](const NodeArg *node_arg) {
    // Check if node_arg is in graph_inputs
    auto iter = std::find_if(
        graph_inputs.begin(), graph_inputs.end(),
        [node_arg](const auto &inp) { return inp.ptr() == node_arg; });
    if (iter == graph_inputs.end()) {
      return false;
    }
    // Check if it's NOT in node_inputs
    bool not_node_input = std::none_of(
        node_inputs.begin(), node_inputs.end(),
        [node_arg](const auto &inp) { return inp.ptr() == node_arg; });
    return not_node_input;
  };
  // Convert output_nodes to NodeConstRef for DFS
  auto output_nodes_cxx = std::vector<morphizen_cxx::NodeConstRef>();
  output_nodes_cxx.reserve(output_nodes.size());
  for (auto *n : output_nodes) {
    output_nodes_cxx.push_back(
        morphizen_cxx::NodeConstRef::from_node(graph, *n));
  }
  auto hit_ceiling = false;
  morphizen_cxx::GraphConstRef(graph).reverse_dfs_from_multi(
      gsl::make_span(output_nodes_cxx),
      nullptr, // enter
      [&body_nodes, &graph, &constant_initializers, &hit_ceiling,
       &trasverse_out_of_bound](morphizen_cxx::NodeConstRef node1) {
        if (!hit_ceiling) {
          body_nodes.push_back(node1.ptr());
          // all_inputs() = explicit operands ++ implicit Loop/If/Scan captures,
          // so captures join both the self-containment (hit_ceiling) check and
          // the constant-initializer collection below.
          for (auto &opt_arg : node1.all_inputs()) {
            if (!opt_arg.has_value()) {
              // node_arg no value mean optionsl argument.
              continue;
            }
            auto node_arg_ref = opt_arg.value();
            // add node_arg_is_exists
            // test case 18,  Resize_496, The second input to resize is
            // optional
            hit_ceiling =
                hit_ceiling || trasverse_out_of_bound(node_arg_ref.ptr());
            if (node_arg_exists(*node_arg_ref.ptr())) {
              if (node_arg_ref.is_constant()) {
                constant_initializers.insert(node_arg_ref.name());
              }
            }
          }
        }
        return false; // leave callback return value
      },
      nullptr, // comp
      [&inputs](morphizen_cxx::NodeConstRef from,
                morphizen_cxx::NodeConstRef to) -> bool {
        // input_nodes.contains(to);
        auto edge_node_arg_names =
            get_edge_node_arg_names(from.ptr(), to.ptr());
        // The condition for stopping the traversal is the edges all included
        // inputs.
        return is_subset(edge_node_arg_names, inputs);
      });
  if (hit_ceiling) {
    /* If the node's outputs traverse upward all the way to the graph_input
     * instead of stop at node's inputs, then the outputs depends on more
     * than the inputs passed in. Therefore, the fuse should fail as it is
     * not self-contained. See issue 740.
     */
    std::string error_comment =
        "hit ceiling [" + find_input_nodes_msg + find_output_nodes_msg + "]";
    return std::make_pair(nullptr, TryFuseError{error_comment, {}, {}, {}, {}});
  }

  // after upgrade onnxruntime 1.18 , onnx.onnx has some DequantizeLinear
  // isolated ops. we need remove island ops from return_valus and add to
  // body_nodes.
  // TODO : now only support one layer of isolated node
  auto isolated_nodes = graph_get_isolated_nodes(graph);
  for (auto &island : isolated_nodes) {
    // island node's all input node in body_nodes => is_body
    auto is_body = true;
    auto node_inputs_1 = node_get_inputs(*island.ptr());
    for (auto &input : node_inputs_1) {
      if (input.node != nullptr &&
          std::find(body_nodes.begin(), body_nodes.end(), input.node) ==
              body_nodes.end()) {
        is_body = false;
        continue;
      }
    }
    if (is_body) {
      body_nodes.push_back(island.ptr());
      // insert island's initalizers input args
      for (auto input : node_inputs_1) {
        if (input.node == nullptr) {
          constant_initializers.insert(node_arg_get_name(*input.node_arg));
        }
      }
    }
  }

  auto return_values = calculate_return_values(graph, body_nodes);
  auto arguments =
      calculate_arguments(graph, body_nodes, constant_initializers);

  auto [return_output_nodes, find_return_values_msg] = node_arg_names_to_nodes(
      graph, return_values, false /* node must be found */);
  auto maybe_loop_path = check_loop(graph, input_nodes, return_output_nodes);
  if (!maybe_loop_path.empty()) {
    return std::make_pair(
        nullptr,
        TryFuseError{std::string("loop detected"), maybe_loop_path, body_nodes,
                     inputs, return_values}); // argument = input so far
  }

  // After excluding graph input and initializer type node_arg, if there is
  // still a situation where the Node cannot be found through node_arg_name, it
  // means that some nodes have been fused to the middle position of other
  // subgraphs (not outputs). This situation is regarded as two subgraphs
  // competing for the same node, and the current fuse needs to be given up.
  if (!find_input_nodes_msg.empty() || !find_output_nodes_msg.empty() ||
      !find_return_values_msg.empty()) {
    std::string error_comment = "can't find producer_node of [" +
                                find_input_nodes_msg + find_output_nodes_msg +
                                find_return_values_msg + "]";
    return std::make_pair(nullptr,
                          TryFuseError{error_comment, maybe_loop_path,
                                       body_nodes, inputs, return_values});
  }
  // return  meta def
  auto meta_def = std::make_unique<MetaDefProto>();
  meta_def->set_id(name);
  auto combined_inputs = get_combined_inputs(inputs, arguments);
  for (auto &input : combined_inputs) {
    meta_def->add_inputs(input);
  }
  for (auto &output : return_values) {
    meta_def->add_outputs(output);
  }
  for (auto &constant_initializer : constant_initializers) {
    meta_def->add_constant_initializers(constant_initializer);
  }
  for (auto node : body_nodes) {
    meta_def->add_nodes(node_get_first_output_name(*node));
  }
  meta_def->set_device(device);
  return std::make_pair(
      std::move(meta_def),
      TryFuseError{std::string("try fuse OK"), {}, body_nodes, {}, {}});
}
} // namespace morphizen
