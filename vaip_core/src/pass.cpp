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
#include "vaip/graph.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <unordered_set>
#include <utility>
namespace vaip_core {

void IPass::copy_fix_info(const std::string& from, const std::string& to) {
  copy_fix_info(from.c_str(), to.c_str());
}

void IPass::copy_fix_info(const Node& from_node, const Node& to_node) {
  auto from_name = node_get_output_name(from_node);
  auto to_name = node_get_output_name(to_node);
  copy_fix_info(from_name, to_name);
}

void IPass::copy_fix_info(const char* from, const char* to) {
  if (has_fix_info(from)) {
    auto fix_info = get_fix_info(from);
    set_fix_info(to, fix_info);
  } else {
    LOG(FATAL) << "cannot find fix info: "
               << "from " << from << " " //
               << "to " << to << " "     //
        ;
  }
}

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

static bool node_arg_is_graph_input(const Graph& graph,
                                    const std::string& node_arg_name) {
  auto graph_inputs = graph_get_inputs(graph);
  bool ret = std::any_of(graph_inputs.begin(), graph_inputs.end(),
                         [&node_arg_name](const NodeArg* node_arg) -> bool {
                           return node_arg_get_name(*node_arg) == node_arg_name;
                         });
  return ret;
}
static bool node_arg_is_initializer(const Graph& graph,
                                    const std::string& node_arg_name) {
  auto all_initializer = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph);
  return all_initializer.count(node_arg_name) > 0;
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
static std::pair<std::vector<const Node*>, std::string>
node_arg_names_to_nodes(const Graph& graph,
                        const std::vector<std::string>& node_arg_names,
                        bool allow_node_not_found) {
  std::stringstream ss;
  auto ret = std::vector<const Node*>();
  ret.reserve(node_arg_names.size());
  for (auto& onnx_node_arg_name : node_arg_names) {
    auto deq = VAIP_ORT_API(graph_producer_node)(graph, onnx_node_arg_name);
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
      if (node_arg_is_node_output) {
        ss << onnx_node_arg_name << ",";
      }
      if (!allow_node_not_found) {
        LOG(FATAL) << "cannot find producer. onnx_node_arg_name="
                   << onnx_node_arg_name;
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

static std::unordered_set<const NodeArg*>
node_args_names_to_node_arg(const Graph& graph,
                            const std::vector<const NodeArg*>& graph_inputs,
                            const std::vector<std::string>& input_names) {
  std::unordered_set<const NodeArg*> ret;
  for (const auto& input_name : input_names) {
    auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, input_name);
    bool intersection = false;
    for (auto graph_input : graph_inputs) {
      intersection = intersection || graph_input == node_arg;
    }
    if (intersection) {
      ret.insert(node_arg);
    }
  }
  return ret;
}

static std::vector<std::string>
calculate_return_values(const Graph& graph, const Node& output_node,
                        const std::vector<const Node*>& body_nodes) {
  auto ret = std::vector<std::string>();
  auto args = node_get_output_node_args(output_node);
  auto graph_outputs = graph_get_outputs(graph);
  for (auto arg : args) {
    auto& arg_name = node_arg_get_name(*arg);
    auto consumers = graph_get_consumer_nodes(graph, arg_name);
    auto num_of_external_out_edges = 0;
    auto is_graph_output = std::find(graph_outputs.begin(), graph_outputs.end(),
                                     arg) != graph_outputs.end();
    if (is_graph_output) {
      num_of_external_out_edges = num_of_external_out_edges + 1;
    }
    for (auto c : consumers) {
      auto found = std::find(body_nodes.begin(), body_nodes.end(), c) !=
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
calculate_arguments(const Graph& graph, const Node& input_node,
                    const std::vector<const Node*>& body_nodes,
                    const std::set<std::string>& initializers) {
  auto ret = std::vector<std::string>();
  auto args = node_get_input_node_args(input_node);
  auto graph_inputs = graph_get_inputs(graph);
  for (auto arg : args) {
    if (!node_arg_exists(*arg)) {
      // testcase : hrnet_w18_small, optional node input
      continue;
    }
    auto& arg_name = node_arg_get_name(*arg);
    auto producer = VAIP_ORT_API(graph_producer_node)(graph, arg_name);
    auto num_of_external_in_edges = 0;
    auto is_graph_input = std::find(graph_inputs.begin(), graph_inputs.end(),
                                    arg) != graph_inputs.end();
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
calculate_return_values(const Graph& graph,
                        const std::vector<const Node*>& body_nodes) {
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
calculate_arguments(const Graph& graph,
                    const std::vector<const Node*>& body_nodes,
                    const std::set<std::string>& initializers) {
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
check_loop(const Graph& graph, const std::vector<const Node*>& input_nodes,
           const std::vector<const Node*>& output_nodes) {
  auto ret = false;
  std::vector<std::string> maybe_loop_path;
  // key : any node in graph
  // value : path from one of `input_nodes` to the `key`
  std::unordered_map<const Node*, std::vector<const Node*>> map_route;
  for (auto input_node : input_nodes) {
    map_route[input_node] = {input_node};
  }
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      // we start traval from inputs_nodes, and if we found any one of
      // input nodes depends on one of output nodes topolocially, then
      // a loop is detected.
      graph, input_nodes,
      [&output_nodes, &ret, &map_route,
       &maybe_loop_path](const Node* current_node) mutable {
        auto hit_output = std::find(output_nodes.begin(), output_nodes.end(),
                                    current_node) != output_nodes.end();
        if (hit_output) {
          ret = true;
          auto route = map_route[current_node];
          for (size_t i = 0; i < route.size(); i++) {
            auto node = route[i];
            maybe_loop_path.push_back(node_get_first_output_name(*node));
          }
        }
      },
      nullptr,
      [&ret, &map_route](const Node* from, const Node* to) -> bool {
        // There may exist multiple paths and we only find one of them
        // ,because we only care if has connection
        CHECK(map_route.find(from) != map_route.end())
            << "path not exists:" << node_as_string(*from);
        if (map_route.find(to) == map_route.end()) {
          auto route = map_route[from];
          route.push_back(to);
          map_route[to] = route;
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
get_combined_inputs(const std::vector<std::string>& inputs,
                    const std::vector<std::string>& return_values) {
  std::vector<std::string> ret;

  std::map<int, std::string> idx_input;
  std::unordered_map<std::string, int> input_idx;

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

static std::vector<std::string> get_edge_node_arg_names(const Node* from,
                                                        const Node* to) {
  auto ret = std::vector<std::string>();
  auto from_input_args = node_get_input_node_args(*from);
  auto to_output_args = node_get_output_node_args(*to);
  for (auto& arg : to_output_args) {
    if (std::find(from_input_args.begin(), from_input_args.end(), arg) !=
        from_input_args.end()) {
      ret.push_back(node_arg_get_name(*arg));
    }
  }
  CHECK(!ret.empty()) << "[try fuse failed] not exist a edge between "
                      << node_as_string(*from) << " and "
                      << node_as_string(*to);
  return ret;
}

static bool is_subset(const std::vector<std::string>& subset,
                      const std::vector<std::string>& superset) {
  std::unordered_set<std::string> s(superset.begin(), superset.end());
  for (const auto& item : subset) {
    if (s.find(item) == s.end()) {
      return false;
    }
  }
  return true;
}
static std::vector<const Node*> graph_get_isolated_nodes(const Graph& graph) {
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
  return all_nodes;
}

VAIP_DLL_SPEC
std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
IPass::try_fuse(const Graph& graph, const std::string& name,
                const std::vector<std::string>& inputs,
                const std::vector<std::string>& outputs,
                const std::vector<std::string>& constant_initializers1,
                const std::string& device) const {
  return IPass_try_fuse(graph, name, inputs, outputs, constant_initializers1,
                        device);
}

std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
IPass_try_fuse(const Graph& graph, const std::string& name,
               const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs,
               const std::vector<std::string>& constant_initializers1,
               const std::string& device) {
  auto constant_initializers = std::set<std::string>(
      constant_initializers1.begin(), constant_initializers1.end());
  auto bodies = std::vector<std::string>();
  auto body_nodes = std::vector<const Node*>();
  // The input can be the graph input as well, see issue 1043 for model and
  // pattern
  auto [input_nodes, find_input_nodes_msg] =
      node_arg_names_to_nodes(graph, inputs, true /* allow node not found*/);
  auto [output_nodes, find_output_nodes_msg] =
      node_arg_names_to_nodes(graph, outputs, false /* node must be found */);
  auto graph_inputs = graph_get_inputs(graph);
  auto node_inputs = node_args_names_to_node_arg(graph, graph_inputs, inputs);
  auto trasverse_out_of_bound = [&graph_inputs,
                                 &node_inputs](const NodeArg* node_arg) {
    auto iter = std::find(graph_inputs.begin(), graph_inputs.end(), node_arg);
    if (iter == graph_inputs.end()) {
      return false;
    }
    bool not_node_input = node_inputs.find(*iter) == node_inputs.end();
    return not_node_input;
  };
  auto hit_ceiling = false;
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      graph, output_nodes,
      [&body_nodes, &graph, &constant_initializers, &hit_ceiling,
       &trasverse_out_of_bound](const Node* node1) {
        if (!hit_ceiling) {
          CHECK(node1 != nullptr);
          body_nodes.push_back(node1);
          auto node_args = node_get_input_node_args(*node1);
          for (auto node_arg : node_args) {
            CHECK(node_arg != nullptr);
            // add node_arg_is_exists
            // test case 18,  Resize_496, The second input to resize is
            // optional
            hit_ceiling = hit_ceiling || trasverse_out_of_bound(node_arg);
            if (VAIP_ORT_API(node_arg_is_exists)(*node_arg) &&
                VAIP_ORT_API(node_arg_is_constant)(graph, *node_arg)) {
              constant_initializers.insert(node_arg_get_name(*node_arg));
            }
          }
        }
      },
      nullptr,
      [&inputs](const Node* from, const Node* to) -> bool {
        // input_nodes.contains(to);
        auto edge_node_arg_names = get_edge_node_arg_names(from, to);
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
  for (auto island : isolated_nodes) {
    // island node's all input node in body_nodes => is_body
    auto is_body = true;
    auto node_inputs = node_get_inputs(*island);
    for (auto& input : node_inputs) {
      if (input.node != nullptr &&
          std::find(body_nodes.begin(), body_nodes.end(), input.node) ==
              body_nodes.end()) {
        is_body = false;
        continue;
      }
    }
    if (is_body) {
      body_nodes.push_back(island);
      // insert island's initalizers input args
      for (auto input : node_inputs) {
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
  for (auto& input : combined_inputs) {
    meta_def->add_inputs(input);
  }
  for (auto& output : return_values) {
    meta_def->add_outputs(output);
  }
  for (auto& constant_initializer : constant_initializers) {
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
} // namespace vaip_core
