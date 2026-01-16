// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "rocm.pb.h"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace vaip_core;

/**
 * Level-1 Pass: ROCm Orchestrator
 *
 * This pass serves as the entry point for ROCm-based operations.
 * It creates and runs Level-2 sub-passes (Conv, Gemm) for pattern matching.
 *
 * The original graph is read-only, so we clone the model and run sub-passes
 * on the cloned graph.
 *
 * Configuration via vaip_config.json pass_generic_param:
 * {
 *   "sub_pass_names": ["vaip-pass_level2_rocm_conv", "vaip-pass_level2_rocm_gemm"]
 * }
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  // Check if a node is a ROCm fused node (created by Level-2 passes)
  bool is_rocm_fused_node(const Node& node) {
    // Fused nodes use the "com.xilinx" domain with EP name in the op type
    auto domain = node_op_domain(node);
    auto op_type = node_op_type(node);
    return domain == "com.xilinx" &&
           (op_type.find("rocm_conv") != std::string::npos ||
            op_type.find("rocm_gemm") != std::string::npos);
  }

  // Get all ROCm fused nodes from the cloned graph in topological order
  std::vector<const Node*> find_rocm_fused_nodes(Graph& graph) {
    std::vector<const Node*> rocm_nodes;
    auto all_nodes = graph_nodes(graph);

    for (auto* node : all_nodes) {
      if (is_rocm_fused_node(*node)) {
        rocm_nodes.push_back(node);
        MY_LOG(2) << "[ROCm EP Level-1] Found ROCm fused node: "
                  << node_op_type(*node);
      }
    }
    return rocm_nodes;
  }

  // Build a map from output node_arg name to the node that produces it
  std::unordered_map<std::string, const Node*>
  build_producer_map(const std::vector<const Node*>& nodes) {
    std::unordered_map<std::string, const Node*> producer_map;
    for (auto* node : nodes) {
      auto outputs = node_get_output_node_args(*node);
      for (auto* output : outputs) {
        if (output) {
          producer_map[node_arg_get_name(*output)] = node;
        }
      }
    }
    return producer_map;
  }

  // Check if two nodes are connected (node_a produces input for node_b)
  bool are_connected(const Node& node_a, const Node& node_b,
                     const std::unordered_map<std::string, const Node*>&
                         producer_map) {
    auto inputs = node_get_input_node_args(node_b);
    for (auto* input : inputs) {
      if (input) {
        auto name = node_arg_get_name(*input);
        auto it = producer_map.find(name);
        if (it != producer_map.end() && it->second == &node_a) {
          return true;
        }
      }
    }
    return false;
  }

  // Find groups of consecutive ROCm nodes that can be merged
  std::vector<std::vector<const Node*>>
  find_mergeable_groups(const std::vector<const Node*>& rocm_nodes,
                        const std::unordered_map<std::string, const Node*>&
                            producer_map) {
    if (rocm_nodes.empty()) {
      return {};
    }

    // Create a set for fast lookup
    std::unordered_set<const Node*> rocm_node_set(rocm_nodes.begin(),
                                                  rocm_nodes.end());

    // Track which nodes have been assigned to a group
    std::unordered_set<const Node*> assigned;
    std::vector<std::vector<const Node*>> groups;

    // For each node, try to extend or create a group
    for (auto* node : rocm_nodes) {
      if (assigned.count(node) > 0) {
        continue;
      }

      // Start a new group with this node
      std::vector<const Node*> group;
      group.push_back(node);
      assigned.insert(node);

      // Try to extend the group by finding connected ROCm nodes
      bool extended = true;
      while (extended) {
        extended = false;
        for (auto* candidate : rocm_nodes) {
          if (assigned.count(candidate) > 0) {
            continue;
          }

          // Check if candidate is connected to any node in the group
          for (auto* group_node : group) {
            if (are_connected(*group_node, *candidate, producer_map) ||
                are_connected(*candidate, *group_node, producer_map)) {
              group.push_back(candidate);
              assigned.insert(candidate);
              extended = true;
              break;
            }
          }
          if (extended)
            break;
        }
      }

      groups.push_back(std::move(group));
    }

    return groups;
  }

  // Collect external inputs (inputs that come from outside the group)
  std::vector<std::string>
  collect_external_inputs(const std::vector<const Node*>& group,
                          const std::unordered_set<std::string>& internal_outputs) {
    std::set<std::string> inputs_ordered; // Use set for ordering
    for (auto* node : group) {
      auto inputs = node_get_input_node_args(*node);
      for (auto* input : inputs) {
        if (input) {
          auto name = node_arg_get_name(*input);
          // Only include if it's not produced by another node in the group
          if (internal_outputs.count(name) == 0) {
            inputs_ordered.insert(name);
          }
        }
      }
    }
    return std::vector<std::string>(inputs_ordered.begin(),
                                    inputs_ordered.end());
  }

  // Collect external outputs (outputs that are consumed outside the group or
  // are graph outputs)
  std::vector<std::string>
  collect_external_outputs(const std::vector<const Node*>& group,
                           const std::unordered_set<std::string>& internal_inputs,
                           Graph& graph) {
    std::set<std::string> outputs_ordered;

    // Get graph output names
    std::unordered_set<std::string> graph_output_names;
    auto graph_outputs = graph_get_outputs(graph);
    for (auto* output : graph_outputs) {
      if (output) {
        graph_output_names.insert(node_arg_get_name(*output));
      }
    }

    for (auto* node : group) {
      auto outputs = node_get_output_node_args(*node);
      for (auto* output : outputs) {
        if (output) {
          auto name = node_arg_get_name(*output);
          // Include if it's not consumed only by nodes in the group
          // or if it's a graph output
          if (internal_inputs.count(name) == 0 ||
              graph_output_names.count(name) > 0) {
            outputs_ordered.insert(name);
          }
        }
      }
    }
    return std::vector<std::string>(outputs_ordered.begin(),
                                    outputs_ordered.end());
  }

  // Collect node info for merged params
  // Note: The actual rocm_param will be retrieved at custom op creation time
  // from the MetaDef's generic_param. Here we just track node info.
  rocm::RocmMergedParamProto
  collect_merged_params(const std::vector<const Node*>& group) {
    rocm::RocmMergedParamProto merged;
    merged.set_op_count(static_cast<int32_t>(group.size()));
    merged.set_implicit_fusion(true);  // All ops can share one HIP stream
    
    for (auto* node : group) {
      rocm::RocmParamProto param;
      auto op_type = node_op_type(*node);
      
      // Set op_type based on the fused node name
      if (op_type.find("conv") != std::string::npos) {
        param.set_op_type("conv");
      } else if (op_type.find("gemm") != std::string::npos) {
        param.set_op_type("gemm");
      } else {
        param.set_op_type(op_type);
      }
      
      *merged.add_rocm_params() = param;
      MY_LOG(2) << "[ROCm EP Level-1] Added node to merged group: " << op_type;
    }

    return merged;
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[ROCm EP Level-1] Starting ROCm pass";

    // Get pass configuration from pass_generic_param (JSON)
    auto json_param = self_.get_pass_generic_param();
    MY_LOG(1) << "[ROCm EP Level-1] pass_generic_param: " << json_param;

    // Parse the JSON to get sub-pass configuration
    rocm::PassRocmConfigProto config;
    auto status =
        google::protobuf::util::JsonStringToMessage(json_param, &config);
    if (!status.ok()) {
      MY_LOG(1) << "[ROCm EP Level-1] Failed to parse pass_generic_param: "
                << status.ToString();
      return;
    }

    // Clone the model so we can modify the graph
    // The original graph is read-only
    auto& model = VAIP_ORT_API(graph_get_model)(graph);
    auto cloned_model = vaip_core::model_clone(model, 64);
    auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);

    MY_LOG(1) << "[ROCm EP Level-1] Cloned model for sub-pass processing";

    // Create PassProto for each sub-pass and run them on cloned graph
    for (const auto& sub_pass_name : config.sub_pass_names()) {
      MY_LOG(1) << "[ROCm EP Level-1] Creating sub-pass: " << sub_pass_name;

      PassProto sub_pass_proto;
      sub_pass_proto.set_plugin(sub_pass_name);
      sub_pass_proto.set_name(sub_pass_name);

      auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
      if (sub_pass) {
        MY_LOG(1) << "[ROCm EP Level-1] Running sub-pass: " << sub_pass_name;
        std::vector<std::shared_ptr<IPass>> passes;
        passes.push_back(std::move(sub_pass));
        IPass::run_passes(passes, cloned_graph);
      }
    }

    // Step 1: Find all ROCm fused nodes in the cloned graph
    auto rocm_nodes = find_rocm_fused_nodes(cloned_graph);
    MY_LOG(1) << "[ROCm EP Level-1] Found " << rocm_nodes.size()
              << " ROCm fused nodes";

    if (rocm_nodes.empty()) {
      MY_LOG(1) << "[ROCm EP Level-1] No ROCm nodes found, nothing to merge";
      return;
    }

    // Step 2: Build producer map and find mergeable groups
    auto producer_map = build_producer_map(rocm_nodes);
    auto groups = find_mergeable_groups(rocm_nodes, producer_map);
    MY_LOG(1) << "[ROCm EP Level-1] Found " << groups.size()
              << " mergeable groups";

    // Step 3: For each group, create a merged fused node in the original graph
    int group_idx = 0;
    for (const auto& group : groups) {
      MY_LOG(1) << "[ROCm EP Level-1] Processing group " << group_idx
                << " with " << group.size() << " nodes";

      // Collect internal outputs (produced by nodes in the group)
      std::unordered_set<std::string> internal_outputs;
      for (auto* node : group) {
        auto outputs = node_get_output_node_args(*node);
        for (auto* output : outputs) {
          if (output) {
            internal_outputs.insert(node_arg_get_name(*output));
          }
        }
      }

      // Collect internal inputs (consumed by nodes in the group)
      std::unordered_set<std::string> internal_inputs;
      for (auto* node : group) {
        auto inputs = node_get_input_node_args(*node);
        for (auto* input : inputs) {
          if (input) {
            internal_inputs.insert(node_arg_get_name(*input));
          }
        }
      }

      // Get external inputs and outputs
      auto external_inputs = collect_external_inputs(group, internal_outputs);
      auto external_outputs =
          collect_external_outputs(group, internal_inputs, cloned_graph);

      MY_LOG(2) << "[ROCm EP Level-1] Group " << group_idx << " has "
                << external_inputs.size() << " external inputs and "
                << external_outputs.size() << " external outputs";

      // Collect merged parameters from all nodes in the group
      auto merged_params = collect_merged_params(group);

      // Create a unique name for the merged node
      std::string merged_name = "rocm_merged_" + std::to_string(group_idx);

      // Try to fuse in the original graph
      auto [meta_def, error] =
          self_.try_fuse(graph, merged_name, external_inputs, external_outputs,
                         {}, "ROCm_EP");

      if (meta_def) {
        // Serialize the merged params and attach to meta_def
        std::string merged_params_str;
        merged_params.SerializeToString(&merged_params_str);
        self_.attach_meta_def_param(*meta_def, merged_params_str.c_str());

        // Create the fused node
        self_.fuse(graph, std::move(*meta_def));
        MY_LOG(1) << "[ROCm EP Level-1] Created merged fused node: "
                  << merged_name;
      } else {
        MY_LOG(1) << "[ROCm EP Level-1] Failed to fuse group " << group_idx
                  << ": " << error.comments;
      }

      group_idx++;
    }

    MY_LOG(1) << "[ROCm EP Level-1] Completed";
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
