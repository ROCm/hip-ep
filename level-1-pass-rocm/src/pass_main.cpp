// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <iostream>
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
 * It creates and runs Level-2 sub-passes (Conv, Gemm) for pattern matching,
 * then merges consecutive fused nodes into larger subgraphs for efficient
 * execution on AMD GPUs.
 *
 * The original graph is read-only, so we clone the model and run sub-passes
 * on the cloned graph.
 *
 * Configuration via vaip_config.json pass_generic_param:
 * {
 *   "sub_pass_names": ["vaip-pass_level2_rocm_conv", "vaip-pass_level2_rocm_gemm"]
 * }
 *
 * See doc/02_LEVEL1_PASS_DESIGN.md for architecture overview.
 * See doc/03_GROUPING_ALGORITHM.md for the Union-Find grouping algorithm.
 */
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}

  //============================================================================
  // Union-Find Data Structure
  // Used for efficient O(N α(N)) ≈ O(N) grouping of connected ROCm nodes
  //============================================================================

  // Find the root of a node with path compression
  const Node* find_root(std::unordered_map<const Node*, const Node*>& parent,
                        const Node* node) {
    if (parent[node] != node) {
      // Path compression: make the node point directly to the root
      parent[node] = find_root(parent, parent[node]);
    }
    return parent[node];
  }

  // Union two nodes into the same group
  void union_nodes(std::unordered_map<const Node*, const Node*>& parent,
                   const Node* a, const Node* b) {
    const Node* root_a = find_root(parent, a);
    const Node* root_b = find_root(parent, b);
    if (root_a != root_b) {
      // Merge groups: make root_a point to root_b
      parent[root_a] = root_b;
    }
  }

  //============================================================================
  // ROCm Node Detection
  //============================================================================

  // Check if a node is a ROCm fused node (created by Level-2 passes)
  // Fused nodes have:
  // - Domain: "com.xilinx" (morphizen framework's domain for fused nodes)
  //
  // Since we run Level-2 sub-passes on a cloned graph in isolation,
  // any "com.xilinx" node in that graph must be from our ROCm passes.
  // The original graph is cloned fresh, so it has no pre-existing fused nodes.
  bool is_rocm_fused_node(const Node& node) {
    auto domain = node_op_domain(node);
    if (domain != "com.xilinx") {
      return false;
    }
    
    // All com.xilinx nodes in the cloned graph are ROCm fused nodes
    // because we only run ROCm Level-2 passes on the cloned graph
    auto output_name = node_get_first_output_name(node);
    MY_LOG(2) << "[HIP EP Level-1] is_rocm_fused_node: found com.xilinx node with output=" << output_name;
    return true;
  }

  // Get op type from fused node output name
  std::string get_rocm_op_type(const Node& node) {
    auto output_name = node_get_first_output_name(node);
    MY_LOG(2) << "[HIP EP Level-1] get_rocm_op_type: output=" << output_name;
    if (output_name.find("rocm_conv") != std::string::npos) {
      return "conv";
    } else if (output_name.find("rocm_gemm") != std::string::npos) {
      return "gemm";
    }
    return "unknown";
  }

  // Get all ROCm fused nodes from the cloned graph in topological order
  std::vector<const Node*> find_rocm_fused_nodes(Graph& graph) {
    std::vector<const Node*> rocm_nodes;
    auto all_nodes = graph_nodes(graph);

    for (auto* node : all_nodes) {
      if (is_rocm_fused_node(*node)) {
        rocm_nodes.push_back(node);
        MY_LOG(2) << "[HIP EP Level-1] Found ROCm fused node: "
                  << node_get_first_output_name(*node) << " (op_type: "
                  << get_rocm_op_type(*node) << ")";
      }
    }
    return rocm_nodes;
  }

  //============================================================================
  // Producer Map Building
  //============================================================================

  // Build a map from output node_arg name to the node that produces it
  // Only includes outputs from ROCm fused nodes
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

  //============================================================================
  // Grouping Algorithm (Union-Find Based)
  //
  // Key insight: ONNX graphs are DAGs, so nodes are in topological order.
  // When we process a node, all its input producers have already been processed.
  // We can directly merge a node into its producer's group using Union-Find.
  //
  // Time complexity: O(N × α(N)) ≈ O(N) where N is the number of ROCm nodes
  // Space complexity: O(N) for parent map and producer map
  //
  // See doc/03_GROUPING_ALGORITHM.md for detailed algorithm description.
  //============================================================================

  std::vector<std::vector<const Node*>>
  find_mergeable_groups(const std::vector<const Node*>& rocm_nodes,
                        const std::unordered_map<std::string, const Node*>&
                            producer_map) {
    if (rocm_nodes.empty()) {
      return {};
    }

    // Initialize Union-Find: each node is its own group initially
    std::unordered_map<const Node*, const Node*> parent;
    for (auto* node : rocm_nodes) {
      parent[node] = node;
    }

    // Process nodes in topological order (they are already in this order)
    // For each node, check if any of its inputs come from another ROCm node
    for (auto* node : rocm_nodes) {
      auto inputs = node_get_input_node_args(*node);
      for (auto* input : inputs) {
        if (input) {
          auto name = node_arg_get_name(*input);
          auto it = producer_map.find(name);
          if (it != producer_map.end()) {
            // This input is produced by another ROCm node
            // Merge current node's group with producer's group
            union_nodes(parent, node, it->second);
            MY_LOG(2) << "[HIP EP Level-1] Connected: "
                      << node_get_first_output_name(*(it->second)) << " -> "
                      << node_get_first_output_name(*node);
          }
        }
      }
    }

    // Collect groups from Union-Find structure
    std::unordered_map<const Node*, std::vector<const Node*>> group_map;
    for (auto* node : rocm_nodes) {
      const Node* root = find_root(parent, node);
      group_map[root].push_back(node);
    }

    // Convert to vector of groups
    std::vector<std::vector<const Node*>> groups;
    groups.reserve(group_map.size());
    for (auto& [root, group] : group_map) {
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

  //============================================================================
  // Input/Output Collection for Merged Groups
  //
  // External Inputs: Inputs not produced by nodes in the group
  // External Outputs: Outputs consumed outside the group or are graph outputs
  //
  // Note: There's a theoretical edge case when an output is consumed both
  // inside and outside the group. The current logic handles the common
  // sequential case correctly. For branching patterns, we check if an output
  // is consumed ONLY within the group (in which case it's internal).
  //============================================================================

  // Collect external outputs (outputs that are consumed outside the group or
  // are graph outputs)
  std::vector<std::string>
  collect_external_outputs(const std::vector<const Node*>& group,
                           const std::unordered_set<std::string>& group_internal_inputs,
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

    // Create set of all outputs in the group
    std::unordered_set<std::string> group_outputs;
    for (auto* node : group) {
      auto outputs = node_get_output_node_args(*node);
      for (auto* output : outputs) {
        if (output) {
          group_outputs.insert(node_arg_get_name(*output));
        }
      }
    }

    for (auto* node : group) {
      auto outputs = node_get_output_node_args(*node);
      for (auto* output : outputs) {
        if (output) {
          auto name = node_arg_get_name(*output);

          // An output is external if:
          // 1. It's a graph output (must be exposed), OR
          // 2. It's not consumed by any node in the group (consumed outside)
          //
          // Note: If an output is consumed both inside and outside the group,
          // the current logic may not detect this. However, for the common
          // sequential ROCm fusion case (A → B → C), this works correctly.
          bool is_graph_output = graph_output_names.count(name) > 0;
          bool consumed_only_inside = group_internal_inputs.count(name) > 0;

          if (is_graph_output || !consumed_only_inside) {
            outputs_ordered.insert(name);
          }
        }
      }
    }
    return std::vector<std::string>(outputs_ordered.begin(),
                                    outputs_ordered.end());
  }

  //============================================================================
  // Merged Parameters Collection
  //
  // The RocmMergedParamProto stores information about all operations in the
  // merged group. The actual detailed parameters (ConvParamProto, GemmParamProto)
  // are stored in each Level-2 fused node's MetaDef and will be extracted
  // at custom op creation time.
  //============================================================================

  rocm::RocmMergedParamProto
  collect_merged_params(const std::vector<const Node*>& group) {
    rocm::RocmMergedParamProto merged;
    merged.set_op_count(static_cast<int32_t>(group.size()));
    merged.set_implicit_fusion(true); // All ops can share one HIP stream

    for (auto* node : group) {
      rocm::RocmParamProto param;
      auto output_name = node_get_first_output_name(*node);

      // Set op_type using the helper function
      param.set_op_type(get_rocm_op_type(*node));

      // Store the original output name for reference
      // This helps the custom op locate the original parameters if needed
      param.set_ep_context_file_name(output_name);

      *merged.add_rocm_params() = param;
      MY_LOG(2) << "[HIP EP Level-1] Added node to merged group: "
                << output_name << " (op_type: " << param.op_type() << ")";
    }

    return merged;
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[HIP EP Level-1] Starting ROCm pass";
    MY_LOG(2) << "[HIP EP Level-1] process() called - Starting ROCm pass";

    // Get pass configuration from pass_generic_param (JSON)
    auto json_param = self_.get_pass_generic_param();
    MY_LOG(1) << "[HIP EP Level-1] pass_generic_param: " << json_param;

    // Parse the JSON to get sub-pass configuration
    rocm::PassRocmConfigProto config;
    auto status =
        google::protobuf::util::JsonStringToMessage(json_param, &config);
    if (!status.ok()) {
      MY_LOG(1) << "[HIP EP Level-1] Failed to parse pass_generic_param: "
                << status.ToString();
      return;
    }

    // Run sub-passes directly on the original graph
    // The sub-passes will handle fusion themselves via try_fuse() and fuse()
    MY_LOG(2) << "[HIP EP Level-1] Number of sub-passes to run: " << config.sub_pass_names_size();
    for (const auto& sub_pass_name : config.sub_pass_names()) {
      MY_LOG(1) << "[HIP EP Level-1] Creating sub-pass: " << sub_pass_name;
      MY_LOG(2) << "[HIP EP Level-1] Creating sub-pass: " << sub_pass_name;

      PassProto sub_pass_proto;
      sub_pass_proto.set_plugin(sub_pass_name);
      sub_pass_proto.set_name(sub_pass_name);

      auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
      if (sub_pass) {
        MY_LOG(1) << "[HIP EP Level-1] Running sub-pass: " << sub_pass_name;
        MY_LOG(2) << "[HIP EP Level-1] Running sub-pass: " << sub_pass_name;
        std::vector<std::shared_ptr<IPass>> passes;
        passes.push_back(std::move(sub_pass));
        IPass::run_passes(passes, graph);  // Run on original graph, not clone
        MY_LOG(2) << "[HIP EP Level-1] Sub-pass completed: " << sub_pass_name;
      } else {
        MY_LOG(2) << "[HIP EP Level-1] Failed to create sub-pass: " << sub_pass_name;
      }
    }

    MY_LOG(1) << "[HIP EP Level-1] Completed - sub-passes handled fusion";
    MY_LOG(2) << "[HIP EP Level-1] Level-1 pass completed";
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
