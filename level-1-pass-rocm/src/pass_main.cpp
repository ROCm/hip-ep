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
#include "morphizen/morphizen.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_attr.hpp"
#include "rocm.pb.h"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
DEF_ENV_PARAM(MORPHIZEN_ROCM_NO_MERGE, "0")  // Set to 1 to create one subgraph per node
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace morphizen;

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

  // Check if a node is a ROCm fused node (created by Level-2 ROCm passes)
  // Detection criteria:
  // - Domain: "com.xilinx" (morphizen framework's domain for fused nodes)
  // - Has attribute: "rocm_param_file" (set by Level-2 ROCm passes)
  //
  // Level-2 passes add a node attribute using NodeAttributesBuilder::merge_into()
  // after calling level_2_fuse(). Level-1 can detect this using NodeConstRef.has_attr().
  bool is_rocm_fused_node(Graph& graph, const Node& node) {
    // Use C++ style NodeConstRef API (cheap - just two pointer copies)
    auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, node);
    
    auto domain = node_ref.op_domain();
    auto op_type = node_ref.op_type();
    auto output_name = node_get_first_output_name(node);
    
    MY_LOG(2) << "[ROCm EP Level-1] Checking node: output=" << output_name
              << ", domain=" << domain << ", op_type=" << op_type;
    
    // Fused nodes created by level_2_fuse() have:
    // - op_type = "call" (morphizen's internal fused op type)
    // - rocm_param_file attribute set by Level-2 passes
    // Note: domain may be empty for level_2_fuse() created nodes
    if (op_type != "call") {
      return false;
    }
    
    // Check for ROCm-specific attribute set by Level-2 passes
    if (!node_ref.has_attr("rocm_param_file")) {
      MY_LOG(2) << "[ROCm EP Level-1] Node " << output_name 
                << " is call but missing rocm_param_file attr";
      return false;
    }
    
    MY_LOG(1) << "[ROCm EP Level-1] is_rocm_fused_node: found ROCm node with output=" 
              << output_name;
    return true;
  }

  // Get op type from fused node output name
  std::string get_rocm_op_type(const Node& node) {
    auto output_name = node_get_first_output_name(node);
    MY_LOG(2) << "[ROCm EP Level-1] get_rocm_op_type: output=" << output_name;
    if (output_name.find("rocm_conv") != std::string::npos) {
      return "conv";
    } else if (output_name.find("rocm_gemm") != std::string::npos) {
      return "gemm";
    } else if (output_name.find("rocm_matmul") != std::string::npos) {
      return "matmul";
    } else if (output_name.find("rocm_mul") != std::string::npos) {
      return "mul";
    } else if (output_name.find("rocm_softmax") != std::string::npos) {
      return "softmax";
    } else if (output_name.find("rocm_reshape") != std::string::npos) {
      return "reshape";
    } else if (output_name.find("rocm_transpose") != std::string::npos) {
      return "transpose";
    } else if (output_name.find("rocm_tile") != std::string::npos) {
      return "tile";
    }
    return "unknown";
  }

  // Get all ROCm fused nodes from the graph in topological order
  std::vector<const Node*> find_rocm_fused_nodes(Graph& graph) {
    std::vector<const Node*> rocm_nodes;
    auto all_nodes = graph_nodes(graph);

    for (auto* node : all_nodes) {
      if (is_rocm_fused_node(graph, *node)) {
        rocm_nodes.push_back(node);
        MY_LOG(2) << "[ROCm EP Level-1] Found ROCm fused node: "
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

    // If NO_MERGE is set, each node becomes its own group (one subgraph per node)
    if (ENV_PARAM(MORPHIZEN_ROCM_NO_MERGE)) {
      MY_LOG(1) << "[ROCm EP Level-1] NO_MERGE mode: creating one subgraph per node";
      std::vector<std::vector<const Node*>> groups;
      groups.reserve(rocm_nodes.size());
      for (auto* node : rocm_nodes) {
        groups.push_back({node});
      }
      return groups;
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
            MY_LOG(2) << "[ROCm EP Level-1] Connected: "
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
  // IMPORTANT: Preserve the original input order from the ONNX graph!
  // Using std::set would sort alphabetically which breaks the mapping between
  // ONNX Runtime's input indices and our external_input_buffers_.
  std::vector<std::string>
  collect_external_inputs(const std::vector<const Node*>& group,
                          const std::unordered_set<std::string>& internal_outputs) {
    std::vector<std::string> inputs_ordered;
    std::unordered_set<std::string> seen;  // For deduplication
    
    for (auto* node : group) {
      auto inputs = node_get_input_node_args(*node);
      for (auto* input : inputs) {
        if (input) {
          auto name = node_arg_get_name(*input);
          // Only include if it's not produced by another node in the group
          // and we haven't seen it yet (deduplication while preserving order)
          if (internal_outputs.count(name) == 0 && seen.count(name) == 0) {
            inputs_ordered.push_back(name);
            seen.insert(name);
          }
        }
      }
    }
    return inputs_ordered;
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
  // Build RocmSubgraphProto
  //
  // Creates a RocmSubgraphProto that represents the complete topology of the
  // fused subgraph. This includes:
  // - RocmNodeProto for each node with parameters and input references
  // - TensorRefProto for internal (node-to-node) and external (from ORT) inputs
  // - ExternalOutputProto for outputs that go to ORT
  //============================================================================

  rocm::RocmSubgraphProto
  build_subgraph_proto(const std::vector<const Node*>& group,
                       const std::unordered_set<std::string>& internal_outputs,
                       const std::vector<std::string>& external_inputs,
                       const std::vector<std::string>& external_outputs,
                       Graph& graph) {
    rocm::RocmSubgraphProto subgraph;
    auto pass_context = self_.get_context();
    
    // Pre-compute unique external inputs (field 1 in proto)
    // This eliminates runtime deduplication overhead in custom op
    for (const auto& input_name : external_inputs) {
      subgraph.add_external_inputs(input_name);
      MY_LOG(2) << "[ROCm EP Level-1] Added external_input: " << input_name;
    }
    
    // Map from output name to (node_id, output_index) for internal references
    std::unordered_map<std::string, std::pair<int32_t, int32_t>> output_producer_map;
    
    // Build nodes in topological order
    for (int32_t i = 0; i < static_cast<int32_t>(group.size()); ++i) {
      const Node* node = group[i];
      auto output_name = node_get_first_output_name(*node);
      auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, *node);
      
      rocm::RocmNodeProto* node_proto = subgraph.add_nodes();
      node_proto->set_node_id(i);
      
      // Read params from cache file and set as node params
      std::string param_filename;
      if (node_ref.has_attr("rocm_param_file")) {
        param_filename = node_ref.get_attr_string("rocm_param_file");
      }
      
      if (!param_filename.empty()) {
        auto reader = pass_context->open_file_for_read(param_filename);
        if (reader) {
          size_t file_size = reader->size();
          std::string param_json(file_size, '\0');
          reader->fread(&param_json[0], file_size);
          
          rocm::RocmParamProto rocm_param;
          auto parse_status = google::protobuf::util::JsonStringToMessage(param_json, &rocm_param);
          if (parse_status.ok()) {
            rocm_param.set_ep_context_file_name(param_filename);
            *node_proto->mutable_params() = rocm_param;
            MY_LOG(2) << "[ROCm EP Level-1] Node " << i << ": loaded params from " << param_filename;
          } else {
            LOG(WARNING) << "[ROCm EP Level-1] Failed to parse param file: " << param_filename;
          }
        }
      }
      
      // Build input references
      auto inputs = node_get_input_node_args(*node);
      for (auto* input : inputs) {
        if (!input) continue;
        auto input_name = node_arg_get_name(*input);
        
        rocm::TensorRefProto* input_ref = node_proto->add_inputs();
        
        auto it = output_producer_map.find(input_name);
        if (it != output_producer_map.end()) {
          // Internal reference - from another node in subgraph
          auto* internal_ref = input_ref->mutable_internal();
          internal_ref->set_producer_node_id(it->second.first);
          internal_ref->set_output_index(it->second.second);
          MY_LOG(2) << "[ROCm EP Level-1] Node " << i << ": input " << input_name 
                    << " -> internal(node=" << it->second.first 
                    << ", out=" << it->second.second << ")";
        } else {
          // External reference - from outside subgraph
          input_ref->set_external_name(input_name);
          MY_LOG(2) << "[ROCm EP Level-1] Node " << i << ": input " << input_name 
                    << " -> external";
        }
      }
      
      // Register outputs for dependency tracking
      auto outputs = node_get_output_node_args(*node);
      for (int32_t j = 0; j < static_cast<int32_t>(outputs.size()); ++j) {
        if (outputs[j]) {
          auto out_name = node_arg_get_name(*outputs[j]);
          output_producer_map[out_name] = {i, j};
          node_proto->add_output_names(out_name);
        }
      }
    }
    
    // Add external outputs with their source node mappings
    for (const auto& ext_out_name : external_outputs) {
      auto it = output_producer_map.find(ext_out_name);
      if (it != output_producer_map.end()) {
        rocm::ExternalOutputProto* ext_output = subgraph.add_outputs();
        ext_output->set_name(ext_out_name);
        ext_output->set_producer_node_id(it->second.first);
        ext_output->set_output_index(it->second.second);
        MY_LOG(2) << "[ROCm EP Level-1] External output: " << ext_out_name 
                  << " from node " << it->second.first;
      }
    }
    
    return subgraph;
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[ROCm EP Level-1] Starting ROCm pass";
    MY_LOG(2) << "[ROCm EP Level-1] process() called - Starting ROCm pass";

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

    // Step 1: Run sub-passes on the graph
    // Level-2 passes use level_2_fuse() which creates fused nodes but doesn't update context.json
    MY_LOG(2) << "[ROCm EP Level-1] Number of sub-passes to run: " << config.sub_pass_names_size();
    for (const auto& sub_pass_name : config.sub_pass_names()) {
      MY_LOG(1) << "[ROCm EP Level-1] Running sub-pass: " << sub_pass_name;

      PassProto sub_pass_proto;
      sub_pass_proto.set_plugin(sub_pass_name);
      sub_pass_proto.set_name(sub_pass_name);

      auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
      if (sub_pass) {
        std::vector<std::shared_ptr<IPass>> passes;
        passes.push_back(std::move(sub_pass));
        IPass::run_passes(passes, graph);
        MY_LOG(1) << "[ROCm EP Level-1] Sub-pass completed: " << sub_pass_name;
      } else {
        LOG(WARNING) << "[ROCm EP Level-1] Failed to create sub-pass: " << sub_pass_name;
      }
    }

    // Step 2: Find all ROCm fused nodes created by Level-2 passes
    auto rocm_nodes = find_rocm_fused_nodes(graph);
    MY_LOG(1) << "[ROCm EP Level-1] Found " << rocm_nodes.size() << " ROCm fused nodes";
    
    if (rocm_nodes.empty()) {
      MY_LOG(1) << "[ROCm EP Level-1] No ROCm nodes to merge, pass completed";
      return;
    }

    // Step 3: Build producer map and find mergeable groups
    auto producer_map = build_producer_map(rocm_nodes);
    auto groups = find_mergeable_groups(rocm_nodes, producer_map);
    MY_LOG(1) << "[ROCm EP Level-1] Found " << groups.size() << " mergeable groups";

    // Step 4 & 5: Process each group - build merged params and call fuse()
    auto pass_context = self_.get_context();
    int group_index = 0;
    
    for (auto& group : groups) {
      MY_LOG(1) << "[ROCm EP Level-1] Processing group " << group_index 
                << " with " << group.size() << " nodes";
      
      // Collect internal outputs (outputs produced within the group)
      std::unordered_set<std::string> internal_outputs;
      for (auto* node : group) {
        auto outputs = node_get_output_node_args(*node);
        for (auto* output : outputs) {
          if (output) {
            internal_outputs.insert(node_arg_get_name(*output));
          }
        }
      }
      
      // Collect internal inputs (inputs consumed within the group)
      std::unordered_set<std::string> internal_inputs;
      for (auto* node : group) {
        auto inputs = node_get_input_node_args(*node);
        for (auto* input : inputs) {
          if (input) {
            auto name = node_arg_get_name(*input);
            if (internal_outputs.count(name) > 0) {
              internal_inputs.insert(name);
            }
          }
        }
      }
      
      // Collect external inputs and outputs
      auto external_inputs = collect_external_inputs(group, internal_outputs);
      auto external_outputs = collect_external_outputs(group, internal_inputs, graph);
      
      MY_LOG(2) << "[ROCm EP Level-1] Group " << group_index 
                << ": external_inputs=" << external_inputs.size()
                << ", external_outputs=" << external_outputs.size();
      
      // Build RocmSubgraphProto with complete topology
      // Note: external_inputs is passed to be stored in proto for runtime use
      rocm::RocmSubgraphProto subgraph = build_subgraph_proto(
          group, internal_outputs, external_inputs, external_outputs, graph);
      
      MY_LOG(1) << "[ROCm EP Level-1] Built subgraph with " 
                << subgraph.nodes_size() << " nodes, "
                << subgraph.outputs_size() << " external outputs";
      
      // Create merged fused node name
      std::string merged_name = "rocm_subgraph_" + std::to_string(group_index);
      
      // Collect constant initializers from all nodes in the group
      // For now, we don't merge constant initializers - they stay with their original nodes
      std::vector<std::string> constant_initializers;
      
      // Try to fuse the merged group
      auto [meta_def, error] = self_.try_fuse(
          graph,
          merged_name,
          external_inputs,
          external_outputs,
          constant_initializers,
          "ROCm_EP"
      );
      
      if (meta_def) {
        // Serialize RocmSubgraphProto to JSON
        std::string subgraph_json;
        auto serialize_status = google::protobuf::util::MessageToJsonString(
            subgraph, &subgraph_json);
        if (serialize_status.ok()) {
          // Write subgraph to cache for troubleshooting
          std::string subgraph_filename = "rocm_subgraph_" + merged_name + ".json";
          auto writer = pass_context->open_file_for_write(subgraph_filename);
          if (writer) {
            writer->fwrite(subgraph_json.data(), subgraph_json.size());
            MY_LOG(1) << "[ROCm EP Level-1] Saved subgraph: " << subgraph_filename;
          }
          
          // Attach subgraph JSON to meta_def (this is what custom_op receives)
          self_.attach_meta_def_param(*meta_def, subgraph_json.c_str());
        } else {
          LOG(ERROR) << "[ROCm EP Level-1] Failed to serialize subgraph: " 
                     << serialize_status.ToString();
        }
        
        // Call fuse() to create the merged node and update context.json
        self_.fuse(graph, std::move(*meta_def));
        MY_LOG(1) << "[ROCm EP Level-1] Created merged fused node: " << merged_name
                  << " with " << group.size() << " operations";
      } else {
        LOG(WARNING) << "[ROCm EP Level-1] Failed to fuse group " << group_index 
                     << ": " << error.comments;
      }
      
      group_index++;
    }

    MY_LOG(1) << "[ROCm EP Level-1] Pass completed - merged " << groups.size() << " groups";
  }

  IPass& self_;
};

DEFINE_MORPHIZEN_PASS(Level1Rocm, morphizen_pass_level1_rocm)
