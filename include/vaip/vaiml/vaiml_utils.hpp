// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vaip/vaiml/vaiml_config.hpp"
#include "vaip/vaiml/vaiml_logging.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <glog/logging.h>
#include <iostream>
#include <list>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vaip_vaiml {
using namespace vaip_core;

namespace vaiml_compile {} // namespace vaiml_compile
using AttributeMapType =
    std::unordered_map<std::string, std::variant<int64_t, float, std::string>>;
using VaimlArgShapeMap = std::unordered_map<std::string, VaimlTensorShape>;

using VaimlStrVec = std::vector<std::string>;
using VaimlTensorShape = std::vector<int64_t>;
using VaimlShapeVec = std::vector<VaimlTensorShape>;
using VaimlShapeDict = std::map<std::string, VaimlTensorShape>;
using VaimlStrPairList = std::list<std::pair<std::string, std::string>>;

using IDxMapVector = std::vector<std::pair<size_t, size_t>>;

struct StatsInfo {
  int total_unique_ops = 0;
  int total_cached_ops = 0;
};
struct NodeTrace {
  std::string subgraph;
  std::string op_type;
  std::string status;
};

struct LLMCompileStats {
  int op_requested_cnt;
  int op_cached_cnt;
  int op_compiled_cnt;
  int op_failed_cnt;
};

struct NodeDimRecord {
  AttributeMapType attrs;
  VaimlShapeVec constants_dims;
  VaimlShapeVec inputs_dims;
  VaimlStrVec inputs_names;
  std::unordered_map<int64_t, VaimlArgShapeMap> inputs_fixed_dims;
  VaimlShapeVec outputs_dims;
  VaimlStrVec outputs_names;
  std::unordered_map<int64_t, VaimlArgShapeMap> outputs_fixed_dims;
  std::unordered_map<size_t, float> scalar_values_map;
  std::string op_compile_out_path;
  std::unordered_map<int64_t, std::string>
      compile_with_fixed_shape_path; // store pad_to
};
struct IndexedSubGraph {
  std::vector<size_t> all_nodes;
  std::vector<std::string> all_nodes_names;
  VaimlStrVec all_inputs;
  VaimlStrVec all_inits;
  VaimlStrVec all_outputs;
  VaimlStrVec input_consumers;
  std::filesystem::path save_path;
  std::string name;
  size_t name_id;
  VaimlShapeVec input_shapes;
  VaimlShapeVec output_shapes;
  VaimlShapeVec init_shapes;
  int64_t total_ops = 0;
  // extra fields for recording fixed dynamic shapes
  std::unordered_map<int64_t, VaimlArgShapeMap> inputs_fixed_dims;
  std::unordered_map<int64_t, VaimlArgShapeMap> outputs_fixed_dims;
  std::unordered_map<int64_t, std::string> compile_with_fixed_shape_path;
};

struct NodeWithNodeArg {
  const onnxruntime::Node* node;
  const onnxruntime::NodeArg* nodeArg;
  NodeWithNodeArg(const onnxruntime::Node* n, const onnxruntime::NodeArg* na)
      : node(n), nodeArg(na) {}
};

std::string getNodeName(const Node* node);
bool isConsumer(const onnxruntime::Graph& graph,
                std::vector<const Node*>& cur_group_nodes, const Node* node);

void getAttrsAndIOs(
    const onnxruntime::Graph& graph, const std::vector<const Node*>& subgraph,
    const std::unordered_map<std::string, ConstantInfo>& constants_map,
    AttributeMapType& attributes_map,
    std::vector<const NodeArg*>& ordered_constants,
    std::vector<const NodeArg*>& ordered_inputs,
    VaimlStrVec& ordered_input_names,
    std::vector<const NodeArg*>& ordered_outputs,
    VaimlStrVec& ordered_output_names, const std::string& custom_op_name);

void getConstArgsInfo(
    const std::vector<const NodeArg*>& const_args,
    const std::unordered_map<std::string, ConstantInfo>& constants_map,
    NodeDimRecord& node_dims_record);

void getArgsDims(const std::vector<const NodeArg*>& ordered_args,
                 const std::vector<int64_t> initMValues, bool is_input,
                 NodeDimRecord& node_dims_record);

std::string getReplacement(
    std::string marker, std::string strToBeReplaced,
    NodeDimRecord& node_dims_record,
    std::unordered_map<std::string, VaimlShapeVec>& local_dim_paddings);

std::vector<std::string> splitString(const std::string& str, char delimiter);

int64_t padInt64(int64_t n, int64_t g);

void replaceSubStr(std::string& srcStr, const std::string& subStrToBeReplaced,
                   const std::string& replaceStr);

std::string GetSignatureFromFile(fs::path sig_file);

std::vector<NodeWithNodeArg>
getConsumerNodesWithSrcNodeArgs(const onnxruntime::Graph& graph,
                                const Node* node);

std::string to_string_with_precision(float f, int n = 6);
std::string string_from_StringVector(std::vector<std::string>& string_vector);
std::string string_from_intVector(std::vector<size_t>& shape_vector);
void genMlirFromBaseMlir(fs::path& custom_op_src_path, fs::path& model_out_path,
                         std::string& op_full_name,
                         std::vector<std::string>& strToBeReplacedVec,
                         std::vector<std::string>& replaceStrVec,
                         VaimlStrPairList& replacementsForAssignements,
                         fs::path& op_mlir_path);

int64_t nodeProfile(const onnxruntime::Graph& graph,
                    std::map<std::string, int64_t>& gops,
                    std::string vaiml_unarchive_path);

void updateNodeDim(int64_t seq_len, NodeDimRecord& node_dims_record);

bool causeLoopDep(
    const onnxruntime::Graph& graph,
    const std::vector<IndexedSubGraph>& custom_op_subgraph,
    const std::unordered_map<std::string, std::pair<size_t, size_t>>&
        cur_outputs_map,
    IndexedSubGraph* cur_custom_op, std::string& connecting_input_arg_name);

bool isOutputConsumedOutsideSubgraph(const onnxruntime::Graph& graph,
                                     std::string& output_arg_name,
                                     std::vector<std::string>& all_node_names);

size_t getArgSize(const onnxruntime::Graph& graph, std::string& arg_name,
                  VaimlTensorShape& arg_shape);

int64_t eval_extend(std::string extend_str, int64_t dim_size);

void dumpNodeTrace(std::map<std::string, NodeTrace>& map_partition_trace,
                   std::string output_dir);

bool hasProducer(size_t node_idx, std::vector<size_t>& node_idx_vec,
                 const Graph& graph);
bool hasConsumer(size_t node_idx, std::vector<size_t>& node_idx_vec,
                 const Graph& graph);

bool isIsolated(size_t idx, std::vector<size_t>& node_idx_vec,
                const onnxruntime::Graph& graph);

void dumpGraphNodes(onnxruntime::Graph& graph, std::string json_file_name);

} // namespace vaip_vaiml
