/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
#include "stat.hpp"
#include "vitis/ai/env_config.hpp"
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

DEF_ENV_PARAM(XLNX_ENABLE_SUMMARY_LOG, "1")
#define ALL_DEVICE_NAME "all"

namespace vaip_core {

typedef struct {
  int node_num;
  std::set<std::string> op_type;
} device_stat;

static std::vector<std::string> get_input(const Node& node) {
  auto inputs = node_get_input_node_args(node);
  std::vector<std::string> ret;
  ret.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (node_arg_exists(*input)) {
      ret.push_back(node_arg_get_name(*input));
    } else {
      ret.push_back("VAIP_NOT_EXISTS");
    }
  }
  return ret;
}

static std::vector<std::string> get_output(const Node& node) {
  auto outputs = node_get_output_node_args(node);
  std::vector<std::string> ret;
  ret.reserve(outputs.size());
  for (const auto& output : outputs) {
    if (output != nullptr) {
      ret.push_back(node_arg_get_name(*output));
    } else {
      ret.push_back("VAIP_NOT_EXISTS");
    }
  }
  return ret;
}
static void
add_node_stat(StatProto& proto, const std::vector<std::string>& input,
              const std::vector<std::string>& output,
              const std::string& op_domain, const std::string& op_type,
              const std::string& comment, const std::string& device) {
  auto* new_node_stat = proto.add_node_stat();
  new_node_stat->mutable_input()->Add(input.begin(), input.end());
  new_node_stat->mutable_output()->Add(output.begin(), output.end());
  new_node_stat->set_op_domain(op_domain);
  new_node_stat->set_op_type(op_type);
  new_node_stat->set_comment(comment);
  new_node_stat->set_device(device);
}

static std::unordered_map<std::string, std::string>
get_all_node_device(const ContextProto& context) {
  std::unordered_map<std::string, std::string> node_to_device_map;
  for (auto& meta_def : context.meta_def()) {
    auto device = meta_def.device();
    for (auto& node_arg_name : meta_def.nodes()) {
      node_to_device_map[node_arg_name] = device;
    }
  }
  return node_to_device_map;
}

static std::string
get_device(std::unordered_map<std::string, std::string>& node_to_device_map,
           const std::string& name) {
  auto iter = node_to_device_map.find(name);
  if (iter != node_to_device_map.end()) {
    return iter->second;
  }
  return "CPU";
}

static void update_device_stat(const std::string& device,
                               std::map<std::string, device_stat>& device_stat,
                               const std::string& op_type) {
  if (device_stat.find(device) == device_stat.end()) {
    device_stat[device] = {};
  }
  ++device_stat[device].node_num;
  device_stat[device].op_type.insert(op_type);
}

static void add_device_stat(StatProto& proto, const std::string& device_name,
                            const device_stat& device_stat) {
  auto* new_device_stat = proto.add_device_stat();
  new_device_stat->set_name(device_name);
  new_device_stat->set_node_num(device_stat.node_num);
  new_device_stat->mutable_supported_op_type()->Add(device_stat.op_type.begin(),
                                                    device_stat.op_type.end());
}

static void write_device_stat(StatProto& proto,
                              std::map<std::string, device_stat>& map) {
  // write "all" first
  add_device_stat(proto, "all", map.find("all")->second);
  map.erase(map.find("all"));
  for (const auto& iter : map) {
    add_device_stat(proto, iter.first, iter.second);
  }
}

static void write_shape_info(StatProto& proto, const Node& node) {
  auto outputs = node_get_output_node_args(node);
  for (const auto& output : outputs) {
    if (output == nullptr) { // optional output node arg is nullptr
      continue;
    }
    auto* new_shape_info = proto.add_shape_info();
    new_shape_info->set_name(node_arg_get_name(*output));
    auto shape = node_arg_get_shape_i64(*output);
    if (!shape) {
      new_shape_info->set_is_unknown(true);
    } else if (shape->empty()) {
      new_shape_info->set_is_scalar(true);
    } else {
      new_shape_info->mutable_shape()->Add(shape->begin(), shape->end());
    }
  }
}

static void collect_subgraph_stat(StatProto& proto,
                                  const ContextProto& context) {
  std::map<std::string, int32_t> subgraph_count;
  for (const auto& count : context.device_subgraph_count()) {
    subgraph_count[count.first] = count.second;
  }
  for (auto& meta_def : context.meta_def()) {
    auto device = meta_def.device();
    auto iter = subgraph_count.find(device);
    if (iter == subgraph_count.end()) {
      subgraph_count[device] = 1;
    } else {
      ++subgraph_count[device];
    }
  }

  auto iter = subgraph_count.find("DPU");
  int actuall_ipu_count = -1;
  if (iter != subgraph_count.end()) {
    actuall_ipu_count = iter->second;
    subgraph_count.erase(iter);
  }

  for (auto iter : subgraph_count) {
    auto* subgraph_stat = proto.add_subgraph_stat();
    if ("IPU" == iter.first || "DOD" == iter.first) {
      subgraph_stat->set_device("NPU");
    } else {
      subgraph_stat->set_device(iter.first);
    }
    subgraph_stat->set_count(iter.second);
  }
  if (actuall_ipu_count != -1) {
    auto* subgraph_stat = proto.add_subgraph_stat();
    subgraph_stat->set_device("Actually running on NPU");
    subgraph_stat->set_count(actuall_ipu_count);
  }
}

static void log_stat(const StatProto& proto) {
  if (!ENV_PARAM(XLNX_ENABLE_SUMMARY_LOG)) {
    return;
  }

  const auto& device_proto = proto.device_stat();
  std::cout << "[Vitis AI EP] No. of Operators :";
  auto all_op_num = 0;
  for (const auto& device_stat : device_proto) {
    if (device_stat.name() == "all") {
      all_op_num = device_stat.node_num();
    }
  }
  CHECK_GT(all_op_num, 0) << " All Op num should be greater than 0.";
  for (const auto& device_stat : device_proto) {
    auto name = device_stat.name() == "DPU" ? "IPU" : device_stat.name();
    if (name != "all") {
      std::cout << std::setw(6) << name << std::setw(6)
                << device_stat.node_num() << " ";
    }
    if (name == "IPU") {
      std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                << (float)device_stat.node_num() / (float)all_op_num * 100
                << "%";
      std::cout.precision(6);
    }
  }
  std::cout << std::endl;

  const auto& subgraph_proto = proto.subgraph_stat();
  if (!subgraph_proto.empty()) {
    std::cout << "[Vitis AI EP] No. of Subgraphs :";
    for (const auto& subgraph_stat : subgraph_proto) {
      std::cout << std::setw(6) << subgraph_stat.device() << std::setw(6)
                << subgraph_stat.count() << " ";
    }
    std::cout << std::endl;
  }
}

// what if different models run in multi-threading environment?
thread_local StatProto stat_proto;
StatProto& get_stat_proto() { return stat_proto; }

void clean_stat() { get_stat_proto().Clear(); }

void collect_stat(const onnxruntime::Graph& graph,
                  const ContextProto& context_proto) {
  StatProto& proto = get_stat_proto();
  auto node_to_device_map = get_all_node_device(context_proto);
  std::map<std::string, device_stat> device_stat_map;
  for (auto index : graph_get_node_in_topoligical_order(graph)) {
    auto node = VAIP_ORT_API(graph_get_node)(graph, index);
    auto input = get_input(*node);
    auto output = get_output(*node);
    auto op_type = node_op_type(*node);
    auto domain = node_op_domain(*node);
    auto device = get_device(node_to_device_map, output[0]);
    if ("DPU" == device || "DOD" == device) {
      device = "NPU";
    }
    auto comment = node_as_string(*node);
    add_node_stat(proto, input, output, domain, op_type, comment, device);
    auto domain_op = domain + "::" + op_type;
    update_device_stat(device, device_stat_map, domain_op);
    update_device_stat(ALL_DEVICE_NAME, device_stat_map, domain_op);
    write_shape_info(proto, *node);
  }
  collect_subgraph_stat(proto, context_proto);
  write_device_stat(proto, device_stat_map);
  log_stat(proto);
}
} // namespace vaip_core
