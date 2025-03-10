// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/vaip.hpp"

#include "vaip/vaiml/vaiml_utils.hpp"

#ifdef _WIN32
#  pragma warning(disable : 4244)
#  pragma warning(disable : 4996)
// #pragma warning(disable : 4840)
#endif

namespace vaip_vaiml {
using namespace vaip_core;
namespace fs = std::filesystem;

std::vector<const Node*> getConsumerNodes(const Graph& graph,
                                          const Node* node) {
  std::vector<const Node*> ret;
  auto outputs = node_get_output_node_args(*node);
  for (auto output : outputs) {
    auto consumers =
        graph_get_consumer_nodes(graph, node_arg_get_name(*output));

    for (auto c : consumers) {
      ret.push_back(c);
    }
  }
  return ret;
}

std::string getNodeName(const Node* node) {
  std::string nodeName = VAIP_ORT_API(node_get_name)(*node);
  // To avoid collisions between existing node names, we do the following
  // 1) If the node name is empty,
  //    we assign "=<op_type>-><name of first output>"
  // 2) else if the node name starts with "=",
  //    we replace that part with "=="
  // 3) else stays as is
  // Keep in sync with assign_unique_names() in vaiml.py!
  if (nodeName.empty()) {
    auto outputs = node_get_output_node_args(*node);
    // Need an output to guarantee uniqueness; but right now ONNX doesn't
    // contain any op without outputs, so this should never happen.
    if (!outputs.empty())
      nodeName =
          "=" + node_op_type(*node) + "->" + node_arg_get_name(*outputs[0]);
  } else if (nodeName[0] == '=') {
    nodeName = "=" + nodeName;
  }
  return nodeName;
}

bool isConsumer(const Graph& graph, std::vector<const Node*>& cur_group_nodes,
                const Node* node) {
  bool res = false;
  for (const auto* cur_node : cur_group_nodes) {
    // get cur_node cosnumer and match with node name
    auto consumers = getConsumerNodes(graph, cur_node);
    for (auto c : consumers) {
      auto consumerName = VAIP_ORT_API(node_get_name)(*c);
      // for (auto it = cur_node->OutputEdgesBegin(), end =
      // cur_node->OutputEdgesEnd(); it != end; ++it) { auto consumerName =
      // it->GetNode().Name();
      auto nodeName = VAIP_ORT_API(node_get_name)(*node);
      if (consumerName == nodeName) {
        res = true;
        break;
      }
    }
    if (res) {
      break;
    }
  }
  return res;
}

void getAttrsAndIOs(
    const Graph& graph, const std::vector<const Node*>& subgraph,
    const std::unordered_map<std::string, ConstantInfo>& constants_map,
    AttributeMapType& attributes_map,
    std::vector<const NodeArg*>& ordered_constants,
    std::vector<const NodeArg*>& ordered_inputs,
    VaimlStrVec& ordered_input_names,
    std::vector<const NodeArg*>& ordered_outputs,
    VaimlStrVec& ordered_output_names, const std::string& custom_op_name) {
  // get attributes and the ordered inputs of the subgraph
  // First, get entire graph's inputs and outputs
  auto graph_outputs_vec = graph_get_outputs(graph);
  std::unordered_set<const NodeArg*> graph_outputs(graph_outputs_vec.cbegin(),
                                                   graph_outputs_vec.cend());
  auto graph_inputs_vec = graph_get_inputs(graph);
  std::unordered_set<const NodeArg*> graph_inputs(graph_inputs_vec.cbegin(),
                                                  graph_inputs_vec.cend());
  std::unordered_set<const NodeArg*> subgraph_nodes_outputs;
  // get all nodes' attributes and outputs in the subgraph
  for (const auto* node : subgraph) {
    // get attributes map
    const auto& node_attrs = node_get_attributes(*node);
    auto node_name = VAIP_ORT_API(node_get_name)(*node);
    for (auto* attr : node_attrs) {
      auto& attr_name = VAIP_ORT_API(attr_proto_get_name)(*attr);
      auto attr_value = node_get_attr(*node, attr_name);
      auto attr_type = VAIP_ORT_API(attr_proto_get_type)(*attr_value);
      if (attr_type == onnx::AttributeProto_AttributeType_INT) {
        auto value = VAIP_ORT_API(attr_proto_get_int)(*attr_value);
        attributes_map[attr_name] = value;
      } else if (attr_type == onnx::AttributeProto_AttributeType_FLOAT) {
        auto value = VAIP_ORT_API(attr_proto_get_float)(*attr_value);
        attributes_map[attr_name] = value;
      } else if (attr_type == onnx::AttributeProto_AttributeType_STRING) {
        auto value = VAIP_ORT_API(attr_proto_get_string)(*attr_value);
        attributes_map[attr_name] = value;
      }
    }

    for (const auto* output : node_get_output_node_args(*node)) {
      if (!node_arg_exists(*output)) {
        continue;
      }
      subgraph_nodes_outputs.insert(output);
    }
  }

  for (const auto* node : subgraph) {
    auto node_name = VAIP_ORT_API(node_get_name)(*node);
    for (const auto* input : node_get_input_node_args(*node)) {
      if (!node_arg_exists(*input)) {
        continue;
      }

      if (subgraph_nodes_outputs.count(input) != 0) {
        // input is generted by the node in the subgraph
        continue;
      }
      // check if input has already been added to the ordered_constants or
      // ordered_inputs
      auto it = std::find(ordered_inputs.begin(), ordered_inputs.end(), input);
      if (it != ordered_inputs.end()) { // already added
        continue;
      }
      it = std::find(ordered_constants.begin(), ordered_constants.end(), input);
      if (it != ordered_constants.end()) {
        continue;
      }
      std::string inputArgName = node_arg_get_name(*input);
      if (constants_map.find(inputArgName) != constants_map.end()) {
        auto const_info = constants_map.at(inputArgName);
        if (const_info.is_scalar) {
          VAIML_DEBUG_PRINT(2, "  subgraph node ", node_name,
                            " has scalar constant input arg ", inputArgName,
                            " with value ", const_info.value);
        } else {
          VAIML_DEBUG_PRINT(2, "  subgraph node ", node_name,
                            " has tensor constant input arg ", inputArgName);
        }
        ordered_constants.push_back(input);
      } else {
        // bmm_qk pushed reshape output as input
        std::string node_type = VAIP_ORT_API(node_op_type)(*node);
        if ((custom_op_name == "bmm_qk" || custom_op_name == "bmm_v" ||
             custom_op_name == "bmm_qk_4d" || custom_op_name == "bmm_v_4d") &&
            node_type == "Reshape") {
          VAIML_DEBUG_PRINT(2, " add subgraph node ", node_name,
                            " output as subgraph Input");
          auto& output_arg = node_get_output_node_arg(*node);
          ordered_inputs.push_back(&output_arg);
        } else {
          VAIML_DEBUG_PRINT(2, " add subgraph node ", node_name,
                            " input as subgraph Input");
          ordered_inputs.push_back(input);
        }
        ordered_input_names.push_back(inputArgName);
      }
    }

    // get the ordered outputs of the subgraph
    for (const auto* output : node_get_output_node_args(*node)) {
      if (!node_arg_exists(*output)) {
        continue;
      }
      auto it =
          std::find(ordered_outputs.begin(), ordered_outputs.end(), output);
      if (it != ordered_outputs.end()) {
        continue; // already added to ordered_outputs
      }
      auto consumers =
          graph_get_consumer_nodes(graph, node_arg_get_name(*output));

      bool is_consumer_outside_subgraph = false;
      for (auto c : consumers) {
        auto it = std::find(subgraph.begin(), subgraph.end(), c);
        if (it == subgraph.end()) {
          is_consumer_outside_subgraph = true;
          break; // Consumed outside this subgraph
        }
      }

      // Declare this NodeArg as subgraph output if it is consumed by an
      // external node or when it is an overall graph output.
      if (is_consumer_outside_subgraph || graph_outputs.count(output)) {
        std::string node_type = VAIP_ORT_API(node_op_type)(*node);
        if ((custom_op_name == "bmm_v" || custom_op_name == "bmm_v_4d") &&
            node_type == "Reshape") {
          VAIML_DEBUG_PRINT(2, " add subgraph node ", node_name,
                            " input as subgraph Output");
          auto input_args = node_get_input_node_args(*node);
          if (!input_args.empty()) {
            ordered_outputs.push_back(input_args[0]);
          }
        } else {
          ordered_outputs.push_back(output);
        }
        ordered_output_names.push_back(node_arg_get_name(*output));
      }
    }
  }
}

void getConstArgsInfo(
    const std::vector<const NodeArg*>& const_args,
    const std::unordered_map<std::string, ConstantInfo>& constants_map,
    NodeDimRecord& node_dims_record) {
  // VaimlShapeVec& const_dims, std::vector<float>& scalar_values) {
  // get the dimensions of the args
  auto& const_dims = node_dims_record.constants_dims;
  auto& scalar_values = node_dims_record.scalar_values_map;

  size_t idx = 0;
  for (const auto* arg : const_args) {
    auto arg_shape_ptr = node_arg_get_shape_i64(*arg);
    if (arg_shape_ptr == nullptr) {
      continue;
    }
    std::vector<int64_t> arg_shape_vec;
    int64_t dim_size = 1;
    auto arg_shape = *(arg_shape_ptr.get());
    if (arg_shape.size() == 1 && arg_shape[0] == 1) {
      // if the const in the model has a single element in a tensor, mark it as
      // a scalar
      arg_shape.clear();
    }
    VAIML_DEBUG_PRINT(2, "Debug const arg shape: ", node_arg_get_name(*arg));
    for (size_t i = 0; i < arg_shape.size(); i++) {
      VAIML_DEBUG_PRINT(2, "        ", arg_shape[i]);
    }
    for (size_t i = 0; i < arg_shape.size(); i++) {
      arg_shape_vec.push_back(arg_shape[i]);
      dim_size *= arg_shape[i];
    }
    const_dims.push_back(arg_shape_vec);
    if (dim_size == 1) {
      scalar_values[idx] = constants_map.at(node_arg_get_name(*arg)).value;
    }
    idx++;
  }
}

void getArgsDims(const std::vector<const NodeArg*>& ordered_args,
                 const std::vector<int64_t> initMValues, bool is_input,
                 NodeDimRecord& node_dims_record) {
  // get the dimensions of the args
  for (const auto* arg : ordered_args) {
    auto arg_shape_ptr = node_arg_get_shape_i64(*arg);
    std::string arg_name = node_arg_get_name(*arg);
    VAIML_DEBUG_PRINT(2, "Debug node arg shape: ", node_arg_get_name(*arg));
    if (arg_shape_ptr == nullptr) {
      continue;
    }
    std::vector<int64_t> arg_shape_vec;
    auto arg_shape = *(arg_shape_ptr.get());
    bool is_2d = (arg_shape.size() == 2);
    bool is_3d = (arg_shape.size() == 3);
    for (size_t i = 0; i < arg_shape.size(); i++) {
      VAIML_DEBUG_PRINT(2, "Debug shape ", arg_shape[i])
      if (arg_shape[i] < 0) {
        if (is_2d && (i == 0)) {
          arg_shape[i] = initMValues[0];
        }
        if (is_3d) {
          arg_shape[i] = (i == 0) ? 1 : (i == 1) ? initMValues[0] : 0;
        }
        VAIML_DEBUG_PRINT(2, "Debug dynamice shape adjusted to ", arg_shape[i])
      }
      arg_shape_vec.push_back(arg_shape[i]);
    }
    if (is_input) {
      node_dims_record.inputs_dims.push_back(arg_shape_vec);
      // node_dims_record.inputs_names is populated in getAttrsAndIOs
    } else {
      node_dims_record.outputs_dims.push_back(arg_shape_vec);
      // node_dims_record.outputs_names is populated in getAttrsAndIOs
    }
  }
}

std::string getReplacement(
    std::string marker, std::string strToBeReplaced,
    NodeDimRecord& node_dims_record,
    std::unordered_map<std::string, VaimlShapeVec>& local_dim_paddings) {

  auto exprVector = splitString(strToBeReplaced, '.');
  auto argType = exprVector[0]; // inputs OR outputs
  size_t outerIndex = std::atoi(exprVector[1].c_str());
  size_t innerIndex = std::atoi(exprVector[2].c_str());
  VAIML_DEBUG_PRINT(3, "Assignment for ", strToBeReplaced);
  VaimlShapeVec paddings;
  assert(argType == "inputs" || argType == "outputs");
  int64_t paddingMultiple = 1;
  if (local_dim_paddings.find(argType) != local_dim_paddings.end()) {
    paddings = local_dim_paddings.at(argType);
    if ((paddings.size() > outerIndex) &&
        (paddings[outerIndex].size() > innerIndex)) {
      paddingMultiple = paddings[outerIndex][innerIndex];
      VAIML_DEBUG_PRINT(3, "PaddingMultiple from custom op json is ",
                        paddingMultiple);
    }
  } else {
    VAIML_DEBUG_PRINT(
        2, "    no padding field in custom op definition for handling", marker);
  }
  int64_t operatorDim = 0;
  if (argType == "inputs") {
    if ((node_dims_record.inputs_dims.size() > outerIndex) &&
        (node_dims_record.inputs_dims[outerIndex].size() > innerIndex)) {
      operatorDim = node_dims_record.inputs_dims[outerIndex][innerIndex];
    } else {
      VAIML_DEBUG_PRINT(2, "    no inputs available for handling", marker);
    }
  } else {
    if ((node_dims_record.outputs_dims.size() > outerIndex) &&
        (node_dims_record.outputs_dims[outerIndex].size() > innerIndex)) {
      operatorDim = node_dims_record.outputs_dims[outerIndex][innerIndex];
    } else {
      VAIML_DEBUG_PRINT(2, "    no outputs available for handling", marker);
    }
  }
  assert(operatorDim != 0);
  VAIML_DEBUG_PRINT(3, "dimensions from the operator instance is ",
                    operatorDim);

  std::string paddedValue =
      std::to_string(padInt64(operatorDim, paddingMultiple));
  VAIML_DEBUG_PRINT(2, "Assignment value for id ", marker,
                    " with specification ", strToBeReplaced, " is ",
                    paddedValue);
  return paddedValue;
}

int64_t padInt64(int64_t n, int64_t g) {
  return int64_t(std::ceil((float)n / (float)g) * (float)g);
}

void replaceSubStr(std::string& srcStr, const std::string& subStrToBeReplaced,
                   const std::string& replaceStr) {
  while (true) {
    size_t len = subStrToBeReplaced.length();
    size_t pos = srcStr.find(subStrToBeReplaced);
    if (pos != std::string::npos)
      srcStr.replace(pos, len, replaceStr);
    else
      break;
  }
}

std::string string_from_StringVector(std::vector<std::string>& string_vector) {
  std::ostringstream os;
  for (auto elem : string_vector) {
    os << " " << elem;
  }
  return os.str();
}
std::string string_from_intVector(std::vector<size_t>& shape_vector) {
  std::ostringstream os;
  for (auto elem : shape_vector) {
    os << " " << elem;
  }
  return os.str();
}

/**
 * @brief Get signarure from a file
 */
std::string GetSignatureFromFile(fs::path sig_file) {
  std::ifstream sig_fs(sig_file);
  if (!sig_fs.is_open()) {
    return "NONE";
  }
  std::string hash_value;
  getline(sig_fs, hash_value);

  return hash_value;
}

std::vector<NodeWithNodeArg> getConsumerNodesWithSrcNodeArgs(const Graph& graph,
                                                             const Node* node) {
  std::vector<NodeWithNodeArg> ret;
  auto outputs = node_get_output_node_args(*node);
  for (auto output : outputs) {
    auto consumers =
        graph_get_consumer_nodes(graph, node_arg_get_name(*output));
    for (auto c : consumers) {
      NodeWithNodeArg result(c, output);
      ret.push_back(result);
    }
  }
  return ret;
}

std::string to_string_with_precision(float f, int n) {
  std::ostringstream oss;
  oss.precision(n);
  oss << f;
  return oss.str();
}

void genMlirFromBaseMlir(fs::path& custom_op_src_path, fs::path& model_out_path,
                         std::string& op_full_name,
                         std::vector<std::string>& strToBeReplacedVec,
                         std::vector<std::string>& replaceStrVec,
                         VaimlStrPairList& replacementsForAssignements,
                         fs::path& op_mlir_path) {
  if (strToBeReplacedVec.size() != replaceStrVec.size()) {
    LOG(FATAL) << "Error: when generating mlir for microkernel, the size of "
                  "strToBeReplacedVec and replaceStrVec should be the same.";
  }

  auto base_mlir_path = custom_op_src_path / "base.onnx.mlir";
  if (!fs::exists(base_mlir_path)) {
    LOG(FATAL) << "Error: " << base_mlir_path << "does not exist.";
  }
  std::ifstream baseMlirIfs(base_mlir_path);
  if (!baseMlirIfs.is_open()) {
    LOG(FATAL) << "Error: Failed to open " << base_mlir_path;
  }

  op_mlir_path = model_out_path / (op_full_name + ".onnx.mlir");
  std::ofstream opMlirOfs(op_mlir_path);
  if (!opMlirOfs.is_open()) {
    LOG(FATAL) << "Error: Failed to open " << op_mlir_path;
  }

  std::string line;
  VAIML_DEBUG_PRINT(2, "Generating following mlir ..")
  while (getline(baseMlirIfs, line)) {
    for (size_t i = 0; i < strToBeReplacedVec.size(); i++) {
      replaceSubStr(line, strToBeReplacedVec[i], replaceStrVec[i]);
    }
    for (auto replacement : replacementsForAssignements) {
      replaceSubStr(line, replacement.first, replacement.second);
    }
    opMlirOfs << line << '\n';
  }
  opMlirOfs.close();

  VAIML_DEBUG_PRINT(2, "End generating mlir file ", op_mlir_path.string());
}

int64_t getInputVolume(const Node& node, int index) {
  auto inputs = node_get_input_node_args(node);
  auto input_shape = node_arg_get_shape_i64(*inputs[index]);
  auto shape = *(input_shape.get());
  int64_t input_ops = 1;
  for (size_t i = 1; i < shape.size(); i++) {
    input_ops *= shape[i];
  }
  return input_ops;
}

int64_t getOutputVolume(const Node& node, int index) {
  auto output_shape = node_get_output_shape(node, index);
  int64_t output_ops = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_ops *= output_shape[i];
  }
  return output_ops;
}

int64_t nodeProfile(const Graph& graph, std::map<std::string, int64_t>& ops,
                    std::string vaiml_unarchive_path) {
  const auto& node_indices = graph_get_node_in_topoligical_order(graph);
  std::vector<std::string> unsupported_node_type;
  std::vector<std::string> nodes_with_unknown_shapes;
  int64_t total_operations = 0;
  for (size_t node_idx : node_indices) {
    const auto* node(VAIP_ORT_API(graph_get_node)(graph, node_idx));
    std::string node_type = VAIP_ORT_API(node_op_type)(*node);
    int64_t DIV_MACS = 4;
    int64_t EXP_MACS = 32;
    int64_t SQRT_MACS = 24;
    int64_t LOG_MACS = 43;
    int64_t POW_MACS = EXP_MACS;
    int64_t TANH_MACS = EXP_MACS + 2 * 1 + DIV_MACS;
    int64_t ATANH_MACS = LOG_MACS + 2 * 1 + DIV_MACS + 1;
    std::string node_name = VAIP_ORT_API(node_get_name)(*node);
    if (node_name.empty()) {
      auto outputs = node_get_output_node_args(*node);
      // Need an output to guarantee uniqueness; but right now ONNX doesn't
      // contain any op without outputs, so this should never happen.
      if (!outputs.empty()) {
        node_name =
            "=" + node_op_type(*node) + "->" + node_arg_get_name(*outputs[0]);
      }
    } else if (node_name[0] == '=') {
      node_name = "=" + node_name;
    }
    if (node_as_string(*node).find("shape=UNKWN") != std::string::npos) {
      // std::cout<<"debug node as string::"<< node_as_string(*node)<<std::endl;
      nodes_with_unknown_shapes.push_back(node_name);
      if (node_type == "Conv" || node_type == "ConvTranspose") {
        int64_t input_ops = 1;
        int64_t output_ops = 1;
        auto inputs = node_get_input_node_args(*node);
        std::string input_shape_string = node_arg_as_string(*inputs[1]);
        auto outputs = node_get_output_node_args(*node);
        std::string output_shape_string = node_arg_as_string(*outputs[0]);
        if (input_shape_string.find("shape=UNKWN") == std::string::npos) {
          if (output_shape_string.find("shape=UNKWN") == std::string::npos) {
            input_ops = getInputVolume(*node, 1);
            output_ops = getOutputVolume(*node, 0);
          } else {
            auto input_shape = node_arg_get_shape_i64(*inputs[1]);
            auto shape = *(input_shape.get());
            for (size_t i = 0; i < shape.size(); i++) {
              input_ops *= shape[i];
            }
          }
        } else {
          if (output_shape_string.find("shape=UNKWN") == std::string::npos) {
            output_ops = getOutputVolume(*node, 0);
          }
        }
        if (sizeof(inputs) > 2) {
          ops[node_name] += output_ops;
        }
        ops[node_name] = input_ops * output_ops;
        VAIML_DEBUG_PRINT(2, "INFO: GOPS calculation of node ", node_name,
                          " has unknown shape, calculate part of it.");
      } else if (node_type == "MatMul" || node_type == "Gemm") {
        int64_t input_ops = 1;
        int64_t output_ops = 1;
        auto inputs = node_get_input_node_args(*node);
        std::string input_shape_string = node_arg_as_string(*inputs[1]);
        auto outputs = node_get_output_node_args(*node);
        std::string output_shape_string = node_arg_as_string(*outputs[0]);
        if (input_shape_string.find("shape=UNKWN") == std::string::npos) {
          auto input_shape = node_arg_get_shape_i64(*inputs[1]);
          auto shape = *(input_shape.get());
          if (output_shape_string.find("shape=UNKWN") == std::string::npos) {
            output_ops = getOutputVolume(*node, 0);
            if (node_type == "MatMul") {
              input_ops *= shape.rbegin()[1];
            } else {
              input_ops *= shape.rbegin()[0];
            }
          } else {
            for (size_t i = 0; i < shape.size(); i++) {
              input_ops *= shape[i];
            }
          }
        } else {
          if (output_shape_string.find("shape=UNKWN") == std::string::npos) {
            output_ops = getOutputVolume(*node, 0);
          }
        }
        if (sizeof(inputs) > 2) {
          ops[node_name] += output_ops;
        }
        ops[node_name] = input_ops * output_ops;
        VAIML_DEBUG_PRINT(2, "INFO: GOPS calculation of node ", node_name,
                          " has unknown shape, calculate part of it.");
      } else {
        VAIML_DEBUG_PRINT(2, "INFO: GOPS calculation of node ", node_name,
                          " has unknown shape is not supported.");
      }
    } else {
      ops[node_name] = 0;
      if (node_type == "Conv" || node_type == "ConvTranspose") {
        // output_image_size * kernel shape * output_channels
        auto inputs = node_get_input_node_args(*node);
        int64_t input_ops = getInputVolume(*node, 1);
        int64_t output_ops = getOutputVolume(*node, 0);
        ops[node_name] = input_ops * output_ops;
        // bias
        if (sizeof(inputs) > 2) {
          ops[node_name] += output_ops;
        }
      } else if (node_type == "MatMul" || node_type == "Gemm") {
        auto inputs = node_get_input_node_args(*node);
        auto input_shape = node_arg_get_shape_i64(*inputs[1]);
        auto shape = *(input_shape.get());
        int64_t input_ops = 1;
        int64_t output_ops = getOutputVolume(*node, 0);
        if (node_type == "MatMul") {
          input_ops *= shape.rbegin()[1];
        } else {
          input_ops *= shape.rbegin()[0];
        }
        // bias
        if (sizeof(inputs) > 2) {
          ops[node_name] += output_ops;
        }
        ops[node_name] += input_ops * output_ops;
      } else if (node_type == "DequantizeLinear" ||
                 node_type == "QuantizeLinear") {
        ops[node_name] = 2 * getOutputVolume(*node, 0);
      } else if (node_type == "Add" || node_type == "Mul" ||
                 node_type == "Relu" || node_type == "Sub") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "Div") {
        ops[node_name] = DIV_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "MaxPool") {
        int64_t output_ops = getOutputVolume(*node, 0);
        if (node_has_attr(*node, "kernel_shape")) {
          auto kernel_shape = node_get_attr_ints(*node, "kernel_shape");
          if (kernel_shape.size() == 2) {
            ops[node_name] += output_ops * kernel_shape[0] * kernel_shape[1];
          } else {
            ops[node_name] += output_ops * kernel_shape[0];
          }
        } else {
          ops[node_name] += output_ops;
        }
      } else if (node_type == "GlobalAveragePool") {
        ops[node_name] =
            getInputVolume(*node, 0) + getOutputVolume(*node, 0) * DIV_MACS;
      } else if (node_type == "Softmax") {
        ops[node_name] = (EXP_MACS + DIV_MACS) * getOutputVolume(*node, 0);
      } else if (node_type == "LayerNormalization") {
        auto inputs = node_get_input_node_args(*node);
        auto input_shape = node_arg_get_shape_i64(*inputs[0]);
        auto shape = *(input_shape.get());
        int64_t axis = node_get_attr_int(*node, "axis");
        if (axis < 0) {
          axis = shape.size() + axis;
        }
        int64_t vol = 1;
        int64_t vol2 = 1;
        for (size_t i = 1; i < shape.size(); i++) {
          vol *= shape[i];
        }
        shape[axis] = 1;
        for (size_t i = 1; i < shape.size(); i++) {
          vol2 *= shape[i];
        }
        ops[node_name] =
            vol * (1 * 3 + +1 * 4) + vol2 * (1 + SQRT_MACS + DIV_MACS);
      } else if (node_type == "BatchNormalization") {
        ops[node_name] =
            (1 + SQRT_MACS + DIV_MACS + 1 + 1) * getOutputVolume(*node, 0);
      } else if (node_type == "InstanceNormalization") {
        ops[node_name] = (3 + DIV_MACS) * getOutputVolume(*node, 0);
      } else if (node_type == "Sigmod" || node_type == "Exp") {
        ops[node_name] = EXP_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Pow") {
        ops[node_name] = POW_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Sqrt") {
        ops[node_name] = SQRT_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Clip" || node_type == "LeakyRelu") {
        ops[node_name] = 2 * getOutputVolume(*node, 0);
      } else if (node_type == "HardSwish") {
        ops[node_name] = 5 * getOutputVolume(*node, 0);
      } else if (node_type == "Reciprocal") {
        ops[node_name] = DIV_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "CumSum" || node_type == "Sign") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "Log") {
        ops[node_name] = LOG_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Sin" || node_type == "Cos") {
        ops[node_name] = 39 * getOutputVolume(*node, 0);
      } else if (node_type == "Tanh") {
        ops[node_name] = 2 * TANH_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Atan") {
        ops[node_name] = 2 * ATANH_MACS * getOutputVolume(*node, 0);
      } else if (node_type == "Max" || node_type == "Min" ||
                 node_type == "Neg") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "ReduceMean" || node_type == "ReduceMin" ||
                 node_type == "ReduceMax" || node_type == "ReduceSum") {
        ops[node_name] = getInputVolume(*node, 0);
      } else if (node_type == "Floor" || node_type == "Equal" ||
                 node_type == "Greater") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "GridSample") {
        if (node_has_attr(*node, "mode")) {
          auto mode = node_get_attr_string(*node, "mode");
          if (mode == "linear") {
            auto output_shape = node_get_output_shape(*node, 0);
            int64_t output_ops = getOutputVolume(*node, 0);
            ops[node_name] = output_ops +
                             output_shape[output_shape.size() - 2] * 50 +
                             output_shape[output_shape.size() - 1] *
                                 output_shape[output_shape.size() - 2] * 18;
          } else if (mode == "bilinear") {
            auto output_shape = node_get_output_shape(*node, 0);
            int64_t output_ops = getOutputVolume(*node, 0);
            ops[node_name] = output_ops +
                             output_shape[output_shape.size() - 2] * 50 +
                             output_shape[output_shape.size() - 1] *
                                 output_shape[output_shape.size() - 2] * 18;
          }
        }
      } else if (node_type == "HardSigmoid") {
        ops[node_name] = 4 * getInputVolume(*node, 0);
      } else if (node_type == "And" || node_type == "Abs" ||
                 node_type == "Equal") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "Resize") {
        int64_t output_ops = getOutputVolume(*node, 0);
        if (node_has_attr(*node, "mode")) {
          auto mode = node_get_attr_string(*node, "mode");
          if (mode == "linear") {
            ops[node_name] = output_ops;
          } else if (mode == "nearest") {
            ops[node_name] = 4 * output_ops;
          } else if (mode == "cubic") {
            ops[node_name] = 8 * output_ops;
          }
        }
      } else if (node_type == "Where") {
        ops[node_name] = getInputVolume(*node, 0);
      } else if (node_type == "erf") {
        ops[node_name] = 9 * getInputVolume(*node, 0);
      } else if (node_type == "Not" || node_type == "Or" ||
                 node_type == "Round") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "LSTM") {
        auto inputs = node_get_input_node_args(*node);
        auto input_xshape = node_arg_get_shape_i64(*inputs[0]);
        auto xshape = *(input_xshape.get());
        auto input_wshape = node_arg_get_shape_i64(*inputs[1]);
        auto wshape = *(input_wshape.get());
        auto input_rshape = node_arg_get_shape_i64(*inputs[2]);
        auto rshape = *(input_rshape.get());
        auto input_bshape = node_arg_get_shape_i64(*inputs[3]);
        auto bshape = *(input_bshape.get());
        int64_t seq = xshape[0];
        int64_t batch = xshape[1];
        int64_t num_dir = wshape[0];
        int64_t h_len = wshape[1];
        int64_t ht_size = batch * seq * h_len;
        int64_t wshape_ops = 1, rshape_ops = 1, bshape_ops = 1;
        for (size_t i = 1; i < wshape.size(); i++) {
          wshape_ops *= wshape[i];
        }
        for (size_t i = 1; i < rshape.size(); i++) {
          rshape_ops *= rshape[i];
        }
        for (size_t i = 1; i < bshape.size(); i++) {
          bshape_ops *= bshape[i];
        }
        int64_t gemm_macs = (wshape_ops + rshape_ops) * batch * seq;
        int64_t gemm_bias_macs = bshape_ops * batch * seq;
        int64_t gemm_add_macs = ht_size * 4;
        int64_t sig_macs = ht_size * EXP_MACS * 3;
        int64_t tanh_macs = ht_size * TANH_MACS * 2;
        int64_t blend_macs = ht_size * (1 + 1 + 1 + 1);
        ops[node_name] = gemm_macs + gemm_bias_macs + sig_macs + tanh_macs +
                         blend_macs + gemm_add_macs;
      } else if (node_type == "ReduceL2") {
        ops[node_name] = (1 + SQRT_MACS) * getInputVolume(*node, 0);
      } else if (node_type == "Concat") {
        ops[node_name] = getInputVolume(*node, 0);
      } else if (node_type == "Flatten" || node_type == "Gather" ||
                 node_type == "Slice" || node_type == "Unsqueeze") {
        ops[node_name] = getOutputVolume(*node, 0);
      } else if (node_type == "Reshape" || node_type == "SpaceToDepth") {
        // No computation for Reshape and SpaceToDepth yet, use 1
        ops[node_name] = 1;
      } else {
        if (std::find(unsupported_node_type.begin(),
                      unsupported_node_type.end(),
                      node_type) == unsupported_node_type.end()) {
          unsupported_node_type.push_back(node_type);
          VAIML_DEBUG_PRINT(2, "INFO: GOPs calculation of node type ",
                            node_type, " is not supported.");
        }
      }
    }
  }
  for (const auto& [key, value] : ops) {
    // ops = 2 * macs
    if (value < 0) {
      ops[key] = -value * 2;
    } else {
      ops[key] = value * 2;
    }
    total_operations += ops[key];
  }
  // save ops to a csv file
  fs::path gops_csv_path = fs::path(vaiml_unarchive_path) / "gops.csv";
  std::ofstream gopsOfs(gops_csv_path);
  if (gopsOfs.is_open()) {
    gopsOfs << "Node,OPs,Note\n";
    for (const auto& [key, value] : ops) {
      if (std::find(nodes_with_unknown_shapes.begin(),
                    nodes_with_unknown_shapes.end(),
                    key) != nodes_with_unknown_shapes.end()) {
        gopsOfs << key << "," << value << ",node has unknown shapes"
                << "\n";
      } else {
        gopsOfs << key << "," << value << "\n";
      }
    }
    gopsOfs.close();
  }

  return total_operations;
}

void updateNodeDim(int64_t seq_len, NodeDimRecord& node_dims_record) {
  for (size_t i = 0; i < node_dims_record.inputs_names.size(); ++i) {
    auto& input_name = node_dims_record.inputs_names[i];
    auto& input_dim = node_dims_record.inputs_dims[i];
    bool updateInput = false;
    for (auto dim : input_dim) {
      if (dim <= 0) {
        updateInput = true;
        break;
      }
    }
    if (!updateInput) {
      continue;
    }
    if (node_dims_record.inputs_fixed_dims.find(seq_len) !=
        node_dims_record.inputs_fixed_dims.end()) {
      if (node_dims_record.inputs_fixed_dims[seq_len].find(input_name) !=
          node_dims_record.inputs_fixed_dims[seq_len].end()) {
        auto fixed_shape =
            node_dims_record.inputs_fixed_dims[seq_len][input_name];
        if (fixed_shape.size() == input_dim.size()) {
          for (auto k = 0; k < input_dim.size(); k++) {
            input_dim[k] = fixed_shape[k];
          }
        }
      }
    }
  }

  for (size_t i = 0; i < node_dims_record.outputs_names.size(); ++i) {
    auto& output_name = node_dims_record.outputs_names[i];
    auto& output_dim = node_dims_record.outputs_dims[i];
    bool updateOutput = false;
    for (auto dim : output_dim) {
      if (dim <= 0) {
        updateOutput = true;
        break;
      }
    }
    if (!updateOutput) {
      continue;
    }
    if (node_dims_record.outputs_fixed_dims.find(seq_len) !=
        node_dims_record.outputs_fixed_dims.end()) {
      if (node_dims_record.outputs_fixed_dims[seq_len].find(output_name) !=
          node_dims_record.outputs_fixed_dims[seq_len].end()) {
        auto fixed_shape =
            node_dims_record.outputs_fixed_dims[seq_len][output_name];
        if (fixed_shape.size() == output_dim.size()) {
          for (auto k = 0; k < output_dim.size(); k++) {
            output_dim[k] = fixed_shape[k];
          }
        }
      }
    }
  }
}

// this function checks if adding cur_custom_op to custom_op_subgraph will cause
// loop dependency. the connecting_input_arg_name is the input arg name of
// cur_custom_op that connects to
//  the output of custom_op_subgraph
//  the idea is to check if the connecting_input_arg_name will also be the
//  output of the formed subgraph and the cur_custom_op also has another input
//  that is the also the input of the formed subgraph
bool causeLoopDep(
    const Graph& graph, const std::vector<IndexedSubGraph>& custom_op_subgraph,
    const std::unordered_map<std::string, std::pair<size_t, size_t>>&
        cur_outputs_map,
    IndexedSubGraph* cur_custom_op, std::string& connecting_input_arg_name) {
  bool res = false;
  bool connecting_input_is_output = false;

  std::vector<std::string> all_nodes_names;
  for (auto subgraph : custom_op_subgraph) {
    for (auto node_idx : subgraph.all_nodes) {
      const auto* node(VAIP_ORT_API(graph_get_node)(graph, node_idx));
      std::string node_name = VAIP_ORT_API(node_get_name)(*node);
      all_nodes_names.push_back(node_name);
    }
  }
  for (auto node_idx : cur_custom_op->all_nodes) {
    const auto* node(VAIP_ORT_API(graph_get_node)(graph, node_idx));
    std::string node_name = VAIP_ORT_API(node_get_name)(*node);
    all_nodes_names.push_back(node_name);
  }
  if (isOutputConsumedOutsideSubgraph(graph, connecting_input_arg_name,
                                      all_nodes_names)) {
    connecting_input_is_output = true;
  }
  if (connecting_input_is_output) {
    for (auto input : cur_custom_op->all_inputs) {
      if (cur_outputs_map.find(input) == cur_outputs_map.end()) {
        res = true;
        break;
      }
    }
  }
  return res;
}

bool isOutputConsumedOutsideSubgraph(const Graph& graph,
                                     std::string& output_arg_name,
                                     std::vector<std::string>& all_node_names) {
  bool res = false;
  auto graph_outputs = graph_get_outputs(graph);
  for (auto& o : graph_outputs) {
    if ((o != nullptr) && (node_arg_get_name(*o) == output_arg_name)) {
      return true;
    }
  }
  auto consumers = graph_get_consumer_nodes(graph, output_arg_name);
  VAIML_DEBUG_PRINT(2,
                    "DEBUG: isOutputConsumedOutsideSubgraph output_arg_name: ",
                    output_arg_name);
  for (auto c : consumers) {
    std::string consumerName = VAIP_ORT_API(node_get_name)(*c);
    VAIML_DEBUG_PRINT(
        2,
        "DEBUG: isOutputConsumedOutsideSubgraph consumerName: ", consumerName);
    if (std::find(all_node_names.begin(), all_node_names.end(), consumerName) ==
        all_node_names.end()) {
      res = true;
      break;
    }
  }
  return res;
}

size_t getArgSize(const Graph& graph, std::string& arg_name,
                  VaimlTensorShape& arg_shape) {
  std::map<int, int> datatype_to_size;
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_FLOAT] = sizeof(float);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_UINT8] =
      sizeof(uint8_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_INT8] = sizeof(int8_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_UINT16] =
      sizeof(uint16_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_INT16] =
      sizeof(int16_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_INT32] =
      sizeof(int32_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_INT64] =
      sizeof(int64_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_DOUBLE] =
      sizeof(double);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_UINT32] =
      sizeof(uint32_t);
  datatype_to_size[ONNX_NAMESPACE::TensorProto_DataType_UINT64] =
      sizeof(uint64_t);

  size_t arg_size = 1;
  auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, arg_name);
  auto type = VAIP_ORT_API(node_arg_get_element_type)(*node_arg);
  for (auto& dim : arg_shape) {
    arg_size *= dim;
  }
  arg_size *= datatype_to_size[type];
  return arg_size;
}

// Function to check if a character is an operator
bool isOperator(char c) {
  // Returns true if the character is an operator
  return c == '+' || c == '-' || c == '*' || c == '/';
}

int64_t eval_int64_op(int64_t a, int64_t b, char op) {
  switch (op) {
  case '+':
    return a + b;
  case '-':
    return a - b;
  case '*':
    return a * b;
  case '/':
    return a / b;
  default:
    LOG(FATAL) << "Error: Invalid operator " << op;
  }
  return -1;
}

/**
 * @brief Replace the placeholder x in extend_str with dim_size and evaluate the
 * expression
 */
int64_t eval_extend(std::string extend_str, int64_t dim_size) {
  std::string dim_size_str = std::to_string(dim_size);
  size_t x_pos = extend_str.find("x");
  while (x_pos != std::string::npos) {
    if ((x_pos > 0) && std::isdigit(extend_str[x_pos - 1])) {
      extend_str.insert(x_pos, " * ");
      x_pos += 3;
    }
    extend_str.replace(x_pos, 1, dim_size_str);
    x_pos = extend_str.find("x");
  }
  // add space between operator and operand
  std::string result;
  for (size_t i = 0; i < extend_str.length(); ++i) {
    char currentChar = extend_str[i];
    if (isOperator(currentChar)) {
      // Add a space before the operator if it's not the first character
      if (i > 0 && !std::isspace(result.back())) {
        result += ' ';
      }
      result += currentChar;
      // Add a space after the operator if the next character is not a space or
      // the end of the string
      if (i < extend_str.length() - 1 && !std::isspace(extend_str[i + 1])) {
        result += ' ';
      }
    } else {
      result += currentChar;
    }
  }
  extend_str = result;
  VAIML_DEBUG_PRINT(2, "    extend_str after replacing x: ", extend_str);

  std::vector<int64_t> operands;
  std::deque<char> operators;
  std::string token;
  std::stringstream ss(extend_str);
  while (getline(ss, token, ' ')) {
    if (token.empty()) {
      continue;
    }
    bool isTokenDigits = std::all_of(token.begin(), token.end(), ::isdigit);
    if (isTokenDigits) {
      operands.push_back(std::stoi(token));
      if (operands.size() < 2) {
        continue;
      }
      int64_t b = operands.back();
      operands.pop_back();
      int64_t a = operands.back();
      operands.pop_back();
      operands.push_back(eval_int64_op(a, b, operators[0]));
      operators.pop_front();
    } else if (isOperator(token[0])) {
      operators.push_back(token[0]);
    } else {
      LOG(FATAL) << "Error: Invalid token " << token;
    }
  }
  if (operands.size() != 1) {
    LOG(FATAL) << "Error: Invalid expression " << extend_str;
  }
  VAIML_DEBUG_PRINT(2, "    extend_str evaluation result: ", operands[0]);
  return (operands[0]);
}

void dumpNodeTrace(std::map<std::string, NodeTrace>& map_partition_trace,
                   std::string output_dir) {
  fs::path all_node_trace_csv_path =
      fs::path(output_dir) / "graph_partition_trace.csv";

  std::ofstream ofs_all_node_trace_csv(all_node_trace_csv_path);
  if (ofs_all_node_trace_csv.is_open()) {
    ofs_all_node_trace_csv << "Node, Type, Subgraph/CustomOp, Status\n";
    for (const auto& [key, value] : map_partition_trace) {
      ofs_all_node_trace_csv << key << "," << value.op_type << ","
                             << value.subgraph << "," << value.status << "\n";
    }
    ofs_all_node_trace_csv.close();
  }
}

// Check if a node has producer
bool hasProducer(size_t node_idx, std::vector<size_t>& node_idx_vec,
                 const Graph& graph) {
  bool producer_found = false;
  const auto* cur_node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
  for (size_t ni : node_idx_vec) {
    const auto* node = VAIP_ORT_API(graph_get_node)(graph, ni);
    for (const auto* input : node_get_input_node_args(*node)) {
      if (!node_arg_exists(*input)) {
        continue;
      }
      std::string inputArgName = node_arg_get_name(*input);
      const Node* producer =
          VAIP_ORT_API(graph_producer_node)(graph, inputArgName);
      if ((producer != nullptr) && (VAIP_ORT_API(node_get_name)(*producer) ==
                                    VAIP_ORT_API(node_get_name)(*cur_node))) {
        producer_found = true;
        break;
      }
    }
    if (producer_found) {
      break;
    }
  }

  return producer_found;
}

// Check if a node has consumer
bool hasConsumer(size_t node_idx, std::vector<size_t>& node_idx_vec,
                 const Graph& graph) {
  bool consumer_found = false;
  const auto* cur_node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
  for (size_t ni : node_idx_vec) {
    const auto* node = VAIP_ORT_API(graph_get_node)(graph, ni);
    auto consumers = getConsumerNodes(graph, node);
    for (auto c : consumers) {
      auto consumerName = VAIP_ORT_API(node_get_name)(*c);
      auto nodeName = VAIP_ORT_API(node_get_name)(*cur_node);
      if (consumerName == nodeName) {
        consumer_found = true;
        break;
      }
    }
    if (consumer_found) {
      break;
    }
  }

  return consumer_found;
}

bool isIsolated(size_t idx, std::vector<size_t>& node_idx_vec,
                const Graph& graph) {
  bool res = true;
  const auto* cur_node = VAIP_ORT_API(graph_get_node)(graph, idx);
  for (size_t i : node_idx_vec) {
    const auto* node = VAIP_ORT_API(graph_get_node)(graph, i);
    for (const auto* input : node_get_input_node_args(*node)) {
      if (!node_arg_exists(*input)) {
        continue;
      }
      std::string inputArgName = node_arg_get_name(*input);
      const Node* producer =
          VAIP_ORT_API(graph_producer_node)(graph, inputArgName);
      if ((producer != nullptr) && (VAIP_ORT_API(node_get_name)(*producer) ==
                                    VAIP_ORT_API(node_get_name)(*cur_node))) {
        res = false;
        break;
      }
    }
    if (!res) {
      break;
    }
    auto consumers = getConsumerNodes(graph, node);
    for (auto c : consumers) {
      auto consumerName = VAIP_ORT_API(node_get_name)(*c);
      auto nodeName = VAIP_ORT_API(node_get_name)(*cur_node);
      if (consumerName == nodeName) {
        res = false;
        break;
      }
    }
    if (!res) {
      break;
    }
  }
  return res;
}

void dumpGraphNodes(Graph& graph, std::string json_file_name) {
  auto nodes = graph_nodes(graph);
  nlohmann::json graph_nodes_json;

  for (auto* node : nodes) {
    if (node == nullptr)
      continue;
    auto node_name = VAIP_ORT_API(node_get_name)(*node);
    auto node_id = VAIP_ORT_API(node_get_index)(*node);

    // dump inputs
    nlohmann::json inputs_json;
    auto inputs = node_get_input_node_args(*node);
    for (auto* input : inputs) {
      if ((input == nullptr) || !node_arg_exists(*input))
        continue;
      auto input_name = node_arg_get_name(*input);
      auto shape_ptr = node_arg_get_shape_i64(*input);
      if (shape_ptr == nullptr)
        continue;
      auto shape = *(shape_ptr.get());
      auto input_shape_str = shapeToString(shape);
      nlohmann::json arg_json = {{"input_name:", input_name},
                                 {"input_dimension:", input_shape_str}};
      inputs_json.push_back(arg_json);
    }

    // dump outputs
    nlohmann::json outputs_json;
    auto outputs = node_get_output_node_args(*node);
    for (auto* output : outputs) {
      if ((output == nullptr) || !node_arg_exists(*output))
        continue;
      auto output_name = node_arg_get_name(*output);
      auto shape_ptr = node_arg_get_shape_i64(*output);
      if (shape_ptr == nullptr)
        continue;
      auto shape = *(shape_ptr.get());
      auto output_shape_str = shapeToString(shape);
      nlohmann::json arg_json = {{"output_name:", output_name},
                                 {"output_dimension:", output_shape_str}};
      outputs_json.push_back(arg_json);
    }
    nlohmann::json node_json = {{"node_name:", node_name},
                                {"node_id:", node_id},
                                {"node_inputs:", inputs_json},
                                {"node_outputs:", outputs_json}};
    graph_nodes_json.push_back(node_json);
  }

  std::ofstream graph_nodes_file(json_file_name, std::ios::binary);
  graph_nodes_file << graph_nodes_json.dump(
      4); // Serialize with 4 spaces indentation
  graph_nodes_file.close();
}

} // namespace vaip_vaiml
