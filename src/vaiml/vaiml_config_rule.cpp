#include "vaip/vaip.hpp"

#include "vaip/vaiml/vaiml_config_rule.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
namespace fs = std::filesystem;

namespace vaip_vaiml {
using namespace vaip_core;

void checkOverlayJson(VaimlConfigOptions& options) {
  bool overlayJsonFound = false;

  auto overlayJsonPath = fs::path(options.overlay_json);
  VAIML_DEBUG_PRINT(2, "DEBUG: Check if ", options.overlay_json, " exists.")
  if (fs::exists(overlayJsonPath)) {
    VAIML_DEBUG_PRINT(2, "DEBUG: Found ", options.overlay_json)
    overlayJsonFound = true;
  }
  if (!overlayJsonFound) {
    overlayJsonPath = options.custom_ops_install_path / options.overlay_json;
    VAIML_DEBUG_PRINT(2, "DEBUG: Check if ", overlayJsonPath.string(),
                      " exists.")
    if (fs::exists(overlayJsonPath)) {
      VAIML_DEBUG_PRINT(2, "DEBUG: Found ", overlayJsonPath.string())
      options.overlay_json = overlayJsonPath.string();
      overlayJsonFound = true;
    }
  }
  if (overlayJsonFound) {
    options.overlay_json_signature = xir::get_md5_of_file(options.overlay_json);
  } else {
    VAIML_DEBUG_PRINT(1, "WARNING: Cannot find overlay_json ",
                      options.overlay_json,
                      " specified in the config file. Skipped.")
  }
}

void dumpConstants(Graph& graph, VaimlConfigOptions& options,
                   std::string& modelPathStr) {
  VAIML_DEBUG_PRINT(2, "Running vaiml_config_rule::dumpConstants...");
  std::ofstream* cnts_file_ptr = nullptr;
  std::ofstream cnts_file;

  fs::path fullCntsFileName =
      fs::path(options.vaiml_unarchive_path) / options.constants_file_name;
  fs::path constInfoFile =
      fs::path(options.vaiml_unarchive_path) / "custom_op_constinfo.json";
  fs::path contextJsonFile =
      fs::path(options.vaiml_unarchive_path) / "context.json";
  if (fs::exists(constInfoFile) && fs::exists(contextJsonFile)) {
    VAIML_DEBUG_PRINT(2, constInfoFile.string(),
                      " exists already. Skip dumping constants.");
    return;
  }

  if (!fs::exists(fs::path(options.vaiml_unarchive_path))) {
    fs::create_directory(fs::path(options.vaiml_unarchive_path));
  }

  if (!fs::exists(fullCntsFileName)) {
    cnts_file.open(fullCntsFileName, std::ios::binary);
    // Check if the file is open
    if (!cnts_file.is_open()) {
      throw std::runtime_error("Failed to open the constant file for writing.");
    }
    cnts_file_ptr = &cnts_file;
  }

  auto modelPath = fs::path(modelPathStr);
  auto parentPath = modelPath.parent_path();
  auto currPath = fs::current_path();
  if (!parentPath.empty()) {
    fs::current_path(
        parentPath); // TODO: this is a workaround, when we move to ORT 1.20,
                     // changing dir might not be required, revisit!
  }
  size_t cnt_offset = 0;
  auto all_constants = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph);
  try {
    for (const auto& constant : all_constants) {
      // constant.first = "siglip-so400m/" + constant.first;
      ConstantInfo cnt_info;
      cnt_info.offset = cnt_offset;
      auto tensor_proto_ptr = constant.second;
      if (tensor_proto_ptr == nullptr) {
        continue;
      }
      auto& tensor_proto = *tensor_proto_ptr;
      cnt_info.type = VAIP_ORT_API(tensor_proto_data_type)(tensor_proto);
      auto tensor_proto_shape = tensor_proto_get_shape(tensor_proto);

      for (auto s : tensor_proto_shape) {
        cnt_info.shape.push_back((int)s);
      }
      if (cnt_info.shape.empty()) {
        // use [1] to mimic scalar
        cnt_info.shape.push_back(1);
      }
      // for some unknown reason, some shape are <=0.
      for (auto& s : cnt_info.shape) {
        if (s <= 0) {
          s = 1;
        }
      }
      auto raw_values = tensor_proto_as_raw(graph, tensor_proto);
      if (cnts_file_ptr != nullptr) {
        cnts_file_ptr->write(raw_values.data(), raw_values.size());
      }
      cnt_info.size = raw_values.size();
      cnt_info.is_scalar = true;
      for (auto dim : cnt_info.shape) {
        if (dim != 1) {
          cnt_info.is_scalar = false;
          break;
        }
      }

      if ((raw_values.size() == 4) && (cnt_info.is_scalar)) { // float32 scalar
        float* f = (float*)raw_values.data();
        cnt_info.value = *f;
      }
      options.constants_map[constant.first] = cnt_info;
      VAIML_DEBUG_PRINT(3, "    Constant ", constant.first, " has ",
                        options.constants_map[constant.first].size,
                        " bytes and saved to offset ",
                        options.constants_map[constant.first].offset);
      cnt_offset += raw_values.size();
    }
  } catch (...) {
    fs::current_path(
        currPath); // first return back to the current path, then throw error
    throw std::runtime_error("Failed to dump constants.");
  }
  fs::current_path(currPath);
  if (cnts_file_ptr != nullptr) {
    cnts_file_ptr->close();
  }
}

VaimlConfigRule::VaimlConfigRule(IPass& pass) : pass_{pass} {}

bool VaimlConfigRule::apply(Graph& graph, VaimlConfigOptions& options) {
  auto res_ok = true;
  for (auto& action : actions_) {
    res_ok = res_ok && action(this, graph, options);
  }
  return res_ok;
}

// TODO: This function is common to unified flow and custom flow. Maybe move it
// to a common place.
VaimlConfigRule& VaimlConfigRule::set_config_options() {
  this->actions_.push_back([=](VaimlConfigRule* self, Graph& graph,
                               VaimlConfigOptions& options) -> bool {
    options.setInstallPaths();
    options.setVaimlConfigOptions(self->pass_);
    checkOverlayJson(options);
    if (options.compile_flow_int == CompileFlow::CUSTOM_OP ||
        options.compile_flow_int == CompileFlow::HYBRID) {
      auto& modelPathString =
          pass_.get_context()->get_config_proto().onnx_path();
      std::string modelPath(modelPathString.data());
      dumpConstants(graph, options, modelPath);
    }
    return true;
  });
  return *this;
}

VaimlConfigRule& VaimlConfigRule::get_model_type() {
  this->actions_.push_back([=](VaimlConfigRule* self, Graph& graph,
                               VaimlConfigOptions& options) -> bool {
    VAIML_DEBUG_PRINT(2, "DEBUG: Auto detect model type...");
    const auto& node_indices = graph_get_node_in_topoligical_order(graph);

    std::unordered_map<std::string, int> graph_op_summary;
    std::unordered_map<std::string, int> graph_key_op_summary;
    std::unordered_map<std::string, float> key_ops;
    std::string conv_datatype = "";
    std::string matmul_datatype = "";
    std::string quantization_datatype = "no";
    int matmul_bits = 0;

    nlohmann::json key_op_data = {{"op_types",
                                   {{{"op_type", "MatMul"}, {"value", 0.5}},
                                    {{"op_type", "Conv"}, {"value", 0.5}},
                                    {{"op_type", "Gemm"}, {"value", 0.5}}}}};

    for (auto& op_type : key_op_data["op_types"]) {
      std::string name = op_type["op_type"];
      key_ops[name] = op_type["value"].get<float>();
      graph_key_op_summary[name] = 0;
    }

    for (size_t node_idx : node_indices) {
      const auto* node(VAIP_ORT_API(graph_get_node)(graph, node_idx));
      std::string node_type = VAIP_ORT_API(node_op_type)(*node);
      // Get node output type
      if (node_type == "Conv") {
        conv_datatype =
            data_type_to_string(node_get_output_element_type(*node));
      } else if (node_type == "QuantizeLinear") {
        auto output_data_type =
            data_type_to_string(node_get_output_element_type(*node));
        quantization_datatype = output_data_type;
      } else if (node_type == "MatMulNBits") {
        if (node_has_attr(*node, "bits")) {
          matmul_bits = (int)node_get_attr_int(*node, "bits");
        }
      } else if ((node_type == "MatMul") || (node_type == "Gemm")) {
        matmul_datatype =
            data_type_to_string(node_get_output_element_type(*node));
      }

      // Build op summary, key op summary
      if (graph_op_summary.find(node_type) == graph_op_summary.end()) {
        graph_op_summary[node_type] = 1;
      } else {
        graph_op_summary[node_type] = graph_op_summary[node_type] + 1;
      }
      if (key_ops.find(node_type) != key_ops.end()) {
        graph_key_op_summary[node_type] = graph_key_op_summary[node_type] + 1;
      }
    }

    VAIML_DEBUG_PRINT(2, "Model Summary:");
    if (graph_op_summary.find("MatMulNBits") != graph_op_summary.end()) {
      quantization_datatype = std::to_string(matmul_bits) + "-bit weight";
    }

    VAIML_DEBUG_PRINT(2, "    Quantization: ", quantization_datatype);
    VAIML_DEBUG_PRINT(2, "    OpType: Count");
    for (const auto& [key, value] : graph_op_summary) {
      VAIML_DEBUG_PRINT(2, "        ", key, ": ", value);
    }
    int total_key_ops = 0;
    VAIML_DEBUG_PRINT(2, "    Key OpType: Count");
    for (const auto& [key, value] : graph_key_op_summary) {
      VAIML_DEBUG_PRINT(2, "        ", key, ": ", value);
      total_key_ops = total_key_ops + value;
    }

    float conv_ratio =
        (float)graph_key_op_summary["Conv"] / (float)total_key_ops;
    float matmul_ratio =
        (float)(graph_key_op_summary["MatMul"] + graph_key_op_summary["Gemm"]) /
        (float)total_key_ops;

    if (conv_ratio >= key_ops["Conv"]) {
      if (graph_op_summary.find("QuantizeLinear") == graph_op_summary.end()) {
        if (conv_datatype == "float16") {
          VAIML_DEBUG_PRINT(2,
                            "DEBUG: Set enable_f16_to_bf16_conversion to true");
          options.enable_f16_to_bf16_conversion = true;
        } else if (conv_datatype == "float32") {
          VAIML_DEBUG_PRINT(2,
                            "DEBUG: Set enable_f32_to_bf16_conversion to true");
          options.enable_f32_to_bf16_conversion = true;
        } else {
          VAIML_DEBUG_PRINT(1, "ERROR: ", conv_datatype, " not supported");
        }
      }
    } else if (matmul_ratio >= key_ops["MatMul"]) {
      if (matmul_datatype == "float16") {
        VAIML_DEBUG_PRINT(2,
                          "DEBUG: Set enable_f16_to_bf16_conversion to true");
        options.enable_f16_to_bf16_conversion = true;
      } else if (matmul_datatype == "float32") {
        VAIML_DEBUG_PRINT(2,
                          "DEBUG: Set enable_f32_to_bf16_conversion to true");
        options.enable_f32_to_bf16_conversion = true;
      } else {
        VAIML_DEBUG_PRINT(1, "ERROR: ", matmul_datatype, " not supported");
      }
    } else if (matmul_ratio >= key_ops["Gemm"]) {
      if (matmul_datatype == "float16") {
        VAIML_DEBUG_PRINT(2,
                          "DEBUG: Set enable_f16_to_bf16_conversion to true");
        options.enable_f16_to_bf16_conversion = true;
      } else if (matmul_datatype == "float32") {
        VAIML_DEBUG_PRINT(2,
                          "DEBUG: Set enable_f32_to_bf16_conversion to true");
        options.enable_f32_to_bf16_conversion = true;
      } else {
        VAIML_DEBUG_PRINT(1, "ERROR: ", matmul_datatype, " not supported");
      }
    }

    if (options.enable_f16_to_bf16_conversion) {
      VAIML_INFO_PRINT("Enable FP16 to BF16 conversion");
    }
    if (options.enable_f32_to_bf16_conversion) {
      VAIML_INFO_PRINT("Enable FP32 to BF16 conversion");
    }

    if ((options.initMValues.size() == 1) && (options.initMValues[0] == -1)) {
      options.initMValues.clear();
    }

    // Resolve options.drop_orphan_nodes value
    if (std::find(options.experiment_features.begin(),
                  options.experiment_features.end(),
                  "KeepOrphanNodes") != options.experiment_features.end()) {
      options.drop_orphan_nodes = false;
    } else if (std::find(options.experiment_features.begin(),
                         options.experiment_features.end(),
                         "DropOrphanNodes") !=
               options.experiment_features.end()) {
      options.drop_orphan_nodes = true;
    } else if (options.enable_f32_to_bf16_conversion ||
               options.enable_f16_to_bf16_conversion) {
      // enable DropOrphanNodes when KeepOrphanNodes is not set in the config
      // file and it is a bf16 model
      options.drop_orphan_nodes = true;
    }
    return true;
  });
  return *this;
}

VaimlConfigRule& VaimlConfigRule::print_options() {
  this->actions_.push_back([=](VaimlConfigRule* self, Graph& graph,
                               VaimlConfigOptions& options) -> bool {
    options.printOptions();
    return true;
  });
  return *this;
}

void applyVaimlConfigRule(IPass& pass, Graph& graph,
                          VaimlConfigOptions& options) {
  std::unique_ptr<VaimlConfigRule> config_rule =
      std::make_unique<VaimlConfigRule>(pass);
  (config_rule->set_config_options());
  (config_rule->get_model_type());
  (config_rule->print_options());

  auto ok = config_rule->apply(graph, options);
  if (!ok) {
    LOG(ERROR) << "Failed to apply VaimlConfigRule";
  } else {
    VAIML_DEBUG_PRINT(2, "Successfully applied VaimlConfigRule");
  }
}

} // namespace vaip_vaiml