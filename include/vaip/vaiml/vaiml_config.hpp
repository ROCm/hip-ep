#pragma once

#include "vaip/pass.hpp"
#include "vaip/vaip.hpp"
#include "vaip/xir_headers.hpp"
#include "vitis/ai/env_config.hpp"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <string>
#include <vector>

#include "vaiml_logging.hpp"

namespace vaip_vaiml {
namespace fs = std::filesystem;
using namespace vaip_core;

enum CompileFlow { DEFAULT = 0, CUSTOM_OP = 1, HYBRID = 2 };

struct ConstantInfo {
  size_t offset;
  size_t size;
  int type;
  std::vector<int> shape;
  bool is_scalar;
  float value; // only valid when the constant is scalar, that is
};

struct VaimlConfigOptions {
  // --------------------------------------------------------------------------
  // Members
  // --------------------------------------------------------------------------
  // Add options that can be configued in vitisai_config.json here
  std::unordered_set<std::string> vaiml_supported_ops;
  bool debug = false;
  std::string vaiml_unarchive_path = ".";
  std::string onnx_model_name = "";
#ifdef _WIN32
  std::string vaiml_model_path = "C:\\amd\\voe\\binary-modules\\vaiml_par_0";
  std::string overlay_json = "bmm_bias\\overlay_4x4.json";
#else
  std::string vaiml_model_path = "./vaiml_par_0";
  std::string overlay_json = "bmm_bias/overlay_4x4.json";
#endif
  std::string user_subgraph_config;
  std::string llm_ops_config;
  bool override_batch_size = false;
  bool enable_batch = false;
  std::string vaiml_archive_file;
  bool run_archive = false;
  std::string device_name = "stx"; // PassVaimlConfigProto 5
  std::vector<std::string> supported_devices = {
      "v70", "phx", "ryzen-ai", "stx", "vek280", "ve2", "cpu"};
  std::string model_type = "default"; // PassVaimlConfigProto 6
  std::string compile_flow = "default";
  vaip_vaiml::CompileFlow compile_flow_int = vaip_vaiml::CompileFlow::DEFAULT;

  int max_num_inputs = -1;
  int max_num_outputs = -1;
  int max_num_partitions = 7;
  std::string output_type = "aie-exe";
  std::string config_filename = "";
  bool transformer_mode = false;
  bool ai_analyzer_profiling = false;
  bool ai_analyzer_visualization = false;
  bool fast_partition_swap = true;
  bool no_failsafe = false;
  std::string priority = "normal";
  std::string aie_unsupported_ops_override;
  bool force = false;
  std::string init_m_values = "-1";
  std::vector<int64_t> initMValues;
  std::vector<std::string> custom_op_repos;
  std::string overlay_json_signature = "";
  std::string tilingEngine = "microkernel";
  fs::path voe_install_path;
  std::string vaip_commit;
  fs::path flexmlrt_install_path;
  fs::path custom_ops_install_path;
  fs::path mllib_root;
  std::vector<std::string> supported_custom_ops = {
      "bmm_bias", "bmm_b1", "bmm_4d", "elwadd", "elwmul"};
  std::vector<std::string> experiment_features;
  bool newPartition = false;
  bool enable_f16_to_bf16_conversion = false;
  bool enable_f32_to_bf16_conversion = false;
  std::vector<std::string> exclude_subgraphs;
  std::vector<std::string> include_subgraphs;
  // subgraph with gops% less than this threshold will be dropped
  int32_t threshold_gops_percent = 2;
  bool keep_outputs = false;
  bool run_vaiml = false;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::vector<std::string>>>
      subgraph_map;
  std::vector<std::string> fusableOps;
  bool single_archive = false; // change default to false now to disable
                               // archive
  // removable_cluster_node_ratio is the ration between the number of nodes in
  // an isolated cluster and the total number of nodes in the graph
  float removable_cluster_node_ratio = 0;
  // minimum number of customops in a subgraph, -1 no limit
  int min_subgraph_size = -1;
  std::string constants_file_name = "wts.bin";
  std::string const_struct_file_name = "custom_op_constinfo.json";
  std::string partition_strategy = "op_count";
  // set default runner type
  std::string runner_type = "hw";
  std::unordered_map<std::string, ConstantInfo> constants_map;
  bool gen_const_json = true;
  int device_batch_size = 1;
  bool enable_local_context = true;
  bool drop_orphan_nodes = false;
  bool enable_update_initializer = false;
  bool fatal_error_on_exception = false;
  bool fatal_error = false;
  bool partition_only = false;
  bool keep_edge_relayout_ops = false;
  std::vector<std::string> edge_relayout_ops = {
      "Reshape", "Flatten", "ScatterND", "Squeeze", "Split", "Unsqueeze"};
  fs::path onnx_external_data_dir;
  std::string microkernel_operator;
  std::string microkernel_overlay_array = "4x4";

  // --------------------------------------------------------------------------
  // Methods
  // --------------------------------------------------------------------------
  bool resolve_install_path(char* env, std::string& install_path) {
    bool result = false;
    install_path = std::string(env);
    // trim trailing \r and \n characters to work with cygwin
    install_path.erase(
        std::find_if(install_path.rbegin(), install_path.rend(),
                     [](unsigned char ch) { return ch != '\r' && ch != '\n'; })
            .base(),
        install_path.end());

    fs::path custom_ops_path;

#ifdef _WIN32
    custom_ops_path = fs::path(install_path) / "Lib" / "site-packages" / "voe" /
                      "vaiml-custom-ops";
#else
    custom_ops_path = fs::path(install_path) / "lib" / "python3.10" /
                      "site-packages" / "voe" / "vaiml-custom-ops";
#endif

    if (fs::is_directory(custom_ops_path)) {
      result = true;
    }

    return result;
  }

  void setInstallPaths() {
#if !defined(_WIN32) && defined(__aarch64_)
    const char* envVarValue = std::getenv("LD_LIBRARY_PATH");
    if (envVarValue == nullptr) {
      LOG(WARNING)
          << "ERROR::  LD_LIBRARY_PATH is not exported to get VOE library.";
      return;
    }
    std::stringstream libPaths(envVarValue);
    std::string lib_path;
    while (getline(libPaths, lib_path, ':')) {
      if ((lib_path.size() > 0) &&
          (lib_path.find("site-packages/voe") != std::string::npos)) {
        if (lib_path.back() == '/') {
          voe_install_path = fs::path(lib_path).parent_path().parent_path();
        } else {
          voe_install_path = fs::path(lib_path).parent_path();
        }
      }
    }
#else
    // x86 Linux and Windows
    auto conda_prefix = getenv("CONDA_PREFIX");
    auto virtual_prefix = getenv("VIRTUAL_ENV");
    std::string install_path = "";
    bool install_path_resolved = false;
    if (virtual_prefix == nullptr && conda_prefix == nullptr) {
      LOG(WARNING) << "WARNING:: CONDA_PREFIX or VIRTUAL_ENV can not be found "
                      "in current environment. Please run the test either in "
                      "conda or python virtual env.";
    }

    if (!install_path_resolved && conda_prefix != nullptr) {
      install_path_resolved = resolve_install_path(conda_prefix, install_path);
    }

    if (!install_path_resolved && virtual_prefix != nullptr) {
      install_path_resolved =
          resolve_install_path(virtual_prefix, install_path);
    }

    if (!install_path_resolved) {
      LOG(WARNING)
          << "WARNING: Cannot find vaiml-custom-ops installation. Please check "
             "if RyzenAI software is installed properly.";
    }

#  ifdef _WIN32
    voe_install_path = fs::path(install_path) / "Lib" / "site-packages" / "voe";
    flexmlrt_install_path =
        fs::path(install_path) / "Lib" / "site-packages" / "flexmlrt";

    VAIML_DEBUG_PRINT(1, "FlexMLRT installation: ", flexmlrt_install_path);
#  else
    voe_install_path =
        fs::path(install_path) / "lib" / "python3.10" / "site-packages" / "voe";
    mllib_root = fs::path(install_path) / "lib" / "python3.10" /
                 "site-packages" / "vitis_mllib";
#  endif
#endif
    // Resolve vaiml-custom-ops installation path
    custom_ops_install_path = voe_install_path / "vaiml-custom-ops";

    VAIML_DEBUG_PRINT(1, "voe installation: ", voe_install_path.string());
    VAIML_DEBUG_PRINT(1,
                      "vaiml-custom-ops: ", custom_ops_install_path.string());
    VAIML_DEBUG_PRINT(1, "mllib installation: ", mllib_root.string());
  }

  void setVaimlConfigOptions(IPass& iPass) {
    auto& vaiml_proto = iPass.get_pass_proto().vaiml_config();
    auto& config_proto = iPass.get_context()->get_config_proto();
    auto& code_versions = config_proto.version();

    for (const auto& iter : code_versions.version_infos()) {
      if (iter.package_name() == "vaip") {
        vaip_commit = iter.commit();
        LOG(INFO) << "VAIP commit: " << vaip_commit;
      }
    }

    // get config filename and parse json
    // change to use single json file:
    // config_filename = vaiml_proto.config_file();
    auto all_session_options = config_proto.provider_options();

    auto it = all_session_options.find("config_file");
    if (it != all_session_options.end() && !it->second.empty()) {
      config_filename = it->second;
    } else {
      LOG(FATAL) << "Error: Key 'config_file' not found in session options ";
      return;
    }

    auto& modelPathString = config_proto.onnx_path();
    std::string modelPath(modelPathString.data());
    std::string origin_cache_key =
        iPass.get_context()->get_config_proto().cache_key();
    onnx_model_name = !fs::exists(fs::path(modelPath))
                          ? origin_cache_key
                          : fs::path(modelPath).stem().string();

    std::ifstream f(config_filename);
    if (!f.is_open()) {
      LOG(FATAL) << "Error: Failed to open file: " << config_filename;
      return;
    }

    // get AI Analyzer user settings
    // NOTE: visualization and profiling options can be set in config_file as
    // well, but the settings in provider_options will take precedence
    if (config_proto.has_ai_analyzer_profiling()) {
      VAIML_DEBUG_PRINT(
          1, "ai_analyzer_profiling: ", config_proto.ai_analyzer_profiling());
      ai_analyzer_profiling = config_proto.ai_analyzer_profiling();
      ai_analyzer_visualization = true;
    }
    // DL Analyzer creating dpu_timestamp_info.json, or not.
    if (!config_proto.has_ai_analyzer_profiling() &&
        config_proto.has_ai_analyzer_visualization()) {
      VAIML_DEBUG_PRINT(1, "ai_analyzer_visualization: ",
                        config_proto.ai_analyzer_visualization());
      ai_analyzer_visualization = config_proto.ai_analyzer_visualization();
    }

    // Qos support
    if (config_proto.has_priority()) {
      VAIML_DEBUG_PRINT(1, "priority: ", config_proto.priority());
      priority = config_proto.priority();
    }

    // New No failsafe partition flow
    if (vaiml_proto.has_no_failsafe()) {
      no_failsafe = vaiml_proto.no_failsafe();
    }

    // Get rest options from json

    if (vaiml_proto.has_vaiml_unarchive_path()) {
      if (vaiml_proto.vaiml_unarchive_path().find("MODEL_PATH") !=
          std::string::npos) {
        std::string custom_vaiml_unarchive_path =
            vaiml_proto.vaiml_unarchive_path();
        // 10 is length of string "MODEL_PATH"
        custom_vaiml_unarchive_path.replace(
            0, 10, (fs::path(modelPath).parent_path()).string());

        vaiml_unarchive_path =
            (fs::path(custom_vaiml_unarchive_path) / onnx_model_name).string();
        VAIML_DEBUG_PRINT(1,
                          "INFO: [VAIP-VAIML] vaiml_unarchive_path is set to ",
                          vaiml_unarchive_path);

      } else {
        vaiml_unarchive_path =
            (fs::path(vaiml_proto.vaiml_unarchive_path()) / onnx_model_name)
                .string();
      }

    } else {

      fs::path default_cache_dir;
      fs::path tmp_dir;
#ifdef _WIN32
      tmp_dir = fs::path("C:\\temp");
#else
      tmp_dir = fs::path("/tmp");
#endif
      auto user_name = std::string();
      if (std::getenv("USERNAME")) {
        user_name = std::getenv("USERNAME");
      } else if (std::getenv("USER")) {
        user_name = std::getenv("USER");
      }
      default_cache_dir = tmp_dir / user_name / "vaip" / ".cache";

      std::string origin_cache_dir =
          iPass.get_context()->get_config_proto().cache_dir();
      std::string origin_cache_key =
          iPass.get_context()->get_config_proto().cache_key();
      if (fs::equivalent(default_cache_dir, fs::path(origin_cache_dir))) {
        vaiml_unarchive_path =
            (fs::path(vaiml_unarchive_path) / onnx_model_name).string();
      } else {
        onnx_model_name = origin_cache_key;
        vaiml_unarchive_path =
            (fs::path(origin_cache_dir) / fs::path(origin_cache_key)).string();
      }
    }

    if (vaiml_proto.has_vaiml_archive_file()) {
      vaiml_archive_file = vaiml_proto.vaiml_archive_file();
    } else {
      vaiml_archive_file = onnx_model_name + ".vai";
    }

    if (vaiml_proto.has_run_archive()) {
      run_archive = vaiml_proto.run_archive();
    }

    if (run_archive) {
      vaiml_unarchive_path = vaiml_proto.vaiml_unarchive_path();
    }

    if (vaiml_proto.has_single_archive()) {
      single_archive = vaiml_proto.single_archive();
    }

    if (vaiml_proto.has_removable_cluster_node_ratio()) {
      removable_cluster_node_ratio = vaiml_proto.removable_cluster_node_ratio();
    }

    if (vaiml_proto.has_min_subgraph_size()) {
      min_subgraph_size = vaiml_proto.min_subgraph_size();
    }

    if (vaiml_proto.has_user_subgraph_config()) {
      user_subgraph_config = vaiml_proto.user_subgraph_config();
    }

    if (vaiml_proto.has_llm_ops_config()) {
      llm_ops_config = vaiml_proto.llm_ops_config();
    }

    if (vaiml_proto.has_override_batch_size()) {
      override_batch_size = vaiml_proto.override_batch_size();
    }

    if (vaiml_proto.has_enable_batch()) {
      enable_batch = vaiml_proto.enable_batch();
    }

    if (vaiml_proto.has_device_name()) {
      device_name = vaiml_proto.device_name();
      std::cout << "INFO: [VAIP-VAIML] device_name option is being deprecated. "
                   "Please use device option instead."
                << std::endl;
    }

    if (vaiml_proto.has_device()) {
      device_name = vaiml_proto.device();
    }

    if (vaiml_proto.has_runner_type()) {
      runner_type = vaiml_proto.runner_type();
    }

    if (std::find(supported_devices.begin(), supported_devices.end(),
                  device_name) == supported_devices.end()) {
      LOG(FATAL) << "Unsupported device " << device_name
                 << ". Supported device are "
                 << stringVectorToString(supported_devices);
    }

    // Will obsolete model_type once tests are updated with compile_flow
    if (vaiml_proto.has_model_type()) {
      model_type = vaiml_proto.model_type();
    }

    if (model_type == "default") {
      compile_flow = "default";
    } else if (model_type == "transformer") {
      compile_flow = "custom_op";
    } else if (model_type == "vit") {
      compile_flow = "hybrid";
    } else {
      LOG(FATAL) << "Supported model_type: default, transformer, vit";
    }

    if (vaiml_proto.has_compile_flow()) {
      compile_flow = vaiml_proto.compile_flow();
    }

    if (compile_flow == "default") {
      compile_flow_int = CompileFlow::DEFAULT;
    } else if (compile_flow == "custom_op") {
      compile_flow_int = CompileFlow::CUSTOM_OP;
      // Set default max_num_partitions. User setting will override this.
      max_num_partitions = -1;
    } else if (compile_flow == "hybrid") {
      compile_flow_int = CompileFlow::HYBRID;
      // Set default max_num_partitions. User setting will override this.
      max_num_partitions = -1;
    } else {
      LOG(FATAL) << "Supported compile_flow: default, custom_op, hybrid";
    }

    if (vaiml_proto.has_force()) {
      force = vaiml_proto.force();
    }

    if (vaiml_proto.has_output_type()) {
      output_type = vaiml_proto.output_type();
    }

    if (vaiml_proto.has_debug()) {
      debug = vaiml_proto.debug();
    }

    if (vaiml_proto.has_max_num_inputs()) {
      max_num_inputs = vaiml_proto.max_num_inputs();
    }

    if (vaiml_proto.has_max_num_outputs()) {
      max_num_outputs = vaiml_proto.max_num_outputs();
    }

    if (vaiml_proto.has_max_num_partitions()) {
      max_num_partitions = vaiml_proto.max_num_partitions();
    }

    if (vaiml_proto.has_aie_unsupported_ops_override()) {
      aie_unsupported_ops_override = vaiml_proto.aie_unsupported_ops_override();
    }

    if (vaiml_proto.has_init_m_values()) {
      init_m_values = vaiml_proto.init_m_values();
    }

    if (vaiml_proto.has_custom_op_repo()) {
      custom_op_repos.push_back(vaiml_proto.custom_op_repo());
    }
    if (fs::is_directory(custom_ops_install_path)) {
      custom_op_repos.push_back(custom_ops_install_path.string());
    } else {
      VAIML_DEBUG_PRINT(1, "WARNING: ", custom_ops_install_path, " is invalid.")
    }

    if (vaiml_proto.supported_custom_ops_size() > 0) {
      supported_custom_ops =
          std::vector<std::string>(vaiml_proto.supported_custom_ops().begin(),
                                   vaiml_proto.supported_custom_ops().end());
    }

    if (vaiml_proto.has_partition_strategy()) {
      partition_strategy = vaiml_proto.partition_strategy();
    }

    if (vaiml_proto.has_overlay_json()) {
      overlay_json = vaiml_proto.overlay_json();
      VAIML_INFO_PRINT("Set default overlay JSON to user specified ",
                       overlay_json);
    }

    if (vaiml_proto.has_enable_f32_to_bf16_conversion()) {
      enable_f32_to_bf16_conversion =
          vaiml_proto.enable_f32_to_bf16_conversion();
    }

    if (vaiml_proto.has_enable_f16_to_bf16_conversion()) {
      enable_f16_to_bf16_conversion =
          vaiml_proto.enable_f16_to_bf16_conversion();
    }

    if (vaiml_proto.experiment_features_size() > 0) {
      experiment_features =
          std::vector<std::string>(vaiml_proto.experiment_features().begin(),
                                   vaiml_proto.experiment_features().end());
    }

    if (vaiml_proto.exclude_subgraphs_size() > 0) {
      exclude_subgraphs =
          std::vector<std::string>(vaiml_proto.exclude_subgraphs().begin(),
                                   vaiml_proto.exclude_subgraphs().end());
    }

    if (vaiml_proto.include_subgraphs_size() > 0) {
      include_subgraphs =
          std::vector<std::string>(vaiml_proto.include_subgraphs().begin(),
                                   vaiml_proto.include_subgraphs().end());
    }

    if (vaiml_proto.has_threshold_gops_percent()) {
      threshold_gops_percent = vaiml_proto.threshold_gops_percent();
    }

    if (vaiml_proto.has_fast_partition_swap()) {
      fast_partition_swap = vaiml_proto.fast_partition_swap();
    }

    if (vaiml_proto.has_keep_outputs()) {
      keep_outputs = vaiml_proto.keep_outputs();
    }

    if (vaiml_proto.has_device_batch_size()) {
      device_batch_size = vaiml_proto.device_batch_size();
    }

    if (vaiml_proto.has_enable_local_context()) {
      enable_local_context = vaiml_proto.enable_local_context();
    }

    if (vaiml_proto.has_enable_update_initializer()) {
      enable_update_initializer = vaiml_proto.enable_update_initializer();
    }

    if (!include_subgraphs.empty()) {
      enable_local_context = false;
    }

    if (vaiml_proto.has_fatal_error_on_exception()) {
      fatal_error_on_exception = vaiml_proto.fatal_error_on_exception();
    }

    // -----------------------------------------------------------------------
    // Resolve final options values
    //------------------------------------------------------------------------
    // Set member variables based on options from the config file
#ifndef _WIN32
    run_vaiml = true;
#endif
    if (std::find(experiment_features.begin(), experiment_features.end(),
                  "CompileOnWindows") != experiment_features.end()) {
      run_vaiml = true;
    }

    if (std::find(experiment_features.begin(), experiment_features.end(),
                  "NewPartition") != experiment_features.end()) {
      newPartition = true;
    }

    // Set initMValues from init_m_values string
    // Use stringstream to split the line by commas
    std::stringstream ss(init_m_values);
    std::string token;
    while (std::getline(ss, token, ',')) {
      initMValues.push_back(std::stoi(token));
    }

    if (std::find(experiment_features.begin(), experiment_features.end(),
                  "PartitionOnly") != experiment_features.end()) {
      partition_only = true;
    }

    // Parse EdgeRelayoutOps:op0:op1:op2...
    for (const auto& ef : experiment_features) {
      std::vector<std::string> tokens;
      std::stringstream ss(ef);
      std::string token;
      bool is_edge_relayout_ops = false;
      while (std::getline(ss, token, ':')) {
        tokens.push_back(token);
      }
      if (tokens.size() > 0) {
        if (tokens[0] == "EdgeRelayoutOps") {
          is_edge_relayout_ops = true;
        }
      }
      if (is_edge_relayout_ops && tokens.size() > 1) {
        for (size_t i = 1; i < tokens.size(); i++) {
          edge_relayout_ops.push_back(tokens[i]);
        }
      }
    }

    if (std::find(experiment_features.begin(), experiment_features.end(),
                  "KeepEdgeRelayoutOps") != experiment_features.end()) {
      keep_edge_relayout_ops = true;
    }

    std::string infos =
        device_name + output_type + std::to_string(max_num_inputs) +
        std::to_string(max_num_outputs) + std::to_string(max_num_partitions) +
        std::to_string(drop_orphan_nodes);

    auto config_hash = xir::get_md5_of_buffer(infos.c_str(), infos.length());

    std::ofstream sig_file("original-info-signature.txt");
    if (sig_file.is_open()) {
      sig_file << config_hash;
      sig_file.close();
    }
  }

  void printOptions() {
    VAIML_INFO_PRINT(
        "VAIP VAIML pass options: ", "\n    vaiml_unarchive_path: ",
        vaiml_unarchive_path,
        "\n    enable_local_context: ", enable_local_context,
        "\n    device: ", device_name, "\n    output_type: ", output_type,
        "\n    config_filename: ", config_filename, "\n    debug: ", debug,
        "\n    force: ", force, "\n    compile_flow: ", compile_flow,
        "\n    enable_f16_to_bf16_conversion: ", enable_f16_to_bf16_conversion,
        "\n    enable_f32_to_bf16_conversion: ", enable_f32_to_bf16_conversion,
        "\n    partition_strategy: ", partition_strategy,
        "\n    threshold_gops_percent: ", threshold_gops_percent,
        "\n    removable_cluster_node_ratio: ", removable_cluster_node_ratio,
        "\n    ai_analyzer_visualization: ", ai_analyzer_visualization,
        "\n    priority: ", priority,
        "\n    ai_analyzer_profiling: ", ai_analyzer_profiling,
        "\n    fast_partition_swap: ", fast_partition_swap,
        "\n    no_failsafe: ", no_failsafe,
        "\n    init_m_values: ", init_m_values,
        "\n    aie_unsupported_ops_override: ", aie_unsupported_ops_override,
        "\n    overlay_json: ", overlay_json,
        "\n    overlay_json_signature_: ", overlay_json_signature,
        "\n    override_batch_size: ", override_batch_size,
        "\n    enable_batch: ", enable_batch,
        "\n    user_subgraph_config: ", user_subgraph_config,
        "\n    llm_ops_config: ", llm_ops_config,
        "\n    drop_orphan_nodes: ", drop_orphan_nodes,
        "\n    run_vaiml: ", run_vaiml, "\n    runner_type: ", runner_type,
        "\n    device_batch_size: ", device_batch_size,
        "\n    fatal_error_on_exception: ", fatal_error_on_exception);

    if (compile_flow != "default") {
      VAIML_INFO_PRINT(
          "VAIP VAIML pass options for custom op flow: ",
          "\n    custom_op_repos: ", stringVectorToString(custom_op_repos),
          "\n    supported_custom_ops: ",
          stringVectorToString(supported_custom_ops));
    }
    if (debug) {
      VAIML_INFO_PRINT(
          "VAIP VAIML pass debug options: ", "\n    vaiml_archive_file: ",
          vaiml_archive_file, "\n    max_num_inputs: ", max_num_inputs,
          "\n    max_num_outputs: ", max_num_outputs,
          "\n    max_num_partitions: ", max_num_partitions,
          "\n    exclude_subgraphs: ", stringVectorToString(exclude_subgraphs),
          "\n    include_subgraphs: ", stringVectorToString(include_subgraphs),
          "\n    experiment_features: ",
          stringVectorToString(experiment_features));
    }
  }
};
} // namespace vaip_vaiml
