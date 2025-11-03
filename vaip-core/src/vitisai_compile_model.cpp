/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS
// clang-format off
#include <cstdint>
#include <glog/logging.h>

#include <hash-library/md5.h>
#include "./cache_dir.hpp"
#include "./config.hpp"
#include "./file_lock.hpp"
#include "./pass_imp.hpp"
#include "./stat.hpp"
#include "profile_utils.hpp"
#include "morphizen/vaip.hpp"
#include "morphizen/util.hpp"
#include "morphizen/env_config.hpp"
#include <codecvt>
#include <errno.h>
#include <google/protobuf/util/json_util.h>
#include <ios>
#include <limits>
#include <locale>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <set>
#include "morphizen/encryption.hpp"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "binary/mem_binary.hpp"
#include "morphizen/config_reader.hpp"
// clang-format on

// this is an experimental feature. When this feature is enabled, the
// `PassContextImp::cache_files_` is not used. Instead, the
// `PassContextImp::tar_file_` is used.  The
// `PassContextImp::cache_files_` is used to store the cache files. It
// creates too many tmp files in disk, per VAI-10873 request, we need
// to reduce the tmp files.  The `PassContextImp::tar_file_` is used
// to store the cache files. It creates only one tmp file in disk or
// open the ep context binary file directly.  limitation: it does not
// when compression or encryption is enabled.
DEF_ENV_PARAM(MORPHIZEN_FEATURE_USE_TAR_FILE, "1")
DEF_ENV_PARAM_2(XLNX_ONNX_EP_REPORT_FILE, "", std::string)
DEF_ENV_PARAM(XLNX_ENABLE_CACHE, "1")
DEF_ENV_PARAM(XLNX_ENABLE_SKIP_FATAL, "1")
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
DEF_ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY, "0")
DEF_ENV_PARAM_2(DEBUG_MD5_SIG, "", std::string)
DEF_ENV_PARAM(DEBUG_VITIS_AI_EP, "1")
DEF_ENV_PARAM(DEBUG_FILE_LOCK, "0")
DEF_ENV_PARAM(DEBUG_EP_CONTEXT, "0")
DEF_ENV_PARAM(XLNX_EP_CONTEXT_ENABLE_COMPRESSION, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_PROFILING, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_VISUALIZATION, "0")
DEF_ENV_PARAM_2(XLNX_VAIML_LEVEL_1_NAME, "vaip-pass_vaiml_partition",
                std::string)

#ifdef _WIN32
#  ifdef ENABLE_PYTHON
// Python is only enabled for VAIML compilation on Windows, which requires
// this threshold to be set to a large value so all constants are cloned for the
// compilation.
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184",
                int64_t)

#  else
// Set the threshold to small value to save memory usage for Windows runtime
// package
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "128", int64_t)
#  endif
#else
// Set the threshold to a large value nn Linux for VAIML compilation
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184",
                int64_t)
#endif
namespace vaip_core {
// this template is to be deprecated. this function is to be deprecated, please
// use get_provier_option, only support XLNX_model_clone_external_data_threshold
// for backward compatibility.
template <>
int64_t PassContext::get_provier_option_with_class<
    ENV_PARAM_XLNX_model_clone_external_data_threshold>() const {
  using env_name = ENV_PARAM_XLNX_model_clone_external_data_threshold;
  const char* name = env_name::get_name();
  const char* defvalue = env_name::get_default_value();
  auto p = get_provider_option(std::string(name), std::string(defvalue));
  using helper =
      typename morphizen::env_config_helper<decltype(env_name::value)>;
  return helper::from_string(p);
} /*
 void force_instantiate_get_provider_options_with_class () {
   // This function is to force the instantiation of the
   // get_provier_option_with_class function template.
   // It is used to ensure that the function is instantiated
   // when the header file is included.
   auto p = PassContext::create();
   auto _ = p->get_provier_option_with_class<
       ENV_PARAM_XLNX_model_clone_external_data_threshold>();
   (void)_;
 }*/
} // namespace vaip_core
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VITIS_AI_EP) >= n)

DEF_ENV_PARAM_2(XLNX_MD5_SIG_SKIP_OPS, "QuantizeLinear,DequantizeLinear",
                std::vector<std::string>)

#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "
#ifdef ENABLE_PYTHON
#  include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
using namespace onnxruntime;

namespace google {
int GetStackTrace(void** result, int max_depth, int skip_count);
bool Symbolize(void* /*pc*/, char* /*out*/, size_t /*out_size*/);
} // namespace google

namespace vaip_core {
static void save_protobuf_message(const fs::path& filename,
                                  const google::protobuf::Message& msg) {
  try {
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    auto json_str = std::string();
    auto status =
        google::protobuf::util::MessageToJsonString(msg, &json_str, options);
    CHECK(status.ok()) << "cannot write json string:" << msg.DebugString();
    CHECK(std::ofstream(filename).write(&json_str[0], json_str.size()).good())
        << "failed to write " << filename;
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}

static void load_protobuf_message(const fs::path& filename,
                                  google::protobuf::Message& msg) {
  auto json_str = slurp(filename);
  auto status = google::protobuf::util::JsonStringToMessage(json_str, &msg);
  CHECK(status.ok()) << "cannot parse json string:" << json_str;
}

static void load_protobuf_message_2(const fs::path& filename,
                                    google::protobuf::Message& msg) {
  auto json_str = slurp_if_exists(filename);
  if (!json_str.empty()) {
    auto status = google::protobuf::util::JsonStringToMessage(json_str, &msg);
    CHECK(status.ok()) << "cannot parse json string:" << json_str;
  }
}

static inline void remove_encryption(ConfigProto& proto) {
  proto.clear_encryption_key();
}

static void print_device_subgraph(const PassContextImp& context) {
  LOG_VERBOSE(2) << "dpu subgraph: " << context.context_proto.meta_def_size();
}

static ContextProto load_context_json_2(PassContextImp& context) {
  ContextProto ctxProto;
  load_protobuf_message_2(get_cache_file_name(context, "context_dod.json"),
                          ctxProto);
  return ctxProto;
}

static void
collect_stat_and_dump(const PassContextImp& context,
                      const onnxruntime::Graph& onnx_graph) noexcept {
  try {
    auto filename = ENV_PARAM(XLNX_ONNX_EP_REPORT_FILE);
    collect_stat(onnx_graph, context.context_proto);
    if (!filename.empty()) {
      save_protobuf_message(get_cache_file_name(context, filename),
                            get_stat_proto());
    }
    clean_stat();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}

static void update_primary_context(std::shared_ptr<PassContextImp> context) {
  auto second_proto = load_context_json_2(*context);
  for (const auto& meta_def : second_proto.meta_def()) {
    context->context_proto.mutable_meta_def()->Add()->CopyFrom(meta_def);
  }
}

static void update_cache(std::shared_ptr<PassContextImp> context,
                         onnxruntime::Graph& graph) {
  auto deferred_write = std::shared_ptr<void>(
      nullptr, [context](void* /*p*/) { context->save_context_json(); });
  auto measure_update_cache = context->measure("update_cache");
  auto passes =
      IPass::create_passes(context, context->get_config_proto().passes());
  IPass::run_passes(passes, graph);
  // ##########
  // Workaround for Shell model compiler
  // Need to load meta_def from secondary json and merge with context.json
  // ##########
  update_primary_context(context);
}

void read_cache(std::shared_ptr<PassContextImp> context) {
  auto measure = context->measure("read_cache");
  context->update_pass_context_from_context_json_in_cache();
}

static std::string get_commit(const AllVersionInfoProto& proto,
                              const std::string& name) {
  for (const auto& iter : proto.version_infos()) {
    if (iter.package_name() == name) {
      return iter.commit();
    }
  }
  return ""; // if a commit is absent in both version info, we assume it matched
}

static bool cache_valid(const PassContextImp& context) {
  ContextProto proto;
  load_protobuf_message(get_cache_file_name(context, "context.json"), proto);
  auto& code_versions = context.get_config_proto().version();
  auto& cache_versions = proto.config().version();
  std::vector<std::string> package_name = {"xcompiler", "vaip"};
  for (const auto& name : package_name) {
    auto code_package_version = get_commit(code_versions, name);
    auto cache_package_version = get_commit(cache_versions, name);
    if (code_package_version != cache_package_version) {
      LOG(WARNING) << name << "'s versions mistached: " << code_package_version
                   << " at code and " << cache_package_version << " at cache";
      return true;
    }
  }
  return true;
}

static bool check_cache_exist(const PassContextImp& context) {
  fs::path cache_file = get_cache_file_name(context, "context.json");
  return file_exists(cache_file);
}
bool check_cache_hit(PassContextImp& context) {
  auto measure_check_cache_hit = context.measure("check_cache_hit");
  auto prebuild_cache_context_name =
      context.get_provider_option("prebuild_cache_context");
  if (prebuild_cache_context_name) {
    MY_LOG(1) << "==== prebuild_cache_context hit ====";
    if (!has_mem_binary(prebuild_cache_context_name.value())) {
      LOG(ERROR) << " " << prebuild_cache_context_name.value()
                 << " does not in mem please check vaip_config.json";

      std::abort();
    }
    auto prebuild_ep_context_in_mem =
        get_mem_binary(prebuild_cache_context_name.value());
    context.create_tar_file_for_prebuild_cache(
        std::move(prebuild_ep_context_in_mem));
    return true;
  }
  auto cache_in_mem = context.cache_in_mem();
  if (cache_in_mem) {
    return false;
  }
  if (ENV_PARAM(XLNX_ENABLE_CACHE)) {
    return check_cache_exist(context) && cache_valid(context);
  }
  return false;
}

void compile_onnx_model_2(std::shared_ptr<PassContextImp> context,
                          const Graph& onnx_graph) {
  bool cache_hit = check_cache_hit(*context);
  if (!cache_hit) {
    auto& model = graph_get_model(onnx_graph);
    int64_t threshold = ENV_PARAM(XLNX_model_clone_external_data_threshold);
    auto po_threshold = context->get_provider_option(
        "XLNX_model_clone_external_data_threshold");
    if (po_threshold) {
      threshold = std::stoll(po_threshold.value());
    }
    auto cloned_model = model_clone(model, threshold);
    auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);
    auto deferred_collect =
        std::shared_ptr<void>(nullptr, [context, &onnx_graph](void* /*p*/) {
          collect_stat_and_dump(*context, onnx_graph);
        });
    update_cache(context, cloned_graph);
  } else {
    MY_LOG(1) << "==== cache hit ====";
  }
  auto encryption_key = context->context_proto.config().encryption_key();
  auto session_configs = context->context_proto.config().session_configs();
  read_cache(context);
  context->context_proto.mutable_config()->set_encryption_key(encryption_key);
  auto session_configs_in_cache =
      context->context_proto.mutable_config()->mutable_session_configs();
  session_configs_in_cache->swap(session_configs);
}

static std::string get_dump_md5_file(const std::string& suffix) {
  auto ret = ENV_PARAM(DEBUG_MD5_SIG);
  if (!ret.empty()) {
    ret = ret + suffix;
  }
  return ret;
}
struct MD5Sig {
public:
  MD5Sig(const std::string& suffix)
      : dump_md5_file{get_dump_md5_file(suffix)} {}
  void add(const void* data, size_t numBytes) {
    md5.add(data, numBytes);
    if (str) {
      CHECK(str->write((const char*)data, numBytes).good())
          << "failed to write to dump_md5_file " << dump_md5_file;
    }
  }
  std::string getHash() {
    if (str) {
      str->close();
    }
    return md5.getHash();
  }

public:
  const std::string dump_md5_file;
  MD5 md5 = MD5();
  std::unique_ptr<std::ofstream> str =
      (dump_md5_file.empty() ? nullptr
                             : std::make_unique<std::ofstream>(dump_md5_file));
};

static std::string
get_model_signature_with_graph_inputs_and_outputs(const Graph& onnx_graph) {
  auto md5 = MD5Sig("_with_io.data");
  auto inputs = graph_get_inputs(onnx_graph);
  for (auto& input : inputs) {
    auto input_name = node_arg_get_name(*input);
    md5.add(input_name.data(), input_name.size());

    auto shape = node_arg_get_shape_i64(*input);
    if (shape && !shape->empty()) {
      md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
    }
  }
  auto outputs = graph_get_outputs(onnx_graph);
  for (auto& output : outputs) {
    auto output_name = node_arg_get_name(*output);
    md5.add(output_name.data(), output_name.size());

    auto shape = node_arg_get_shape_i64(*output);
    if (shape && !shape->empty()) {
      md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
    }
  }
  return md5.getHash();
}

static std::string get_model_signature(const Graph& onnx_graph) {
  auto md5 = MD5Sig(".data");
  for (auto node_idx : graph_get_node_in_topoligical_order(onnx_graph)) {
    auto node = VAIP_ORT_API(graph_get_node)(onnx_graph, node_idx);
    auto op_type = node_op_type(*node);
    const auto& skip_op = ENV_PARAM(XLNX_MD5_SIG_SKIP_OPS);
    if (std::find(skip_op.begin(), skip_op.end(), op_type) != skip_op.end()) {
      continue;
    }
    CHECK(node != nullptr) << "node_idx " << node_idx << " ";
    auto output = node_get_output_node_args(*node);
    for (auto& node_arg : output) {
      if (node_arg == nullptr) {
        continue;
      }
      if (!node_arg_exists(*node_arg)) {
        continue;
      }
      auto node_arg_name = node_arg_get_name(*node_arg);
      md5.add(node_arg_name.data(), node_arg_name.size());

      auto shape = node_arg_get_shape_i64(*node_arg);
      if (shape && !shape->empty()) {
        md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
      }
    }
  }
  return md5.getHash();
}

static std::pair<const std::string, const MepConfigTable*>
find_signature_in_meptabel(const ConfigProto& proto,
                           const std::string md5_file_base,
                           const std::string md5_in_memory_a,
                           const std::string md5_in_memory_b,
                           int32_t node_count) {
  for (auto& mep : proto.mep_table()) {
    if (md5_in_memory_a == mep.md5sum_in_memory()) {
      MY_LOG(1) << "find signature in meptable : "             //
                << "model_name :  " << mep.model_name() << " " //
                << "md5sum_in_memory : " << mep.md5sum_in_memory();
      return std::make_pair(md5_in_memory_a, &mep);
    }
  }
  for (auto& mep : proto.mep_table()) {
    if (md5_in_memory_b == mep.md5sum_in_memory_with_io()) {
      MY_LOG(1) << "find signature in meptable : "             //
                << "model_name :  " << mep.model_name() << " " //
                << "md5sum_in_memory_with_io : "
                << mep.md5sum_in_memory_with_io()
                << " model node_count : " << node_count
                << " mep node_count : " << mep.node_count();
      // Also match node count if it's specified in vaip_config.json
      if ((!mep.has_node_count()) || (node_count == mep.node_count())) {
        return std::make_pair(md5_in_memory_b, &mep);
      }
    }
  }
  for (auto& mep : proto.mep_table()) {
    if (!md5_file_base.empty() && md5_file_base == mep.md5sum_on_disk()) {
      MY_LOG(1) << "find signature in meptable : "             //
                << "model_name :  " << mep.model_name() << " " //
                << "md5sum_on_disk : " << mep.md5sum_on_disk();
      return std::make_pair(md5_file_base, &mep);
    }
  }
  MY_LOG(1) << "Can not find signature in meptable , use in memory signature "
            << md5_in_memory_a;
  return std::make_pair(md5_in_memory_a, nullptr);
}
static std::pair<const std::string, const MepConfigTable*>
get_signature_with_meptable(const std::string& model_path,
                            const Graph& onnx_graph, ConfigProto& proto) {
  auto md5_file_base =
      model_path.empty() ? "" : vaip_core::get_md5_of_file(model_path);
  auto md5_in_memory_a = get_model_signature(onnx_graph);
  auto md5_in_memory_b =
      get_model_signature_with_graph_inputs_and_outputs(onnx_graph);

  *proto.mutable_onnx_md5_file() = md5_file_base;
  *proto.mutable_onnx_md5_a() = md5_in_memory_a;
  *proto.mutable_onnx_md5_b() = md5_in_memory_b;
  const auto& node_indices = graph_get_node_in_topoligical_order(onnx_graph);
  int32_t node_count = (int32_t)node_indices.size();

  MY_LOG(1) << "File base signature : " << md5_file_base;
  MY_LOG(1) << "Algorithm-A: based on topologically ordered signature : "
            << md5_in_memory_a;
  MY_LOG(1) << "Algorithm-B: based on graph inputs/outputs signature : "
            << md5_in_memory_b;
  MY_LOG(1) << "Algorithm-B: node count: " << node_count;
  return find_signature_in_meptabel(proto, md5_file_base, md5_in_memory_a,
                                    md5_in_memory_b, node_count);
}

std::shared_ptr<PassContextImp>
initialize_context(const std::string& model_path, const Graph& onnx_graph,
                   const std::vector<vaip_cxx::NodeConstRef>& ep_context_nodes,
                   const onnxruntime::ProviderOptions& options) {

  std::shared_ptr<PassContextImp> context =
      PassContextImp::create_pass_context(options);
  // "session.model_external_initializers_file_folder_path/virtual_model.onnx
  // would be passed for in-mem model when this happen, a invalid path is
  // passed, we use the model_path == empty to differentiate if the model is
  // in-mem
  if (std::filesystem::is_regular_file(model_path)) {
    context->model_path = model_path;
  }
  context->is_ep_context_model = !ep_context_nodes.empty();
  auto& model = graph_get_model(onnx_graph);
  auto [md5, mep_table] =
      get_signature_with_meptable(context->model_path.string(), onnx_graph,
                                  *context->context_proto.mutable_config());

  if (!context->context_proto.config().cache_key().empty()) {
    MY_LOG(1) << "use cache key specified by user "
              << context->context_proto.config().cache_key();
  } else if (VAIP_ORT_API(model_has_meta_data)(model, "vaip_model_md5sum")) {
    auto new_cache_key =
        *VAIP_ORT_API(model_get_meta_data)(model, "vaip_model_md5sum");
    MY_LOG(1) << "use cache key in meta-data " << new_cache_key;
    *context->context_proto.mutable_config()->mutable_cache_key() =
        new_cache_key;
  } else if (ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY) &&
             (!context->model_path.empty())) {
    auto new_cache_key =
        vaip_core::get_md5_of_file(context->model_path.string());
    MY_LOG(1) << "use cache key on-disk " << new_cache_key;
    *context->context_proto.mutable_config()->mutable_cache_key() =
        new_cache_key;
  } else {
    auto new_cache_key = md5;
    LOG_VERBOSE(1) << "use cache key in memory signature " << new_cache_key;
    *context->context_proto.mutable_config()->mutable_cache_key() =
        new_cache_key;
  }
  // Algorithm-A : based on names of node-args tensor0-names of
  // topologically ordered model-graph Algorithm-B : based on
  // input/output-tensor names overall auto-mapping mechanism will be to use
  // Algorithm-A first, if that fails then use Algorithm-B to identify the
  // model/target
  if (mep_table) {
    context->mep_config_proto_ = std::make_unique<MepConfigTable>(*mep_table);
    context->context_proto.mutable_config()->mutable_provider_options()->insert(
        {"model_name", mep_table->model_name()});
    std::string model_category = "";
    if (mep_table->has_model_category()) {
      model_category = mep_table->model_category();
    }
    context->context_proto.mutable_config()->mutable_provider_options()->insert(
        {"model_category", model_category});

    std::string model_variant = "";
    if (mep_table->has_model_variant()) {
      model_variant = mep_table->model_variant();
    }
    context->context_proto.mutable_config()->mutable_provider_options()->insert(
        {"model_variant", model_variant});

    std::string is_preemptible = "0";
    if (mep_table->has_is_preemptible()) {
      is_preemptible = mep_table->is_preemptible() ? "1" : "0";

      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"is_preemptible", is_preemptible});
    }

    std::string dd_use_lazy_scratch_bo = "1";
    if (mep_table->has_dd_use_lazy_scratch_bo()) {
      dd_use_lazy_scratch_bo = mep_table->dd_use_lazy_scratch_bo() ? "1" : "0";
    }

    context->context_proto.mutable_config()->mutable_provider_options()->insert(
        {"dd_use_lazy_scratch_bo", dd_use_lazy_scratch_bo});

    std::string qos_priority = "";
    if (mep_table->has_qos_priority()) {
      qos_priority = mep_table->qos_priority();

      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"qos_priority", qos_priority});
    }

    std::string perf_pref = "";
    if (mep_table->has_perf_pref()) {
      perf_pref = mep_table->perf_pref();

      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"perf_pref", perf_pref});
    }

    auto qos_gen_params = mep_table->qos_generic_params();
    for (const auto& pair : qos_gen_params) {
      context->context_proto.mutable_config()
          ->mutable_qos_provider_options()
          ->insert({pair.first, pair.second});
    }

    std::string dd_use_lazy_const_bo = "0";
    if (mep_table->has_dd_use_lazy_const_bo()) {
      dd_use_lazy_const_bo = mep_table->dd_use_lazy_const_bo() ? "1" : "0";
      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"dd_use_lazy_const_bo", dd_use_lazy_const_bo});
    }

    std::string dealloc_scratch_bo = "0";
    if (mep_table->has_dd_dealloc_scratch_bo()) {
      dealloc_scratch_bo = mep_table->dd_dealloc_scratch_bo() ? "1" : "0";
      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"dd_dealloc_scratch_bo", dealloc_scratch_bo});
    }

    std::string constbo_sharing_key = "";
    if (mep_table->has_constbo_sharing_key()) {
      constbo_sharing_key = mep_table->constbo_sharing_key();
      context->context_proto.mutable_config()
          ->mutable_provider_options()
          ->insert({"constbo_sharing_key", constbo_sharing_key});
    }
  }
  context->target_auto_discovery(model);
  if (!context->is_ep_context_model) {
    vaip_core::update_config_by_target(*context->context_proto.mutable_config(),
                                       mep_table, context->target_proto_.get(),
                                       context);
  }

  auto onnx_path = model_path.empty() ? std::string("N/A") : model_path;
  *context->context_proto.mutable_config()->mutable_onnx_path() = onnx_path;

  if (VAIP_ORT_API(model_has_meta_data)(model, "suffix_counter")) {
    context->suffix_counter =
        std::stoi(*VAIP_ORT_API(model_get_meta_data)(model, "suffix_counter"));
  }
  update_cache_dir(*context);
  // DANGER!
  Model& mutable_model = const_cast<Model&>(model);
  model_set_meta_data(mutable_model, "vaip_log_dir",
                      context->get_log_dir().u8string());
  // log version of binary
  context->print_version_info("EXEC VERSION: ");
  if (!context->is_ep_context_model) {
    context->maybe_create_tar_file_for_write();
  }
  return context;
}
static void get_ep_cache_context_common(PassContextImp& context,
                                        IStreamWriter& dst) {
  auto measure_get_ep_cache_context_embed_mode =
      context.measure("get_ep_cache_context_common");
  auto reader = context_cache_files_to_tar_stream(context);
  if (ENV_PARAM(XLNX_EP_CONTEXT_ENABLE_COMPRESSION)) {
    auto measure_compression = context.measure("vaip_core::compress");
    LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
        << " start compressing ep context ";
    reader = vaip_core::compress(*reader);
  }
  auto encryption_key = context.context_proto.config().encryption_key();
  if (!encryption_key.empty()) {
    reader = stream_filter(
        *reader,
        [](const IStreamReader& src, IStreamWriter& dst,
           const std::string& encryption_key) {
          vaip_encryption::aes_encryption(src, dst, encryption_key);
        },
        encryption_key);
  }
  stream_copy(*reader, dst);
  return;
}

std::string get_ep_cache_context_embed_mode(PassContextImp& context) {
  auto measure_get_ep_cache_context_embed_mode =
      context.measure("get_ep_cache_context_embed_mode");
  if (context.tar_file_ != nullptr) {
    auto out = std::string();
    out.resize(context.tar_file_->current_size());
    auto ok = context.tar_file_->dump_to(out.data(), out.size());
    CHECK(ok) << "cannot dump, size=" << out.size();
    return out;
  } else {
    std::vector<char> out;
    auto dst = IStreamWriter::from_bytes(out); // TODO: add from_string.
    get_ep_cache_context_common(context, *dst);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
        << "embed mode = 1, load cache directory  to tar memory " << out.size()
        << " bytes";
    return std::string(out.begin(), out.end());
  }
}

static std::string get_ep_cache_context_nonembed_mode(PassContextImp& context) {
  auto measure_get_ep_cache_context_embed_mode =
      context.measure("get_ep_cache_context_nonembed_mode");
  auto OrtSessionOptionEpContextFilePath_binay = std::filesystem::path();
  // return a file name for the binary file, the file name is written into the
  // atttribute "ep.cache_context"
  if (context.tar_file_) {
    CHECK(!context.tar_file_file_name_.empty())
        << "tar_file_file_name_ is empty, please check the context";
    OrtSessionOptionEpContextFilePath_binay = context.tar_file_file_name_;
    // do nothing
  } else {
    OrtSessionOptionEpContextFilePath_binay =
        context.get_dir_of_ep_context_model() /
        context.get_basename_of_ep_context_binary_file();
    auto dst =
        IStreamWriter::from_path(OrtSessionOptionEpContextFilePath_binay);
    get_ep_cache_context_common(context, *dst);
  }
  CHECK(OrtSessionOptionEpContextFilePath_binay.has_filename())
      << "OrtSessionOptionEpContextFilePath_binay has no filename";
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
      << "embed mode = 0, save cache directory to tar file "
      << OrtSessionOptionEpContextFilePath_binay.filename();
  auto binary_name =
      OrtSessionOptionEpContextFilePath_binay.filename().u8string();
  return binary_name;
}

static std::string get_ep_cache_context(PassContextImp& context,
                                        bool embed_mode) {
  auto ret = std::string();
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
      << "start to create ep context, embed mode = " << embed_mode;
  if (embed_mode) {
    ret = get_ep_cache_context_embed_mode(context);
  } else {
    ret = get_ep_cache_context_nonembed_mode(context);
  }
  return ret;
}

#if VAIP_ORT_API_MAJOR < 6
static std::string escape_json(const std::string& s) {
  std::ostringstream o;
  for (auto c = s.cbegin(); c != s.cend(); c++) {
    if (*c == '"' || *c == '\\' || ('\x00' <= *c && *c <= '\x1f')) {
      o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
        << static_cast<int>(*c);
    } else {
      o << *c;
    }
  }
  return o.str();
}
static std::string get_nodes(PassContextImp& context) {
  std::ostringstream o;
  auto log_dir = context.get_log_dir();
  auto cache_dir = log_dir.parent_path().u8string();
  auto cache_key = log_dir.filename().u8string();
  o << "{"
       "\"backend_cache_dir\":"                  //
    << "\"" << escape_json(cache_dir) << "\",\n" //
    << "\"backend_cache_key\":"                  //
    << "\"" << escape_json(cache_key) << "\"\n"  //
    << "}";
  return o.str();
}
#endif
template <typename T> static std::string combine_outputs_name(T& collection) {
  std::ostringstream oss;
  auto sorted_names =
      std::set<std::string>(collection.begin(), collection.end());
  for (auto& name : sorted_names) {
    oss << " " << name;
  }
  return oss.str();
}
static std::vector<std::optional<vaip_cxx::NodeArgConstRef>>
convert_to_node_arg_const_ref(vaip_cxx::GraphRef g,
                              const std::vector<std::string>& names) {
  auto ret = std::vector<std::optional<vaip_cxx::NodeArgConstRef>>();
  std::transform(
      names.begin(), names.end(), std::back_inserter(ret),
      [&g](
          const std::string& name) -> std::optional<vaip_cxx::NodeArgConstRef> {
        if (name.empty()) {
          return std::nullopt;
        }
        // find node_arg by name frist, beacuse maybe an EPContext's output is
        // the input to another EPContext node
        auto node_arg = g.find_node_arg(name);
        if (node_arg.has_value()) {
          return node_arg;
        }
        return std::optional<vaip_cxx::NodeArgConstRef>(g.new_node_arg(
            name, {}, onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
      });
  return ret;
}

static onnxruntime::Node*
create_ep_context_node(vaip_core::ExecutionProviderConcrete* ep, int index) {
  CHECK(ep != nullptr);
  auto p_context = dynamic_cast<PassContextImp*>(ep->get_context().get());
  CHECK(p_context != nullptr);
  auto& context = *p_context;

  if (ENV_PARAM(DEBUG_EP_CONTEXT) >= 2) {
    LOG(INFO) << "create ep context node , index=" << index;
    LOG(INFO) << "Input meta-defs: "
              << vaip_core::combine_outputs_name(*ep->get_meta_def_inputs());
    LOG(INFO) << "Output meta-defs: "
              << vaip_core::combine_outputs_name(*ep->get_meta_def_outputs());
  }

  if (!context.ep_context_model_) {
    context.ep_context_model_ =
        vaip_cxx::Model::create(context.model_path, {{"ai.onnx", 21}});
  }
  auto ep_context_graph = context.ep_context_model_->main_graph();
  auto op_type = "EPContext";
  auto op_domain = "com.microsoft";
  auto description = "description";
  auto fused_node = ep->get_fused_node();
  auto input_args =
      fused_node
          ? vaip_cxx::NodeConstRef::from_node(ep_context_graph, *fused_node)
                .inputs()
          : convert_to_node_arg_const_ref(vaip_cxx::GraphRef(ep_context_graph),
                                          *ep->get_meta_def_inputs());
  auto output_args =
      fused_node
          ? vaip_cxx::NodeConstRef::from_node(ep_context_graph, *fused_node)
                .outputs()
          : convert_to_node_arg_const_ref(vaip_cxx::GraphRef(ep_context_graph),
                                          *ep->get_meta_def_outputs());
  // for new ABI EP, fused_node is nullptr
  auto name = fused_node ? vaip_cxx::NodeConstRef::from_node(ep_context_graph,
                                                             *fused_node)
                               .name()
                         : "" /* for new ABI EP, name is not used*/;
  if (fused_node) {
    // strictly speaking, it is probably not necessary if we assume
    // that ORT would keep ep and fused_node in the same order.
    index = (int)node_get_attr_int(*fused_node, "index");
  }
  auto attrs = NodeAttributesBuilder();
  attrs.add("index", (int64_t)index);
  int64_t main_context = index == 0 ? 1 : 0;
  attrs.add("main_context", main_context);
  int64_t embed_mode =
      context.get_session_config("ep.context_embed_mode", "1") == "1" ? 1 : 0;
  attrs.add("embed_mode", embed_mode);
  attrs.add("source", std::string("VitisAIExecutionProvider"));
  attrs.add("log_dir", context.get_log_dir().u8string());
  attrs.add("onnx_model_filename", context.model_path.u8string());
  attrs.add("partition_name", name);
  attrs.add("enable_compression",
            (int64_t)ENV_PARAM(XLNX_EP_CONTEXT_ENABLE_COMPRESSION));
  auto enable_encryption = 0;
#ifdef WITHOPENSSL
  enable_encryption =
      context.context_proto.config().encryption_key().empty() ? 0 : 1;
#endif
  attrs.add("enable_encryption", (int64_t)enable_encryption);
  attrs.add("cache_file_use_cache_key_prefix",
            (int64_t)context.cache_file_use_cache_key_prefix_);
  attrs.add("cache_file_prefix", context.get_config_proto().cache_key());
  auto& version_infos = context.get_config_proto().version();
  for (const auto& version_info : version_infos.version_infos()) {
    auto lib_name = "version_of_" + version_info.package_name();
    attrs.add(lib_name, version_info.version());
    lib_name = "version_id_of_" + version_info.package_name();
    attrs.add(lib_name, version_info.commit());
  }
#if VAIP_ORT_API_MAJOR < 6
  auto notes = get_nodes(context);
  attrs.add("notes", notes);
#endif
  auto ep_cache_context = std::string();
  if (main_context) {
    ep_cache_context = get_ep_cache_context(context, embed_mode != 0);
  }
  attrs.add("ep_cache_context", ep_cache_context);
  auto ret = vaip_cxx::GraphRef(ep_context_graph)
                 .add_node(name, op_domain, op_type, description, input_args,
                           output_args, attrs.build());
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT)) << "add ep node:" << ret;
  return ret.ptr();
}
static void init_ep_context_model_inputs(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps) {
  // guess graph input
  std::set<std::string> all_ep_inputs;
  std::set<std::string> all_ep_outputs;
  for (const auto& ep : eps) {
    auto ep_concrete =
        dynamic_cast<const vaip_core::ExecutionProviderConcrete*>(ep.get());
    if (ep_concrete) {
      auto inputs = ep_concrete->get_meta_def_inputs();
      all_ep_inputs.insert(inputs->begin(), inputs->end());

      auto outputs = ep_concrete->get_meta_def_outputs();
      all_ep_outputs.insert(outputs->begin(), outputs->end());
    }
  }

  std::vector<std::string> graph_inputs;
  std::set_difference(all_ep_inputs.begin(), all_ep_inputs.end(),
                      all_ep_outputs.begin(), all_ep_outputs.end(),
                      std::back_inserter(graph_inputs));
  // set graph input
  if (auto ep = dynamic_cast<vaip_core::ExecutionProviderConcrete*>(
          eps.front().get())) {
    if (auto p_context =
            dynamic_cast<PassContextImp*>(ep->get_context().get())) {
      if (!p_context->ep_context_model_) {
        p_context->ep_context_model_ =
            vaip_cxx::Model::create(p_context->model_path, {{"ai.onnx", 21}});
      }
      auto ep_context_graph = p_context->ep_context_model_->main_graph();
      auto optional_inputs =
          convert_to_node_arg_const_ref(ep_context_graph, graph_inputs);
      std::vector<vaip_cxx::NodeArgConstRef> actual_inputs;
      for (const auto& opt_input : optional_inputs) {
        if (opt_input.has_value()) {
          actual_inputs.push_back(opt_input.value());
        }
      }
      vaip_cxx::GraphRef(ep_context_graph).set_inputs(actual_inputs);
    }
  }
}
extern "C" VAIP_DLL_SPEC int create_ep_context_nodes(
#if VAIP_ORT_API_MAJOR < 6
    onnxruntime::Graph& /*ep_context_graph unused to deleted*/,
#endif
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    vaip_core::DllSafe<std::vector<Node*>>* ret_value) {
  std::vector<Node*> ret;
  if (eps.empty()) {
    *ret_value =
        vaip_core::DllSafe<std::vector<Node*>>(new std::vector<Node*>());
    return 1;
  }
  auto ep =
      dynamic_cast<vaip_core::ExecutionProviderConcrete*>(eps.front().get());

  auto p_context = dynamic_cast<PassContextImp*>(ep->get_context().get());
  CHECK(p_context != nullptr);
  if (p_context->is_ep_context_model) {
    // cannot create a ep context model when it is already a ep context
    *ret_value =
        vaip_core::DllSafe<std::vector<Node*>>(new std::vector<Node*>());
    return 1;
  }
  auto& context = *p_context;
  auto deferred_write = std::shared_ptr<void>(nullptr, [&context](void* /*p*/) {
    if (0)
      context.save_context_json();
  });
  auto measure_create_ep_context_nodes =
      context.measure("create_ep_context_nodes");
  // Inputs need to be set beforehand because the MLIR workflow requires them to
  // be defined before add node
  init_ep_context_model_inputs(eps);
  ret.reserve(eps.size());
  auto ep_index = 0;
  for (auto& ep_1 : eps) {
    ret.push_back(create_ep_context_node(
        dynamic_cast<vaip_core::ExecutionProviderConcrete*>(ep_1.get()),
        ep_index++));
  }
  *ret_value = vaip_core::DllSafe<std::vector<Node*>>(
      new std::vector<Node*>(std::move(ret)));
  return 0;
}

static std::vector<vaip_cxx::NodeConstRef>
get_ep_context_nodes(vaip_cxx::GraphConstRef onnx_graph) {
  auto ret = std::vector<vaip_cxx::NodeConstRef>();
  auto nodes = onnx_graph.nodes();
  for (auto node : nodes) {
    if (node.op_type() == "EPContext" && node.op_domain() == "com.microsoft") {
      if (node.has_attr("source") &&
          node.get_attr_string("source") == "VitisAIExecutionProvider") {
        ret.push_back(node);
      }
    }
  }
  return ret;
}

static void update_meta_def_from_ep_node(vaip_cxx::NodeConstRef node,
                                         MetaDefProto& meta_def) {
  // There are legacy issues, and it's unclear why metadef inputs and outputs
  // were changed. The inputs/outputs order between ORT FuseNode (EPContext
  // node) and metadef may not match. Test case: running PSI ctx model with
  // new ABI.
  /*
  meta_def.mutable_inputs()->Clear();
  for (auto input : node.inputs()) {
    if (input.has_value()) {
      meta_def.add_inputs(input->name());
    }
  }
  meta_def.mutable_outputs()->Clear();
  */
  auto output_name = std::string();
  for (auto output : node.outputs()) {
    if (output.has_value()) {
      if (output_name.empty()) {
        // use the first node arg name.
        output_name = output->name();
      }
      //  meta_def.add_outputs(output->name());
    }
  }

  // here is to trace back the ExecutionProvider from the EPContext node.
  // The nodes and constant_initializers in metadef are from the original
  // graph, but these nodes and constant_initializers do not exist in the EP
  // context model, so they need to be cleared here.
  meta_def.mutable_nodes()->Clear();
  CHECK(!output_name.empty())
      << "EPContext node must have at least one output.";
  meta_def.add_nodes(output_name);
  meta_def.mutable_constant_initializers()->Clear();
  return;
}
static std::optional<vaip_cxx::NodeConstRef> get_main_ep_context_node(
    std::vector<vaip_cxx::NodeConstRef>& ep_context_nodes) {
  std::optional<vaip_cxx::NodeConstRef> ret = std::nullopt;
  auto count_main_context = 0;
  for (auto node : ep_context_nodes) {
    if (node.has_attr("main_context") && node.get_attr_int("main_context")) {
      count_main_context++;
      ret = node;
    }
  }
  CHECK_EQ(count_main_context, 1)
      << "There must be exactly one main EPContext node. The EP context "
         "model "
         "have "
      << count_main_context << " main EPContext nodes.";
  return ret;
}

static void
store_cache_directory_from_main_node(PassContextImp& context,
                                     vaip_cxx::NodeConstRef main_node) {
  int64_t enable_encryption = main_node.get_attr_int("enable_encryption", 0);
  int64_t enable_compression = main_node.get_attr_int("enable_compression", 0);
  int64_t ep_embed_mode = main_node.get_attr_int("embed_mode", 1);
  context.cache_file_use_cache_key_prefix_ =
      main_node.get_attr_int("cache_file_use_cache_key_prefix", 0) != 0;
  *context.get_context_proto().mutable_config()->mutable_cache_key() =
      main_node.get_attr_string("cache_file_prefix", "");
  if (context.cache_file_use_cache_key_prefix_) {
    CHECK_NE(context.get_context_proto().config().cache_key(), "")
        << "cache_key "
        << "should not be empty when cache_file_use_cache_key_prefix_ is set "
           "to "
           "true";
  }
#if VAIP_ORT_API_MAJOR >= 12
  auto ep_cache_context = main_node.release_attr_string("ep_cache_context");
#else
#  error "not supported any more"
#endif
  auto ep_context_size = ep_cache_context->size();

  if (ENV_PARAM(MORPHIZEN_FEATURE_USE_TAR_FILE) //
      && !enable_compression && !enable_encryption) {
    context.create_tar_file_for_read(std::move(ep_cache_context),
                                     ep_embed_mode != 0);
  } else {
    std::unique_ptr<IStreamReader> ep_context_file;
    auto context_holder = std::make_shared<TempFile>();
    if (ep_embed_mode) {
      std::unique_ptr<IStreamReader> src =
          IStreamReader::from_bytes(ep_cache_context->data(), ep_context_size);
      auto dst = context_holder->build_writer();
      stream_copy(*src, *dst);
      std::tie(ep_context_file, ep_context_size) =
          context_holder->build_reader();
      LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
          << "embed mode = 1, load ep context " << ep_context_size << " bytes";
    } else {
      auto ep_context_binary_file = context.get_dir_of_ep_context_model() /
                                    std::filesystem::u8path(*ep_cache_context);
      ep_context_file = IStreamReader::from_path(ep_context_binary_file);
    }
    if (enable_encryption) {
      auto encryption_key = context.context_proto.config().encryption_key();
      if (encryption_key.empty()) {
        LOG(ERROR) << "enable_encryption is set, but encryption_key is empty";
        std::abort();
      }
      try {
        ep_context_file = stream_filter(
            *ep_context_file,
            [](const IStreamReader& src, IStreamWriter& dst,
               const std::string& encryption_key) {
              vaip_encryption::aes_decryption(src, dst, encryption_key);
            },
            encryption_key);
      } catch (std::runtime_error& e) {
        LOG(ERROR) << "exception occurs when decryption: " << e.what();
        std::abort();
      }
    }
    if (enable_compression) {
      ep_context_file = uncompress(*ep_context_file);
    }

    context.tar_file_to_cache_files(*ep_context_file);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
        << " extract memory cache files to  "
        << context.get_log_dir().u8string();
  }
}

static int64_t get_ep_context_index(const vaip_cxx::NodeConstRef& node) {
  CHECK(node.has_attr("index"))
      << "EPContext Node has no index attr, EPContext node : " << node;
  return node.get_attr_int("index");
}

static std::vector<std::unique_ptr<ExecutionProvider>>
create_execution_providers_from_ep_context_nodes(
    std::shared_ptr<PassContextImp> context,
    std::vector<vaip_cxx::NodeConstRef> ep_context_nodes) {
  CHECK_EQ(ep_context_nodes.size(), context->context_proto.meta_def_size());
  auto size = ep_context_nodes.size();
  auto ret = std::vector<std::unique_ptr<ExecutionProvider>>();
  ret.reserve(size);

  std::sort(
      ep_context_nodes.begin(), ep_context_nodes.end(),
      [](const vaip_cxx::NodeConstRef& a, const vaip_cxx::NodeConstRef& b) {
        return get_ep_context_index(a) < get_ep_context_index(b);
      });
  for (auto idx = 0u; idx < size; ++idx) {
    auto node = ep_context_nodes[idx];
    auto index = get_ep_context_index(node);
    CHECK_EQ(index, idx) << "EPContext Node index mismatch, EPContext node : "
                         << node;
    auto meta_def_index = idx;
    auto& meta_def = *context->context_proto.mutable_meta_def(meta_def_index);
    update_meta_def_from_ep_node(node, meta_def);
    auto device = meta_def.device();
    auto plugin_name = std::string("vaip_custom_op_") + device;
    ret.emplace_back(
        ExecutionProviderConcrete::create(plugin_name, context, meta_def));
  }
  return ret;
}
std::vector<std::unique_ptr<ExecutionProvider>>
restore_execution_providers_from_ep_context_model(
    vaip_cxx::GraphConstRef /*onnx_graph*/,
    std::shared_ptr<PassContextImp> context,
    std::vector<vaip_cxx::NodeConstRef> ep_context_nodes) {
  auto measture =
      context->measure("restore_execution_providers_from_ep_context_model");
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
      << "Running EP context onnx model , restore ExecutionProviders from EP "
         "context model";
  // untar the cache directory.
  auto main_node = get_main_ep_context_node(ep_context_nodes);
  CHECK(main_node) << " no main EPContext node";
  store_cache_directory_from_main_node(*context, main_node.value());
  context->update_pass_context_from_context_json_in_cache();
  return create_execution_providers_from_ep_context_nodes(context,
                                                          ep_context_nodes);
}
static void dirty_hack_for_model_clone_external_data_threshold(
    const ConfigProto& config_proto) {
  //  check each pass in passes, if vaiml plugin is eanbled , disable the
  //  optimization for model clone.
  for (auto& pass : config_proto.passes()) {
    if (pass.plugin() == ENV_PARAM(XLNX_VAIML_LEVEL_1_NAME)) {
      // effective disable the optimization for model clone.
      ENV_PARAM(XLNX_model_clone_external_data_threshold) = 17179869184;
      break;
    }
  }
}
static bool is_compiling_on_non_npu_platform(PassContextImp& context) {
  auto is_compiling_on_non_npu_platform_provider_option =
      context.get_provider_option(kProviderOptionIsCompilingOnNonNpuPlatform);
  if (is_compiling_on_non_npu_platform_provider_option) {
    // it takes the precedence over the EP context enable option. this is only
    // for internal use, for debugging and testing purpose.
    return is_compiling_on_non_npu_platform_provider_option.value() == "1";
  }
  auto is_ep_context_enabled =
      context.get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  if (!is_ep_context_enabled) {
    // if EP context is not enabled, it is not a compilation flow.
    return false;
  }
  // TODO: vai-rt need to upgrade onnxruntime
  static const char* kOrtSessionOptionsDisableModelCompile_local =
      "session.disable_model_compile";
  if (context.get_session_config(kOrtSessionOptionsDisableModelCompile_local,
                                 "1") == "0") {
    return true; // see also MicroSoft/Onnxruntime#24416
  }
#if defined(_WIN32)
  std::filesystem::path xilinx_dll =
      std::filesystem::path("C:\\Windows\\System32\\xrt_coreutil.dll");
  if (!std::filesystem::exists(xilinx_dll)) {
    return true; // assume compiling on non-npu platform if xrt_coreutil.dll
                 // does not exist.
  }
#elif !defined(__aarch64__)
  // If XILINX_XRT is not set on Linux, it's a compile only run
  auto xilinx_xrt = getenv("XILINX_XRT");
  if (xilinx_xrt == nullptr) {
    return true; // assume compiling on non-npu platform if XILINX_XRT is not
                 // set.
  }
#endif
  return false;
}
static void log_stat_subgraph(const ContextProto& context_proto) {
  auto stat = std::map<std::string, int>{};
  for (auto& meta_def : context_proto.meta_def()) {
    stat[meta_def.device()]++;
  }
  std::cout << "[Vitis AI EP] No. of Subgraphs supported by Vitis AI EP:";
  for (const auto& subgraph_stat : stat) {
    std::cout << std::setw(6) << subgraph_stat.first << std::setw(6)
              << subgraph_stat.second << " ";
  }
  std::cout << std::endl;
}
static std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_internal(
    const Graph& onnx_graph,
    const std::vector<vaip_cxx::NodeConstRef>& ep_context_nodes,
    std::shared_ptr<PassContextImp> context) {
  auto measure_compile_onnx_model_internal =
      context->measure("compile_onnx_model_internal");
  auto ret = std::vector<std::unique_ptr<ExecutionProvider>>();
  auto is_ep_context_model = !ep_context_nodes.empty();
  auto measure_after_compile_onnx_model_2 = std::unique_ptr<PassContextTimer>();
  if (is_ep_context_model) {
    ret = restore_execution_providers_from_ep_context_model(onnx_graph, context,
                                                            ep_context_nodes);
    log_stat_subgraph(context->get_context_proto());
  } else {
    dirty_hack_for_model_clone_external_data_threshold(
        context->get_config_proto());
    auto measure_before_compile_onnx_model_2 =
        context->measure("before_compile_onnx_model_internal");
    compile_onnx_model_2(context, onnx_graph);
    measure_after_compile_onnx_model_2 =
        context->measure("after_compile_onnx_model_internal");
    ret.reserve(context->context_proto.meta_def_size());
    auto enable_generic_custom_op = is_compiling_on_non_npu_platform(*context);
    if (enable_generic_custom_op) {
      LOG(INFO) << "detect running on Non-NPU platform, compilation only";
    }
    for (auto& meta_def : *context->context_proto.mutable_meta_def()) {
      std::string device = meta_def.device();
      if (enable_generic_custom_op) {
        // ovewrite default device for offline compilation flow.
        device = "GENERIC";
        meta_def.set_fallback_cpu(false);
      }
      auto plugin_name = std::string("vaip_custom_op_") + device;
      ret.emplace_back(
          ExecutionProviderConcrete::create(plugin_name, context, meta_def));
    }
  }
  return ret;
}

static std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack_strings;

  void* stack[32];
  // +2 to exclude this function and compile_fatal_func.
  const int depth =
      google::GetStackTrace(stack, sizeof(stack) / sizeof(stack[0]), 2);
  for (auto i = 0; i < depth; ++i) {
    auto pc = stack[i];
    const char* symbol = "(unknown)";
    char symbolized[1024]; // Big enough for a sane symbol.
    // Symbolizes the previous address of pc because pc may be in the
    // next function.
    if (google::Symbolize(reinterpret_cast<char*>(pc) - 1, symbolized,
                          sizeof(symbolized))) {
      symbol = symbolized;
    }
    stack_strings.push_back(std::string(symbol));
  }
  return stack_strings;
}

struct GlogFatalException : public std::exception {
public:
  virtual const char* what() const throw() { return m.c_str(); }
  std::string m;
  std::vector<std::string> stacks;
};

static void compile_fatal_func() {
  GlogFatalException e;
  e.stacks = GetStackTrace();
  for (auto&& t : e.stacks) {
    e.m += std::string(t) + "\n";
  }
  throw e;
}

static std::ostream& operator<<(std::ostream& s,
                                const std::vector<int64_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << "x";
    }
    s << v[c];
  }
  s << "]";
  return s;
}
static bool is_cpu_only_inference(const PassContextImp& context) {
  auto ret = false;
  if (context.context_proto.meta_def_size() == 0) {
    ret = true;
  }
  return ret;
}

static void print_graph_input_and_output(const Graph& onnx_graph) {
#ifdef _WIN32
  _setmaxstdio(8192);
#endif

  auto graph_inputs = graph_get_inputs(onnx_graph);
  auto graph_outputs = graph_get_outputs(onnx_graph);

  LOG(INFO) << "Vitis AI EP Load ONNX Model Success";
  LOG(INFO) << "Graph Input Node Name/Shape (" << graph_inputs.size() << ")";
  for (auto& input : graph_inputs) {
    auto shape = node_arg_get_shape_i64(*input);
    if (shape != nullptr) {
      LOG(INFO) << "\t " << node_arg_get_name(*input) << " : "
                << *(shape.get());
    } else {
      LOG(INFO) << "\t " << node_arg_get_name(*input) << " : []";
    }
  }
  LOG(INFO) << "Graph Output Node Name/Shape (" << graph_outputs.size() << ")";
  for (auto& output : graph_outputs) {
    auto shape = node_arg_get_shape_i64(*output);
    if (shape != nullptr) {
      LOG(INFO) << "\t " << node_arg_get_name(*output) << " : "
                << *(shape.get());
    } else {
      LOG(INFO) << "\t " << node_arg_get_name(*output) << " : []";
    }
  }
}
std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3(const std::string& model_path, const Graph& onnx_graph,
                     const onnxruntime::ProviderOptions& options) {
  print_graph_input_and_output(onnx_graph);
  static std::mutex mtx;
  std::lock_guard<std::mutex> t_lock(mtx);
  auto ep_context_nodes = get_ep_context_nodes(onnx_graph);
  auto context =
      initialize_context(model_path, onnx_graph, ep_context_nodes, options);
  auto measture_compile_onnx_model_3 = context->measure("compile_onnx_model_3");
  // we cannot use get_cache_filename because cache might be a tar file in
  // memory instead of a physical directory.
  bool in_mem = context->cache_in_mem();
  std::unique_ptr<WithFileLock> lock;
  if (!in_mem) {
    lock = std::make_unique<WithFileLock>(
        (context->get_log_dir() / ".lock").u8string().c_str());
  }
  (void)lock;
  auto p_cpu_usage = CreateICPUUsage();
  std::vector<std::unique_ptr<ExecutionProvider>> ret{};
  if (ENV_PARAM(XLNX_ENABLE_SKIP_FATAL)) {
    // clang warning: cannot initialize a parameter of type
    // 'google::logging_fail_func_t' (aka 'void (*)()
    // __attribute__((noreturn))') with an rvalue of type 'void
    // (*)()'
    google::InstallFailureFunction(
        (google::logging_fail_func_t)&compile_fatal_func);
  }
  try {
    ret = compile_onnx_model_internal(onnx_graph, ep_context_nodes, context);
  } catch (const GlogFatalException& e) {
    for (auto&& s : e.stacks) {
      context->context_proto.add_stacks(s);
    }
    LOG(INFO) << "Catch fatal exception, skip this subgraph. Set "
                 "XLNX_ENABLE_SKIP_FATAL=0 to stop skip.\n"
              << e.what();
  }
#ifdef ENABLE_PYTHON
  catch (py::error_already_set& e) {
    (void)e; // suppress unused variable
    if (ENV_PARAM(XLNX_ENABLE_SKIP_FATAL)) {
      LOG(INFO) << " catch pybind11 exception, skip this subgraph:  maybe not "
                   "found vaip python module";
    } else {
      LOG(INFO) << " catch pybind11 exception, maybe not found vaip python "
                   "module , please throw detail message for "
                   "developer";
      abort();
    }
  }
#endif
  catch (const std::exception& e) {
    if (ENV_PARAM(XLNX_ENABLE_SKIP_FATAL)) {
      LOG(INFO) << " catch other exception, skip this subgraph: " << e.what();
    } else {
      LOG(INFO) << " catch exception : " << e.what();
      abort();
    }
  } catch (...) {
    if (ENV_PARAM(XLNX_ENABLE_SKIP_FATAL)) {
      LOG(INFO) << " unknow exception";
    } else {
      LOG(INFO) << " unknow exception";
      abort();
    }
  }

  {
    context->context_proto.clear_cpu_usage();
    auto usage = context->context_proto.add_cpu_usage();
    auto avg_cpu_usage = p_cpu_usage->GetUsage();
    auto peak_working_set_size =
        (float)GetPeakWorkingSetSize() / 1024 / 1024; // MB
    usage->set_avg_cpu_util(avg_cpu_usage);
    usage->set_mem_peak_working_set_size(peak_working_set_size);
    LOG(INFO) << "AVG CPU Usage " << avg_cpu_usage << "%";
    LOG(INFO) << "Peak Working Set size " << peak_working_set_size << " MB";
  }

  print_device_subgraph(*context);
  auto disable_cpu_only =
      context->get_provider_option("vaip_disable_cpu_only_inference", "0");
  if (disable_cpu_only == "1") {
    if (is_cpu_only_inference(*context)) {
      LOG(ERROR) << "[Vitis AI EP][DISABLE CPU ONLY] The model's NPU "
                    "offload is 0";
      abort();
    }
  }
  return ret;
}

thread_local const void* g_state = nullptr;
thread_local vaip_core::DllSafe<std::string> (*g_get_config_entry)(
    const void* state, const char* entry_name) = nullptr;

int vitisai_ep_on_run_start(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const void* state,
    vaip_core::DllSafe<std::string> (*get_config_entry)(
        const void* state, const char* entry_name)) {
  if (eps.empty()) {
    return 0;
  }
  auto ep =
      dynamic_cast<vaip_core::ExecutionProviderConcrete*>(eps.front().get());
  auto p_context =
      dynamic_cast<vaip_core::PassContextImp*>(ep->get_context().get());
  CHECK(p_context != nullptr);
  g_state = state;
  g_get_config_entry = get_config_entry;
  return 0;
}

int vitisai_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const char* const* keys, const char* const* values, size_t kv_len) {
  if (eps.empty()) {
    return 1;
  }
  auto ep =
      dynamic_cast<vaip_core::ExecutionProviderConcrete*>(eps.front().get());
  auto p_context =
      dynamic_cast<vaip_core::PassContextImp*>(ep->get_context().get());
  CHECK(p_context != nullptr);
  std::lock_guard<std::mutex> lock(p_context->ep_dynamic_options_lock);
  for (auto i = 0; i < kv_len; i++) {
    auto key = std::string(keys[i]);
    auto value = std::string(values[i]);
    if (key == "ep.dynamic.workload_type")
      p_context->update_all_qos(value);
  }
  return 0;
}
} // namespace vaip_core

extern "C" VAIP_DLL_SPEC int vitisai_ep_on_run_start(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const void* state,
    vaip_core::DllSafe<std::string> (*get_config_entry)(
        const void* state, const char* entry_name)) {
  return vaip_core::vitisai_ep_on_run_start(eps, state, get_config_entry);
}

extern "C" VAIP_DLL_SPEC int vitisai_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const char* const* keys, const char* const* values, size_t kv_len) {
  return vaip_core::vitisai_ep_set_ep_dynamic_options(eps, keys, values,
                                                      kv_len);
}
