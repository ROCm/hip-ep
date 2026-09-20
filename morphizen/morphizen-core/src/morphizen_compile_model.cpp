/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS
// clang-format off
#include <cstdint>
#include <glog/logging.h>

#include "md5.h"
#include "sha256.h"
#include "./config.hpp"
#include "./file_lock.hpp"
#include "./logger_adapter.hpp"
#include "./pass_imp.hpp"
#include "./version_info.hpp"
#include "profile_utils.hpp"
#include "morphizen/cache_identity.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/symbolic_dims.hpp"
#include "morphizen/util.hpp"
#include "morphizen-foundation/env_config.hpp"
#include <cctype>
#include <codecvt>
#include <errno.h>
#include <functional>
#include <google/protobuf/util/json_util.h>
#include <ios>
#include <limits>
#include <locale>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <sstream>
#include <set>
#include "morphizen-foundation/encryption.hpp"
#include <onnxruntime_session_options_config_keys.h>
#include "morphizen-foundation/mem_binary.hpp"
#include "morphizen/config_reader.hpp"
// clang-format on

DEF_ENV_PARAM_2(XLNX_ONNX_EP_REPORT_FILE, "", std::string)
DEF_ENV_PARAM(XLNX_ENABLE_CACHE, "1")
DEF_ENV_PARAM(XLNX_ENABLE_SKIP_FATAL, "1")
DEF_ENV_PARAM(HIP_EP_VERBOSE, "0")
DEF_ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY, "0")
DEF_ENV_PARAM_2(DEBUG_MD5_SIG, "", std::string)
DEF_ENV_PARAM(DEBUG_VITIS_AI_EP, "1")
DEF_ENV_PARAM(DEBUG_FILE_LOCK, "0")
DEF_ENV_PARAM(DEBUG_EP_CONTEXT, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_PROFILING, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_VISUALIZATION, "0")
DEF_ENV_PARAM_2(XLNX_VAIML_LEVEL_1_NAME, "morphizen-pass_vaiml_partition",
                std::string)

#ifdef _WIN32
#ifdef ENABLE_PYTHON
// Python is only enabled for VAIML compilation on Windows, which requires
// this threshold to be set to a large value so all constants are cloned for the
// compilation.
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184",
                int64_t)

#else
// Set the threshold to small value to save memory usage for Windows runtime
// package
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "128", int64_t)
#endif
#else
// Set the threshold to a large value nn Linux for VAIML compilation
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184",
                int64_t)
#endif
namespace morphizen {
// this template is to be deprecated. this function is to be deprecated, please
// use get_provier_option, only support XLNX_model_clone_external_data_threshold
// for backward compatibility.
template <>
int64_t PassContext::get_provier_option_with_class<
    ENV_PARAM_XLNX_model_clone_external_data_threshold>() const {
  using env_name = ENV_PARAM_XLNX_model_clone_external_data_threshold;
  const char *name = env_name::get_name();
  const char *defvalue = env_name::get_default_value();
  auto p = get_provider_option(std::string(name), std::string(defvalue));
  using helper = typename morphizen::foundation::env_config_helper<
      decltype(env_name::value)>;
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
} // namespace morphizen
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VITIS_AI_EP) >= n)

DEF_ENV_PARAM_2(XLNX_MD5_SIG_SKIP_OPS, "QuantizeLinear,DequantizeLinear",
                std::vector<std::string>)

#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(HIP_EP_VERBOSE) >= n) << "[HIP_EP_VERBOSE] "
#ifdef ENABLE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
using namespace onnxruntime;

namespace morphizen {

static inline void remove_encryption(ConfigProto & /*proto*/) {
  // encryption_key removed from ConfigProto (Issue #004)
  // Now read from provider_options directly when needed
}

static void print_device_subgraph(const PassContextImp &context) {
  LOG_VERBOSE(2) << "dpu subgraph: " << context.context_proto.meta_def_size();
}

static void update_cache(std::shared_ptr<PassContextImp> context,
                         onnxruntime::Graph &graph) {
  auto deferred_write = std::shared_ptr<void>(
      nullptr, [context](void * /*p*/) { context->save_context_json(); });
  auto measure_update_cache = context->measure("update_cache");
  auto effective_passes = context->compute_effective_passes();
  auto passes = IPass::create_passes(context, effective_passes);
  IPass::run_passes(passes, graph);
}

void read_cache(std::shared_ptr<PassContextImp> context) {
  auto measure = context->measure("read_cache");
  context->update_pass_context_from_context_json_in_cache();
}

bool check_cache_hit(PassContextImp &context) {
  auto measure_check_cache_hit = context.measure("check_cache_hit");
  auto prebuild_cache_context_name =
      context.get_provider_option("prebuild_cache_context");
  if (prebuild_cache_context_name) {
    bool available = has_mem_binary(prebuild_cache_context_name.value());
    std::string_view expected_key = context.context_proto.cache_key();
    CacheLoadAction action = select_cache_load_action(
        CacheLoadKind::Prebuilt, context.initializer_digest_finalized,
        available, expected_key, expected_key);
    if (action == CacheLoadAction::Recompile) {
      if (!context.initializer_digest_finalized) {
        LOG(WARNING)
            << "ignoring prebuilt cache because initializer bytes were "
               "not included in the finalized key";
      } else {
        LOG(WARNING) << "prebuilt cache '"
                     << prebuild_cache_context_name.value()
                     << "' is unavailable; compiling a fresh artifact";
      }
      return false;
    }
    try {
      auto prebuild_ep_context_in_mem =
          get_mem_binary(prebuild_cache_context_name.value());
      context.create_tar_file_for_prebuild_cache(
          std::move(prebuild_ep_context_in_mem));
      MY_LOG(1) << "==== prebuild_cache_context candidate ====";
      return true;
    } catch (const CacheIntegrityError &error) {
      LOG(WARNING) << "prebuilt cache rejected: " << error.what();
      context.maybe_create_tar_file_for_write();
      return false;
    }
  }
  // No file-based cache - always miss for non-prebuild scenarios
  return false;
}

static int64_t compute_model_clone_threshold(const ConfigProto &config_proto,
                                             PassContext *context) {
  // Check provider option first (user override)
  auto po_threshold =
      context->get_provider_option("XLNX_model_clone_external_data_threshold");
  if (po_threshold) {
    return std::stoll(po_threshold.value());
  }

  // Check if VAIML plugin is enabled - VAIML needs all constants cloned
  // (effectively disables the optimization by using very large threshold)
  for (auto &pass : config_proto.passes()) {
    if (pass.plugin() == ENV_PARAM(XLNX_VAIML_LEVEL_1_NAME)) {
      return 17179869184; // Large threshold for VAIML
    }
  }

  // Default from ENV_PARAM
  return ENV_PARAM(XLNX_model_clone_external_data_threshold);
}

void compile_onnx_model_2(std::shared_ptr<PassContextImp> context,
                          const Graph &onnx_graph) {
  bool cache_hit = check_cache_hit(*context);
  if (cache_hit) {
    try {
      read_cache(context);
      MY_LOG(1) << "==== cache hit ====";
      return;
    } catch (const CacheIntegrityError &error) {
      CacheLoadAction action = select_cache_load_action(
          CacheLoadKind::Prebuilt, context->initializer_digest_finalized,
          /*cache_available=*/true, context->context_proto.cache_key(),
          /*loaded_key=*/{});
      if (action != CacheLoadAction::Recompile)
        throw;
      LOG(WARNING) << "prebuilt cache rejected: " << error.what()
                   << "; compiling a fresh artifact";
      context->maybe_create_tar_file_for_write();
    }
  }

  auto &model = morphizen_cxx::GraphConstRef(onnx_graph).model();
  int64_t threshold =
      compute_model_clone_threshold(context->get_config_proto(), context.get());
  auto cloned_model = morphizen::model_clone(model, threshold);
  auto &cloned_graph = morphizen::model_main_graph(*cloned_model);
  update_cache(context, cloned_graph);
  auto encryption_key = context->get_provider_option("encryption_key", "");
  read_cache(context);
}

static std::string get_dump_md5_file(const std::string &suffix) {
  auto ret = ENV_PARAM(DEBUG_MD5_SIG);
  if (!ret.empty()) {
    ret = ret + suffix;
  }
  return ret;
}
struct MD5Sig {
public:
  MD5Sig(const std::string &suffix)
      : dump_md5_file{get_dump_md5_file(suffix)} {}
  void add(const void *data, size_t numBytes) {
    md5.add(data, numBytes);
    if (str) {
      CHECK(str->write((const char *)data, numBytes).good())
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
get_model_signature_with_graph_inputs_and_outputs(const Graph &onnx_graph) {
  auto md5 = MD5Sig("_with_io.data");
  auto inputs = morphizen_cxx::GraphConstRef(onnx_graph).inputs();
  for (auto &input : inputs) {
    auto input_name = input.name();
    md5.add(input_name.data(), input_name.size());

    auto shape = node_arg_get_shape_i64(*input.ptr());
    if (shape && !shape->empty()) {
      md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
    }
  }
  auto outputs = morphizen_cxx::GraphConstRef(onnx_graph).outputs();
  for (auto &output : outputs) {
    auto output_name = output.name();
    md5.add(output_name.data(), output_name.size());

    auto shape = output.shape();
    if (shape && !shape->empty()) {
      md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
    }
  }
  return md5.getHash();
}

static std::string get_model_signature(const Graph &onnx_graph) {
  auto md5 = MD5Sig(".data");
  auto graph_ref = morphizen_cxx::GraphConstRef(onnx_graph);
  for (auto &node_ref : graph_ref.nodes_in_topological_order()) {
    auto op_type = node_ref.op_type();
    const auto &skip_op = ENV_PARAM(XLNX_MD5_SIG_SKIP_OPS);
    if (std::find(skip_op.begin(), skip_op.end(), op_type) != skip_op.end()) {
      continue;
    }
    auto output = node_ref.outputs();
    for (auto &node_arg_opt : output) {
      if (!node_arg_opt.has_value()) {
        continue;
      }
      auto &node_arg = node_arg_opt.value();
      auto node_arg_name = node_arg.name();
      md5.add(node_arg_name.data(), node_arg_name.size());

      auto shape = node_arg.shape();
      if (shape && !shape->empty()) {
        md5.add(shape->data(), shape->size() * sizeof(shape->at(0)));
      }
    }
  }
  return md5.getHash();
}

static std::string get_signature(const std::string &model_path,
                                 const Graph &onnx_graph,
                                 const ConfigProto & /*proto*/) {
  auto md5_file_base =
      model_path.empty() ? "" : morphizen::get_md5_of_file(model_path);
  auto md5_in_memory_a = get_model_signature(onnx_graph);
  auto md5_in_memory_b =
      get_model_signature_with_graph_inputs_and_outputs(onnx_graph);

  auto nodes_topo =
      morphizen_cxx::GraphConstRef(onnx_graph).nodes_in_topological_order();
  int32_t node_count = (int32_t)nodes_topo.size();

  MY_LOG(1) << "File base signature : " << md5_file_base;
  MY_LOG(1) << "Algorithm-A: based on topologically ordered signature : "
            << md5_in_memory_a;
  MY_LOG(1) << "Algorithm-B: based on graph inputs/outputs signature : "
            << md5_in_memory_b;
  MY_LOG(1) << "Algorithm-B: node count: " << node_count;
  return md5_in_memory_a;
}

static std::string
get_complete_graph_digest(const std::filesystem::path &model_path,
                          const Graph &graph, const Model &model) {
  if (!model_path.empty()) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open model for cache identity: " +
                               model_path.string());
    SHA256 sha256;
    char buffer[64 * 1024];
    while (input) {
      input.read(buffer, sizeof(buffer));
      sha256.add(buffer, static_cast<size_t>(input.gcount()));
    }
    if (!input.eof())
      throw std::runtime_error("cannot read model for cache identity: " +
                               model_path.string());
    return sha256.getHash();
  }

  if (morphizen_cxx::ModelConstRef(model).has_metadata(
          kCompilerGraphDigestMetadataKey))
    return morphizen_cxx::ModelConstRef(model).get_metadata(
        kCompilerGraphDigestMetadataKey);

  auto serialized = morphizen_cxx::GraphConstRef(graph).save_string();
  if (!serialized.get() || serialized->empty())
    throw std::runtime_error(
        "cannot serialize graph for cache identity finalization");
  return SHA256()(*serialized);
}

static CacheKeyInputs get_cache_key_inputs(const std::string &base_key,
                                           const std::string &graph_digest,
                                           const std::string &compiler_contract,
                                           const Model &model,
                                           std::string &symbolic_bytes,
                                           std::string &initializer_digest,
                                           std::string &compiler_graph_digest) {
  symbolic_bytes = std::string(kOnnxDimParamsEncodingVersion) + "\n0\n";
  auto model_ref = morphizen_cxx::ModelConstRef(model);
  if (model_ref.has_metadata(kOnnxDimParamsMetadataKey))
    symbolic_bytes = model_ref.get_metadata(kOnnxDimParamsMetadataKey);

  if (model_ref.has_metadata(kInitializerDataDigestMetadataKey))
    initializer_digest =
        model_ref.get_metadata(kInitializerDataDigestMetadataKey);
  if (model_ref.has_metadata(kCompilerGraphDigestMetadataKey))
    compiler_graph_digest =
        model_ref.get_metadata(kCompilerGraphDigestMetadataKey);
  return {
      base_key,       graph_digest,       compiler_contract,
      symbolic_bytes, initializer_digest, compiler_graph_digest,
  };
}

std::shared_ptr<PassContextImp> initialize_context(
    const std::string &model_path, const Graph &onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs,
    std::unique_ptr<LoggerAdapter> logger_adapter) {

  std::shared_ptr<PassContextImp> context =
      PassContextImp::create_pass_context(options, session_configs);

  // Store logger and logger_adapter to prolong their lifetime
  if (logger_adapter) {
    context->logger_adapter_ = std::move(logger_adapter);
  }
  // "session.model_external_initializers_file_folder_path/virtual_model.onnx
  // would be passed for in-mem model when this happen, a invalid path is
  // passed, we use the model_path == empty to differentiate if the model is
  // in-mem
  if (std::filesystem::is_regular_file(model_path)) {
    context->model_path = model_path;
  }
  context->is_ep_context_model = !ep_context_nodes.empty();
  auto &model = morphizen_cxx::GraphConstRef(onnx_graph).model();
  auto md5 =
      get_signature(context->model_path.string(), onnx_graph, context->config_);
  std::string complete_graph_digest;
  if (!context->is_ep_context_model)
    complete_graph_digest =
        get_complete_graph_digest(context->model_path, onnx_graph, model);

  // Target selection and effective pass configuration are compiler inputs, so
  // resolve them before freezing the cache identity.
  context->target_auto_discovery(model);
  std::string compiler_contract;
  if (!context->is_ep_context_model) {
    std::vector<std::string> contract_fields;
    contract_fields.push_back(serialize_deterministically(context->config_));
    if (context->target_proto_)
      contract_fields.push_back(
          serialize_deterministically(*context->target_proto_));
    contract_fields.push_back(
        serialize_deterministically(context->context_proto.version()));
    for (const PassProto &pass : context->compute_effective_passes())
      contract_fields.push_back(serialize_deterministically(pass));
    static const std::set<std::string> non_compiler_options = {
        "cache_key",      "cacheKey", "config_file",
        "encryption_key", "target",   "prebuild_cache_context",
    };
    for (const auto &[key, value] : context->get_all_provider_options()) {
      if (non_compiler_options.count(key))
        continue;
      contract_fields.push_back(key);
      contract_fields.push_back(value);
    }
    compiler_contract = compute_framed_sha256(contract_fields);
  }

  if (!context->context_proto.cache_key().empty()) {
    MY_LOG(1) << "use cache key specified by user "
              << context->context_proto.cache_key();
  } else if (morphizen_cxx::ModelConstRef(model).has_metadata(
                 "morphizen_model_md5sum")) {
    auto new_cache_key = morphizen_cxx::ModelConstRef(model).get_metadata(
        "morphizen_model_md5sum");
    MY_LOG(1) << "use cache key in meta-data " << new_cache_key;
    *context->context_proto.mutable_cache_key() = new_cache_key;
  } else if (ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY) &&
             (!context->model_path.empty())) {
    auto new_cache_key =
        morphizen::get_md5_of_file(context->model_path.string());
    MY_LOG(1) << "use cache key on-disk " << new_cache_key;
    *context->context_proto.mutable_cache_key() = new_cache_key;
  } else {
    auto new_cache_key = md5;
    LOG_VERBOSE(1) << "use cache key in memory signature " << new_cache_key;
    *context->context_proto.mutable_cache_key() = new_cache_key;
  }
  if (!context->is_ep_context_model) {
    std::string symbolic_bytes;
    std::string initializer_digest;
    std::string compiler_graph_digest;
    CacheKeyInputs inputs = get_cache_key_inputs(
        context->context_proto.cache_key(), complete_graph_digest,
        compiler_contract, model, symbolic_bytes, initializer_digest,
        compiler_graph_digest);
    auto finalized = finalize_cache_key(inputs);
    MY_LOG(1) << "finalized symbolic-aware cache key " << finalized;
    *context->context_proto.mutable_cache_key() = std::move(finalized);
    context->cache_key_finalized = true;
    context->initializer_digest_finalized =
        morphizen_cxx::ModelConstRef(model).has_metadata(
            kInitializerDataDigestMetadataKey);
  }
  // log version of binary
  context->print_version_info("EXEC VERSION: ");
  if (!context->is_ep_context_model) {
    context->maybe_create_tar_file_for_write();
  }
  return context;
}
static void get_ep_cache_context_common(PassContextImp &context,
                                        std::ostream &dst) {
  auto measure_get_ep_cache_context_embed_mode =
      context.measure("get_ep_cache_context_common");
  auto reader_temp = context_cache_files_to_tar_stream(context);
  std::istream &reader = *reader_temp;

  auto encryption_key = context.get_provider_option("encryption_key", "");
  if (!encryption_key.empty()) {
    auto filtered = stream_filter(
        reader,
        [](std::istream &src, std::ostream &dst,
           const std::string &encryption_key) {
          morphizen_encryption::aes_encryption(src, dst, encryption_key);
        },
        encryption_key);
    stream_copy(*filtered, dst);
  } else {
    stream_copy(reader, dst);
  }
}

std::string get_ep_cache_context_embed_mode(PassContextImp &context) {
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
    MemoryOutputStreambuf membuf(out);
    std::ostream dst(&membuf);
    get_ep_cache_context_common(context, dst);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
        << "embed mode = 1, load cache directory  to tar memory " << out.size()
        << " bytes";
    return std::string(out.begin(), out.end());
  }
}

static std::string get_ep_cache_context_nonembed_mode(PassContextImp &context) {
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
    std::ofstream dst(OrtSessionOptionEpContextFilePath_binay,
                      std::ios::binary);
    get_ep_cache_context_common(context, dst);
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

static std::string get_ep_cache_context(PassContextImp &context,
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

#if MORPHIZEN_ORT_API_MAJOR < 6
static std::string escape_json(const std::string &s) {
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
static std::string get_nodes(PassContextImp &context) {
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
template <typename T> static std::string combine_outputs_name(T &collection) {
  std::ostringstream oss;
  auto sorted_names =
      std::set<std::string>(collection.begin(), collection.end());
  for (auto &name : sorted_names) {
    oss << " " << name;
  }
  return oss.str();
}
static std::vector<std::optional<morphizen_cxx::NodeArgConstRef>>
convert_to_node_arg_const_ref(morphizen_cxx::GraphRef g,
                              const std::vector<std::string> &names) {
  auto ret = std::vector<std::optional<morphizen_cxx::NodeArgConstRef>>();
  std::transform(
      names.begin(), names.end(), std::back_inserter(ret),
      [&g](const std::string &name)
          -> std::optional<morphizen_cxx::NodeArgConstRef> {
        if (name.empty()) {
          return std::nullopt;
        }
        // find node_arg by name frist, beacuse maybe an EPContext's output is
        // the input to another EPContext node
        auto node_arg = g.find_node_arg(name);
        if (node_arg.has_value()) {
          return node_arg;
        }
        return std::optional<morphizen_cxx::NodeArgConstRef>(g.new_node_arg(
            name, {}, onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
      });
  return ret;
}

static onnxruntime::Node *
create_ep_context_node(morphizen::ExecutionProviderConcrete *ep, int index) {
  CHECK(ep != nullptr);
  auto p_context = dynamic_cast<PassContextImp *>(ep->get_context().get());
  CHECK(p_context != nullptr);
  auto &context = *p_context;

  if (ENV_PARAM(DEBUG_EP_CONTEXT) >= 2) {
    LOG(INFO) << "create ep context node , index=" << index;
    LOG(INFO) << "Input meta-defs: "
              << morphizen::combine_outputs_name(*ep->get_meta_def_inputs());
    LOG(INFO) << "Output meta-defs: "
              << morphizen::combine_outputs_name(*ep->get_meta_def_outputs());
  }

  if (!context.ep_context_model_) {
    context.ep_context_model_ =
        morphizen_cxx::Model::create(context.model_path, {{"ai.onnx", 21}});
  }
  auto ep_context_graph = context.ep_context_model_->main_graph();
  auto op_type = "EPContext";
  auto op_domain = "com.microsoft";
  auto description = "description";
  auto fused_node = ep->get_fused_node();
  auto input_args = fused_node ? morphizen_cxx::NodeConstRef::from_node(
                                     ep_context_graph, *fused_node)
                                     .inputs()
                               : convert_to_node_arg_const_ref(
                                     morphizen_cxx::GraphRef(ep_context_graph),
                                     *ep->get_meta_def_inputs());
  auto output_args = fused_node ? morphizen_cxx::NodeConstRef::from_node(
                                      ep_context_graph, *fused_node)
                                      .outputs()
                                : convert_to_node_arg_const_ref(
                                      morphizen_cxx::GraphRef(ep_context_graph),
                                      *ep->get_meta_def_outputs());
  // for new ABI EP, fused_node is nullptr
  auto name = fused_node ? morphizen_cxx::NodeConstRef::from_node(
                               ep_context_graph, *fused_node)
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
  attrs.add("source", std::string("MorphiZenExecutionProvider"));
  attrs.add("dump_dir", context.get_dump_directory().u8string());
  attrs.add("onnx_model_filename", context.model_path.u8string());
  attrs.add("partition_name", name);
  bool enable_encryption =
      morphizen_encryption::has_encryption_support() &&
      (!context.get_provider_option("encryption_key", "").empty());
  attrs.add("enable_encryption", (int64_t)enable_encryption);
  // Always use cache_key prefix - only store the prefix itself
  attrs.add("cache_file_prefix", context.get_context_proto().cache_key());
  auto &version_infos = context.get_context_proto().version();
  for (const auto &version_info : version_infos.version_infos()) {
    auto lib_name = "version_of_" + version_info.package_name();
    attrs.add(lib_name, version_info.version());
    lib_name = "version_id_of_" + version_info.package_name();
    attrs.add(lib_name, version_info.commit());
  }
#if MORPHIZEN_ORT_API_MAJOR < 6
  auto notes = get_nodes(context);
  attrs.add("notes", notes);
#endif
  auto ep_cache_context = std::string();
  if (main_context) {
    ep_cache_context = get_ep_cache_context(context, embed_mode != 0);
  }
  attrs.add("ep_cache_context", ep_cache_context);

  // Add detailed version information from DLL resource
  // (3rd-party/ryzenai_bin_metadata/version.rc.in) These values are read
  // directly from the DLL's embedded version resource
  attrs.add("CompanyName", morphizen::get_dll_company_name());
  attrs.add("ProductName", morphizen::get_dll_product_name());
  attrs.add("LegalCopyright", morphizen::get_dll_legal_copyright());
  attrs.add("FileVersion", morphizen::get_dll_file_version());
  attrs.add("ProductVersion", morphizen::get_dll_product_version());
  attrs.add("FileDescription", morphizen::get_dll_file_description());

  // Add library name and commit ID
  attrs.add("LibraryName", morphizen::get_lib_name());
  attrs.add("LibraryCommitId", morphizen::get_lib_id());

  auto ret = morphizen_cxx::GraphRef(ep_context_graph)
                 .add_node(name, op_domain, op_type, description, input_args,
                           output_args, attrs.build());
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT)) << "add ep node:" << ret;
  return ret.ptr();
}
static void init_ep_context_model_inputs(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps) {
  // guess graph input
  std::set<std::string> all_ep_inputs;
  std::set<std::string> all_ep_outputs;
  for (const auto &ep : eps) {
    auto ep_concrete =
        dynamic_cast<const morphizen::ExecutionProviderConcrete *>(ep.get());
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
  if (auto ep = dynamic_cast<morphizen::ExecutionProviderConcrete *>(
          eps.front().get())) {
    if (auto p_context =
            dynamic_cast<PassContextImp *>(ep->get_context().get())) {
      if (!p_context->ep_context_model_) {
        p_context->ep_context_model_ = morphizen_cxx::Model::create(
            p_context->model_path, {{"ai.onnx", 21}});
      }
      auto ep_context_graph = p_context->ep_context_model_->main_graph();
      auto optional_inputs =
          convert_to_node_arg_const_ref(ep_context_graph, graph_inputs);
      std::vector<morphizen_cxx::NodeArgConstRef> actual_inputs;
      for (const auto &opt_input : optional_inputs) {
        if (opt_input.has_value()) {
          actual_inputs.push_back(opt_input.value());
        }
      }
      morphizen_cxx::GraphRef(ep_context_graph).set_inputs(actual_inputs);
    }
  }
}
extern "C" MORPHIZEN_DLL_SPEC int create_ep_context_nodes(
#if MORPHIZEN_ORT_API_MAJOR < 6
    onnxruntime::Graph & /*ep_context_graph unused to deleted*/,
#endif
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    morphizen::DllSafe<std::vector<Node *>> *ret_value) {
  std::vector<Node *> ret;
  if (eps.empty()) {
    *ret_value =
        morphizen::DllSafe<std::vector<Node *>>(new std::vector<Node *>());
    return 1;
  }
  auto ep =
      dynamic_cast<morphizen::ExecutionProviderConcrete *>(eps.front().get());

  auto p_context = dynamic_cast<PassContextImp *>(ep->get_context().get());
  CHECK(p_context != nullptr);
  if (p_context->is_ep_context_model) {
    // cannot create a ep context model when it is already a ep context
    *ret_value =
        morphizen::DllSafe<std::vector<Node *>>(new std::vector<Node *>());
    return 1;
  }
  auto &context = *p_context;
  auto deferred_write =
      std::shared_ptr<void>(nullptr, [&context](void * /*p*/) {
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
  for (auto &ep_1 : eps) {
    ret.push_back(create_ep_context_node(
        dynamic_cast<morphizen::ExecutionProviderConcrete *>(ep_1.get()),
        ep_index++));
  }
  *ret_value = morphizen::DllSafe<std::vector<Node *>>(
      new std::vector<Node *>(std::move(ret)));
  return 0;
}

static std::vector<morphizen_cxx::NodeConstRef>
get_ep_context_nodes(morphizen_cxx::GraphConstRef onnx_graph) {
  auto ret = std::vector<morphizen_cxx::NodeConstRef>();
  auto nodes = onnx_graph.nodes();
  for (auto node : nodes) {
    if (node.op_type() == "EPContext" && node.op_domain() == "com.microsoft") {
      if (node.has_attr("source") &&
          node.get_attr_string("source") == "MorphiZenExecutionProvider") {
        ret.push_back(node);
      }
    }
  }
  return ret;
}

static void update_meta_def_from_ep_node(morphizen_cxx::NodeConstRef node,
                                         MetaDefProto &meta_def) {
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
static std::optional<morphizen_cxx::NodeConstRef> get_main_ep_context_node(
    std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes) {
  std::optional<morphizen_cxx::NodeConstRef> ret = std::nullopt;
  auto count_main_context = 0;
  for (auto node : ep_context_nodes) {
    if (node.has_attr("main_context") && node.get_attr_int("main_context")) {
      count_main_context++;
      ret = node;
    }
  }
  if (count_main_context != 1)
    throw CacheIntegrityError(
        "EPContext model must contain exactly one main context node");
  return ret;
}

static void
store_cache_directory_from_main_node(PassContextImp &context,
                                     morphizen_cxx::NodeConstRef main_node) {
  int64_t enable_encryption = main_node.get_attr_int("enable_encryption", 0);
  int64_t ep_embed_mode = main_node.get_attr_int("embed_mode", 1);
  // Always use cache_key prefix - validate it exists
  auto loaded_cache_key = main_node.get_attr_string("cache_file_prefix", "");
  if (!is_finalized_cache_key(loaded_cache_key))
    throw CacheIntegrityError(
        "EPContext node has an invalid finalized cache_file_prefix");
  *context.context_proto.mutable_cache_key() = loaded_cache_key;
  context.cache_key_finalized = true;
#if MORPHIZEN_ORT_API_MAJOR >= 12
  auto ep_cache_context = main_node.release_attr_string("ep_cache_context");
#else
#error "not supported any more"
#endif
  auto ep_context_size = ep_cache_context->size();

  // Always use tar_file_, decrypt if needed
  if (enable_encryption) {
    // Decrypt the encrypted tar data first
    auto encryption_key = context.get_provider_option("encryption_key", "");
    if (encryption_key.empty()) {
      throw morphizen_encryption::EncryptionError(
          "enable_encryption is set, but encryption_key is empty");
    }

    // Create input stream from encrypted data
    std::istringstream encrypted_src(
        std::string(ep_cache_context->data(), ep_context_size),
        std::ios::binary);

    // Decrypt using stream_filter (returns std::unique_ptr<std::istream>)
    auto decrypted_reader = stream_filter(
        encrypted_src,
        [](std::istream &src, std::ostream &dst,
           const std::string &encryption_key) {
          morphizen_encryption::aes_decryption(src, dst, encryption_key);
        },
        encryption_key);

    // Read decrypted data into buffer
    std::vector<char> decrypted_buffer;
    char read_buffer[8192];
    while (decrypted_reader->read(read_buffer, sizeof(read_buffer)) ||
           decrypted_reader->gcount() > 0) {
      decrypted_buffer.insert(decrypted_buffer.end(), read_buffer,
                              read_buffer + decrypted_reader->gcount());
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
        << "Decrypted ep context: " << decrypted_buffer.size() << " bytes";

    // Load decrypted tar data into tar_file_
    context.create_tar_file_for_read(
        std::string(decrypted_buffer.begin(), decrypted_buffer.end()),
        ep_embed_mode != 0);
  } else {
    // No encryption: Load tar data directly
    context.create_tar_file_for_read(std::move(*ep_cache_context),
                                     ep_embed_mode != 0);
  }
}

static int64_t get_ep_context_index(const morphizen_cxx::NodeConstRef &node) {
  if (!node.has_attr("index"))
    throw CacheIntegrityError("EPContext node has no index attribute");
  return node.get_attr_int("index");
}

static std::vector<std::unique_ptr<ExecutionProvider>>
create_execution_providers_from_ep_context_nodes(
    std::shared_ptr<PassContextImp> context,
    std::vector<morphizen_cxx::NodeConstRef> ep_context_nodes) {
  if (ep_context_nodes.size() !=
      static_cast<size_t>(context->context_proto.meta_def_size()))
    throw CacheIntegrityError(
        "EPContext node count does not match cached metadata");
  auto size = ep_context_nodes.size();
  auto ret = std::vector<std::unique_ptr<ExecutionProvider>>();
  ret.reserve(size);

  std::sort(ep_context_nodes.begin(), ep_context_nodes.end(),
            [](const morphizen_cxx::NodeConstRef &a,
               const morphizen_cxx::NodeConstRef &b) {
              return get_ep_context_index(a) < get_ep_context_index(b);
            });
  for (auto idx = 0u; idx < size; ++idx) {
    auto node = ep_context_nodes[idx];
    auto index = get_ep_context_index(node);
    if (index != static_cast<int64_t>(idx))
      throw CacheIntegrityError("EPContext node index mismatch");
    auto meta_def_index = idx;
    auto &meta_def = *context->context_proto.mutable_meta_def(meta_def_index);
    update_meta_def_from_ep_node(node, meta_def);
    auto device = meta_def.device();
    auto plugin_name = std::string("morphizen_custom_op_") + device;
    ret.emplace_back(
        ExecutionProviderConcrete::create(plugin_name, context, meta_def));
  }
  return ret;
}
std::vector<std::unique_ptr<ExecutionProvider>>
restore_execution_providers_from_ep_context_model(
    morphizen_cxx::GraphConstRef /*onnx_graph*/,
    std::shared_ptr<PassContextImp> context,
    std::vector<morphizen_cxx::NodeConstRef> ep_context_nodes) {
  auto measture =
      context->measure("restore_execution_providers_from_ep_context_model");
  LOG_IF(INFO, ENV_PARAM(DEBUG_EP_CONTEXT))
      << "Running EP context onnx model , restore ExecutionProviders from EP "
         "context model";
  // untar the cache directory.
  auto main_node = get_main_ep_context_node(ep_context_nodes);
  // print some attr in main_node if attr is exist
  if (main_node.has_value()) {
    std::vector<std::string> attr_names = {
        "ProductVersion", "onnx_model_filename", "cache_file_prefix"};
    for (auto &attr_name : attr_names) {
      if (main_node.value().has_attr(attr_name)) {
        MY_LOG(1) << "main_node attr " << attr_name << " = "
                  << main_node.value().get_attr_string(attr_name);
      }
    }
  }
  if (!main_node)
    throw CacheIntegrityError("EPContext model has no main context node");

  auto user_cache_key = context->get_provider_option("cache_key");
  if (!user_cache_key.has_value()) {
    user_cache_key = context->get_provider_option("cacheKey");
  }
  if (user_cache_key.has_value() && !user_cache_key->empty()) {
    LOG(WARNING)
        << "User provided cache key '" << user_cache_key.value()
        << "' in provider options. "
        << "You should not provide a cache key when using EP context model.";
  }

  store_cache_directory_from_main_node(*context, main_node.value());

  auto ep_context_cache_key = context->context_proto.cache_key();
  if (user_cache_key.has_value() && !user_cache_key->empty() &&
      is_finalized_cache_key(*user_cache_key) &&
      select_cache_load_action(CacheLoadKind::EpContext,
                               /*initializer_digest_finalized=*/true,
                               /*cache_available=*/true, *user_cache_key,
                               ep_context_cache_key) ==
          CacheLoadAction::Reject) {
    throw CacheIntegrityError(
        "provider cache key does not match the EPContext cache key");
  }

  context->update_pass_context_from_context_json_in_cache();
  return create_execution_providers_from_ep_context_nodes(context,
                                                          ep_context_nodes);
}
static void log_stat_subgraph(const ContextProto &context_proto) {
  auto stat = std::map<std::string, int>{};
  for (auto &meta_def : context_proto.meta_def()) {
    stat[meta_def.device()]++;
  }
  // as per AIESW-11754 request, use LOG(INFO) instead of std::cout
  LOG(INFO) << "[MorphiZen EP] No. of Subgraphs supported by MorphiZen EP:";
  for (const auto &subgraph_stat : stat) {
    LOG(INFO) << std::setw(6) << subgraph_stat.first << std::setw(6)
              << subgraph_stat.second << " ";
  }
}
static std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_internal(
    const Graph &onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
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
    auto measure_before_compile_onnx_model_2 =
        context->measure("before_compile_onnx_model_internal");
    compile_onnx_model_2(context, onnx_graph);
    measure_after_compile_onnx_model_2 =
        context->measure("after_compile_onnx_model_internal");
    ret.reserve(context->context_proto.meta_def_size());
    for (auto &meta_def : *context->context_proto.mutable_meta_def()) {
      std::string device = meta_def.device();
      auto plugin_name = std::string("morphizen_custom_op_") + device;
      ret.emplace_back(
          ExecutionProviderConcrete::create(plugin_name, context, meta_def));
    }
  }
  return ret;
}

static std::vector<std::string> GetStackTrace() {
  // Stack trace functionality disabled
  return std::vector<std::string>();
}

struct GlogFatalException : public std::exception {
public:
  virtual const char *what() const throw() { return m.c_str(); }
  std::string m;
  std::vector<std::string> stacks;
};

static void compile_fatal_func() {
  GlogFatalException e;
  e.stacks = GetStackTrace();
  for (auto &&t : e.stacks) {
    e.m += std::string(t) + "\n";
  }
  throw e;
}

static std::ostream &operator<<(std::ostream &s,
                                const std::vector<int64_t> &v) {
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
static bool is_cpu_only_inference(const PassContextImp &context) {
  auto ret = false;
  if (context.context_proto.meta_def_size() == 0) {
    ret = true;
  }
  return ret;
}

static void print_graph_input_and_output(const Graph &onnx_graph) {
#ifdef _WIN32
  _setmaxstdio(8192);
#endif

  auto graph_ref = morphizen_cxx::GraphConstRef(onnx_graph);
  auto graph_inputs = graph_ref.inputs();
  auto graph_outputs = graph_ref.outputs();

  LOG(INFO) << "MorphiZen EP Load ONNX Model Success";
  LOG(INFO) << "Graph Input Node Name/Shape (" << graph_inputs.size() << ")";
  for (auto &input : graph_inputs) {
    auto shape = input.shape();
    if (shape != nullptr) {
      LOG(INFO) << "\t " << input.name() << " : " << *(shape.get());
    } else {
      LOG(INFO) << "\t " << input.name() << " : []";
    }
  }
  LOG(INFO) << "Graph Output Node Name/Shape (" << graph_outputs.size() << ")";
  for (auto &output : graph_outputs) {
    auto shape = output.shape();
    if (shape != nullptr) {
      LOG(INFO) << "\t " << output.name() << " : " << *(shape.get());
    } else {
      LOG(INFO) << "\t " << output.name() << " : []";
    }
  }
}
// Internal helper that accepts logger_adapter for lifetime management
std::vector<std::unique_ptr<ExecutionProvider>> compile_onnx_model_3_internal(
    const std::string &model_path, const Graph &onnx_graph,
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs,
    std::unique_ptr<LoggerAdapter> logger_adapter,
    std::function<void(int, const char *)> set_ort_status) {
  print_graph_input_and_output(onnx_graph);
  static std::mutex mtx;
  std::lock_guard<std::mutex> t_lock(mtx);
  auto ep_context_nodes = get_ep_context_nodes(onnx_graph);
  auto context =
      initialize_context(model_path, onnx_graph, ep_context_nodes, options,
                         session_configs, std::move(logger_adapter));
  auto measture_compile_onnx_model_3 = context->measure("compile_onnx_model_3");
  // Cache is always in memory (tar_file_), no file lock needed
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
  } catch (const GlogFatalException &e) {
    for (auto &&s : e.stacks) {
      context->context_proto.add_stacks(s);
    }
    LOG(INFO) << "Catch fatal exception, skip this subgraph. Set "
                 "XLNX_ENABLE_SKIP_FATAL=0 to stop skip.\n"
              << e.what();
  }
#ifdef ENABLE_PYTHON
  catch (py::error_already_set &e) {
    (void)e; // suppress unused variable
    if (ENV_PARAM(XLNX_ENABLE_SKIP_FATAL)) {
      LOG(INFO) << " catch pybind11 exception, skip this subgraph:  maybe not "
                   "found morphizen python module";
    } else {
      LOG(INFO)
          << " catch pybind11 exception, maybe not found morphizen python "
             "module , please throw detail message for "
             "developer";
      abort();
    }
  }
#endif
  catch (const CacheIntegrityError &e) {
    if (set_ort_status)
      set_ort_status(ORT_FAIL, e.what());
    return {};
  } catch (const morphizen_encryption::EncryptionError &e) {
    if (set_ort_status) {
      set_ort_status(1, e.what());
    }
    return {};
  } catch (const std::exception &e) {
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
      context->get_provider_option("morphizen_disable_cpu_only_inference", "0");
  if (disable_cpu_only == "1") {
    if (is_cpu_only_inference(*context)) {
      LOG(ERROR) << "[MorphiZen EP][DISABLE CPU ONLY] The model's NPU "
                    "offload is 0";
      abort();
    }
  }
  return ret;
}

// Public API - calls internal version without logger
std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3(const std::string &model_path, const Graph &onnx_graph,
                     const onnxruntime::ProviderOptions &options,
                     const std::map<std::string, std::string> &session_configs,
                     std::function<void(int, const char *)> set_ort_status) {
  return compile_onnx_model_3_internal(model_path, onnx_graph, options,
                                       session_configs, nullptr,
                                       set_ort_status);
}

thread_local const void *g_state = nullptr;
thread_local morphizen::DllSafe<std::string> (*g_get_config_entry)(
    const void *state, const char *entry_name) = nullptr;

int morphizen_ep_on_run_start(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const void *state,
    morphizen::DllSafe<std::string> (*get_config_entry)(
        const void *state, const char *entry_name)) {
  if (eps.empty()) {
    return 0;
  }
  auto ep =
      dynamic_cast<morphizen::ExecutionProviderConcrete *>(eps.front().get());
  auto p_context =
      dynamic_cast<morphizen::PassContextImp *>(ep->get_context().get());
  CHECK(p_context != nullptr);
  g_state = state;
  g_get_config_entry = get_config_entry;
  return 0;
}

int morphizen_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const char *const *keys, const char *const *values, size_t kv_len) {
  if (eps.empty()) {
    return 1;
  }
  auto ep =
      dynamic_cast<morphizen::ExecutionProviderConcrete *>(eps.front().get());
  auto p_context =
      dynamic_cast<morphizen::PassContextImp *>(ep->get_context().get());
  CHECK(p_context != nullptr);
  std::lock_guard<std::mutex> lock(p_context->ep_dynamic_options_lock);
  for (size_t i = 0; i < kv_len; i++) {
    auto key = std::string(keys[i]);
    auto value = std::string(values[i]);
    if (key == "ep.dynamic.workload_type")
      p_context->update_all_qos(value);
  }
  return 0;
}
} // namespace morphizen

extern "C" MORPHIZEN_DLL_SPEC int morphizen_ep_on_run_start(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const void *state,
    morphizen::DllSafe<std::string> (*get_config_entry)(
        const void *state, const char *entry_name)) {
  return morphizen::morphizen_ep_on_run_start(eps, state, get_config_entry);
}

extern "C" MORPHIZEN_DLL_SPEC int morphizen_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const char *const *keys, const char *const *values, size_t kv_len) {
  return morphizen::morphizen_ep_set_ep_dynamic_options(eps, keys, values,
                                                        kv_len);
}
