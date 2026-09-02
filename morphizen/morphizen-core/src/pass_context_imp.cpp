/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <onnxruntime_session_options_config_keys.h>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif

#include "config.hpp"
#include "ep_shared_context_workspace.hpp"
#include "mem_stream_buffer.hpp"
#include "morphizen-foundation/env_config.hpp"
#include "morphizen-foundation/mem_binary.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/util.hpp"
#include "morphizen/weak.hpp"
#include "pass_context_imp.hpp"
#include "profile_utils.hpp"
#include <map>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY, "0")
DEF_ENV_PARAM(HIP_EP_VERBOSE, "0")
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(HIP_EP_VERBOSE) >= n) << "[HIP_EP_VERBOSE] "

namespace morphizen {

namespace {
thread_local const void *g_run_option_state = nullptr;
thread_local DllSafe<std::string> (*g_get_run_option_entry)(
    const void *state, const char *entry_name) = nullptr;
} // namespace

void set_run_option_accessor(const void *state,
                             DllSafe<std::string> (*get_entry)(
                                 const void *state, const char *entry_name)) {
  g_run_option_state = state;
  g_get_run_option_entry = get_entry;
}

/// struct WithPass
PassContextImp::WithPass::WithPass(PassContextImp &context, IPass &pass)
    : _context(&context) {
  _context->current_pass_stack.push_back(&pass);
}
PassContextImp::WithPass::~WithPass() {
  _context->current_pass_stack.pop_back();
}

/// static
static MemUsageProto convert_to_chrome_event(const MemUsageProto &mem_usage) {
  auto ret = MemUsageProto();
  {
    std::stringstream stream;
    stream << std::hex << mem_usage.current_memory_in_bytes();
    ret.set_current_memory(stream.str());
  }
  {
    std::stringstream stream;
    stream << std::hex << mem_usage.peak_memory_in_bytes();
    ret.set_peak_memory(stream.str());
  }
  return ret;
}
static FILE *write_to_tmp_file(gsl::span<const char> data) {
  FILE *tmp_file = create_tmpfile();
  CHECK(tmp_file != nullptr) << "tmpfile creation error";
  auto write_size = std::fwrite(data.data(), 1, data.size(), tmp_file);
  CHECK_EQ((size_t)write_size, data.size());
  return tmp_file;
}

static std::string msg_to_json_string(const google::protobuf::Message &msg) {
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  auto json_str = std::string();
  auto status =
      google::protobuf::util::MessageToJsonString(msg, &json_str, options);
  CHECK(status.ok()) << "cannot write json string:" << msg.DebugString();
  return json_str;
}
std::unique_ptr<PassContextImp>
PassContextImp::create_pass_context(const ConfigProto &config_proto1) {
  auto ret = std::make_unique<PassContextImp>(PrivateTag{},
                                              ConfigProto(config_proto1));
  Config::add_version_info(ret->context_proto);
  ret->update_config_proto_root_field();
  return ret;
}
std::unique_ptr<PassContextImp> PassContextImp::create_pass_context(
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs) {
  auto json_config_string = get_config_json_str(options);
  const char *json_config = json_config_string.c_str();
  auto config_proto = ConfigProto();
  if (json_config != nullptr && !std::string(json_config).empty()) {
    Config::merge_config_proto(config_proto, json_config);
  }
  auto ret = create_pass_context(config_proto);
  ret->provider_option_origin_.insert(options.begin(), options.end());
  ret->session_configs_.insert(session_configs.begin(), session_configs.end());
  ret->update_config_proto_root_field();
  return ret;
}
/// struct PassContextImp
int PassContextImp::allocate_suffix()
    const { // it is not a big deal to update suffix_counter
  suffix_counter = suffix_counter + 1;
  return suffix_counter;
}

PassContextImp::WithPass PassContextImp::with_current_pass(IPass &pass) {
  return WithPass(*this, pass);
}

std::filesystem::path PassContextImp::get_dump_directory() const {
  // Check provider option override first
  auto dump_dir_option = get_provider_option("dump_dir", "");
  std::filesystem::path dump_dir;

  if (!dump_dir_option.empty()) {
    dump_dir = std::filesystem::path(dump_dir_option);
  } else {
    // Default: temp/morphizen_dumps/cache_key
    auto temp_dir =
#ifdef _WIN32
        std::filesystem::path("C:\\temp");
#else
        std::filesystem::path("/tmp");
#endif
    dump_dir = temp_dir / "morphizen_dumps" / get_context_proto().cache_key();
  }

  // Create the directory if it doesn't exist
  // Use create_directories which is idempotent (safe if directory exists)
  std::error_code ec;
  std::filesystem::create_directories(dump_dir, ec);
  if (ec && ec != std::errc::file_exists) {
    LOG(WARNING) << "Failed to create dump directory: " << dump_dir
                 << " - Error: " << ec.message();
  }

  return dump_dir;
}

template <typename T1, typename T2>
std::optional<std::string>
PassContextImp::get_provider_option_impl(const T1 &option_names,
                                         const T2 &privider_options) const {
  auto ret = std::optional<std::string>();
  if (privider_options) {
    for (auto &option_name : option_names) {
      auto it = privider_options->find(option_name);
      if (it != privider_options->end()) {
        ret = it->second;
        break;
      }
    }
  }
  return ret;
}
template <typename T1, typename T, typename... T2>
std::optional<std::string> PassContextImp::get_provider_option_impl(
    const T1 &option_names, const T &options1, const T2 &...options) const {
  auto ret = get_provider_option_impl(option_names, options1);
  if (ret) {
    return ret;
  }
  return get_provider_option_impl(option_names, options...);
}

template <typename T1, typename... T2>
std::optional<std::string> PassContextImp::get_provider_option_with_priority(
    const T1 &option_names) const {
  // priority order:
  // 0. provider_option provided by user
  // 1. context_proto
  // 2. target_proto, from target discovery
  //    Target priority
  //        1. provider option
  //        2. heuristic process or method
  //        3. default target in config file
  //
  //  3. default value
  return get_provider_option_impl(
      option_names,
      &provider_option_origin_,                                    //
      &config_.provider_options(),                                 //
      target_proto_ ? &target_proto_->provider_options() : nullptr //
  );
}
std::map<std::string, std::string>
PassContextImp::get_all_provider_options() const {
  auto ret = std::map<std::string, std::string>();
  get_all_provider_option_impl(
      ret,
      &provider_option_origin_,                                    //
      &config_.provider_options(),                                 //
      target_proto_ ? &target_proto_->provider_options() : nullptr //
  );
  return ret;
}

std::vector<PassProto> PassContextImp::compute_effective_passes() const {
  std::vector<PassProto> result;

  // Build pass library map from ConfigProto.passes
  std::unordered_map<std::string, PassProto> pass_map;
  for (const auto &pass : config_.passes()) {
    pass_map[pass.name()] = pass;
  }

  // Target-based pass selection
  if (target_proto_) {
    for (const auto &pass_name : target_proto_->pass()) {
      auto iter = pass_map.find(pass_name);
      CHECK(iter != pass_map.end())
          << "Pass not found in library: " << pass_name;
      result.push_back(iter->second);
    }
  }

  // Note: Anonymous plugin passes are handled differently via
  // IPass::create_pass() which now creates PassProto locally without mutating
  // ConfigProto

  return result;
}

template <typename T>
void PassContextImp::get_all_provider_option_impl(
    std::map<std::string, std::string> &ret, const T &provider_options) const {
  if (provider_options) {
    for (auto &kv : *provider_options) {
      ret.insert({kv.first, kv.second});
    }
  };
}
template <typename T, typename... T1>
void PassContextImp::get_all_provider_option_impl(
    std::map<std::string, std::string> &ret, const T &options1,
    const T1 &...options) const {
  get_all_provider_option_impl(ret, options1);
  get_all_provider_option_impl(ret, options...);
}

std::optional<std::string>
PassContextImp::get_provider_option(const std::string &option_name) const {
  return get_provider_option_with_priority(
      std::array<std::string, 1>{option_name});
}
std::optional<std::string>
PassContextImp::get_session_config(const std::string &option_name) const {
  auto it = session_configs_.find(option_name);
  if (it != session_configs_.end()) {
    return it->second;
  }
  return std::nullopt;
}
std::string
PassContextImp::get_provider_option(const std::string &option_name,
                                    const std::string &default_value) const {
  auto option_value = get_provider_option(option_name);
  if (option_value.has_value()) {
    return option_value.value();
  }
  return default_value;
}

bool PassContextImp::cache_in_mem() const {
  // Cache is always in memory (using tmpfile() via tar_file_)
  return true;
}
PassContextImp::~PassContextImp() {
  // No cleanup needed - tar_file_ cleans itself up
}

int64_t PassContextImp::get_provider_option_i64(const std::string &option_name,
                                                int64_t default_value) const {
  auto config_value = get_provider_option(option_name);
  auto ret = default_value;
  if (config_value.has_value()) {
    ret = std::stoll(config_value.value());
  } else {
    ret = default_value;
  }
  return ret;
}

std::string
PassContextImp::get_session_config(const std::string &option_name,
                                   const std::string &default_value) const {
  auto option_value = get_session_config(option_name);
  if (option_value.has_value()) {
    return option_value.value();
  }
  return default_value;
}

std::string
PassContextImp::get_run_option(const std::string &option_name,
                               const std::string &default_value) const {
  auto value = DllSafe<std::string>();
  if (g_get_run_option_entry != nullptr && g_run_option_state != nullptr) {
    value = g_get_run_option_entry(g_run_option_state, option_name.c_str());
  }
  // OrtRunOptions has no API to enumerate its config entries, so logging every
  // lookup is the only way to see which run options actually reach us.
  const bool found = value.get() != nullptr;
  auto ret = found ? *value : default_value;
  LOG_VERBOSE(1) << "run_option: " << option_name << " = " << ret
                 << (found ? "" : " (default)");
  return ret;
}
std::string
PassContextImp::get_meta_def_param(const MetaDefProto &meta_def) const {
  auto json_str = std::string();
  auto status =
      google::protobuf::util::MessageToJsonString(meta_def.param(), &json_str);
  if (!status.ok()) {
    LOG(FATAL) << "failed to get meta_def param: " << status.ToString();
  }
  return json_str;
}
std::string
PassContextImp::get_ep_dynamic_option(const std::string &option_name,
                                      const std::string &default_value) const {
  std::lock_guard<std::mutex> lock(this->ep_dynamic_options_lock);
  auto it = ep_dynamic_options.find(option_name);
  if (it == ep_dynamic_options.end()) {
    return default_value;
  } else {
    return it->second;
  }
}

void PassContextImp::remove_QosUpdater(QoSUpdateInterface *updater) {
  qos_updaters_.erase(
      std::remove_if(
          qos_updaters_.begin(), qos_updaters_.end(),
          [updater](const std::shared_ptr<QoSUpdateInterface> &item) {
            return item.get() == updater;
          }),
      qos_updaters_.end());
}

void PassContextImp::add_QosUpdater(
    const std::shared_ptr<QoSUpdateInterface> &updater) const {
  CHECK(updater) << "Null QoS updater cannot be added to PassContext";
  qos_updaters_.push_back(updater);
}

void PassContextImp::update_all_qos(const std::string &workload_type) const {
  if (workload_type == "Efficient" || workload_type == "Default") {
    for (const auto &updater : qos_updaters_) {
      CHECK(updater) << "Found null QoS updater in qos_updaters_";
      updater->update_qos(workload_type);
    }
  } else {
    throw std::runtime_error("Invalid workload type: " + workload_type);
  }
}

template <typename T>
std::optional<std::vector<T>>
PassContextImp::read_file_generic(const std::string &filename) const {
  std::optional<std::vector<T>> ret;
  auto stream = open_file_for_read(filename);
  if (stream == nullptr) {
    return std::nullopt;
  }
  constexpr size_t buffer_size = 8192;
  char tmp[buffer_size];
  ret = std::vector<T>();
  ret.value().reserve(buffer_size);
  size_t read_count = 0;
  do {
    read_count = stream->fread(&tmp, buffer_size);
    ret.value().insert(ret.value().end(), tmp, tmp + read_count);
  } while (read_count != 0);
  LOG_IF(FATAL, !ret.has_value())
      << "can't read " << filename << " in the cache object.";
  return ret;
}

std::filesystem::path PassContextImp::get_model_path() const {
  return model_path;
}

/**
 * @brief Retrieves the directory of the execution provider (EP) context model.
 *
 * input : session config `ep.context_file_path` [optional]
 * input : model_path [optional]
 * output : directory of the EP context model
 *
 * This function checks the session configuration for the "ep.context_file_path"
 * key. If it exists and is not empty, it returns the parent directory of the
 * specified path. If the key is not set, it returns the parent directory of the
 * model path.
 * If both the session configuration and model path are empty, it returns the
 * current working directory.
 *
 * @return std::filesystem::path The directory of the EP context model.
 */
std::filesystem::path PassContextImp::get_dir_of_ep_context_model() {
  // get the directory of the ep context onnx model
  auto ret = std::filesystem::path();
  auto ep_context_file_path = get_session_config("ep.context_file_path");
  // For same with onnxruntime (graph_partitioner.cc::GetValidatedEpContextPath)
  // ep.context_file_path validated in onnxruntime
  // ep.context_file_path is a file path, not a directory path.
  // support absolute path and relative path
  // relative path is relative to current working directory
  if (ep_context_file_path.has_value()) {
    ret = std::filesystem::u8path(ep_context_file_path.value()).parent_path();
  } else if (!model_path.empty()) {
    ret = model_path.parent_path();
  } else {
    ret = std::filesystem::path(".");
  }
  return ret;
}

/**
 * @brief Retrieves the basename of the execution provider (EP) context model.
 *
 * input1 : session config `ep.context_file_path` [optional]
 * input2 : model_path  [optional]
 * output : ep_context_onnx_file_path.filename()
 * The input2 and input2 must not be both empty.
 *
 * This function checks the session configuration for the "ep.context_file_path"
 * key. If it exists and is not empty, it returns the filename of the specified
 * path. If the key is not set, it constructs the filename by appending "_ctx"
 * to the stem of the model path.
 * If both the session configuration and model path are empty, it throws a
 * runtime error.
 *
 * @return std::filesystem::path The basename of the EP context model.
 *
 * @throws std::runtime_error If both "ep.context_file_path" and the model
 * path are empty.
 *
 */
std::filesystem::path PassContextImp::get_basename_of_ep_context_model() {
  auto ep_context_onnx_file_path = std::filesystem::path();
  auto ep_context_file_path = get_session_config("ep.context_file_path");
  // For same with onnxruntime
  // (graph_partitioner.cc::GetValidatedEpContextPath) ep.context_file_path
  // validated in onnxruntime ep.context_file_path is a file path, not a
  // directory path. support absolute path and relative path relative path is
  // relative to current working directory
  if (ep_context_file_path.has_value()) {
    ep_context_onnx_file_path =
        std::filesystem::u8path(ep_context_file_path.value());
  } else if (!model_path.empty()) {
    ep_context_onnx_file_path =
        model_path.parent_path() /
        std::filesystem::u8path(model_path.stem().u8string() + "_ctx.onnx");
  } else {
    // ??? Is it possible for both model_path and ep.context_file_path are
    // empty at same time
    LOG(FATAL) << "ep.context_file_path and model_path are both empty.";
  }
  return ep_context_onnx_file_path.filename();
}

/**
 * @brief Retrieves the basename of the EP context binary file.
 *
 * This function constructs the basename of the EP context binary file by
 * appending "_MORPHIZEN.bin" to the basename of the EP context model.
 *
 * @return std::filesystem::path The basename of the EP context binary file.
 */
std::filesystem::path PassContextImp::get_basename_of_ep_context_binary_file() {
  auto ctx_model_basename = get_basename_of_ep_context_model();
  if (ctx_model_basename.empty()) {
    LOG(FATAL) << "get_basename_of_ep_context_model() returned empty path.";
  }
  return std::filesystem::u8path(ctx_model_basename.u8string() +
                                 "_MORPHIZEN.bin");
}

std::optional<std::vector<char>>
PassContextImp::read_file_c8(const std::string &filename) const {
  return read_file_generic<char>(filename);
}

std::optional<std::vector<uint8_t>>
PassContextImp::read_file_u8(const std::string &filename) const {
  return read_file_generic<uint8_t>(filename);
}

std::unique_ptr<CacheFileReader>
PassContextImp::open_file_for_read(const std::string &filename) const {
  CHECK(tar_file_ != nullptr) << "tar_file_ should always exist";
  return open_file_for_read_with_tar_file(filename);
}

std::string
PassContextImp::get_cache_filename(const std::string &filename) const {
  auto cache_key = get_context_proto().cache_key();
  CHECK(!cache_key.empty()) << "cache_key required for cache file operations";
  return cache_key + "/" + filename;
}

std::unique_ptr<CacheFileReader>
PassContextImp::open_file_for_read_with_tar_file(
    const std::string &filename1) const {
  auto filename = get_cache_filename(filename1);
  CHECK(tar_file_ != nullptr) << "tar_file_ is nullptr";
  auto stream = tar_file_->open_for_read(filename);
  if (stream == nullptr) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "cannot open " << filename << " in the tar file";
    return nullptr;
  }
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << filename << " in the tar file";
  return std::make_unique<CacheFileReaderStreamImp>(filename, stream->size(),
                                                    *stream);
}
std::unique_ptr<CacheFileWriter>
PassContextImp::open_file_for_write_with_tar_file(
    const std::string &filename1) {
  auto filename = get_cache_filename(filename1);
  CHECK(tar_file_ != nullptr) << "tar_file_ is nullptr";
  auto stream = tar_file_->open_for_write(filename);
  if (stream == nullptr) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "cannot write " << filename << " in the tar file";
    return nullptr;
  }
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "write " << filename << " in the tar file";
  return std::make_unique<CacheFileWriterStreamImp>(filename,
                                                    std::move(stream));
}
std::unique_ptr<CacheFileWriter>
PassContextImp::open_file_for_write(const std::string &filename) {
  CHECK(tar_file_ != nullptr) << "tar_file_ should always exist";
  return open_file_for_write_with_tar_file(filename);
}

bool write_to_cache_files(std::map<std::string, FILE *> &cache_files,
                          const std::string &filename,
                          gsl::span<const char> data) {
  auto iter = cache_files.find(filename);
  if (iter != cache_files.end()) {
    fclose(iter->second);
  }
  cache_files[filename] = write_to_tmp_file(data);
  return true;
}
bool PassContextImp::write_file(const std::string &filename,
                                gsl::span<const char> data) {
  bool ret = true;
  auto stream = open_file_for_write(filename);
  CHECK(stream != nullptr) << "cannot open " << filename << " for write";
  if (!data.empty()) {
    CHECK(stream->fwrite(data.data(), data.size()) == data.size())
        << "failed to write " << filename;
  }
  stream = nullptr; // close file
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "write " << filename << " " << data.size()
      << " bytes to the cache files";
  return ret;
}

bool PassContextImp::has_cache_file(const std::string &filename1) const {
  auto filename = get_cache_filename(filename1);
  CHECK(tar_file_ != nullptr) << "tar_file_ should always exist";
  return tar_file_->has_file(filename);
}

std::vector<std::string> PassContextImp::get_cache_file_names() const {
  CHECK(tar_file_ != nullptr) << "tar_file_ should always exist";
  auto ret = std::vector<std::string>{};
  const auto &entries = tar_file_->entries();
  ret.reserve(entries.size());
  for (const auto &entry : entries) {
    if (entry && !entry->is_symlink()) {
      ret.push_back(entry->path());
    }
  }
  return ret;
}
const ConfigProto &PassContextImp::get_config_proto() const { return config_; }
const ContextProto &PassContextImp::get_context_proto() const {
  return context_proto;
}
ContextProto &PassContextImp::get_context_proto() { return context_proto; }
void PassContextImp::save_context_json() const {
  ContextProto proto;
  proto.CopyFrom(this->context_proto);
  // NO config manipulation - config not in ContextProto anymore!
  try {
    // When the GENERIC device is used, set fallback_cpu to true. When
    // inferencing a cached model, either from the cache directory or the EP
    // cache context file, we should not enable fallback_cpu. Otherwise, a
    // considerable amount of overhead is incurred for creating an
    // `Ort::Session` object for the subgraph behind the scenes.

    // There is a pitfall: if a custom op really needs to fall back to the
    // CPU, and the GENERIC device is enabled for model compilation, it is a
    // bug. However, this combination is not in used for now. We can fix it
    // later.
    for (auto &meta_def : *proto.mutable_meta_def()) {
      if (meta_def.device() == "GENERIC") {
        meta_def.set_fallback_cpu(true);
      }
    }
    auto json_str = msg_to_json_string(proto);
    const_cast<PassContextImp *>(this)->write_file("context.json", json_str);
  } catch (const std::exception &e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}

void PassContextImp::add_context_resource(const std::string &name,
                                          std::shared_ptr<void> resource) {
  pass_resources[name] = resource;
}

std::shared_ptr<void>
PassContextImp::get_context_resource(const std::string &name) const {
  auto it = pass_resources.find(name);
  auto ret = std::shared_ptr<void>();
  if (it != pass_resources.end()) {
    ret = it->second;
  }
  return ret;
}

std::unique_ptr<PassContextTimer>
PassContextImp::measure(const std::string &label) {
  return std::unique_ptr<PassContextTimer>(
      new PassContextTimerImp(label, *this));
}

void PassContextImp::on_custom_op_create_end() {
  created_customop_count++;
  LOG_VERBOSE(2) << "on_custom_op_create_end: " << created_customop_count
                 << " of " << this->context_proto.meta_def_size();
  if (created_customop_count == this->context_proto.meta_def_size()) {
    bool is_embed_mode = tar_file_file_name_.empty();
    LOG_VERBOSE(2) << "delete_flag=" << delete_tar_file_on_session_created_
                   << ", tar_file_=" << (tar_file_.get() != nullptr)
                   << ", is_embed_mode=" << is_embed_mode;
    if (delete_tar_file_on_session_created_ && tar_file_ && !is_embed_mode) {
      tar_file_.reset();
    }
  }
}
/// struct PassContextTimerImp
PassContextTimerImp::PassContextTimerImp(const std::string &label,
                                         PassContextImp &context)
    : PassContextTimer(), label_{label}, context_{context},
      start_{std::chrono::steady_clock::now()}, mem_usage_{GetMemUsage()} {}
PassContextTimerImp::~PassContextTimerImp() {
  auto end_tp = std::chrono::steady_clock::now();
  auto end_mem_usage = GetMemUsage();
  auto event = context_.context_proto.mutable_events()->Add();
  int64_t thead_id = morphizen::get_tid();
  int64_t process_id = morphizen::get_pid();
  event->set_name(label_);
  event->set_ph("X");
  event->set_pid(process_id);
  event->set_tid(thead_id);
  auto start = std::chrono::duration_cast<std::chrono::microseconds>(
                   start_ - context_.start_)
                   .count();
  event->set_ts(start);
  auto interval =
      std::chrono::duration_cast<std::chrono::microseconds>(end_tp - start_)
          .count();
  event->set_dur(interval);
  *event->mutable_args()->mutable_mem_usage() = mem_usage_;
  event->mutable_args()->mutable_mem_usage()->set_current_memory_in_bytes(
      end_mem_usage.current_memory_in_bytes() -
      event->args().mem_usage().current_memory_in_bytes());
  // memory usage at start
  event = context_.context_proto.mutable_events()->Add();
  event->set_id(label_ + "_mem_usage_1");
  event->set_ph("v");
  event->set_pid(process_id);
  event->set_ts(std::chrono::duration_cast<std::chrono::microseconds>(
                    start_ - context_.start_)
                    .count());
  *event->mutable_args()->mutable_dumps()->mutable_process_totals() =
      convert_to_chrome_event(mem_usage_);
  // memory usage at end
  event = context_.context_proto.mutable_events()->Add();
  event->set_id(label_ + "_mem_usage_2");
  event->set_ph("v");
  event->set_pid(process_id);
  event->set_ts(std::chrono::duration_cast<std::chrono::microseconds>(
                    end_tp - context_.start_)
                    .count());
  *event->mutable_args()->mutable_dumps()->mutable_process_totals() =
      convert_to_chrome_event(end_mem_usage);
}

/// struct PassContext
std::unique_ptr<PassContext> PassContext::create() {
  return std::make_unique<PassContextImp>();
}

void PassContextImp::load_plugins() {
  auto str_backends = get_provider_option("backends", "");

  auto split = [](const std::string &str,
                  char delimiter) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  };
  auto backends = split(str_backends, ';');
  auto plugins = new std::vector<std::shared_ptr<Plugin>>();
  for (const auto &backend : backends) {
    auto plugin = load_plugin(backend);
    plugins->push_back(plugin);
  }
  this->add_context_resource(
      "__all_plugins__", std::shared_ptr<void>((void *)plugins, [](void *p) {
        delete (std::vector<std::shared_ptr<Plugin>> *)p;
      }));
}
std::shared_ptr<Plugin>
PassContextImp::load_plugin(const std::string &plugin_name) {
  auto plugin = morphizen::WeakStore<std::string, Plugin>::create(
      plugin_name, plugin_name.c_str());
  return plugin;
}
CacheFileReaderImp::CacheFileReaderImp(bool in_mem, const std::string &filename,
                                       FILE *fp)
    : CacheFileReader(), in_mem_(in_mem), name_{filename}, fp_{fp} {

  std::rewind(fp);
  CHECK(fseek64(fp, 0, SEEK_SET) == 0);
  CHECK(fseek64(fp, 0, SEEK_END) == 0);
  size_ = ftell64(fp);
  CHECK(fseek64(fp, 0, SEEK_SET) == 0);
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << filename << " for read";
}

CacheFileReaderImp::~CacheFileReaderImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for read";
}

std::size_t CacheFileReaderImp::fread(void *buffer, std::size_t size) const {
  auto ret = std::fread(buffer, 1u, size, fp_);
  return ret;
}

size_t CacheFileReaderImp::size() const { return size_; }

void CacheFileReaderImp::rewind() const { std::rewind(fp_); }

CacheFileReaderStreamImp::CacheFileReaderStreamImp(const std::string &name,
                                                   size_t size,
                                                   TarEntryInputStream &stream)
    : name_{name}, size_{size}, stream_{stream} {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << name << " for read";
}

CacheFileReaderStreamImp::~CacheFileReaderStreamImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for read";
}

std::size_t CacheFileReaderStreamImp::fread(void *buffer,
                                            std::size_t size) const {
  CHECK(!stream_.read(static_cast<char *>(buffer), size).bad())
      << "failed to read " << name_;
  auto ret = stream_.gcount();
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= 9)
      << "read " << name_ << " " << ret << " bytes "
      << " size_ =" << size_;
  return ret;
}

void *CacheFileReaderStreamImp::mmap() { return stream_.mmap(); }

size_t CacheFileReaderStreamImp::size() const { return size_; }

void CacheFileReaderStreamImp::rewind() const {
  CHECK(!stream_.seekg(0, std::ios::beg).fail())
      << "failed to seek to the beginning of the stream";
}

CacheFileWriterImp::CacheFileWriterImp(bool in_mem, const std::string &filename,
                                       FILE *fp)
    : CacheFileWriter(), in_mem_(in_mem), name_{filename}, fp_{fp} {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << filename << " for write";
}

CacheFileWriterImp::~CacheFileWriterImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for write";
  std::fflush(fp_);
}

std::size_t CacheFileWriterImp::fwrite(const void *buffer,
                                       std::size_t size) const {
  auto ret = std::fwrite(buffer, 1u, size, fp_);
  return ret;
}

CacheFileWriterStreamImp::CacheFileWriterStreamImp(
    const std::string &name, std::unique_ptr<std::ostream> stream)
    : CacheFileWriter(), name_{name}, stream_{std::move(stream)} {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << name_ << "for write";
}

CacheFileWriterStreamImp::~CacheFileWriterStreamImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for write";
  stream_->flush();
  stream_.reset();
}

std::size_t CacheFileWriterStreamImp::fwrite(const void *buffer,
                                             std::size_t size) const {
  auto pos = stream_->tellp();
  stream_->write(static_cast<const char *>(buffer), size);
  if (stream_->bad()) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "failed to write " << name_;
    return 0;
  }
  auto ret = stream_->tellp() - pos;
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= 3)
      << "failed to write " << name_;
  return ret;
}

FileSystemImp::FileSystemImp(PassContext &context) : context_(context) {}

FileReader *FileSystemImp::create_reader(const char *path) {
  return context_.open_file_for_read(path).release();
}

FileWriter *FileSystemImp::create_writer(const char *path) {
  return context_.open_file_for_write(path).release();
}

void FileSystemImp::destroy_reader(FileReader *reader) { delete reader; }

void FileSystemImp::destroy_writer(FileWriter *writer) { delete writer; }

std::unique_ptr<FileSystem> PassContextImp::get_file_system() {
  return std::make_unique<FileSystemImp>(*this);
}

void PassContextImp::maybe_create_tar_file_for_write() {
  auto is_shared_context_enabled =
      get_session_config(kOrtSessionOptionShareEpContexts, "0") == "1";
  auto is_stop_shared_context =
      get_session_config(kOrtSessionOptionStopShareEpContexts, "0") == "1";
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  auto is_ep_context_embed_mode =
      get_session_config(kOrtSessionOptionEpContextEmbedMode, "1") == "1";

  // Note: Cache files serve TWO purposes:
  // 1. Inter-pass communication (Level 1 passes, custom ops) - ALWAYS needed
  // 2. EP context persistence (external save) - Only when enabled
  //
  // We ALWAYS create tar_file_ for cache storage (purpose #1).
  // The EP context flag only controls WHERE to persist (external file vs
  // in-memory/discard).
  if (!is_ep_context_enabled) {
    // EP disabled: Create in-memory tar (discarded after session)
    // Used for inter-pass communication only, not persisted
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "EP context disabled, creating tmpfile/in-memory tar for cache";
    tar_file_file_name_.clear();
    tar_file_ = TarFile::create_from_tmpfile();
    CHECK(tar_file_ != nullptr)
        << "Failed to create tar file for cache (EP context disabled)";
    return;
  }

  // EP enabled: Persist to external file or embed in model
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "EP context enabled, creating tar file for external persistence";

  if (!is_ep_context_embed_mode) {
    // Non-embed mode: Create persistent external file
    auto ep_context_binary_file = get_dir_of_ep_context_model() /
                                  get_basename_of_ep_context_binary_file();
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "open tar file for write: " << ep_context_binary_file;

    if (is_shared_context_enabled) {
      auto &shared_workspace =
          SharedContextContextWorkspace::create_workspace_or_get(
              ep_context_binary_file);
      ep_context_binary_file = shared_workspace.get_ep_context_binary_file();
      if (is_stop_shared_context) {
        shared_workspace.close_workspace();
      }
    }

    auto open_mode = std::ios::binary | std::ios::in | std::ios::out;
    if (!is_shared_context_enabled) {
      // overwrite existing file if it is not shared.
      open_mode |= std::ios::trunc;
    }
    if (!std::filesystem::exists(ep_context_binary_file)) {
      open_mode |= std::ios::trunc;
    }
    auto stream = std::unique_ptr<std::fstream>();
    tar_file_file_name_ = ep_context_binary_file;
    stream = std::make_unique<std::fstream>(ep_context_binary_file, open_mode);

    CHECK(stream->is_open())
        << "failed to open ep context file " << ep_context_binary_file;
    CHECK(!get_context_proto().cache_key().empty())
        << "cache_key should be empty when using tar file";
    tar_file_ = TarFile::create(std::move(stream));
  } else {
    // Embed mode: Create tmpfile (serialized to model later)
    tar_file_file_name_.clear();
    tar_file_ = TarFile::create_from_tmpfile();
    CHECK(tar_file_ != nullptr)
        << "Failed to create tar file for write in embed mode";
  }

  // Cache key prefix configuration
  if (is_shared_context_enabled) {
    // Always use cache_key prefix for consistent behavior
  }
}
void PassContextImp::create_tar_file_for_read(std::string &&ep_context_binary,
                                              bool embed_mode) {
  // Check provider option for mmap enablement (applies to both modes)
  bool enable_mmap =
      get_provider_option(kProviderOptionEpContextEnableMmap, "1") == "1";

  if (!embed_mode) {
    // non-embed mode: To ensure the mmap functionality, the non-embed mode does
    // not delete the tar file.
    disable_delete_tar_file_in_session_created();
    auto ep_context_binary_file = get_dir_of_ep_context_model() /
                                  std::filesystem::u8path(ep_context_binary);
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "open tar file for read: " << ep_context_binary_file;
    CHECK(std::filesystem::exists(ep_context_binary_file))
        << "ep context binary file does not exist at path: "
        << ep_context_binary_file;
    if (!std::filesystem::is_regular_file(ep_context_binary_file)) {
      LOG(FATAL) << "ep context binary does not exist at path: "
                 << ep_context_binary_file;
    }
    tar_file_ = TarFile::create_from_path(ep_context_binary_file, enable_mmap);
    CHECK(tar_file_ != nullptr)
        << "failed to open ep context file " << ep_context_binary_file;
  } else {
    // embed mode: create from buffer with mmap support
    tar_file_ =
        TarFile::create_from_buffer(std::move(ep_context_binary), enable_mmap);
  }
}

void PassContextImp::create_tar_file_for_prebuild_cache(
    std::vector<char> &&buffer) {
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  auto is_ep_context_embed_mode =
      get_session_config(kOrtSessionOptionEpContextEmbedMode, "1") == "1";

  if (!is_ep_context_enabled) {
    // when ep.context is not enabled, we don't need to worry to much about
    // how to save tar_file_
    tar_file_ = TarFile::create_from_buffer(std::move(buffer));
    CHECK(tar_file_ != nullptr) << " create a tar file from memory ";
  } else {
    if (is_ep_context_embed_mode) {
      // for embeded mode, it works similar to is_ep_context_enable = false;
      tar_file_ = TarFile::create_from_buffer(std::move(buffer));
      CHECK(tar_file_ != nullptr) << " create a tar file from memory ";
    } else {
      auto binary_file_path = get_dir_of_ep_context_model() /
                              get_basename_of_ep_context_binary_file();
      auto open_mode =
          std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc;
      auto stream = std::unique_ptr<std::fstream>();
      stream = std::make_unique<std::fstream>(binary_file_path, open_mode);
      CHECK(stream->is_open())
          << "failed to open ep context file " << binary_file_path;
      CHECK(stream->write(buffer.data(), buffer.size()).good())
          << "fail to write to " << binary_file_path;
      stream->seekg(0);
      stream->seekp(0);
      stream->flush();
      tar_file_ = TarFile::create(std::move(stream));
      tar_file_file_name_ = binary_file_path;
    }
  }
  // Always use cache_key prefix - verify context.json exists with prefix
  auto prefix = get_context_proto().cache_key();
  CHECK(!prefix.empty()) << "cache_key required for prebuild cache";
  CHECK(tar_file_->has_file(prefix + "/context.json"))
      << "tar file does not have " << prefix << "/context.json, "
      << "please check prebuild ep context generation";
}
void PassContextImp::print_version_info(const char *prefix) {
  auto &context = get_context_proto();
  for (auto version_info : context.version().version_infos()) {
    LOG_VERBOSE(1) << prefix << version_info.package_name() << " ("
                   << version_info.version() << ")"
                   << (version_info.commit().empty()
                           ? std::string()
                           : " :" + version_info.commit());
  }
  auto print_kv = [](int level, const char *prefix,
                     std::pair<const std::string, std::string> &kv) {
    if (kv.first != "encryption_key") {
      LOG_VERBOSE(level) << prefix << ": " << kv.first << " = " << kv.second;
    } else {
      LOG_VERBOSE(level) << prefix << ": " << kv.first << " = "
                         << "******";
    }
  };
  LOG_VERBOSE(1) << prefix << "cache_key: " << get_context_proto().cache_key();
  LOG_VERBOSE(1) << prefix << "dump_dir: " << get_dump_directory();
  for (auto &kv : provider_option_origin_) {
    print_kv(3, "provider_option_from_origin", kv);
  }
  for (auto &kv :
       // print sorted keys
       std::map<std::string, std::string>(config_.provider_options().begin(),
                                          config_.provider_options().end())) {
    print_kv(3, "provider_options_in_config", kv);
  }
  if (target_proto_) {
    for (auto &kv : std::map<std::string, std::string>(
             target_proto_->provider_options().begin(),
             target_proto_->provider_options().end())) {
      print_kv(3, "provider_options_in_target_proto", kv);
    }
  }
  for (auto &kv : session_configs_) {
    LOG_VERBOSE(3) << "session_config: " << kv.first << " = " << kv.second;
  }
  auto all_po = get_all_provider_options();
  for (auto &kv : all_po) {
    print_kv(1, "provider_option", kv);
  }
}
void PassContextImp::pass_context_update_context_json(
    gsl::span<char> json_str) {
  // parse the context proto
  ContextProto context_proto_in_cache;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      &json_str[0], &context_proto_in_cache, options);

  CHECK(status.ok()) << "cannot parse json string:" << status.message()
                     << &json_str[0];
  // Note: Old cache files with config field in ContextProto are not supported
  // after this refactoring (breaking change). The config field is now
  // runtime-only.
  this->context_proto.Swap(&context_proto_in_cache);
  this->update_config_proto_root_field();
  // update_cache_dir(*this);
  print_version_info("CACHE VERSION: ");
}

void PassContextImp::update_pass_context_from_context_json_in_cache() {
  auto context_context_json = read_file_c8("context.json");
  CHECK(context_context_json.has_value())
      << "cannot read context.json from ep context";
  auto context_context_json_text = dos2unix(*context_context_json);
  pass_context_update_context_json(context_context_json_text);
}
void PassContextImp::update_config_proto_root_field() {
  // ADD_CUSTOM_FIELD∆
  // NOTE:
  //  1. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  //  2. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  //  3. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  auto get_provider_option_local =
      [this](
          const std::vector<std::string> &names) -> std::optional<std::string> {
    auto ret = std::optional<std::string>();
    return this->get_provider_option_with_priority(names);
  };
  if (auto cache_key = get_provider_option_local({"cache_key", "cacheKey"})) {
    context_proto.set_cache_key(*cache_key);
  }
  // cache_dir removed by Issue #006 (PR #80)
  // encryption_key removed - now read from provider_options directly (Issue
  // #004)
  // target copying removed by Issue #007 - ConfigProto.target is immutable
}
template <typename T>
static std::optional<std::string>
get_provider_option_internal(const std::string &name, const T &options) {
  auto it = options.find(name);
  if (it != options.end()) {
    return it->second;
  }
  return std::nullopt;
}

const TargetProto *
PassContextImp::find_target_proto(const std::string &target_name) {
  auto &targets = config_.targets();
  auto it = std::find_if(targets.begin(), targets.end(),
                         [&target_name](const TargetProto &target) {
                           return target.name() == target_name;
                         });
  if (it != targets.end()) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "Found target proto for target: " << target_name;
    return &(*it); // Return raw pointer to element in ConfigProto
  }
  return nullptr; // not found
}

std::string PassContextImp::get_valid_target_names() {
  std::ostringstream valid_names;
  int c = 0;
  auto &targets = config_.targets();

  for (const auto &target : targets) {
    if (c++ > 0) {
      valid_names << ", ";
    }
    valid_names << '"' << target.name() << '"';
  }
  return valid_names.str();
}

bool PassContextImp::has_user_config_file() const {
  return get_provider_option_internal(kProviderOptionConfigFile,
                                      provider_option_origin_)
      .has_value();
}

bool PassContextImp::try_initialize_target_proto(const std::string &target_name,
                                                 bool thorow_if_not_found) {
  target_proto_ = find_target_proto(target_name);
  if (target_proto_ == nullptr) {
    auto valid_target_names = get_valid_target_names();
    if (thorow_if_not_found) {
      LOG(ERROR) << "Target auto-discovery: target proto not found for "
                    "target: "
                 << target_name
                 << ", valid target names: " << valid_target_names;
      throw std::invalid_argument("not a valid target name, valid names:" +
                                  valid_target_names);
    } else {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "Target auto-discovery: target proto not found for target: "
          << target_name << ", valid target names: " << valid_target_names;
    }
  }
  return target_proto_ != nullptr;
}
static std::optional<std::string> discover_target(const ConfigProto &proto,
                                                  const Model &model) {
  typedef std::optional<std::string> (*discovery_function_t)(
      const ConfigProto &, const Model &);
  auto all_plugin_functions =
      morphizen::Plugin::get_all_symbols("morphizen_target_discovery");
  std::sort(all_plugin_functions.begin(), all_plugin_functions.end(),
            [](const std::pair<std::string, void *> &a,
               const std::pair<std::string, void *> &b) {
              return a.first < b.first;
            });
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
      << "discover_target: all_plugin_functions size: "
      << all_plugin_functions.size();
  for (auto &plugin : all_plugin_functions) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "discover_target: plugin name: " << plugin.first
        << " model id:" << (void *)(&model) << " id: " << plugin.second;
    auto target_discovery_func = (discovery_function_t)plugin.second;
    auto target = target_discovery_func(proto, model);
    if (target.has_value()) {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "discover_target: plugin name: " << plugin.first
          << " target: " << target.value();
      return target;
    } else {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "discover_target: plugin name: " << plugin.first
          << " cannot guess target, continue";
    }
  }
  return std::nullopt;
}
void PassContextImp::target_auto_discovery(const Model &model) {
  bool using_builtin_config = !has_user_config_file();

  // Priority 1: User explicit override (both paths)
  auto target_specified_by_end_user = get_provider_option_internal(
      kProviderOptionTarget, provider_option_origin_);
  if (target_specified_by_end_user) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "Target specified by user: " << *target_specified_by_end_user;
    if (!try_initialize_target_proto(*target_specified_by_end_user, true)) {
      // try_initialize_target_proto already throws if not found
      return;
    }
    return;
  }

  if (using_builtin_config) {
    // Path A: Built-in config - auto-discovery REQUIRED
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "Using built-in config - attempting auto-discovery";

    auto discovered_target_name = discover_target(config_, model);
    if (discovered_target_name.has_value()) {
      if (try_initialize_target_proto(*discovered_target_name, false)) {
        LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
            << "Auto-discovery: detected target: " << *discovered_target_name;
        return;
      }
    }

    // Priority 3: Fallback to built-in ConfigProto.target
    auto &target_name = config_.target();
    if (!target_name.empty()) {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "Auto-discovery failed, using built-in config default: "
          << target_name;
      if (try_initialize_target_proto(target_name, true)) {
        return;
      }
    }

    // Fatal error - built-in config MUST have working auto-discovery or default
    throw std::runtime_error("Auto-discovery failed with built-in config and "
                             "no default target - this is a fatal error");

  } else {
    // Path B: User config file - use config target directly (no auto-discovery)
    auto &target_name = config_.target();
    if (target_name.empty()) {
      throw std::invalid_argument("User config file must specify target field");
    }
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "Using target from user config file: " << target_name;
    if (!try_initialize_target_proto(target_name, true)) {
      // try_initialize_target_proto already throws if not found
      return;
    }
  }
}

void PassContextImp::append_compiled_model_compatibility_info(
    const std::string &backend_name, const std::string &compatibility_info) {
  // Validate that compatibility_info is not empty
  if (compatibility_info.empty()) {
    LOG(WARNING) << "Backend '" << backend_name
                 << "' provided empty compatibility info. Ignoring.";
    return;
  }

  // Check if backend already has compatibility info
  auto it = compiled_model_compatibility_info_.find(backend_name);
  if (it != compiled_model_compatibility_info_.end()) {
    LOG(WARNING) << "Compatibility info for backend '" << backend_name
                 << "' already exists. Overwriting previous value.";
  }

  compiled_model_compatibility_info_[backend_name] = compatibility_info;
}

const std::map<std::string, std::string> &
PassContextImp::get_compiled_model_compatibility_info() const {
  return compiled_model_compatibility_info_;
}
void PassContextImp::disable_delete_tar_file_in_session_created() {
  delete_tar_file_on_session_created_ = false;
}

} // namespace morphizen
