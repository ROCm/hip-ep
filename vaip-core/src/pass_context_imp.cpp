/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "core/session/onnxruntime_session_options_config_keys.h"
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <stdexcept>

#include "./binary/mem_binary.hpp"
#include "./cache_dir.hpp"
#include "config.hpp"
#include "ep_shared_context_workspace.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/util.hpp"
#include "morphizen/vaip_io.hpp"
#include "morphizen/weak.hpp"
#include "pass_context_imp.hpp"
#include "profile_utils.hpp"
#include "tar_ball.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
DEF_ENV_PARAM(MORPHIZEN_FEATURE_USE_TAR_FILE, "1")
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "

namespace vaip_core {

/// struct WithPass
PassContextImp::WithPass::WithPass(PassContextImp& context, IPass& pass)
    : _context(&context) {
  _context->current_pass_stack.push_back(&pass);
}
PassContextImp::WithPass::~WithPass() {
  _context->current_pass_stack.pop_back();
}

/// static
static MemUsageProto convert_to_chrome_event(const MemUsageProto& mem_usage) {
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
static std::vector<char>
read_file_to_buffer(const std::filesystem::path& path) {
  std::ifstream is(path, std::ios::binary);
  CHECK(is.good());
  CHECK(is.seekg(0, std::ios_base::end).good());
  auto size = is.tellg();
  CHECK_NE(size, -1);
  CHECK(is.seekg(0, std::ios_base::beg).good());
  auto buffer = std::vector<char>((size_t)size);
  CHECK(is.read(buffer.data(), size).good());
  return buffer;
}
static FILE* write_to_tmp_file(gsl::span<const char> data) {
#if _WIN32
  FILE* tmp_file = nullptr;
  auto err = tmpfile_s(&tmp_file);
  CHECK_EQ(err, 0) << "tmpfile_s error";
#else
  FILE* tmp_file = tmpfile();
  CHECK(tmp_file != nullptr) << "cannot create tmp file";
#endif
  auto write_size = std::fwrite(data.data(), 1, data.size(), tmp_file);
  CHECK_EQ((size_t)write_size, data.size());
  return tmp_file;
}

static std::string msg_to_json_string(const google::protobuf::Message& msg) {
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  auto json_str = std::string();
  auto status =
      google::protobuf::util::MessageToJsonString(msg, &json_str, options);
  CHECK(status.ok()) << "cannot write json string:" << msg.DebugString();
  return json_str;
}
std::unique_ptr<PassContextImp>
PassContextImp::create_pass_context(const ConfigProto& config_proto1) {
  auto ret = std::make_unique<PassContextImp>();
  auto config_proto = ConfigProto(config_proto1);
  Config::add_version_info(config_proto);
  ret->cache_dir_set = (config_proto.cache_dir().size() > 0);
  ret->context_proto.mutable_config()->Swap(&config_proto);
  ret->update_config_proto_root_field();
  return ret;
}
std::unique_ptr<PassContextImp> PassContextImp::create_pass_context(
    const onnxruntime::ProviderOptions& options) {
  auto json_config_string = get_config_json_str(options);
  const char* json_config = json_config_string.c_str();
  auto config_proto = ConfigProto();
  if (json_config != nullptr && !std::string(json_config).empty()) {
    Config::merge_config_proto(config_proto, json_config);
  }
  auto ret = create_pass_context(config_proto);
  ret->provider_option_origin_.insert(options.begin(), options.end());
  ret->update_config_proto_root_field();
  return ret;
}
/// struct PassContextImp
int PassContextImp::allocate_suffix()
    const { // it is not a big deal to update suffix_counter
  suffix_counter = suffix_counter + 1;
  return suffix_counter;
}

PassContextImp::WithPass PassContextImp::with_current_pass(IPass& pass) {
  return WithPass(*this, pass);
}

const std::filesystem::path& PassContextImp::get_log_dir() const {
  return pass_context_log_dir_;
}

template <typename T1, typename T2>
std::optional<std::string>
PassContextImp::get_provider_option_impl(const T1& option_names,
                                         const T2& privider_options) const {
  auto ret = std::optional<std::string>();
  if (privider_options) {
    for (auto& option_name : option_names) {
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
    const T1& option_names, const T& options1, const T2&... options) const {
  auto ret = get_provider_option_impl(option_names, options1);
  if (ret) {
    return ret;
  }
  return get_provider_option_impl(option_names, options...);
}

template <typename T1, typename... T2>
std::optional<std::string> PassContextImp::get_provider_option_with_priority(
    const T1& option_names) const {
  // priority order:
  // 0. provider_option privded by user
  // 1. provider_option in cache
  // 1. context_proto,
  // 2. mep_config_proto, from known models.
  // 3. target_proto, from target discovery
  //    Target priority
  //        1. provider option
  //        2. meptable
  //        3. heuristic process or method:
  //        4. default target in config file
  //
  //  4. default value
  return get_provider_option_impl(
      option_names,
      &provider_option_origin_,                                             //
      &provider_option_from_cache_,                                         //
      &context_proto.config().provider_options(),                           //
      mep_config_proto_ ? &mep_config_proto_->provider_options() : nullptr, //
      target_proto_ ? &target_proto_->provider_options() : nullptr          //
  );
}
std::map<std::string, std::string>
PassContextImp::get_all_provider_options() const {
  auto ret = std::map<std::string, std::string>();
  get_all_provider_option_impl(
      ret,
      &provider_option_origin_,                                             //
      &provider_option_from_cache_,                                         //
      &context_proto.config().provider_options(),                           //
      mep_config_proto_ ? &mep_config_proto_->provider_options() : nullptr, //
      target_proto_ ? &target_proto_->provider_options() : nullptr          //
  );
  return ret;
}

template <typename T>
void PassContextImp::get_all_provider_option_impl(
    std::map<std::string, std::string>& ret, const T& provider_options) const {
  if (provider_options) {
    for (auto& kv : *provider_options) {
      ret.insert({kv.first, kv.second});
    }
  };
}
template <typename T, typename... T1>
void PassContextImp::get_all_provider_option_impl(
    std::map<std::string, std::string>& ret, const T& options1,
    const T1&... options) const {
  get_all_provider_option_impl(ret, options1);
  get_all_provider_option_impl(ret, options...);
}

std::optional<std::string>
PassContextImp::get_provider_option(const std::string& option_name) const {
  return get_provider_option_with_priority(
      std::array<std::string, 1>{option_name});
}
std::optional<std::string>
PassContextImp::get_session_config(const std::string& option_name) const {
  const auto& config = context_proto.config();
  auto it = config.session_configs().find(option_name);
  if (it != config.session_configs().end()) {
    return it->second;
  }
  return std::nullopt;
}
std::string
PassContextImp::get_provider_option(const std::string& option_name,
                                    const std::string& default_value) const {
  auto option_value = get_provider_option(option_name);
  if (option_value.has_value()) {
    return option_value.value();
  }
  return default_value;
}

bool PassContextImp::cache_in_mem() const {
  return this->get_provider_option("enable_cache_file_io_in_mem", "1") == "1";
}
PassContextImp::~PassContextImp() {
  for (auto iter : cache_files_) {
    fclose(iter.second);
  }
}

int64_t PassContextImp::get_provider_option_i64(const std::string& option_name,
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
PassContextImp::get_session_config(const std::string& option_name,
                                   const std::string& default_value) const {
  auto option_value = get_session_config(option_name);
  if (option_value.has_value()) {
    return option_value.value();
  }
  return default_value;
}

std::string
PassContextImp::get_run_option(const std::string& option_name,
                               const std::string& default_value) const {
  auto ret = default_value;
  if (get_run_options_) {
    // if the function exists. TODO, it might be a stale
    // function.
    std::shared_lock<std::shared_mutex> lock(
        const_cast<PassContextImp*>(this)->rw_mutex_);
    auto maybe_value = get_run_options_(option_name);
    if (maybe_value) {
      ret = maybe_value.value();
    }
  }
  return ret;
}
std::string
PassContextImp::get_meta_def_param(const MetaDefProto& meta_def) const {
  auto json_str = std::string();
  auto status =
      google::protobuf::util::MessageToJsonString(meta_def.param(), &json_str);
  if (!status.ok()) {
    LOG(FATAL) << "failed to get meta_def param: " << status.ToString();
  }
  return json_str;
}
std::string
PassContextImp::get_ep_dynamic_option(const std::string& option_name,
                                      const std::string& default_value) const {
  std::lock_guard<std::mutex> lock(this->ep_dynamic_options_lock);
  auto it = ep_dynamic_options.find(option_name);
  if (it == ep_dynamic_options.end()) {
    return default_value;
  } else {
    return it->second;
  }
}

void PassContextImp::add_QosUpdater(
    const std::shared_ptr<QoSUpdateInterface>& updater) const {
  CHECK(updater) << "Null QoS updater cannot be added to PassContext";
  qos_updaters_.push_back(updater);
}

void PassContextImp::update_all_qos(const std::string& workload_type) const {
  if (workload_type == "Efficient" || workload_type == "Default") {
    for (const auto& updater : qos_updaters_) {
      CHECK(updater) << "Found null QoS updater in qos_updaters_";
      updater->update_qos(workload_type);
    }
  } else {
    throw std::runtime_error("Invalid workload type: " + workload_type);
  }
}

template <typename T>
std::optional<std::vector<T>>
PassContextImp::read_file_generic(const std::string& filename) const {
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
 * appending "_VITISAI.bin" to the basename of the EP context model.
 *
 * @return std::filesystem::path The basename of the EP context binary file.
 */
std::filesystem::path PassContextImp::get_basename_of_ep_context_binary_file() {
  auto ctx_model_basename = get_basename_of_ep_context_model();
  if (ctx_model_basename.empty()) {
    LOG(FATAL) << "get_basename_of_ep_context_model() returned empty path.";
  }
  return std::filesystem::u8path(ctx_model_basename.u8string() +
                                 "_VITISAI.bin");
}

std::optional<std::vector<char>>
PassContextImp::read_file_c8(const std::string& filename) const {
  return read_file_generic<char>(filename);
}

std::optional<std::vector<uint8_t>>
PassContextImp::read_file_u8(const std::string& filename) const {
  return read_file_generic<uint8_t>(filename);
}

std::unique_ptr<CacheFileReader>
PassContextImp::open_file_for_read(const std::string& filename) const {
  if (tar_file_) {
    return open_file_for_read_with_tar_file(filename);
  }
  std::unique_ptr<CacheFileReader> ret = nullptr;
  auto in_mem = cache_in_mem();
  auto& cace_files =
      const_cast<std::remove_cv_t<decltype(cache_files_)&>>(cache_files_);
  auto it = cace_files.find(filename);
  if (it != cace_files.end()) {
    if (in_mem) {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
          << "tmp file opened: " << filename;
      ret = std::unique_ptr<CacheFileReader>(
          new CacheFileReaderImp(in_mem, filename, it->second));
    } else {
#ifdef _WIN32
      FILE* fp =
          _wfreopen((get_log_dir() / filename).c_str(), L"rb+", it->second);
#else
      FILE* fp =
          std::freopen((get_log_dir() / filename).c_str(), "rb+", it->second);
#endif //  _WIN32
      if (fp == nullptr) {
        LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
            << " cannot freopen " << filename;
      } else {
        it->second = fp;
        ret = std::unique_ptr<CacheFileReader>(
            new CacheFileReaderImp(in_mem, filename, it->second));
      }
    }
  } else {
    if (!in_mem) {
#ifdef _WIN32
      FILE* fp = _wfopen((get_log_dir() / filename).c_str(), L"rb+");
#else
      FILE* fp = std::fopen((get_log_dir() / filename).c_str(), "rb+");
#endif //  _WIN32
      if (fp == nullptr) {
        LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
            << " cannot freopen " << filename;
      } else {
        cace_files[filename] = fp;
        ret = std::unique_ptr<CacheFileReader>(
            new CacheFileReaderImp(in_mem, filename, fp));
      }
    } else {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
          << "tmp file open failed: cannot found " << filename
          << ". try to use write_file_for_write before reading.";
      ret = nullptr;
    }
  }
  return ret;
}

std::unique_ptr<CacheFileReader>
PassContextImp::open_file_for_read_with_tar_file(
    const std::string& filename1) const {
  auto prefix = get_config_proto().cache_key();
  auto filename =
      cache_file_use_cache_key_prefix_ ? prefix + "/" + filename1 : filename1;
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
    const std::string& filename1) {
  auto prefix = get_config_proto().cache_key();
  auto filename =
      cache_file_use_cache_key_prefix_ ? prefix + "/" + filename1 : filename1;
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
PassContextImp::open_file_for_write(const std::string& filename) {
  std::unique_ptr<CacheFileWriter> ret = nullptr;
  if (tar_file_) {
    ret = open_file_for_write_with_tar_file(filename);
  } else {
    auto it = cache_files_.find(filename);
    FILE* tmp_file = nullptr;
    auto in_mem = cache_in_mem();
    if (it != cache_files_.end()) {
      if (in_mem) {
        fclose(it->second);
        LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
            << "tmp file write: " << filename;
        tmp_file = tmpfile();
        if (tmp_file == nullptr) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
              << " cannot create tmp file " << filename;
        } else {
          it->second = tmp_file;
          return std::unique_ptr<CacheFileWriter>(
              new CacheFileWriterImp(in_mem, filename, it->second));
        }
      } else {
#ifdef _WIN32
        FILE* fp =
            _wfreopen((get_log_dir() / filename).c_str(), L"wb+", it->second);
#else
        FILE* fp =
            std::freopen((get_log_dir() / filename).c_str(), "wb+", it->second);
#endif //  _WIN32
        if (fp == nullptr) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
              << " cannot freopen " << filename;
        } else {
          it->second = fp;
          return std::unique_ptr<CacheFileWriter>(
              new CacheFileWriterImp(in_mem, filename, fp));
        }
      }
    } else {
      if (in_mem) {
        LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
            << "tmp file write: " << filename;
        tmp_file = tmpfile();
        if (tmp_file == nullptr) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
              << "cannot create tmp file" << filename;
        } else {
          cache_files_[filename] = tmp_file;
          this->context_proto.add_cache_files(filename);
          ret = std::unique_ptr<CacheFileWriter>(
              new CacheFileWriterImp(in_mem, filename, tmp_file));
        }
      } else {
        std::filesystem::path tmp_dir =
            (get_log_dir() / filename).parent_path();
        if (!std::filesystem::exists(tmp_dir)) {
          std::filesystem::create_directories(tmp_dir);
        }
#ifdef _WIN32
        tmp_file = _wfopen((get_log_dir() / filename).c_str(), L"wb+");
#else
        tmp_file = std::fopen((get_log_dir() / filename).c_str(), "wb+");
#endif //  _WIN32
        if (tmp_file == nullptr) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
              << " fopen failed. " << filename;
        } else {
          cache_files_[filename] = tmp_file;
          this->context_proto.add_cache_files(filename);
          ret = std::unique_ptr<CacheFileWriter>(
              new CacheFileWriterImp(in_mem, filename, tmp_file));
        }
      }
    }
  }
  return ret;
}

bool write_to_cache_files(std::map<std::string, FILE*>& cache_files,
                          const std::string& filename,
                          gsl::span<const char> data) {
  auto iter = cache_files.find(filename);
  if (iter != cache_files.end()) {
    fclose(iter->second);
  }
  cache_files[filename] = write_to_tmp_file(data);
  return true;
}
bool PassContextImp::write_file(const std::string& filename,
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

void PassContextImp::restore_cache_files() {
  if (tar_file_) {
    // special optimization
    // TODO replace;
  } else {
    for (const auto& str : this->context_proto.cache_files()) {
      open_file_for_read(str);
    }
  }
}

bool PassContextImp::has_cache_file(const std::string& filename1) const {
  auto filename = filename1;
  if (cache_file_use_cache_key_prefix_) {
    filename = get_config_proto().cache_key() + "/" + filename1;
  }
  if (tar_file_) {
    return tar_file_->has_file(filename);
  }
  return cache_files_.find(filename1) != cache_files_.end();
}

std::vector<std::string> PassContextImp::get_cache_file_names() const {
  auto ret = std::vector<std::string>{};
  ret.reserve(cache_files_.size());
  for (const auto& files : cache_files_) {
    ret.push_back(files.first);
  }
  return ret;
}
std::vector<char> PassContextImp::cache_files_to_tar_mem() const {
  std::vector<char> ret;
  {
    auto p = IStreamWriter::from_bytes(ret);
    CHECK(cache_files_to_tar_file(*p)) << "ok";
  }
  return ret;
}

bool PassContextImp::cache_files_to_tar_file(IStreamWriter& writer) const {
  TarWriter tar_writer(writer);
  auto file_names = get_cache_file_names();
  for (const auto& file_name : file_names) {
    tar_writer.write(CacheFileStreamReader(open_file_for_read(file_name)),
                     file_name);
  }
  return true;
}

size_t CacheFileStreamWriter::write(const char* data, size_t size) {
  auto write_size = writer_->fwrite(data, size);
  CHECK_EQ((size_t)write_size, size);
  return write_size;
}

std::unique_ptr<IStreamWriter>
CacheFileStreamWriterBuilder::build(const std::string& filename) {
  auto stream = context->open_file_for_write(filename);
  CHECK(stream != nullptr) << "cannot open " << filename << " for write";
  return std::make_unique<CacheFileStreamWriter>(std::move(stream));
}

std::optional<std::vector<char>>
CacheFileStreamReader::read(size_t size_hint) const {
  auto ret = std::vector<char>();
  ret.resize(size_hint);
  auto read_size = reader_->fread(&ret[0], size_hint);
  if (read_size == 0) {
    return std::nullopt;
  } else {
    ret.resize(read_size);
  }
  return ret;
}

bool PassContextImp::tar_file_to_cache_files(IStreamReader& src) {
  TarReader tar_reader(src);
  CacheFileStreamWriterBuilder build(this);
  for (;;) {
    bool is_continue = tar_reader.read(build);
    if (!is_continue) {
      break;
    }
  }
  return true;
}

std::filesystem::path PassContextImp::xclbin_path_to_cache_files(
    const std::filesystem::path& path) const {
  auto filename = path.filename().u8string();
  auto ret = get_log_dir() / filename;

  bool in_mem = cache_in_mem();
  std::error_code ec;
  // already done
  if (in_mem && has_cache_file(filename)) {
    return ret;
  } else if ((!in_mem) && std::filesystem::is_regular_file(ret, ec)) {
    return ret;
  }

  std::vector<char> buffer;
  if (has_mem_binary(filename)) {
    buffer = get_mem_binary(filename);
  } else if (std::filesystem::is_regular_file(path, ec)) {
    buffer = read_file_to_buffer(path);
  } else {
    LOG(WARNING) << "Xclbin path doesn't exist, are you running with cpu "
                    "runner? Path: "
                 << path.string();
    return path;
  }
  const_cast<PassContextImp*>(this)->write_file(filename, buffer);
  return ret;
}

std::optional<std::vector<char>>
PassContextImp::read_xclbin(const std::filesystem::path& path) const {

  std::optional<std::vector<char>> ret;
  auto reader = open_file_for_read(path.filename().u8string());
  if (!reader) {
    return ret;
  }

  ret = std::vector<char>(reader->size());
  reader->fread(ret->data(), reader->size());
  return ret;
}

const ConfigProto& PassContextImp::get_config_proto() const {
  return context_proto.config();
}
const ContextProto& PassContextImp::get_context_proto() const {
  return context_proto;
}
ContextProto& PassContextImp::get_context_proto() { return context_proto; }
void PassContextImp::save_context_json() const {
  ContextProto proto;
  proto.CopyFrom(this->context_proto);
  proto.mutable_config()->clear_encryption_key();
  auto all_provider_options = get_all_provider_options();
  proto.mutable_config()->mutable_provider_options()->clear();
  proto.mutable_config()->mutable_provider_options()->insert(
      all_provider_options.begin(), all_provider_options.end());
  try {
    if (std::find(proto.mutable_cache_files()->begin(),
                  proto.mutable_cache_files()->end(),
                  "context.json") == proto.mutable_cache_files()->end()) {
      proto.add_cache_files("context.json");
    }
    // When the GENERIC device is used, set fallback_cpu to true. When
    // inferencing a cached model, either from the cache directory or the EP
    // cache context file, we should not enable fallback_cpu. Otherwise, a
    // considerable amount of overhead is incurred for creating an
    // `Ort::Session` object for the subgraph behind the scenes.

    // There is a pitfall: if a custom op really needs to fall back to the
    // CPU, and the GENERIC device is enabled for model compilation, it is a
    // bug. However, this combination is not in used for now. We can fix it
    // later.
    for (auto& meta_def : *proto.mutable_meta_def()) {
      if (meta_def.device() == "GENERIC") {
        meta_def.set_fallback_cpu(true);
      }
    }
    auto json_str = msg_to_json_string(proto);
    const_cast<PassContextImp*>(this)->write_file("context.json", json_str);
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}

void PassContextImp::add_context_resource(const std::string& name,
                                          std::shared_ptr<void> resource) {
  pass_resources[name] = resource;
}

std::shared_ptr<void>
PassContextImp::get_context_resource(const std::string& name) const {
  auto it = pass_resources.find(name);
  auto ret = std::shared_ptr<void>();
  if (it != pass_resources.end()) {
    ret = it->second;
  }
  return ret;
}

std::unique_ptr<PassContextTimer>
PassContextImp::measure(const std::string& label) {
  return std::unique_ptr<PassContextTimer>(
      new PassContextTimerImp(label, *this));
}

void PassContextImp::on_custom_op_create_end() {
  created_customop_count++;
  if (created_customop_count == this->context_proto.meta_def_size()) {
    for (auto iter : cache_files_) {
      fclose(iter.second);
    }
    cache_files_.clear();
  }
}
/// struct PassContextTimerImp
PassContextTimerImp::PassContextTimerImp(const std::string& label,
                                         PassContextImp& context)
    : PassContextTimer(), label_{label}, context_{context},
      start_{std::chrono::steady_clock::now()}, mem_usage_{GetMemUsage()} {}
PassContextTimerImp::~PassContextTimerImp() {
  auto end_tp = std::chrono::steady_clock::now();
  auto end_mem_usage = GetMemUsage();
  auto event = context_.context_proto.mutable_events()->Add();
  int64_t thead_id = vaip_core::get_tid();
  int64_t process_id = vaip_core::get_pid();
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

  auto split = [](const std::string& str,
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
  for (const auto& backend : backends) {
    auto plugin = load_plugin(backend);
    plugins->push_back(plugin);
  }
  this->add_context_resource("__all_plugins__",
                             std::shared_ptr<void>((void*)plugins, [](void* p) {
                               delete (std::vector<std::shared_ptr<Plugin>>*)p;
                             }));
}
std::shared_ptr<Plugin>
PassContextImp::load_plugin(const std::string& plugin_name) {
  auto plugin = morphizen::WeakStore<std::string, Plugin>::create(
      plugin_name, plugin_name.c_str());
  return plugin;
}
CacheFileReaderImp::CacheFileReaderImp(bool in_mem, const std::string& filename,
                                       FILE* fp)
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

std::size_t CacheFileReaderImp::fread(void* buffer, std::size_t size) const {
  auto ret = std::fread(buffer, 1u, size, fp_);
  return ret;
}

size_t CacheFileReaderImp::size() const { return size_; }

void CacheFileReaderImp::rewind() const { std::rewind(fp_); }

CacheFileReaderStreamImp::CacheFileReaderStreamImp(const std::string& name,
                                                   size_t size,
                                                   TarEntryInputStream& stream)
    : name_{name}, size_{size}, stream_{stream} {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << name << " for read";
}

CacheFileReaderStreamImp::~CacheFileReaderStreamImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for read";
}

std::size_t CacheFileReaderStreamImp::fread(void* buffer,
                                            std::size_t size) const {
  CHECK(!stream_.read(static_cast<char*>(buffer), size).bad())
      << "failed to read " << name_;
  auto ret = stream_.gcount();
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= 9)
      << "read " << name_ << " " << ret << " bytes "
      << " size_ =" << size_;
  return ret;
}

void* CacheFileReaderStreamImp::mmap() { return stream_.mmap(); }

size_t CacheFileReaderStreamImp::size() const { return size_; }

void CacheFileReaderStreamImp::rewind() const {
  CHECK(!stream_.seekg(0, std::ios::beg).fail())
      << "failed to seek to the beginning of the stream";
}

CacheFileWriterImp::CacheFileWriterImp(bool in_mem, const std::string& filename,
                                       FILE* fp)
    : CacheFileWriter(), in_mem_(in_mem), name_{filename}, fp_{fp} {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "open " << filename << " for write";
}

CacheFileWriterImp::~CacheFileWriterImp() {
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "close " << name_ << " for write";
  std::fflush(fp_);
}

std::size_t CacheFileWriterImp::fwrite(const void* buffer,
                                       std::size_t size) const {
  auto ret = std::fwrite(buffer, 1u, size, fp_);
  return ret;
}

CacheFileWriterStreamImp::CacheFileWriterStreamImp(
    const std::string& name, std::unique_ptr<std::ostream> stream)
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

std::size_t CacheFileWriterStreamImp::fwrite(const void* buffer,
                                             std::size_t size) const {
  auto pos = stream_->tellp();
  stream_->write(static_cast<const char*>(buffer), size);
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

void PassContextImp::maybe_create_tar_file_for_write() {
  if (ENV_PARAM(MORPHIZEN_FEATURE_USE_TAR_FILE) != 1) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "MORPHIZEN_FEATURE_USE_TAR_FILE=1, disabled by user explicily";
    return;
  }
  auto is_shared_context_enabled =
      get_session_config(kOrtSessionOptionShareEpContexts, "0") == "1";
  auto is_stop_shared_context =
      get_session_config(kOrtSessionOptionStopShareEpContexts, "0") == "1";
  auto is_encryption_enabled = !context_proto.config().encryption_key().empty();
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  auto is_ep_context_embed_mode =
      get_session_config(kOrtSessionOptionEpContextEmbedMode, "1") == "1";

  if (is_encryption_enabled) {
    // TODO: remove this, after tar_file_ also supports encryption.
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "TODO: tar_file_ does not support encryption yet, "
           "please use a different cache file format.";
    return;
  }
  if (cache_in_mem() == false) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "no need to create create tar file for write";
    return;
  }
  if (!is_ep_context_enabled) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "ep.context is not enabled, no need to create tar file for write";
    return;
  }
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << "now, create tar file for write for ep context";
  // the tar_file_ is created upon the follow
  // 1. std::tmpfile(), i.e. TarFile::create(), for embed mode.
  // 2. ep_context_binary_file.
  if (!is_ep_context_embed_mode) {
    auto ep_context_binary_file = get_dir_of_ep_context_model() /
                                  get_basename_of_ep_context_binary_file();
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "open tar file for write: " << ep_context_binary_file;

    if (is_shared_context_enabled) {
      auto& shared_workspace =
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
    CHECK(!get_config_proto().cache_key().empty())
        << "cache_key should be empty when using tar file";
    tar_file_ = TarFile::create(std::move(stream));
  } else {
    tar_file_file_name_.clear();
    tar_file_ = TarFile::create();
    CHECK(tar_file_ != nullptr)
        << " create a tar file for write in embed mode, but tar_file_ is "
           "nullptr";
  }
  if (is_shared_context_enabled) {
    // for shared ep context, we must enable file prefix.
    cache_file_use_cache_key_prefix_ = true;
  } else {
    // for non-shared ep context, we can use cache_key_prefix or not, it is
    // and we prefer to enable it.
    cache_file_use_cache_key_prefix_ =
        get_provider_option("use_cache_key_prefix", "1") == "1";
  }
}
void PassContextImp::create_tar_file_for_read(std::string&& ep_context_binary,
                                              bool embed_mode) {
  if (!embed_mode) {
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
    tar_file_ = TarFile::create_from_path(
        ep_context_binary_file,
        get_provider_option(kProviderOptionEpContextEnableMmap, "1") == "1");
    CHECK(tar_file_ != nullptr)
        << "failed to open ep context file " << ep_context_binary_file;
  } else {
    tar_file_ = TarFile::create(std::move(ep_context_binary));
  }
}

void PassContextImp::create_tar_file_for_prebuild_cache(
    std::vector<char>&& buffer) {
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  auto is_ep_context_embed_mode =
      get_session_config(kOrtSessionOptionEpContextEmbedMode, "1") == "1";

  if (!is_ep_context_enabled) {
    // when ep.context is not enabled, we don't need to worry to much about
    // how to save tar_file_
    tar_file_ = TarFile::create(std::move(buffer));
    CHECK(tar_file_ != nullptr) << " create a tar file from memory ";
  } else {
    if (is_ep_context_embed_mode) {
      // for embeded mode, it works similar to is_ep_context_enable = false;
      tar_file_ = TarFile::create(std::move(buffer));
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
  if (tar_file_->has_file("context.json")) {
    cache_file_use_cache_key_prefix_ = false;
  } else {
    cache_file_use_cache_key_prefix_ = true;
    auto prefix = get_config_proto().cache_key();
    CHECK(tar_file_->has_file(prefix + "/context.json"))
        << "tar file does not have context.json, cache_key_prefix is "
           "enabled, but context.json is not found in the tar file, please "
           "check prebuild ep context generation";
  }
}
void PassContextImp::print_version_info(const char* prefix) {
  auto& config = get_config_proto();
  for (auto version_info : config.version().version_infos()) {
    LOG_VERBOSE(1) << prefix << version_info.package_name() << " ("
                   << version_info.version() << ") :" + version_info.commit();
  }
  auto print_kv = [](int level, const char* prefix,
                     std::pair<const std::string, std::string>& kv) {
    if (kv.first != "encryption_key") {
      LOG_VERBOSE(level) << prefix << ": " << kv.first << " = " << kv.second;
    } else {
      LOG_VERBOSE(level) << prefix << ": " << kv.first << " = "
                         << "******";
    }
  };
  LOG_VERBOSE(1) << prefix << "cache_dir: " << config.cache_dir();
  LOG_VERBOSE(1) << prefix << "cache_key: " << config.cache_key();
  LOG_VERBOSE(1) << prefix << "log_dir: " << get_log_dir();
  for (auto& kv : provider_option_origin_) {
    print_kv(3, "provider_option_from_origin", kv);
  }
  for (auto& kv : provider_option_from_cache_) {
    print_kv(3, "provider_options_from_cache", kv);
  }
  for (auto& kv :
       // print sorted keys
       std::map<std::string, std::string>(config.provider_options().begin(),
                                          config.provider_options().end())) {
    print_kv(3, "provider_options_in_config", kv);
  }
  if (mep_config_proto_) {
    for (auto& kv : std::map<std::string, std::string>(
             mep_config_proto_->provider_options().begin(),
             mep_config_proto_->provider_options().end())) {
      print_kv(3, "provider_options_in_mep_table", kv);
    }
  }
  if (target_proto_) {
    for (auto& kv : std::map<std::string, std::string>(
             target_proto_->provider_options().begin(),
             target_proto_->provider_options().end())) {
      print_kv(3, "provider_options_in_target_proto", kv);
    }
  }
  for (auto& kv : config.session_configs()) {
    LOG_VERBOSE(3) << "session_config: " << kv.first << " = " << kv.second;
  }
  auto all_po = get_all_provider_options();
  for (auto& kv : all_po) {
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
  // save cached provider options
  provider_option_from_cache_.insert(
      context_proto_in_cache.config().provider_options().begin(),
      context_proto_in_cache.config().provider_options().end());
  this->context_proto.Swap(&context_proto_in_cache);
  auto& context_proto_origin =
      context_proto_in_cache; // give it a new name after swap.
  // restore session options
  this->context_proto.mutable_config()->mutable_session_configs()->swap(
      *context_proto_origin.mutable_config()->mutable_session_configs());
  this->context_proto.mutable_config()->mutable_provider_options()->swap(
      *context_proto_origin.mutable_config()->mutable_provider_options());
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
  restore_cache_files();
}
void PassContextImp::update_config_proto_root_field() {
  // ADD_CUSTOM_FIELD∆
  // NOTE:
  //  1. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  //  2. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  //  3. FOR BACKWARD COMPATIBILITY, NO MORE NEW FIELD PLEASE
  auto get_provider_option_local =
      [this](
          const std::vector<std::string>& names) -> std::optional<std::string> {
    auto ret = std::optional<std::string>();
    return this->get_provider_option_with_priority(names);
  };
  if (auto cache_key = get_provider_option_local({"cache_key", "cacheKey"})) {
    context_proto.mutable_config()->set_cache_key(*cache_key);
  }
  if (auto cache_dir = get_provider_option_local({"cache_dir", "cacheDir"})) {
    context_proto.mutable_config()->set_cache_dir(*cache_dir);
  }
  if (auto encryption_key =
          get_provider_option_local({"encryption_key", "encryptionKey"})) {
    context_proto.mutable_config()->set_encryption_key(*encryption_key);
  }
  if (auto target = get_provider_option_local({"target", "xlnx_target_name"})) {
    context_proto.mutable_config()->set_target(*target);
  }
  if (auto priority = get_provider_option_local({"priority"})) {
    context_proto.mutable_config()->set_priority(*priority);
  }
  if (auto no_failsafe =
          get_provider_option_local({"no_fail_safe", "noFailSafe"})) {
    context_proto.mutable_config()->set_no_failsafe(*no_failsafe == "1");
  }
  if (auto enable_preemption = get_provider_option_local(
          {"enable_preemption", "enablePreemption"})) {
    context_proto.mutable_config()->set_enable_preemption(*enable_preemption ==
                                                          "1");
  }
  if (auto max_spill_buffer_size = get_provider_option_local(
          {"max_spill_buffer_size", "maxSpillBufferSize"})) {
    context_proto.mutable_config()->set_max_spill_buffer_size(
        std::stoull(*max_spill_buffer_size));
  }
}
template <typename T>
static std::optional<std::string>
get_provider_option_internal(const std::string& name, const T& options) {
  auto it = options.find(name);
  if (it != options.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::unique_ptr<TargetProto>
PassContextImp::find_target_proto(const std::string& target_name) {
  auto& targets = context_proto.config().targets();
  auto it = std::find_if(targets.begin(), targets.end(),
                         [&target_name](const TargetProto& target) {
                           return target.name() == target_name;
                         });
  if (it != targets.end()) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "Found target proto for target: " << target_name;
    return std::make_unique<TargetProto>(*it);
  }
  return nullptr; // not found
}

std::string PassContextImp::get_valid_target_names() {
  std::ostringstream valid_names;
  int c = 0;
  auto& targets = context_proto.config().targets();

  for (const auto& target : targets) {
    if (c++ > 0) {
      valid_names << ", ";
    }
    valid_names << '"' << target.name() << '"';
  }
  return valid_names.str();
}
bool PassContextImp::try_initialize_target_proto(const std::string& target_name,
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
static std::optional<std::string> discover_target(const ConfigProto& proto,
                                                  const Model& model) {
  typedef std::optional<std::string> (*discovery_function_t)(const ConfigProto&,
                                                             const Model&);
  auto all_plugin_functions =
      vaip_core::Plugin::get_all_symbols("morphizen_target_discovery");
  std::sort(
      all_plugin_functions.begin(), all_plugin_functions.end(),
      [](const std::pair<std::string, void*>& a,
         const std::pair<std::string, void*>& b) { return a.first < b.first; });
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
      << "discover_target: all_plugin_functions size: "
      << all_plugin_functions.size();
  for (auto& plugin : all_plugin_functions) {
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
        << "discover_target: plugin name: " << plugin.first
        << " model id:" << (void*)(&model) << " id: " << plugin.second;
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
void PassContextImp::target_auto_discovery(const Model& model) {
  auto target_specified_by_end_user = get_provider_option_internal(
      kProviderOptionTarget, provider_option_origin_);
  auto config_file = get_provider_option_internal(kProviderOptionConfigFile,
                                                  provider_option_origin_);
  do {
    // 1. `provider_options["target"]` set explicitly by users
    if (target_specified_by_end_user) {
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "Target auto-discovery: target proto specified by end user: "
          << *target_specified_by_end_user;
      if (try_initialize_target_proto(*target_specified_by_end_user, true)) {
        break;
      }
    }
    // 2. `provider_options["target"]` not set, but `config_file` is set.
    // try to find target proto from config file.
    if (mep_config_proto_) {
      auto& target_name = mep_config_proto_->target();
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "Target auto-discovery: target proto specified by MEP table: "
          << target_name << "; MEP table = " << mep_config_proto_->model_name();
      if (try_initialize_target_proto(target_name, true)) {
        if (mep_config_proto_->has_xclbin()) {
          // FIXME: we need to find xclbin in a proper way.
          target_proto_->set_xclbin(mep_config_proto_->xclbin());
        }
        break;
      }
    }
    {
      // 3. auto target discovery
      //
      // - when the build-in config file is used, the plugin must return
      // a valid target name, otherwise it is a fatal error, because it
      // means that the source code is not consistent with the built-in
      // config file, and the built-in config file is regarded as the
      // part source code.
      auto discoveried_target_name =
          discover_target(context_proto.config(), model);
      if (discoveried_target_name.has_value()) {
        if (try_initialize_target_proto(*discoveried_target_name, true)) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
              << "Target auto-discovery: target proto discovered: "
              << *discoveried_target_name;
          break;
        }
      }
    }
    { // 4. default target in config file
      auto& target_name = context_proto.config().target();
      LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY))
          << "Target auto-discovery: target proto specified by config file: "
          << target_name;
      if (try_initialize_target_proto(target_name, true)) {
        break;
      }
    }
  } while (0);
}
} // namespace vaip_core
