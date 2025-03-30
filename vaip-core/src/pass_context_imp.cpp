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
#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include "morphizen/env_config.hpp"
#include "morphizen/mem_xclbin.hpp"
#include "morphizen/util.hpp"
#include "morphizen/vaip_io.hpp"
#include "morphizen/weak.hpp"
#include "pass_context_imp.hpp"
#include "profile_utils.hpp"
#include "tar_ball.hpp"
DEF_ENV_PARAM(DEBUG_TAR_CACHE, "0")

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

/// struct PassContextImp
int PassContextImp::allocate_suffix()
    const { // it is not a big deal to update suffix_counter
  suffix_counter = suffix_counter + 1;
  return suffix_counter;
}

PassContextImp::WithPass PassContextImp::with_current_pass(IPass& pass) {
  return WithPass(*this, pass);
}

std::filesystem::path PassContextImp::get_log_dir() const { return log_dir; }
std::optional<std::string>
PassContextImp::get_provider_option(const std::string& option_name) const {
  const auto& config = context_proto.config();
  auto it = config.provider_options().find(option_name);
  if (it != config.provider_options().end()) {
    return it->second;
  }
  return std::nullopt;
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
  auto context_enable_option = this->get_session_config("ep.context_enable");
  bool cache_inside_model = context_enable_option.has_value() &&
                            (context_enable_option.value() == "1");
  bool use_cache_model = cache_inside_model || is_ep_context_model;
  if (!use_cache_model) {
#ifdef WIN24_BUILD
    // session option set
    if (cache_dir_set) {
      return false;
    }
    return get_config_proto().enable_cache_file_io_in_mem();
#else
    return false;
#endif
  } else {
    return get_config_proto().enable_cache_file_io_in_mem();
  }
}
PassContextImp::~PassContextImp() {
  for (auto iter : cache_files_) {
    fclose(iter.second);
  }
}
void PassContextImp::set_is_ep_context_model(bool is_ep_context_model) {
  this->is_ep_context_model = is_ep_context_model;
}

bool PassContextImp::get_is_ep_context_model() {
  return this->is_ep_context_model;
}

void PassContextImp::set_cache_file_md5_map(
    const std::map<std::string, std::string>& cache_file_md5) {
  cache_file_md5s_ = cache_file_md5;
}
std::map<std::string, std::string> PassContextImp::get_cache_file_md5_map() {
  return cache_file_md5s_;
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

template <typename char_type> struct binary_io {
  static std::vector<char_type> slurp_binary(FILE* file) {
    CHECK(fseek64(file, 0, SEEK_SET) == 0);
    CHECK(fseek64(file, 0, SEEK_END) == 0);
    auto size = ftell64(file);
    CHECK(fseek64(file, 0, SEEK_SET) == 0);
    auto buffer = std::vector<char_type>((size_t)size / sizeof(char_type));
    if (size != 0) {
      CHECK(fread(buffer.data(), 1, size, file) == static_cast<size_t>(size));
    }
    return buffer;
  }
};

template <typename T>
std::optional<std::vector<T>>
PassContextImp::read_file_generic(const std::string& filename) const {
  std::optional<std::vector<T>> ret;
  auto stream = open_file_for_read(filename);
  if (stream == nullptr) {
    return std::nullopt;
  }
  constexpr size_t buffer_size = 8196;
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
  std::unique_ptr<CacheFileReader> ret = nullptr;
  auto in_mem = cache_in_mem();
  auto& cace_files =
      const_cast<std::remove_cv_t<decltype(cache_files_)&>>(cache_files_);
  auto it = cace_files.find(filename);
  if (it != cace_files.end()) {
    if (in_mem) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
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
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
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
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
            << " cannot freopen " << filename;
      } else {
        cace_files[filename] = fp;
        ret = std::unique_ptr<CacheFileReader>(
            new CacheFileReaderImp(in_mem, filename, fp));
      }
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
          << "tmp file open failed: cannot found " << filename
          << ". try to use write_file_for_write before reading.";
      ret = nullptr;
    }
  }
  return ret;
}

std::unique_ptr<CacheFileWriter>
PassContextImp::open_file_for_write(const std::string& filename) {
  std::unique_ptr<CacheFileWriter> ret = nullptr;
  auto it = cache_files_.find(filename);
  FILE* tmp_file = nullptr;
  auto in_mem = cache_in_mem();
  if (it != cache_files_.end()) {
    if (in_mem) {
      fclose(it->second);
      LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
          << "tmp file write: " << filename;
      tmp_file = tmpfile();
      if (tmp_file == nullptr) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
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
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
            << " cannot freopen " << filename;
      } else {
        it->second = fp;
        return std::unique_ptr<CacheFileWriter>(
            new CacheFileWriterImp(in_mem, filename, fp));
      }
    }
  } else {
    if (in_mem) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
          << "tmp file write: " << filename;
      tmp_file = tmpfile();
      if (tmp_file == nullptr) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
            << "cannot create tmp file" << filename;
      } else {
        cache_files_[filename] = tmp_file;
        this->context_proto.add_cache_files(filename);
        ret = std::unique_ptr<CacheFileWriter>(
            new CacheFileWriterImp(in_mem, filename, tmp_file));
      }
    } else {
      std::filesystem::path tmp_dir = (get_log_dir() / filename).parent_path();
      if (!std::filesystem::exists(tmp_dir)) {
        std::filesystem::create_directories(tmp_dir);
      }
#ifdef _WIN32
      tmp_file = _wfopen((get_log_dir() / filename).c_str(), L"wb+");
#else
      tmp_file = std::fopen((get_log_dir() / filename).c_str(), "wb+");
#endif //  _WIN32
      if (tmp_file == nullptr) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
            << " fopen failed. " << filename;
      } else {
        cache_files_[filename] = tmp_file;
        this->context_proto.add_cache_files(filename);
        ret = std::unique_ptr<CacheFileWriter>(
            new CacheFileWriterImp(in_mem, filename, tmp_file));
      }
    }
  }
  return ret;
}

FILE* PassContextImp::open_file(const std::string& filename) const {
  auto it = cache_files_.find(filename);
  if (it != cache_files_.end()) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE)) << "tmp file opened: " << filename;
    return it->second;
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
      << "tmp file open failed: " << filename;
  return nullptr;
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
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
      << "write " << filename << " " << data.size()
      << " bytes to the cache files";
  return ret;
}

void PassContextImp::restore_cache_files() {
  for (const auto& str : this->context_proto.cache_files()) {
    open_file_for_read(str);
  }
}

bool PassContextImp::has_cache_file(const std::string& filename) const {
  return cache_files_.find(filename) != cache_files_.end();
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

bool PassContextImp::tar_mem_to_cache_files(const char* buffer, size_t size) {
  // todo: is this function can be delete
  // auto p = buffer;
  // for (;;) {
  //   auto [filename, data] = tarball_read_file_from_memory(p, size);
  //   if (filename.empty()) {
  //     break;
  //   }
  //   LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
  //       << "load " << filename << " " << data.size() << " bytes";
  //   gsl::span<char> data_span = gsl::span<char>(data);
  //   write_file(filename, data_span);
  // }
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
  if (has_mem_xclbin(filename)) {
    buffer = get_mem_xclbin(filename);
  } else if (std::filesystem::is_regular_file(path, ec)) {
    buffer = read_file_to_buffer(path);
  } else {
    LOG(WARNING)
        << "Xclbin path doesn't exist, are you running with cpu runner? Path: "
        << path.string();
    return path;
  }
  const_cast<PassContextImp*>(this)->write_file(filename, buffer);
  return ret;
}

std::optional<std::vector<char>>
PassContextImp::read_xclbin(const std::filesystem::path& path) const {
  auto filename = path.filename().u8string();
  return read_file_c8(filename);
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
  try {
    if (std::find(proto.mutable_cache_files()->begin(),
                  proto.mutable_cache_files()->end(),
                  "context.json") == proto.mutable_cache_files()->end()) {
      proto.add_cache_files("context.json");
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
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
      << "open " << filename << " for read";
}

CacheFileReaderImp::~CacheFileReaderImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE)) << "close " << name_ << " for read";
}

std::size_t CacheFileReaderImp::fread(void* buffer, std::size_t size) const {
  auto ret = std::fread(buffer, 1u, size, fp_);
  return ret;
}

size_t CacheFileReaderImp::size() const { return size_; }

void CacheFileReaderImp::rewind() const { std::rewind(fp_); }

CacheFileWriterImp::CacheFileWriterImp(bool in_mem, const std::string& filename,
                                       FILE* fp)
    : CacheFileWriter(), in_mem_(in_mem), name_{filename}, fp_{fp} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE))
      << "open " << filename << " for write";
}

CacheFileWriterImp::~CacheFileWriterImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TAR_CACHE)) << "close " << name_ << " for write";
  std::fflush(fp_);
}

std::size_t CacheFileWriterImp::fwrite(const void* buffer,
                                       std::size_t size) const {
  auto ret = std::fwrite(buffer, 1u, size, fp_);
  return ret;
}

} // namespace vaip_core
