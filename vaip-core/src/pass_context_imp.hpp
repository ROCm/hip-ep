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
#pragma once
#include <deque>
#include <shared_mutex>

#include <vaip/custom_op.h>
#include <vaip/dll_safe.h>

#include "./tar_file.hpp"
#include "morphizen/model.hpp"
#include "morphizen/pass.hpp"
#include "morphizen/pass_context.hpp"
#include "morphizen/vaip_io.hpp"
#include "morphizen/vaip_plugin.hpp"
namespace vaip_core {
class CacheFileReaderImp : public CacheFileReader {
public:
  CacheFileReaderImp(bool in_mem, const std::string& filename, FILE* fp);
  virtual ~CacheFileReaderImp();

private:
  size_t size() const override final;
  void rewind() const override final;
  virtual std::size_t fread(void* buffer,
                            std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  size_t size_;
  FILE* fp_;
};
class CacheFileReaderStreamImp : public CacheFileReader {
public:
  CacheFileReaderStreamImp(const std::string& name, size_t size,
                           std::istream& stream);
  virtual ~CacheFileReaderStreamImp();

private:
  size_t size() const override final;
  void rewind() const override final;
  virtual std::size_t fread(void* buffer,
                            std::size_t size) const override final;

private:
  const std::string name_; // for debugging purpuse
  const size_t size_;
  std::istream& stream_;
};
class CacheFileWriterImp : public CacheFileWriter {
public:
  CacheFileWriterImp(bool in_mem, const std::string& fileanme, FILE* fp);
  virtual ~CacheFileWriterImp();

private:
  virtual std::size_t fwrite(const void* buffer,
                             std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  FILE* fp_;
};
class CacheFileWriterStreamImp : public CacheFileWriter {
public:
  CacheFileWriterStreamImp(const std::string& name,
                           std::unique_ptr<std::ostream> stream);
  virtual ~CacheFileWriterStreamImp();

private:
  virtual std::size_t fwrite(const void* buffer,
                             std::size_t size) const override final;

private:
  std::string name_;
  std::unique_ptr<std::ostream> stream_;
};
class CacheFileStreamWriter : public IStreamWriter {
public:
  CacheFileStreamWriter(std::unique_ptr<CacheFileWriter>&& writer)
      : writer_(std::move(writer)) {}

private:
  virtual size_t write(const char* data, size_t size) override final;

private:
  std::unique_ptr<CacheFileWriter> writer_;
};

class CacheFileStreamWriterBuilder : public IStreamWriterBuilder {
public:
  CacheFileStreamWriterBuilder(PassContext* ctx) : context(ctx) {}

private:
  virtual std::unique_ptr<IStreamWriter>
  build(const std::string& filename) override final;

private:
  PassContext* context;
};

class CacheFileStreamReader : public IStreamReader {
public:
  CacheFileStreamReader(std::unique_ptr<CacheFileReader> reader)
      : reader_(std::move(reader)) {}

private:
  virtual std::optional<std::vector<char>>
  read(size_t size_hint) const override final;

private:
  std::unique_ptr<CacheFileReader> reader_;
};

static void
store_cache_directory_from_main_node(class PassContextImp& context,
                                     vaip_cxx::NodeConstRef main_node);
class ExecutionProviderConcrete;
static onnxruntime::Node* create_ep_context_node(ExecutionProviderConcrete* ep);
class PassContextImp : public PassContext {
public:
  std::vector<char> const_data_;
  std::map<std::string, std::shared_ptr<std::function<void(gsl::span<char>)>>>
      const_lazy_;
  std::filesystem::path log_dir;
  std::map<std::string, std::vector<AttributeProtoPtr>> node_extra_attrs;
  std::deque<IPass*> current_pass_stack;
  ContextProto context_proto;
  bool is_ep_context_model = false;
  bool cache_dir_set = false;
  std::filesystem::path model_path;
  std::unique_ptr<vaip_cxx::Model> ep_context_model_;
  std::chrono::time_point<std::chrono::steady_clock> start_ =
      std::chrono::steady_clock::now();
  mutable int suffix_counter = 0;
  std::unordered_map<std::string, std::shared_ptr<void>> pass_resources;

public:
  ~PassContextImp();
  int allocate_suffix() const;
  virtual std::filesystem::path get_log_dir() const override final;
  virtual std::optional<std::string>
  get_provider_option(const std::string& option_name) const override final;
  virtual std::optional<std::string>
  get_session_config(const std::string& option_name) const override final;
  virtual std::string
  get_provider_option(const std::string& option_name,
                      const std::string& default_value) const override final;
  virtual int64_t get_provider_option_i64(const std::string& option_name,
                                          int64_t default_value) const;
  virtual bool cache_in_mem() const override final;
  virtual void set_is_ep_context_model(bool is_ep_context_model) override final;
  virtual bool get_is_ep_context_model() override final;
  virtual std::string
  get_session_config(const std::string& option_name,
                     const std::string& default_value) const override final;
  virtual std::string
  get_run_option(const std::string& option_name,
                 const std::string& default_value) const override final;

  virtual std::string
  get_ep_dynamic_option(const std::string& option_name,
                        const std::string& default_value) const override final;

  virtual void add_QosUpdater(
      const std::shared_ptr<QoSUpdateInterface>& updater) const override final;
  virtual void
  update_all_qos(const std::string& workload_type) const override final;
  virtual const ConfigProto& get_config_proto() const override final;
  virtual const ContextProto& get_context_proto() const override final;
  virtual ContextProto& get_context_proto() override final;
  void load_plugins();
  std::shared_ptr<Plugin> load_plugin(const std::string& plugin_name);

private:
  template <typename T>
  std::optional<std::vector<T>>
  read_file_generic(const std::string& filename) const;

public:
  virtual std::filesystem::path get_model_path() const override final;
  virtual std::optional<std::vector<char>>
  read_file_c8(const std::string& filename) const override final;
  std::optional<std::vector<uint8_t>>
  read_file_u8(const std::string& filename) const override final;
  virtual std::unique_ptr<CacheFileReader>
  open_file_for_read(const std::string& filename) const override final;
  std::unique_ptr<CacheFileReader>
  open_file_for_read_with_tar_file(const std::string& filename) const;
  virtual std::unique_ptr<CacheFileWriter>
  open_file_for_write(const std::string& filename) override final;
  std::unique_ptr<CacheFileWriter>
  open_file_for_write_with_tar_file(const std::string& filename);
  virtual FILE* open_file(const std::string& filename) const override final;
  virtual bool write_file(const std::string& filename,
                          gsl::span<const char> data) override final;
  virtual void restore_cache_files() override final;
  virtual bool has_cache_file(const std::string& filename) const override final;
  virtual std::vector<std::string> get_cache_file_names() const override final;
  virtual std::vector<char> cache_files_to_tar_mem() const override final;

  virtual bool
  cache_files_to_tar_file(IStreamWriter& writer) const override final;
  virtual bool tar_mem_to_cache_files(const char* data,
                                      size_t size) override final;
  virtual bool tar_file_to_cache_files(class IStreamReader& src) override final;

  virtual std::shared_ptr<void>
  get_context_resource(const std::string& name) const override final;
  virtual std::filesystem::path xclbin_path_to_cache_files(
      const std::filesystem::path& path) const override final;
  virtual std::optional<std::vector<char>>
  read_xclbin(const std::filesystem::path& path) const override final;
  virtual std::unique_ptr<PassContextTimer>
  measure(const std::string& label) override final;
  virtual void on_custom_op_create_end() override final;
  virtual void set_cache_file_md5_map(
      const std::map<std::string, std::string>& cache_file_md5) override final;
  virtual std::map<std::string, std::string>
  get_cache_file_md5_map() override final;
  // helper class
  struct WithPass {
    WithPass(PassContextImp& context, IPass& pass);
    WithPass(const WithPass&) = delete;
    ~WithPass();
    PassContextImp* _context;
  };
  WithPass with_current_pass(IPass& pass);
  void add_context_resource(const std::string& name,
                            std::shared_ptr<void> resource);
  virtual void save_context_json() const override final;

private:
  // use std::map to keep filename ordered.
  std::map<std::string, FILE*> cache_files_;
  std::map<std::string, std::string> cache_file_md5s_;
  std::function<std::optional<std::string>(std::string)> get_run_options_;
  std::shared_mutex rw_mutex_;
  friend int vitisai_ep_on_run_start(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
      const void* state,
      vaip_core::DllSafe<std::string> (*get_config_entry)(
          const void* state, const char* entry_name));
  friend int vitisai_ep_set_ep_dynamic_options(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
      const char* const* keys, const char* const* values, size_t kv_len);
  std::map<std::string, std::string> ep_dynamic_options;
  mutable std::mutex ep_dynamic_options_lock;
  // for share context, many context may be same. may need to change container
  // to set.
  mutable std::vector<std::shared_ptr<QoSUpdateInterface>> qos_updaters_;
  int created_customop_count = 0;
  std::unique_ptr<TarFile> tar_file_ = nullptr;
  // cache_file_use_cache_key_prefix_ is only enabled for shared ep context is
  // enabled.
  // when this feature is enabled, open_file_for_read and
  // open_file_for_write, the file name will be prefixed with cache_key_prefix_.
  // this feature only tested when tar_file_ is not null.
  bool cache_file_use_cache_key_prefix_ = false;

private:
  friend void
  store_cache_directory_from_main_node(PassContextImp& context,
                                       vaip_cxx::NodeConstRef main_node);
  friend std::shared_ptr<PassContextImp> initialize_context(
      const std::string& model_path, const Graph& onnx_graph,
      const std::vector<vaip_cxx::NodeConstRef>& ep_context_nodes,
      const char* json_config);
  friend onnxruntime::Node*
  create_ep_context_node(vaip_core::ExecutionProviderConcrete* ep);
  friend std::string
  get_ep_cache_context_nonembed_mode(PassContextImp& context);
};

struct PassContextTimerImp : public PassContextTimer {
  PassContextTimerImp(const std::string& label, PassContextImp& context);
  virtual ~PassContextTimerImp();
  std::string label_;
  PassContextImp& context_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
  MemUsageProto mem_usage_;
};
} // namespace vaip_core
