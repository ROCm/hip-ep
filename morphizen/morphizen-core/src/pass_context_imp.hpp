/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/morphizen.hpp"
#include <deque>
#include <mutex>

#include <morphizen/custom_op.h>
#include <morphizen/dll_safe.h>

#include "./tar_file.hpp"
#include "logger_adapter.hpp"
#include "morphizen/model.hpp"
#include "morphizen/pass.hpp"
#include "morphizen/pass_context.hpp"
#include "morphizen/plugin.hpp"

namespace morphizen {
// Forward declaration for logger adapter
class LoggerAdapter;

// Publishes the accessor that PassContextImp::get_run_option() uses to reach
// the OrtRunOptions of the run in progress. Installed by
// morphizen_ep_on_run_start() and cleared by passing nulls when the run ends.
// The accessor is per-thread, because a session may be run concurrently from
// several threads with different OrtRunOptions.
void set_run_option_accessor(const void *state,
                             DllSafe<std::string> (*get_entry)(
                                 const void *state, const char *entry_name));

class CacheFileReaderImp : public CacheFileReader {
public:
  CacheFileReaderImp(bool in_mem, const std::string &filename, FILE *fp);
  virtual ~CacheFileReaderImp();

private:
  size_t size() const override final;
  void rewind() const override final;
  virtual std::size_t fread(void *buffer,
                            std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  size_t size_;
  FILE *fp_;
};
class CacheFileReaderStreamImp : public CacheFileReader {
public:
  CacheFileReaderStreamImp(const std::string &name, size_t size,
                           TarEntryInputStream &stream);
  virtual ~CacheFileReaderStreamImp();

private:
  size_t size() const override final;
  void rewind() const override final;
  virtual std::size_t fread(void *buffer,
                            std::size_t size) const override final;
  virtual void *mmap() override final;

private:
  const std::string name_; // for debugging purpuse
  const size_t size_;
  TarEntryInputStream &stream_;
};
class CacheFileWriterImp : public CacheFileWriter {
public:
  CacheFileWriterImp(bool in_mem, const std::string &fileanme, FILE *fp);
  virtual ~CacheFileWriterImp();

private:
  virtual std::size_t fwrite(const void *buffer,
                             std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  FILE *fp_;
};
class CacheFileWriterStreamImp : public CacheFileWriter {
public:
  CacheFileWriterStreamImp(const std::string &name,
                           std::unique_ptr<std::ostream> stream);
  virtual ~CacheFileWriterStreamImp();

private:
  virtual std::size_t fwrite(const void *buffer,
                             std::size_t size) const override final;

private:
  std::string name_;
  std::unique_ptr<std::ostream> stream_;
};

class FileSystemImp : public FileSystem {
public:
  explicit FileSystemImp(PassContext &context);
  FileReader *create_reader(const char *path) override;
  FileWriter *create_writer(const char *path) override;
  void destroy_reader(FileReader *reader) override;
  void destroy_writer(FileWriter *writer) override;

private:
  PassContext &context_;
};

// iostream adapters for CacheFile

class CacheFileOstreambuf : public std::streambuf {
public:
  explicit CacheFileOstreambuf(std::unique_ptr<CacheFileWriter> &&writer)
      : writer_(std::move(writer)) {}

protected:
  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      char ch = static_cast<char>(c);
      writer_->fwrite(&ch, 1);
    }
    return c;
  }

  std::streamsize xsputn(const char *s, std::streamsize n) override {
    writer_->fwrite(s, n);
    return n;
  }

private:
  std::unique_ptr<CacheFileWriter> writer_;
};

class CacheFileIstreambuf : public std::streambuf {
public:
  explicit CacheFileIstreambuf(std::unique_ptr<CacheFileReader> &&reader)
      : reader_(std::move(reader)) {}

protected:
  int_type underflow() override {
    if (gptr() == egptr()) {
      buffer_.resize(4096);
      auto read_size = reader_->fread(buffer_.data(), buffer_.size());
      if (read_size == 0) {
        return traits_type::eof();
      }
      setg(buffer_.data(), buffer_.data(), buffer_.data() + read_size);
    }
    return gptr() == egptr() ? traits_type::eof()
                             : traits_type::to_int_type(*gptr());
  }

private:
  std::unique_ptr<CacheFileReader> reader_;
  std::vector<char> buffer_;
};

class CacheFileOstreamAdapter : public std::ostream {
public:
  explicit CacheFileOstreamAdapter(std::unique_ptr<CacheFileWriter> &&writer)
      : std::ostream(&buf_), buf_(std::move(writer)) {}

private:
  CacheFileOstreambuf buf_;
};

class CacheFileIstreamAdapter : public std::istream {
public:
  explicit CacheFileIstreamAdapter(std::unique_ptr<CacheFileReader> &&reader)
      : std::istream(&buf_), buf_(std::move(reader)) {}

private:
  CacheFileIstreambuf buf_;
};

static void
store_cache_directory_from_main_node(class PassContextImp &context,
                                     morphizen_cxx::NodeConstRef main_node);
class ExecutionProviderConcrete;
static onnxruntime::Node *create_ep_context_node(ExecutionProviderConcrete *ep,
                                                 int index);
static std::string get_ep_cache_context_nonembed_mode(PassContextImp &context);
static std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_internal(
    const Graph &onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
    std::shared_ptr<PassContextImp> context);
class PassContextImp : public PassContext {
public:
  static std::unique_ptr<PassContextImp>
  create_pass_context(const ConfigProto &config);
  static std::unique_ptr<PassContextImp> create_pass_context(
      const onnxruntime::ProviderOptions &options,
      const std::map<std::string, std::string> &session_configs);

  // Private tag pattern - see docs/technical/privatetag-factory-pattern.md
  struct PrivateTag {};

  // Default constructor for PassContext::create()
  PassContextImp() : config_(ConfigProto()) {}

  // Constructor with private tag - allows make_unique in factory methods
  explicit PassContextImp(PrivateTag, ConfigProto config)
      : config_(std::move(config)) {}

private:
public:
  std::map<std::string, std::vector<AttributeProtoPtr>> node_extra_attrs;
  std::deque<IPass *> current_pass_stack;
  ContextProto context_proto;
  const ConfigProto config_; // Runtime-only INPUT (never serialized, immutable)
  bool is_ep_context_model = false;
  std::filesystem::path model_path;
  std::unique_ptr<morphizen_cxx::Model> ep_context_model_;
  std::chrono::time_point<std::chrono::steady_clock> start_ =
      std::chrono::steady_clock::now();
  mutable int suffix_counter = 0;
  std::unordered_map<std::string, std::shared_ptr<void>> pass_resources;
  // Logger integration - keep these alive for the duration of PassContext
  std::unique_ptr<LoggerAdapter> logger_adapter_;

public:
  ~PassContextImp();
  int allocate_suffix() const;

  /**
   * @brief Returns the directory path for debugging and troubleshooting dumps.
   *
   * This directory is used ONLY for debugging output files such as graph dumps
   * (.txt, .onnx), fix info, and diagnostic reports. It is NOT used for cache
   * persistence (which uses EP context tar-based cache).
   *
   * The dump directory can be overridden via the 'dump_dir' provider option.
   * Otherwise, defaults to: temp/morphizen_dumps/<cache_key>
   *
   * Directory is created on-demand when dump operations are performed.
   *
   * @return The directory path where dump files should be written.
   *
   * @note This replaces the legacy get_log_dir() for dump operations.
   * @see maybe_dump_txt(), maybe_dump_onnx(), dump_fix_info()
   */
  virtual std::filesystem::path get_dump_directory() const override final;
  virtual std::optional<std::string>
  get_provider_option(const std::string &option_name) const override final;
  virtual std::optional<std::string>
  get_session_config(const std::string &option_name) const override final;
  virtual std::string
  get_provider_option(const std::string &option_name,
                      const std::string &default_value) const override final;
  virtual int64_t get_provider_option_i64(const std::string &option_name,
                                          int64_t default_value) const;
  virtual bool cache_in_mem() const override final;
  virtual std::string
  get_session_config(const std::string &option_name,
                     const std::string &default_value) const override final;
  virtual std::string
  get_run_option(const std::string &option_name,
                 const std::string &default_value) const override final;
  virtual std::string
  get_meta_def_param(const MetaDefProto &meta_def) const override final;

  virtual std::string
  get_ep_dynamic_option(const std::string &option_name,
                        const std::string &default_value) const override final;
  virtual void remove_QosUpdater(QoSUpdateInterface *) override final;
  virtual void add_QosUpdater(
      const std::shared_ptr<QoSUpdateInterface> &updater) const override final;
  virtual void
  update_all_qos(const std::string &workload_type) const override final;
  virtual const ConfigProto &get_config_proto() const override final;
  virtual const ContextProto &get_context_proto() const override final;
  virtual ContextProto &get_context_proto() override final;
  void load_plugins();
  std::shared_ptr<Plugin> load_plugin(const std::string &plugin_name);
  virtual std::map<std::string, std::string>
  get_all_provider_options() const override final;

  // Compute the effective pass list for compilation.
  // Combines ConfigProto.passes library with target-based selection.
  // Returns local vector - passes are not stored as member variable.
  std::vector<PassProto> compute_effective_passes() const;

private:
  template <typename T1, typename T2>
  std::optional<std::string>
  get_provider_option_impl(const T1 &option_names,
                           const T2 &privider_options) const;
  template <typename T1, typename T, typename... T2>
  std::optional<std::string>
  get_provider_option_impl(const T1 &option_names, const T &options1,
                           const T2 &...options) const;

  template <typename T>
  void get_all_provider_option_impl(std::map<std::string, std::string> &ret,
                                    const T &source) const;
  template <typename T, typename... T1>
  void get_all_provider_option_impl(std::map<std::string, std::string> &ret,
                                    const T &source,
                                    const T1 &...options) const;

  template <typename T1, typename... T2>
  std::optional<std::string>
  get_provider_option_with_priority(const T1 &option_names) const;
  template <typename T>
  std::optional<std::vector<T>>
  read_file_generic(const std::string &filename) const;

  std::filesystem::path get_dir_of_ep_context_model();
  std::filesystem::path get_basename_of_ep_context_model();
  std::filesystem::path get_basename_of_ep_context_binary_file();

  void maybe_create_tar_file_for_write();
  void create_tar_file_for_read(std::string &&ep_context_binary_file_name,
                                bool embed_mode);

  void print_version_info(const char *prefix);
  void pass_context_update_context_json(gsl::span<char> json_str);
  void update_pass_context_from_context_json_in_cache();
  void create_tar_file_for_prebuild_cache(std::vector<char> &&buffer);
  void update_config_proto_root_field();
  std::string get_cache_filename(const std::string &filename) const;
  void target_auto_discovery(const Model &model);
  const TargetProto *find_target_proto(const std::string &target_name);
  bool try_initialize_target_proto(const std::string &target_name,
                                   bool thorow_if_not_found);
  std::string get_valid_target_names();
  bool has_user_config_file() const;

public:
  virtual std::filesystem::path get_model_path() const override final;
  virtual std::optional<std::vector<char>>
  read_file_c8(const std::string &filename) const override final;
  std::optional<std::vector<uint8_t>>
  read_file_u8(const std::string &filename) const override final;
  virtual std::unique_ptr<CacheFileReader>
  open_file_for_read(const std::string &filename) const override final;
  std::unique_ptr<CacheFileReader>
  open_file_for_read_with_tar_file(const std::string &filename) const;
  virtual std::unique_ptr<CacheFileWriter>
  open_file_for_write(const std::string &filename) override final;
  std::unique_ptr<CacheFileWriter>
  open_file_for_write_with_tar_file(const std::string &filename);
  virtual bool write_file(const std::string &filename,
                          gsl::span<const char> data) override final;
  virtual bool has_cache_file(const std::string &filename) const override final;
  virtual std::vector<std::string> get_cache_file_names() const override final;

  virtual std::shared_ptr<void>
  get_context_resource(const std::string &name) const override final;
  virtual std::unique_ptr<PassContextTimer>
  measure(const std::string &label) override final;
  virtual void on_custom_op_create_end() override final;
  // helper class
  struct WithPass {
    WithPass(PassContextImp &context, IPass &pass);
    WithPass(const WithPass &) = delete;
    ~WithPass();
    PassContextImp *_context;
  };
  WithPass with_current_pass(IPass &pass);
  void add_context_resource(const std::string &name,
                            std::shared_ptr<void> resource);
  virtual void save_context_json() const override final;

  virtual void append_compiled_model_compatibility_info(
      const std::string &backend_name,
      const std::string &compatibility_info) override final;
  virtual const std::map<std::string, std::string> &
  get_compiled_model_compatibility_info() const override final;
  virtual void disable_delete_tar_file_in_session_created() override final;
  virtual std::unique_ptr<FileSystem> get_file_system() override final;

private:
  friend int morphizen_ep_set_ep_dynamic_options(
      const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
      const char *const *keys, const char *const *values, size_t kv_len);
  friend bool check_cache_hit(PassContextImp &context);
  std::map<std::string, std::string> ep_dynamic_options;
  mutable std::mutex ep_dynamic_options_lock;
  // for share context, many context may be same. may need to change container
  // to set.
  mutable std::vector<std::shared_ptr<QoSUpdateInterface>> qos_updaters_;
  int created_customop_count = 0;
  std::unique_ptr<TarFile> tar_file_ = nullptr;
  bool delete_tar_file_on_session_created_ = true;
  std::filesystem::path tar_file_file_name_;
  const TargetProto *target_proto_ = nullptr;
  std::map<std::string, std::string> provider_option_origin_ = {};
  std::map<std::string, std::string> compiled_model_compatibility_info_ = {};
  std::map<std::string, std::string> session_configs_ = {};

private:
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
  friend void
  store_cache_directory_from_main_node(PassContextImp &context,
                                       morphizen_cxx::NodeConstRef main_node);
  friend std::shared_ptr<PassContextImp> initialize_context(
      const std::string &model_path, const Graph &onnx_graph,
      const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
      const onnxruntime::ProviderOptions &options,
      const std::map<std::string, std::string> &session_configs,
      std::unique_ptr<LoggerAdapter> logger_adapter);
  friend onnxruntime::Node *
  create_ep_context_node(morphizen::ExecutionProviderConcrete *ep, int index);
  friend std::string
  get_ep_cache_context_nonembed_mode(PassContextImp &context);
  friend std::vector<std::unique_ptr<ExecutionProvider>>
  compile_onnx_model_internal(
      const Graph &onnx_graph,
      const std::vector<morphizen_cxx::NodeConstRef> &ep_context_nodes,
      std::shared_ptr<PassContextImp> context);
  friend void read_cache(std::shared_ptr<PassContextImp> context);
  friend std::vector<std::unique_ptr<ExecutionProvider>>
  restore_execution_providers_from_ep_context_model(
      morphizen_cxx::GraphConstRef /*onnx_graph*/,
      std::shared_ptr<PassContextImp> context,
      std::vector<morphizen_cxx::NodeConstRef> ep_context_nodes);
  friend std::string get_ep_cache_context_embed_mode(PassContextImp &context);
  friend std::unique_ptr<std::istream>
  context_cache_files_to_tar_stream(PassContext &context);
  friend class PassContextConfigTest; // for unit test.
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
};

struct PassContextTimerImp : public PassContextTimer {
  PassContextTimerImp(const std::string &label, PassContextImp &context);
  virtual ~PassContextTimerImp();
  std::string label_;
  PassContextImp &context_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
  MemUsageProto mem_usage_;
};
} // namespace morphizen
