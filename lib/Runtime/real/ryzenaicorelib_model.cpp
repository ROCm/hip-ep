/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ryzenaicorelib_model.cpp - Whole models through RyzenAI CoreLib ----===//
//
// The native side of corelib_model.h: loads ryzenai_corelib.dll at runtime
// (rather than linking it, so the EP still loads on machines without it),
// checks its version against the headers, and creates and runs CoreLib model
// sessions. See docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#include "corelib_model.h"

// The C headers only: <ryzenai/corelib.h> also pulls in the C++20 wrappers.
#include <ryzenai/corelib/base.h>
#include <ryzenai/corelib/model.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

// std::getenv cannot see the process environment from the static-CRT EP DLL.
std::string read_env(const char *name) {
#ifdef _WIN32
  char buf[1024];
  DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf))
    return {};
  return std::string(buf, n);
#else
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string();
#endif
}

#define HIPDNN_EP_CORELIB_FUNCTIONS(X)                                         \
  X(ryzenai_corelib_get_version)                                               \
  X(ryzenai_corelib_cleanup)                                                   \
  X(ryzenai_corelib_npu_usable)                                                \
  X(ryzenai_corelib_get_last_error_message)                                    \
  X(ryzenai_corelib_object_release)                                            \
  X(ryzenai_corelib_create_model_from_path)                                    \
  X(ryzenai_corelib_model_create_session)                                      \
  X(ryzenai_corelib_model_get_metadata)                                        \
  X(ryzenai_corelib_model_run)

struct CorelibApi {
#define HIPDNN_EP_CORELIB_MEMBER(name) decltype(&::name) name = nullptr;
  HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_MEMBER)
#undef HIPDNN_EP_CORELIB_MEMBER
};

void *open_library(const std::string &path) {
#ifdef _WIN32
  // Search the DLL's own directory first so its dependencies beside it load.
  DWORD flags =
      path.find_first_of("/\\") != std::string::npos
          ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
          : 0;
  return reinterpret_cast<void *>(LoadLibraryExA(path.c_str(), nullptr, flags));
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void *find_symbol(void *lib, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

// Process-wide; the library is never unloaded: a CoreLib stream must not
// outlive it.
const CorelibApi *load_corelib() {
  static CorelibApi api;
  static const CorelibApi *loaded = []() -> const CorelibApi * {
#ifdef _WIN32
    const char *kSubdir = "/bin/";
    const char *kName = "ryzenai_corelib.dll";
#else
    const char *kSubdir = "/lib/";
    const char *kName = "libryzenai_corelib.so";
#endif
    std::vector<std::string> candidates;
    std::string root = read_env("RYZENAICORELIB_ROOT");
    if (!root.empty()) {
      candidates.push_back(root + kSubdir + kName);
      candidates.push_back(root + "/install" + kSubdir + kName);
    }
    candidates.push_back(kName);

    void *lib = nullptr;
    std::string loaded_path;
    for (const std::string &c : candidates)
      if ((lib = open_library(c))) {
        loaded_path = c;
        break;
      }
    if (!lib) {
      fprintf(stderr, "[corelib] %s not found (RYZENAICORELIB_ROOT='%s')\n",
              kName, root.c_str());
      return nullptr;
    }

#define HIPDNN_EP_CORELIB_RESOLVE(name)                                        \
  api.name = reinterpret_cast<decltype(api.name)>(find_symbol(lib, #name));    \
  if (!api.name) {                                                             \
    fprintf(stderr, "[corelib] %s has no symbol %s\n", loaded_path.c_str(),    \
            #name);                                                            \
    return nullptr;                                                            \
  }
    // The version comes first: another version's library may lack symbols,
    // and a mismatch names the real problem better than a missing symbol.
    // Below 1.0 any minor release may break the ABI.
    HIPDNN_EP_CORELIB_RESOLVE(ryzenai_corelib_get_version)
    uint32_t major = 0, minor = 0, patch = 0;
    api.ryzenai_corelib_get_version(&major, &minor, &patch);
    if (major != RYZENAI_CORELIB_VERSION_MAJOR ||
        (major == 0 && minor != RYZENAI_CORELIB_VERSION_MINOR)) {
      fprintf(stderr,
              "[corelib] %s is version %u.%u.%u, but hip-ep was built against "
              "%d.%d.%d\n",
              loaded_path.c_str(), major, minor, patch,
              RYZENAI_CORELIB_VERSION_MAJOR, RYZENAI_CORELIB_VERSION_MINOR,
              RYZENAI_CORELIB_VERSION_PATCH);
      return nullptr;
    }
    HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_RESOLVE)
#undef HIPDNN_EP_CORELIB_RESOLVE
    return &api;
  }();
  return loaded;
}

// ryzenai_corelib_cleanup must run before process exit rather than at DLL
// unload under the loader lock, so the last session to go calls it.
std::mutex g_sessions_mu;
int g_live_sessions = 0;

} // namespace

namespace hipdnn_ep::corelib {

class ModelSession {
public:
  ~ModelSession() {
    if (report_)
      fprintf(stderr,
              "[corelib] model report: %s, %llu forwards on the NPU (%llu "
              "prefills, %llu decodes, %llu rows)\n",
              arch_.c_str(), (unsigned long long)(prefills_ + decodes_),
              (unsigned long long)prefills_, (unsigned long long)decodes_,
              (unsigned long long)rows_);
    if (session_)
      api_->ryzenai_corelib_object_release(session_);
    if (model_)
      api_->ryzenai_corelib_object_release(model_);
    std::lock_guard<std::mutex> lock(g_sessions_mu);
    if (--g_live_sessions == 0)
      api_->ryzenai_corelib_cleanup();
  }

  const CorelibApi *api_ = nullptr;
  ryzenai_corelib_model_ptr model_ = nullptr;
  ryzenai_corelib_model_session_ptr session_ = nullptr;
  std::string arch_;
  int64_t vocab_ = 0;
  bool report_ = false;
  uint64_t prefills_ = 0, decodes_ = 0, rows_ = 0;
};

std::shared_ptr<ModelSession> create_model_session(const std::string &arch,
                                                   const std::string &path,
                                                   int64_t max_sequence_length,
                                                   std::string *why) {
  const CorelibApi *api = load_corelib();
  if (!api) {
    *why = "corelib is not loadable (see the [corelib] line above)";
    return nullptr;
  }
  if (!api->ryzenai_corelib_npu_usable()) {
    *why = "no usable AIE4 NPU (corelib's models run on the NPU only on AIE4, "
           "with XRT 2.26.0 or later)";
    return nullptr;
  }
  ryzenai_corelib_model_ptr model = nullptr;
  if (api->ryzenai_corelib_create_model_from_path(arch.c_str(), path.c_str(),
                                                  &model) !=
      ryzenai_corelib_status_success) {
    *why = "corelib's " + arch + " refused " + path + ": " +
           api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  // Counted from here: the destructor releases the model and may clean up.
  {
    std::lock_guard<std::mutex> lock(g_sessions_mu);
    ++g_live_sessions;
  }
  auto s = std::make_shared<ModelSession>();
  s->api_ = api;
  s->model_ = model;
  s->arch_ = arch;
  s->report_ = read_env("RYZENAICORELIB_REPORT") == "1";

  ryzenai_corelib_model_session_options options{};
  options.size = sizeof(options);
  options.max_sequence_length = max_sequence_length;
  // hip-ep owns the GPU: CoreLib's GPU tier would load its own HIP and
  // register its buffers alongside the EP's. The CPU stays, for the token
  // pick and gathers CoreLib runs there.
  options.enabled_backends =
      ryzenai_corelib_backend_npu | ryzenai_corelib_backend_cpu;
  if (api->ryzenai_corelib_model_create_session(
          model, &options, &s->session_) != ryzenai_corelib_status_success) {
    *why = "creating a " + arch +
           " session failed: " + api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  ryzenai_corelib_model_metadata metadata{};
  metadata.size = sizeof(metadata);
  if (api->ryzenai_corelib_model_get_metadata(s->session_, &metadata) !=
          ryzenai_corelib_status_success ||
      metadata.vocab_size <= 0) {
    *why = "reading the " + arch + " model's metadata failed: " +
           api->ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  s->vocab_ = metadata.vocab_size;
  fprintf(stderr,
          "[corelib] model: %s from %s on the NPU (%lld layers, hidden %lld, "
          "vocab %lld, context %lld)\n",
          arch.c_str(), path.c_str(), (long long)metadata.layer_count,
          (long long)metadata.hidden_size, (long long)metadata.vocab_size,
          (long long)metadata.context_length);
  return s;
}

int64_t vocab_size(const ModelSession &session) { return session.vocab_; }

bool forward(ModelSession &s, const int32_t *tokens, size_t rows,
             int64_t position, float *logits, std::string *why) {
  ryzenai_corelib_model_value inputs[2] = {};
  inputs[0].size = sizeof(ryzenai_corelib_model_value);
  inputs[0].name = "input_ids";
  inputs[0].data_type = ryzenai_corelib_data_type_int32;
  inputs[0].rank = 1;
  inputs[0].dims[0] = static_cast<int64_t>(rows);
  inputs[0].data = const_cast<int32_t *>(tokens);
  inputs[1].size = sizeof(ryzenai_corelib_model_value);
  inputs[1].name = "position";
  inputs[1].data_type = ryzenai_corelib_data_type_int64;
  inputs[1].rank = 0;
  inputs[1].data = &position;

  ryzenai_corelib_model_value output{};
  output.size = sizeof(output);
  output.name = "logits";
  output.data_type = ryzenai_corelib_data_type_fp32;
  output.data = logits;
  output.capacity = size_t(s.vocab_) * sizeof(float);

  if (s.api_->ryzenai_corelib_model_run(s.session_, "forward", inputs, 2,
                                        &output, 1,
                                        0) != ryzenai_corelib_status_success) {
    *why = s.api_->ryzenai_corelib_get_last_error_message();
    return false;
  }
  (rows == 1 ? s.decodes_ : s.prefills_)++;
  s.rows_ += rows;
  return true;
}

} // namespace hipdnn_ep::corelib
