/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ryzenaicorelib_model.cpp - Whole models through RyzenAI CoreLib ----===//
//
// The native side of corelib_model.h: loads ryzenai_corelib.dll at runtime
// (rather than linking it, so the EP still loads on machines without it),
// checks its version against the headers, finds the models it ships, and
// creates and runs their sessions. See docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#include "corelib_model.h"

// The C headers only: <ryzenai/corelib.h> also pulls in the C++20 wrappers.
#include <ryzenai/corelib/base.h>
#include <ryzenai/corelib/model.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  X(ryzenai_corelib_has_feature)                                               \
  X(ryzenai_corelib_get_last_error_message)                                    \
  X(ryzenai_corelib_object_release)                                            \
  X(ryzenai_corelib_create_model_from_path)                                    \
  X(ryzenai_corelib_model_create_session)                                      \
  X(ryzenai_corelib_model_get_metadata)                                        \
  X(ryzenai_corelib_model_get_method)                                          \
  X(ryzenai_corelib_model_run)

struct CorelibApi {
#define HIPDNN_EP_CORELIB_MEMBER(name) decltype(&::name) name = nullptr;
  HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_MEMBER)
#undef HIPDNN_EP_CORELIB_MEMBER
  // Where the library was loaded from: its model libraries sit beside it.
  std::filesystem::path dir;
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

// The directory a loaded library came from, however it was found.
std::filesystem::path library_dir(void *lib, void *symbol) {
#ifdef _WIN32
  (void)symbol;
  wchar_t buf[MAX_PATH];
  DWORD n = GetModuleFileNameW(reinterpret_cast<HMODULE>(lib), buf, MAX_PATH);
  if (n == 0 || n >= MAX_PATH)
    return {};
  return std::filesystem::path(buf).parent_path();
#else
  (void)lib;
  Dl_info info{};
  if (!dladdr(symbol, &info) || !info.dli_fname)
    return {};
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

// Below 1.0 any CoreLib release may change the API, so only the minor version
// the EP was built against is taken; from 1.0 the C ABI only grows within a
// major version, so that minor or any later one is.
bool version_compatible(uint32_t major, uint32_t minor) {
  if (major != RYZENAI_CORELIB_VERSION_MAJOR)
    return false;
  return major == 0 ? minor == RYZENAI_CORELIB_VERSION_MINOR
                    : minor >= RYZENAI_CORELIB_VERSION_MINOR;
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
    HIPDNN_EP_CORELIB_RESOLVE(ryzenai_corelib_get_version)
    uint32_t major = 0, minor = 0, patch = 0;
    api.ryzenai_corelib_get_version(&major, &minor, &patch);
    if (!version_compatible(major, minor)) {
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
    api.dir = library_dir(
        lib, reinterpret_cast<void *>(api.ryzenai_corelib_get_version));
    return &api;
  }();
  return loaded;
}

// The models CoreLib ships: each is a library ryzenai_corelib_model_<name>
// beside ryzenai_corelib, which create_model_from_path loads by name.
// RYZENAICORELIB_MODELS (comma-separated) names them instead, for a CoreLib
// whose models are built in.
std::vector<std::string> model_names(const CorelibApi &api) {
  std::vector<std::string> names;
  std::string listed = read_env("RYZENAICORELIB_MODELS");
  if (!listed.empty()) {
    size_t pos = 0;
    while (pos <= listed.size()) {
      size_t end = listed.find(',', pos);
      std::string n = listed.substr(pos, end - pos);
      // cmd keeps a trailing space after `set VAR=value `.
      const size_t first = n.find_first_not_of(" \t");
      n = first == std::string::npos
              ? std::string()
              : n.substr(first, n.find_last_not_of(" \t") - first + 1);
      if (!n.empty())
        names.push_back(n);
      if (end == std::string::npos)
        break;
      pos = end + 1;
    }
    return names;
  }
#ifdef _WIN32
  const std::string prefix = "ryzenai_corelib_model_", suffix = ".dll";
#else
  const std::string prefix = "libryzenai_corelib_model_", suffix = ".so";
#endif
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(api.dir, ec)) {
    std::string f = entry.path().filename().string();
    std::transform(f.begin(), f.end(), f.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (f.size() > prefix.size() + suffix.size() && f.rfind(prefix, 0) == 0 &&
        f.compare(f.size() - suffix.size(), suffix.size(), suffix) == 0)
      names.push_back(
          f.substr(prefix.size(), f.size() - prefix.size() - suffix.size()));
  }
  std::sort(names.begin(), names.end());
  return names;
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
    if (report_ && prefills_ + decodes_ > 0)
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

  // One `forward` input: its name and the types it accepts.
  struct Input {
    std::string name;
    ryzenai_corelib_data_type preferred;
    uint64_t accepted;
  };

  const CorelibApi *api_ = nullptr;
  ryzenai_corelib_model_ptr model_ = nullptr;
  ryzenai_corelib_model_session_ptr session_ = nullptr;
  std::string arch_;
  int64_t vocab_ = 0;
  std::vector<std::string> input_names_; // the graph inputs forward takes
  std::vector<Input> inputs_;            // likewise, with their types
  bool report_ = false;
  uint64_t prefills_ = 0, decodes_ = 0, rows_ = 0;
  // Converted inputs, reused across calls.
  std::vector<std::vector<uint8_t>> staging_;
};

namespace {

// A session of model `arch` from `path` that fits `graph`, or null with
// `*why` set.
std::shared_ptr<ModelSession>
try_model(const CorelibApi &api, const std::string &arch,
          const std::string &path, int64_t max_sequence_length,
          const GraphSignature &graph, std::string *why) {
  ryzenai_corelib_model_ptr model = nullptr;
  if (api.ryzenai_corelib_create_model_from_path(arch.c_str(), path.c_str(),
                                                 &model) !=
      ryzenai_corelib_status_success) {
    *why = api.ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(g_sessions_mu);
    ++g_live_sessions;
  }
  auto s = std::make_shared<ModelSession>();
  s->api_ = &api;
  s->model_ = model;
  s->arch_ = arch;

  ryzenai_corelib_model_session_options options{};
  options.size = sizeof(options);
  options.max_sequence_length = max_sequence_length;
  // hip-ep owns the GPU: CoreLib's GPU tier would load its own HIP and
  // register its buffers alongside the EP's. The CPU stays, for the token
  // pick and gathers CoreLib runs there.
  options.enabled_backends =
      ryzenai_corelib_backend_npu | ryzenai_corelib_backend_cpu;
  ryzenai_corelib_status status =
      api.ryzenai_corelib_model_create_session(model, &options, &s->session_);
  // onnxruntime-genai configs often ask for the base model's whole context
  // while a model holds less from this file; a session at the model's own
  // capacity still serves every generation that fits it, and one that does
  // not fails on the forward past it, saying so.
  std::string capped;
  if (status != ryzenai_corelib_status_success && max_sequence_length > 0) {
    capped = api.ryzenai_corelib_get_last_error_message();
    options.max_sequence_length = 0;
    status =
        api.ryzenai_corelib_model_create_session(model, &options, &s->session_);
  }
  if (status != ryzenai_corelib_status_success) {
    *why = std::string("creating a session failed: ") +
           api.ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  ryzenai_corelib_model_metadata metadata{};
  metadata.size = sizeof(metadata);
  if (api.ryzenai_corelib_model_get_metadata(s->session_, &metadata) !=
          ryzenai_corelib_status_success ||
      metadata.vocab_size <= 0) {
    *why = "it has no text model";
    return nullptr;
  }
  // A model that takes the file need not be the one it was exported for
  // (CoreLib's test model reads any file); the logits' width tells.
  if (graph.vocab_size > 0 && metadata.vocab_size != graph.vocab_size) {
    *why = "its vocabulary is " + std::to_string(metadata.vocab_size) +
           ", the graph's logits " + std::to_string(graph.vocab_size);
    return nullptr;
  }
  s->vocab_ = metadata.vocab_size;

  // Method 0 is forward. Its inputs come from the graph by name, but for the
  // position, which the op derives from attention_mask.
  ryzenai_corelib_model_method_info method{};
  if (api.ryzenai_corelib_model_get_method(s->session_, 0, &method) !=
      ryzenai_corelib_status_success) {
    *why = std::string("reading its forward failed: ") +
           api.ryzenai_corelib_get_last_error_message();
    return nullptr;
  }
  for (size_t i = 0; i < method.input_count; ++i) {
    const ryzenai_corelib_model_value_info &in = method.inputs[i];
    const std::string name = in.name;
    if (name == "position")
      continue;
    const bool in_graph = std::find(graph.inputs.begin(), graph.inputs.end(),
                                    name) != graph.inputs.end();
    if (in_graph) {
      s->input_names_.push_back(name);
      s->inputs_.push_back({name, in.data_type, in.accepted_types});
    } else if (!(in.flags & RYZENAI_CORELIB_MODEL_VALUE_OPTIONAL)) {
      *why = "its forward needs " + name + ", which the graph lacks";
      return nullptr;
    }
  }
  if (s->inputs_.empty()) {
    *why = "its forward takes none of the graph's inputs";
    return nullptr;
  }
  s->staging_.resize(s->inputs_.size());
  s->report_ = read_env("RYZENAICORELIB_REPORT") == "1";
  std::string taken;
  for (const std::string &n : s->input_names_)
    taken += (taken.empty() ? "" : ", ") + n;
  fprintf(stderr,
          "[corelib] model: %s from %s on the NPU (%lld layers, hidden %lld, "
          "vocab %lld, context %lld; forward takes %s)\n",
          arch.c_str(), path.c_str(), (long long)metadata.layer_count,
          (long long)metadata.hidden_size, (long long)metadata.vocab_size,
          (long long)metadata.context_length, taken.c_str());
  if (!capped.empty())
    fprintf(stderr,
            "[corelib] model: holding the model's own context, not the %lld "
            "positions genai_config.json's max_length asks for (%s)\n",
            (long long)max_sequence_length, capped.c_str());
  return s;
}

ryzenai_corelib_data_type corelib_type(ElementType t) {
  switch (t) {
  case ElementType::Int32:
    return ryzenai_corelib_data_type_int32;
  case ElementType::Int64:
    return ryzenai_corelib_data_type_int64;
  case ElementType::Float16:
    return ryzenai_corelib_data_type_fp16;
  default:
    return ryzenai_corelib_data_type_fp32;
  }
}

float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Converts `n` elements of `in` to `to` in `out`: integers between int64 and
// int32, floats from fp16 to fp32. False for any other pair.
bool convert(const HostTensor &in, size_t n, ryzenai_corelib_data_type to,
             std::vector<uint8_t> &out) {
  if (in.type == ElementType::Int64 && to == ryzenai_corelib_data_type_int32) {
    out.resize(n * sizeof(int32_t));
    auto *dst = reinterpret_cast<int32_t *>(out.data());
    for (size_t i = 0; i < n; ++i)
      dst[i] = int32_t(static_cast<const int64_t *>(in.data)[i]);
    return true;
  }
  if (in.type == ElementType::Int32 && to == ryzenai_corelib_data_type_int64) {
    out.resize(n * sizeof(int64_t));
    auto *dst = reinterpret_cast<int64_t *>(out.data());
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<const int32_t *>(in.data)[i];
    return true;
  }
  if (in.type == ElementType::Float16 && to == ryzenai_corelib_data_type_fp32) {
    out.resize(n * sizeof(float));
    auto *dst = reinterpret_cast<float *>(out.data());
    for (size_t i = 0; i < n; ++i)
      dst[i] = half_to_float(static_cast<const uint16_t *>(in.data)[i]);
    return true;
  }
  return false;
}

} // namespace

std::shared_ptr<ModelSession> create_model_session(const std::string &arch,
                                                   const std::string &path,
                                                   int64_t max_sequence_length,
                                                   const GraphSignature &graph,
                                                   std::string *why) {
  const CorelibApi *api = load_corelib();
  if (!api) {
    *why = "corelib is not loadable (see the [corelib] line above)";
    return nullptr;
  }
  if (!api->ryzenai_corelib_has_feature(ryzenai_corelib_feature_npu_aiem)) {
    *why = "no NPU corelib can use here (ryzenai_corelib_feature_npu_aiem is "
           "not available)";
    return nullptr;
  }
  // Discovery leans on the vocabulary check to turn away a model that takes
  // any file, so it needs the graph's.
  if (arch.empty() && graph.vocab_size <= 0) {
    *why = "the graph's logits have no static width to match a model by";
    return nullptr;
  }
  std::vector<std::string> names =
      arch.empty() ? model_names(*api) : std::vector<std::string>{arch};
  // The model the export was made for goes first: a model offered a file it
  // was not exported for should refuse it, but need not do so cheaply (or at
  // all safely).
  if (arch.empty() && !graph.model_type.empty()) {
    std::string hint = graph.model_type;
    std::transform(hint.begin(), hint.end(), hint.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    auto it = std::find(names.begin(), names.end(), hint);
    if (it != names.end())
      std::rotate(names.begin(), it, it + 1);
  }
  if (names.empty()) {
    *why = "corelib ships no model libraries beside it (" + api->dir.string() +
           "; RYZENAICORELIB_MODELS names built-in ones)";
    return nullptr;
  }
  std::string refusals;
  for (const std::string &name : names) {
    std::string reason;
    if (auto s =
            try_model(*api, name, path, max_sequence_length, graph, &reason))
      return s;
    // One line for every model: a model listing all its problems with the
    // file would otherwise fill the screen.
    constexpr size_t kMaxReason = 200;
    if (reason.size() > kMaxReason)
      reason = reason.substr(0, kMaxReason) + "...";
    refusals += (refusals.empty() ? "" : "; ") + name + ": " + reason;
  }
  *why = "no corelib model takes " + path + " (" + refusals + ")";
  return nullptr;
}

const std::string &architecture(const ModelSession &session) {
  return session.arch_;
}

int64_t vocab_size(const ModelSession &session) { return session.vocab_; }

const std::vector<std::string> &forward_inputs(const ModelSession &session) {
  return session.input_names_;
}

bool forward(ModelSession &s, const std::vector<HostTensor> &inputs,
             int64_t position, float *logits, std::string *why) {
  std::vector<ryzenai_corelib_model_value> values(inputs.size() + 1);
  int64_t rows = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const HostTensor &in = inputs[i];
    auto it = std::find_if(s.inputs_.begin(), s.inputs_.end(),
                           [&](const auto &x) { return x.name == in.name; });
    if (it == s.inputs_.end() ||
        in.dims.size() > RYZENAI_CORELIB_MODEL_MAX_DIMS) {
      *why = "forward does not take " + in.name;
      return false;
    }
    size_t n = 1;
    for (int64_t d : in.dims)
      n *= size_t(d);
    ryzenai_corelib_model_value &v = values[i];
    v.size = sizeof(v);
    v.name = in.name.c_str();
    v.rank = uint32_t(in.dims.size());
    std::copy(in.dims.begin(), in.dims.end(), v.dims);
    const ryzenai_corelib_data_type own = corelib_type(in.type);
    if (it->accepted & RYZENAI_CORELIB_DATA_TYPE_BIT(own)) {
      v.data_type = own;
      v.data = const_cast<void *>(in.data);
    } else {
      std::vector<uint8_t> &buf = s.staging_[size_t(it - s.inputs_.begin())];
      if (!convert(in, n, it->preferred, buf)) {
        *why = in.name + " has a type forward does not accept";
        return false;
      }
      v.data_type = it->preferred;
      v.data = buf.data();
    }
    if (rows == 0 && !in.dims.empty())
      rows = in.dims[0];
  }
  ryzenai_corelib_model_value &pos = values.back();
  pos.size = sizeof(pos);
  pos.name = "position";
  pos.data_type = ryzenai_corelib_data_type_int64;
  pos.rank = 0;
  pos.data = &position;

  ryzenai_corelib_model_value output{};
  output.size = sizeof(output);
  output.name = "logits";
  output.data_type = ryzenai_corelib_data_type_fp32;
  output.data = logits;
  output.capacity = size_t(s.vocab_) * sizeof(float);

  if (s.api_->ryzenai_corelib_model_run(s.session_, "forward", values.data(),
                                        values.size(), &output, 1,
                                        0) != ryzenai_corelib_status_success) {
    *why = s.api_->ryzenai_corelib_get_last_error_message();
    return false;
  }
  (rows == 1 ? s.decodes_ : s.prefills_)++;
  s.rows_ += uint64_t(rows);
  return true;
}

} // namespace hipdnn_ep::corelib
