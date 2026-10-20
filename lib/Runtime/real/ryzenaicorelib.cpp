/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ryzenaicorelib.cpp - RyzenAI CoreLib (NPU) wrappers ----------------===//
//
// Native implementation of the wrap_corelib_* entry points declared in
// hipdnn_ep_runtime.h. See docs/design/ryzenai-corelib-backend.md.
//
// corelib is loaded at runtime rather than linked, so an EP built against its
// header still loads on a machine without ryzenai_corelib.dll; every call then
// returns HIPDNN_EP_CORELIB_NOT_HANDLED and the generated code runs the GPU
// wrapper instead.
//
// corelib exposes no device pointer, so every call stages through host memory:
// GPU -> host -> corelib tensor -> NPU -> corelib tensor -> host -> GPU.
//
//===----------------------------------------------------------------------===//

#include "../hipdnn_ep_errors.h"
#include "../hipdnn_ep_runtime.h"
#include "../runtime_state_internal.h"

#include <ryzenai/corelib.h>
#include <ryzenai/corelib/testing.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
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

// Default PDI pair: the one corelib's Qwen2 model is built for
// (models/Qwen2.cpp). Each corelib model ships kernels for its own pair, and a
// stream on the wrong pair silently sends shapes to the host, so other models
// set RYZENAICORELIB_PDI.
constexpr int kDefaultPrefillPdi = 0;
constexpr int kDefaultTokenPdi = 16;

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

//===----------------------------------------------------------------------===//
// Dynamically loaded corelib C API
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_CORELIB_FUNCTIONS(X)                                         \
  X(ryzenai_corelib_get_version)                                               \
  X(ryzenai_corelib_cleanup)                                                   \
  X(ryzenai_corelib_get_hardware_context)                                      \
  X(ryzenai_corelib_get_last_error_message)                                    \
  X(ryzenai_corelib_object_release)                                            \
  X(ryzenai_corelib_create_stream)                                             \
  X(ryzenai_corelib_stream_synchronize)                                        \
  X(ryzenai_corelib_stream_get_kernels_root)                                   \
  X(ryzenai_corelib_stream_enable_kernel_bundle)                               \
  X(ryzenai_corelib_create_device_tensor)                                      \
  X(ryzenai_corelib_tensor_write)                                              \
  X(ryzenai_corelib_tensor_read)                                               \
  X(ryzenai_corelib_matmul_bf16_weights_create)                                \
  X(ryzenai_corelib_matmul_bf16_enum_kernels)                                  \
  X(ryzenai_corelib_matmul_bf16)                                               \
  X(ryzenai_corelib_rmsnorm_bf16_weights_create)                               \
  X(ryzenai_corelib_rmsnorm_bf16_enum_kernels)                                 \
  X(ryzenai_corelib_rmsnorm_bf16)                                              \
  X(ryzenai_corelib_add_rmsnorm_bf16)                                          \
  X(ryzenai_corelib_testing_stream_counters_get)

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

// Process-wide; the library is never unloaded (see release_session).
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
    for (const std::string &c : candidates)
      if ((lib = open_library(c)))
        break;
    if (!lib) {
      fprintf(stderr,
              "[corelib] %s not found (RYZENAICORELIB_ROOT='%s'); "
              "corelib ops run on the GPU\n",
              kName, root.c_str());
      return nullptr;
    }

#define HIPDNN_EP_CORELIB_RESOLVE(name)                                        \
  api.name = reinterpret_cast<decltype(api.name)>(find_symbol(lib, #name));    \
  if (!api.name) {                                                             \
    fprintf(stderr,                                                            \
            "[corelib] missing symbol %s; corelib ops run on the GPU\n",       \
            #name);                                                            \
    return nullptr;                                                            \
  }
    HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_RESOLVE)
#undef HIPDNN_EP_CORELIB_RESOLVE

    // Below 1.0 any minor release may break the ABI.
    uint32_t major = 0, minor = 0, patch = 0;
    api.ryzenai_corelib_get_version(&major, &minor, &patch);
    if (major != RYZENAI_CORELIB_VERSION_MAJOR ||
        (major == 0 && minor != RYZENAI_CORELIB_VERSION_MINOR)) {
      fprintf(stderr,
              "[corelib] runtime version %u.%u.%u does not match headers "
              "%d.%d.%d; corelib ops run on the GPU\n",
              major, minor, patch, RYZENAI_CORELIB_VERSION_MAJOR,
              RYZENAI_CORELIB_VERSION_MINOR, RYZENAI_CORELIB_VERSION_PATCH);
      return nullptr;
    }
    return &api;
  }();
  return loaded;
}

// Parses RYZENAICORELIB_PDI ("<prefill>,<token>"); false when malformed.
bool parse_pdi_pair(const std::string &value, int *prefill, int *token) {
  int p = 0, t = 0;
  char trailing = 0;
  if (std::sscanf(value.c_str(), " %d , %d %c", &p, &t, &trailing) != 2 ||
      p < 0 || t < 0)
    return false;
  *prefill = p;
  *token = t;
  return true;
}

// ryzenai_corelib_cleanup must run before process exit rather than at DLL
// unload under the loader lock, so the last session to go calls it.
std::mutex g_sessions_mu;
int g_live_sessions = 0;

//===----------------------------------------------------------------------===//
// Element conversion (host side, for constant weights only)
//===----------------------------------------------------------------------===//

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

uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF)
    return uint16_t(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 0x1F)
    return uint16_t(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10)
      return uint16_t(sign);
    mant |= 0x800000;
    uint32_t shift = uint32_t(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1)))
      ++half;
    return uint16_t(sign | half);
  }
  uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1)))
    ++half;
  return uint16_t(half);
}

uint16_t float_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  if ((x & 0x7FFFFFFF) > 0x7F800000)
    return uint16_t((x >> 16) | 0x40);
  return uint16_t((x + 0x7FFF + ((x >> 16) & 1)) >> 16);
}

float to_float(int64_t data_type, const void *base, size_t i) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return static_cast<const float *>(base)[i];
  case HIPDNN_EP_DATATYPE_HALF:
    return half_to_float(static_cast<const uint16_t *>(base)[i]);
  default: { // HIPDNN_EP_DATATYPE_BFLOAT16
    uint32_t bits = uint32_t(static_cast<const uint16_t *>(base)[i]) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
  }
}

bool activation_type(int64_t data_type, ryzenai_corelib_data_type *out,
                     size_t *elem_size) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    *out = ryzenai_corelib_data_type_fp32;
    *elem_size = 4;
    return true;
  case HIPDNN_EP_DATATYPE_HALF:
    *out = ryzenai_corelib_data_type_fp16;
    *elem_size = 2;
    return true;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    *out = ryzenai_corelib_data_type_bf16;
    *elem_size = 2;
    return true;
  default:
    return false;
  }
}

ryzenai_corelib_weights_source
make_source(ryzenai_corelib_weights_role role, const void *data, uint64_t size,
            int32_t data_type, std::initializer_list<int64_t> dims) {
  ryzenai_corelib_weights_source s{};
  s.data = data;
  s.role = role;
  s.size = size;
  s.data_type = data_type;
  s.rank = static_cast<uint32_t>(dims.size());
  size_t i = 0;
  for (int64_t d : dims)
    s.dims[i++] = d;
  s.memory = ryzenai_corelib_weights_memory_copy;
  return s;
}

//===----------------------------------------------------------------------===//
// Per-session context
//===----------------------------------------------------------------------===//

// One packed op instance, keyed by its constant weight pointer (stable for the
// session), op kind and epsilon. `rows_on_cpu` records
// padded row counts corelib ran on the host, which are routed to the GPU from
// then on.
struct OpEntry {
  ryzenai_corelib_weights_ptr weights = nullptr;
  bool unsupported = false;
  std::set<int64_t> shipped_rows;
  std::set<int64_t> rows_on_cpu;
  // Padded rows -> BF16 device tensors, in argument order of the op.
  std::map<int64_t, std::vector<ryzenai_corelib_tensor_ptr>> tensors;
};

struct CorelibContext {
  const CorelibApi *api = nullptr;
  ryzenai_corelib_stream_ptr stream = nullptr;
  bool usable = false;
  bool strict = false;
  std::map<std::tuple<const void *, int, float>, OpEntry> ops;
  std::vector<uint8_t> staging_in, staging_aux, staging_out, staging_out2;
};

void destroy_context(void *p) {
  auto *ctx = static_cast<CorelibContext *>(p);
  if (ctx->usable) {
    const CorelibApi &api = *ctx->api;
    api.ryzenai_corelib_stream_synchronize(ctx->stream);
    for (auto &[key, entry] : ctx->ops) {
      for (auto &[rows, ts] : entry.tensors)
        for (ryzenai_corelib_tensor_ptr t : ts)
          api.ryzenai_corelib_object_release(t);
      if (entry.weights)
        api.ryzenai_corelib_object_release(entry.weights);
    }
    api.ryzenai_corelib_object_release(ctx->stream);
    std::lock_guard<std::mutex> lock(g_sessions_mu);
    if (--g_live_sessions == 0)
      api.ryzenai_corelib_cleanup();
  }
  delete ctx;
}

CorelibContext *get_context(RuntimeState *state) {
  if (state->corelib_ctx)
    return static_cast<CorelibContext *>(state->corelib_ctx);

  auto *ctx = new CorelibContext();
  state->corelib_ctx = ctx;
  state->corelib_ctx_destroy = destroy_context;
  ctx->strict = read_env("HIPDNN_EP_STRICT") == "1";

  ctx->api = load_corelib();
  if (!ctx->api)
    return ctx;
  const CorelibApi &api = *ctx->api;
  if (!api.ryzenai_corelib_get_hardware_context()) {
    fprintf(stderr, "[corelib] no NPU; corelib ops run on the GPU\n");
    return ctx;
  }
  int prefill_pdi = kDefaultPrefillPdi;
  int token_pdi = kDefaultTokenPdi;
  const std::string pdi = read_env("RYZENAICORELIB_PDI");
  if (!pdi.empty() && !parse_pdi_pair(pdi, &prefill_pdi, &token_pdi)) {
    fprintf(stderr,
            "[corelib] RYZENAICORELIB_PDI='%s' is not '<prefill>,<token>'; "
            "corelib ops run on the GPU\n",
            pdi.c_str());
    return ctx;
  }
  if (api.ryzenai_corelib_create_stream(prefill_pdi, token_pdi, &ctx->stream) !=
      ryzenai_corelib_status_success) {
    fprintf(stderr, "[corelib] create_stream(%d, %d) failed: %s\n", prefill_pdi,
            token_pdi, api.ryzenai_corelib_get_last_error_message());
    return ctx;
  }
  auto drop_stream = [&] {
    api.ryzenai_corelib_object_release(ctx->stream);
    ctx->stream = nullptr;
  };

  // A model's kernel bundle is linked into its corelib model library and
  // works with or without a kernel store.
  const std::string bundle = read_env("RYZENAICORELIB_KERNEL_BUNDLE");
  if (!bundle.empty() &&
      api.ryzenai_corelib_stream_enable_kernel_bundle(
          ctx->stream, bundle.c_str()) != ryzenai_corelib_status_success) {
    fprintf(stderr,
            "[corelib] enabling kernel bundle '%s' failed: %s; corelib ops "
            "run on the GPU\n",
            bundle.c_str(), api.ryzenai_corelib_get_last_error_message());
    drop_stream();
    return ctx;
  }

  // With neither a bundle nor a kernel store every op would run on corelib's
  // host fallback, which is slower than the GPU path.
  const char *root = api.ryzenai_corelib_stream_get_kernels_root(ctx->stream);
  if (bundle.empty() && (!root || !*root)) {
    fprintf(stderr, "[corelib] no kernels (set RYZENAI_CORELIB_KERNELS_ROOT or "
                    "RYZENAICORELIB_KERNEL_BUNDLE); corelib ops run on the "
                    "GPU\n");
    drop_stream();
    return ctx;
  }
  fprintf(stderr,
          "[corelib] NPU stream: PDI (%d, %d), kernel bundle '%s', kernels "
          "root '%s'\n",
          prefill_pdi, token_pdi, bundle.c_str(), root ? root : "");
  ctx->usable = true;
  std::lock_guard<std::mutex> lock(g_sessions_mu);
  ++g_live_sessions;
  return ctx;
}

// Smallest shipped row count that holds `rows`, or 0 when none does.
int64_t pick_rows(const OpEntry &entry, int64_t rows) {
  auto it = entry.shipped_rows.lower_bound(rows);
  if (it == entry.shipped_rows.end() || entry.rows_on_cpu.count(*it))
    return 0;
  return *it;
}

bool d2h(RuntimeState *state, void *dst, const void *src, size_t bytes) {
  hipStream_t stream = state->stream;
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream) ==
             hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess;
}

bool h2d(RuntimeState *state, void *dst, const void *src, size_t bytes) {
  hipStream_t stream = state->stream;
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream) ==
             hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess;
}

uint64_t cpu_fallbacks(const CorelibContext &ctx) {
  ryzenai_corelib_testing_stream_counters c{};
  ctx.api->ryzenai_corelib_testing_stream_counters_get(ctx.stream, &c);
  return c.cpu_fallbacks;
}

// corelib ran the op on the host: the result is correct but slow. Under
// HIPDNN_EP_STRICT=1 that is an error surfaced through the device error flag
// (read by inference_compute); otherwise this shape moves to the GPU.
int note_cpu_fallback(RuntimeState *state, CorelibContext &ctx, OpEntry &entry,
                      const char *op, int64_t rows) {
  entry.rows_on_cpu.insert(rows);
  if (!ctx.strict) {
    fprintf(stderr,
            "[corelib] %s ran on the CPU at %lld rows; using the GPU for that "
            "shape from now on\n",
            op, (long long)rows);
    return 0;
  }
  fprintf(stderr,
          "[corelib] %s ran on the CPU at %lld rows (HIPDNN_EP_STRICT)\n", op,
          (long long)rows);
  int code = HIPDNN_EP_ERR_CORELIB_CPU_FALLBACK;
  if (state->device_error_flag)
    h2d(state, state->device_error_flag, &code, sizeof(code));
  return code;
}

// Creates (once) the BF16 device tensors for `rows` padded rows.
const std::vector<ryzenai_corelib_tensor_ptr> *
tensors_for(CorelibContext &ctx, OpEntry &entry, int64_t rows,
            std::initializer_list<int64_t> widths) {
  auto it = entry.tensors.find(rows);
  if (it != entry.tensors.end())
    return &it->second;
  std::vector<ryzenai_corelib_tensor_ptr> ts;
  for (int64_t w : widths) {
    int64_t shape[2] = {rows, w};
    ryzenai_corelib_tensor_ptr t = nullptr;
    if (ctx.api->ryzenai_corelib_create_device_tensor(
            ryzenai_corelib_data_type_bf16, shape, 2, &t) !=
        ryzenai_corelib_status_success) {
      fprintf(stderr, "[corelib] create_device_tensor failed: %s\n",
              ctx.api->ryzenai_corelib_get_last_error_message());
      for (ryzenai_corelib_tensor_ptr p : ts)
        ctx.api->ryzenai_corelib_object_release(p);
      return nullptr;
    }
    ts.push_back(t);
  }
  return &entry.tensors.emplace(rows, std::move(ts)).first->second;
}

bool ok(const CorelibContext &ctx, ryzenai_corelib_status st,
        const char *what) {
  if (st == ryzenai_corelib_status_success)
    return true;
  fprintf(stderr, "[corelib] %s failed: %s\n", what,
          ctx.api->ryzenai_corelib_get_last_error_message());
  return false;
}

//===----------------------------------------------------------------------===//
// MatMulNBits
//===----------------------------------------------------------------------===//

struct MatMulKernelQuery {
  int64_t k, n, block_size;
  bool has_bias;
  std::set<int64_t> *rows;
};

bool collect_matmul_rows(void *p, int /*pdi*/, int64_t m, int64_t k, int64_t n,
                         int64_t group_size, bool has_bias, int64_t /*heads*/,
                         int64_t /*max_seq*/) {
  auto *q = static_cast<MatMulKernelQuery *>(p);
  // corelib derives the kernel group from the model's: divisible by 64 takes
  // g64, otherwise g32. Accept either spelling of the group.
  int64_t kernel_group = q->block_size % 64 == 0 ? 64 : 32;
  if (k == q->k && n == q->n && has_bias == q->has_bias &&
      (group_size == q->block_size || group_size == kernel_group))
    q->rows->insert(m);
  return true;
}

bool pack_matmul(RuntimeState *state, CorelibContext &ctx, OpEntry &entry,
                 const void *B, const void *scales, const void *zero_points,
                 const void *bias, int64_t N, int64_t K, int64_t block_size,
                 int64_t scale_elem_size, int64_t data_type) {
  const int64_t groups = K / block_size;
  const size_t qweight_bytes = size_t(N) * size_t(K / 2);
  const size_t zp_bytes = size_t(N) * size_t((groups + 1) / 2);
  size_t elem = 0;
  ryzenai_corelib_data_type unused;
  activation_type(data_type, &unused, &elem);

  std::vector<uint8_t> qweight(qweight_bytes);
  std::vector<uint8_t> scales_raw(size_t(N) * size_t(groups) *
                                  size_t(scale_elem_size));
  std::vector<uint8_t> zp;
  std::vector<uint8_t> bias_raw;
  if (!d2h(state, qweight.data(), B, qweight.size()) ||
      !d2h(state, scales_raw.data(), scales, scales_raw.size()))
    return false;
  if (zero_points) {
    zp.resize(zp_bytes);
    if (!d2h(state, zp.data(), zero_points, zp.size()))
      return false;
  }
  if (bias) {
    bias_raw.resize(size_t(N) * elem);
    if (!d2h(state, bias_raw.data(), bias, bias_raw.size()))
      return false;
  }

  // corelib takes FP16 scales and an FP32 bias whatever the activation type.
  // Scales are FP16 (2 bytes) or FP32 (4 bytes) independently of A.
  std::vector<uint16_t> scales_fp16(size_t(N) * size_t(groups));
  for (size_t i = 0; i < scales_fp16.size(); ++i)
    scales_fp16[i] =
        scale_elem_size == 2
            ? reinterpret_cast<const uint16_t *>(scales_raw.data())[i]
            : float_to_half(
                  to_float(HIPDNN_EP_DATATYPE_FLOAT, scales_raw.data(), i));
  std::vector<float> bias_fp32(bias ? size_t(N) : 0);
  for (size_t i = 0; i < bias_fp32.size(); ++i)
    bias_fp32[i] = to_float(data_type, bias_raw.data(), i);

  std::vector<ryzenai_corelib_weights_source> sources = {
      make_source(ryzenai_corelib_weights_role_qweight, qweight.data(),
                  qweight.size(), ryzenai_corelib_weights_data_type_onnx_uint8,
                  {N, K / 2}),
      make_source(ryzenai_corelib_weights_role_scales, scales_fp16.data(),
                  scales_fp16.size() * sizeof(uint16_t),
                  ryzenai_corelib_weights_data_type_onnx_float16, {N, groups}),
  };
  if (zero_points)
    sources.push_back(make_source(
        ryzenai_corelib_weights_role_qzeros, zp.data(), zp.size(),
        ryzenai_corelib_weights_data_type_onnx_uint8, {N, (groups + 1) / 2}));
  if (bias)
    sources.push_back(
        make_source(ryzenai_corelib_weights_role_bias, bias_fp32.data(),
                    bias_fp32.size() * sizeof(float),
                    ryzenai_corelib_weights_data_type_onnx_float, {N}));

  ryzenai_corelib_matmul_bf16_weights_desc desc{};
  desc.k = K;
  desc.n = N;
  desc.group_size = static_cast<uint32_t>(block_size);
  desc.has_bias = bias != nullptr;
  desc.format = ryzenai_corelib_matmul_bf16_weights_format_mladf;
  if (!ok(ctx,
          ctx.api->ryzenai_corelib_matmul_bf16_weights_create(
              &desc, sources.data(), sources.size(), nullptr, &entry.weights),
          "matmul_bf16_weights_create"))
    return false;

  MatMulKernelQuery query{K, N, block_size, bias != nullptr,
                          &entry.shipped_rows};
  ctx.api->ryzenai_corelib_matmul_bf16_enum_kernels(
      ctx.stream, collect_matmul_rows, &query);
  if (entry.shipped_rows.empty()) {
    fprintf(stderr,
            "[corelib] no matmul kernel for K=%lld N=%lld group=%lld bias=%d; "
            "using the GPU\n",
            (long long)K, (long long)N, (long long)block_size, bias != nullptr);
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// RmsNorm / SkipRmsNorm (shared weights: epsilon + BF16 scale)
//===----------------------------------------------------------------------===//

bool collect_norm_rows(void *p, int /*pdi*/, int64_t m, int64_t k) {
  auto *q = static_cast<std::pair<int64_t, std::set<int64_t> *> *>(p);
  if (k == q->first)
    q->second->insert(m);
  return true;
}

bool pack_norm(RuntimeState *state, CorelibContext &ctx, OpEntry &entry,
               const void *scale, int64_t k, float epsilon, int64_t data_type) {
  size_t elem = 0;
  ryzenai_corelib_data_type unused;
  activation_type(data_type, &unused, &elem);
  std::vector<uint8_t> raw(size_t(k) * elem);
  if (!d2h(state, raw.data(), scale, raw.size()))
    return false;
  std::vector<uint16_t> scale_bf16(static_cast<size_t>(k));
  for (size_t i = 0; i < scale_bf16.size(); ++i)
    scale_bf16[i] = data_type == HIPDNN_EP_DATATYPE_BFLOAT16
                        ? reinterpret_cast<const uint16_t *>(raw.data())[i]
                        : float_to_bf16(to_float(data_type, raw.data(), i));

  ryzenai_corelib_weights_source source =
      make_source(ryzenai_corelib_weights_role_scale, scale_bf16.data(),
                  scale_bf16.size() * sizeof(uint16_t),
                  ryzenai_corelib_weights_data_type_onnx_bfloat16, {k});
  ryzenai_corelib_rmsnorm_bf16_weights_desc desc{};
  desc.k = k;
  desc.epsilon = epsilon;
  if (!ok(ctx,
          ctx.api->ryzenai_corelib_rmsnorm_bf16_weights_create(
              &desc, &source, 1, nullptr, &entry.weights),
          "rmsnorm_bf16_weights_create"))
    return false;

  // add_rmsnorm has no enumeration of its own; its grid is narrower, which
  // the CPU-fallback check catches per shape.
  std::pair<int64_t, std::set<int64_t> *> query{k, &entry.shipped_rows};
  ctx.api->ryzenai_corelib_rmsnorm_bf16_enum_kernels(ctx.stream,
                                                     collect_norm_rows, &query);
  if (entry.shipped_rows.empty()) {
    fprintf(stderr, "[corelib] no rmsnorm kernel for k=%lld; using the GPU\n",
            (long long)k);
    return false;
  }
  return true;
}

enum OpKind { kMatMulNBits, kRmsNorm, kSkipRmsNorm };

// Returns the entry for these weights, packing it on first use; null when the
// op cannot run on corelib.
template <class PackFn>
OpEntry *entry_for(CorelibContext &ctx, const void *weights, OpKind kind,
                   float epsilon, PackFn pack) {
  auto [it, inserted] =
      ctx.ops.try_emplace(std::make_tuple(weights, int(kind), epsilon));
  OpEntry &entry = it->second;
  if (inserted && !pack(entry))
    entry.unsupported = true;
  return entry.unsupported ? nullptr : &entry;
}

} // namespace

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

extern "C" {

int wrap_corelib_matmul_nbits(RuntimeState *state, int /*op_state_slot*/,
                              const void *A, const void *B, const void *scales,
                              const void *zero_points, const void *g_idx,
                              const void *bias, void *output, int64_t M,
                              int64_t N, int64_t K, int64_t batch_count,
                              int64_t bits, int64_t block_size,
                              int64_t /*elem_size*/, int64_t zp_elem_size,
                              int64_t scale_elem_size, int64_t data_type) {
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  if (!state || bits != 4 || g_idx || block_size % 32 != 0 ||
      K % block_size != 0 || (zero_points && zp_elem_size != 1) ||
      (scale_elem_size != 2 && scale_elem_size != 4) ||
      !activation_type(data_type, &act_type, &elem))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  OpEntry *entry = entry_for(*ctx, B, kMatMulNBits, 0.0f, [&](OpEntry &e) {
    return pack_matmul(state, *ctx, e, B, scales, zero_points, bias, N, K,
                       block_size, scale_elem_size, data_type);
  });
  if (!entry)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  const int64_t rows = M * batch_count;
  const int64_t padded = pick_rows(*entry, rows);
  if (padded == 0)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  const auto *ts = tensors_for(*ctx, *entry, padded, {K, N});
  if (!ts)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  const CorelibApi &api = *ctx->api;

  ctx->staging_in.resize(size_t(rows * K) * elem);
  ctx->staging_out.resize(size_t(rows * N) * elem);
  if (!d2h(state, ctx->staging_in.data(), A, ctx->staging_in.size()))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  const uint64_t before = cpu_fallbacks(*ctx);
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write(
              (*ts)[0], act_type, ctx->staging_in.data(), size_t(rows * K), 0),
          "tensor_write") ||
      !ok(*ctx,
          api.ryzenai_corelib_matmul_bf16(ctx->stream, (*ts)[0], entry->weights,
                                          (*ts)[1]),
          "matmul_bf16") ||
      !ok(*ctx, api.ryzenai_corelib_stream_synchronize(ctx->stream),
          "stream_synchronize") ||
      !ok(*ctx,
          api.ryzenai_corelib_tensor_read(
              (*ts)[1], act_type, ctx->staging_out.data(), size_t(rows * N), 0),
          "tensor_read"))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  if (!h2d(state, output, ctx->staging_out.data(), ctx->staging_out.size()))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  if (cpu_fallbacks(*ctx) != before)
    return note_cpu_fallback(state, *ctx, *entry, "matmul_nbits", padded);
  return HIPDNN_EP_SUCCESS;
}

int wrap_corelib_rms_norm(RuntimeState *state, void *input, void *scale,
                          void *output, int64_t input_num_elements,
                          int64_t scale_num_elements,
                          int64_t norm_num_elements,
                          int64_t /*element_size_bytes*/, int64_t /*axis*/,
                          float epsilon, int64_t /*stash_type*/,
                          int64_t data_type) {
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  const int64_t k = norm_num_elements;
  // corelib's scale spans exactly the normalized width.
  if (!state || k <= 0 || scale_num_elements != k ||
      input_num_elements % k != 0 ||
      !activation_type(data_type, &act_type, &elem))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  OpEntry *entry = entry_for(*ctx, scale, kRmsNorm, epsilon, [&](OpEntry &e) {
    return pack_norm(state, *ctx, e, scale, k, epsilon, data_type);
  });
  if (!entry)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  const int64_t rows = input_num_elements / k;
  const int64_t padded = pick_rows(*entry, rows);
  if (padded == 0)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  // In-place is unsafe at large M, so input and output are separate tensors.
  const auto *ts = tensors_for(*ctx, *entry, padded, {k, k});
  if (!ts)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  const CorelibApi &api = *ctx->api;

  const size_t bytes = size_t(input_num_elements) * elem;
  ctx->staging_in.resize(bytes);
  ctx->staging_out.resize(bytes);
  if (!d2h(state, ctx->staging_in.data(), input, bytes))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  const uint64_t before = cpu_fallbacks(*ctx);
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[0], act_type,
                                           ctx->staging_in.data(),
                                           size_t(input_num_elements), 0),
          "tensor_write") ||
      !ok(*ctx,
          api.ryzenai_corelib_rmsnorm_bf16(ctx->stream, (*ts)[0],
                                           entry->weights, (*ts)[1]),
          "rmsnorm_bf16") ||
      !ok(*ctx, api.ryzenai_corelib_stream_synchronize(ctx->stream),
          "stream_synchronize") ||
      !ok(*ctx,
          api.ryzenai_corelib_tensor_read((*ts)[1], act_type,
                                          ctx->staging_out.data(),
                                          size_t(input_num_elements), 0),
          "tensor_read"))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  if (!h2d(state, output, ctx->staging_out.data(), bytes))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  if (cpu_fallbacks(*ctx) != before)
    return note_cpu_fallback(state, *ctx, *entry, "rms_norm", padded);
  return HIPDNN_EP_SUCCESS;
}

int wrap_corelib_skip_rms_norm(RuntimeState *state, void *input, void *skip,
                               void *gamma, void *bias, void *output,
                               void *input_skip_bias_sum,
                               int64_t input_num_elements,
                               int64_t gamma_num_elements,
                               int64_t /*element_size_bytes*/, float epsilon,
                               int64_t data_type) {
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  const int64_t k = gamma_num_elements;
  // corelib's fused add + RMSNorm has no bias input.
  if (!state || bias || k <= 0 || input_num_elements % k != 0 ||
      !activation_type(data_type, &act_type, &elem))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  OpEntry *entry =
      entry_for(*ctx, gamma, kSkipRmsNorm, epsilon, [&](OpEntry &e) {
        return pack_norm(state, *ctx, e, gamma, k, epsilon, data_type);
  });
  if (!entry)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  const int64_t rows = input_num_elements / k;
  const int64_t padded = pick_rows(*entry, rows);
  if (padded == 0)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  // x, residual, total, normed.
  const auto *ts = tensors_for(*ctx, *entry, padded, {k, k, k, k});
  if (!ts)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  const CorelibApi &api = *ctx->api;

  const size_t bytes = size_t(input_num_elements) * elem;
  const size_t count = size_t(input_num_elements);
  ctx->staging_in.resize(bytes);
  ctx->staging_aux.resize(bytes);
  ctx->staging_out.resize(bytes);
  if (!d2h(state, ctx->staging_in.data(), input, bytes) ||
      !d2h(state, ctx->staging_aux.data(), skip, bytes))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  const uint64_t before = cpu_fallbacks(*ctx);
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[0], act_type,
                                           ctx->staging_in.data(), count, 0),
          "tensor_write") ||
      !ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[1], act_type,
                                           ctx->staging_aux.data(), count, 0),
          "tensor_write") ||
      !ok(*ctx,
          api.ryzenai_corelib_add_rmsnorm_bf16(ctx->stream, (*ts)[0], (*ts)[1],
                                               entry->weights, (*ts)[2],
                                               (*ts)[3]),
          "add_rmsnorm_bf16") ||
      !ok(*ctx, api.ryzenai_corelib_stream_synchronize(ctx->stream),
          "stream_synchronize") ||
      !ok(*ctx,
          api.ryzenai_corelib_tensor_read((*ts)[3], act_type,
                                          ctx->staging_out.data(), count, 0),
          "tensor_read"))
    return HIPDNN_EP_CORELIB_NOT_HANDLED;

  if (!h2d(state, output, ctx->staging_out.data(), bytes))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  if (input_skip_bias_sum) {
    ctx->staging_out2.resize(bytes);
    if (!ok(*ctx,
            api.ryzenai_corelib_tensor_read((*ts)[2], act_type,
                                            ctx->staging_out2.data(), count, 0),
            "tensor_read"))
      return HIPDNN_EP_CORELIB_NOT_HANDLED;
    if (!h2d(state, input_skip_bias_sum, ctx->staging_out2.data(), bytes))
      return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  }
  if (cpu_fallbacks(*ctx) != before)
    return note_cpu_fallback(state, *ctx, *entry, "skip_rms_norm", padded);
  return HIPDNN_EP_SUCCESS;
}

} // extern "C"
