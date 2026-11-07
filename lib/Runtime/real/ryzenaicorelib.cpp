/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ryzenaicorelib.cpp - RyzenAI CoreLib (NPU) wrappers ----------------===//
//
// Native implementation of the wrap_corelib_* entry points declared in
// hipdnn_ep_runtime.h. See docs/design/ryzenai-corelib-backend.md.
//
// corelib is loaded at runtime (corelib_loader.h) rather than linked, so an EP
// built against its header still loads on a machine without
// ryzenai_corelib.dll; every call then returns HIPDNN_EP_CORELIB_NOT_HANDLED
// and the generated code runs the GPU wrapper instead.
//
// corelib exposes no device pointer, so every call stages through host memory:
// GPU -> host -> corelib tensor -> NPU -> corelib tensor -> host -> GPU.
//
//===----------------------------------------------------------------------===//

#include "msvc_builtins.h"

#include "../hipdnn_ep_errors.h"
#include "../hipdnn_ep_runtime.h"
#include "../runtime_state_internal.h"

#include "corelib_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

// Default PDIs: those of corelib's Qwen3, Llama and Phi-4-mini models. Each
// corelib model ships kernels on its own PDIs, and a shape no listed PDI ships
// runs on the GPU, so other models set RYZENAICORELIB_PDI.
constexpr const char *kDefaultPdis = "1;16";

using hipdnn_ep::corelib::CorelibApi;
using hipdnn_ep::corelib::read_env;

// Parses a comma-separated list of non-negative PDIs; false when malformed.
bool parse_pdi_list(const std::string &value, std::vector<int32_t> *out) {
  out->clear();
  size_t pos = 0;
  while (true) {
    size_t end = value.find(',', pos);
    std::string item = value.substr(pos, end - pos);
    int v = -1;
    char trailing = 0;
    if (std::sscanf(item.c_str(), " %d %c", &v, &trailing) != 1 || v < 0)
      return false;
    out->push_back(v);
    if (end == std::string::npos)
      return true;
    pos = end + 1;
  }
}

// Parses RYZENAICORELIB_PDI, "<prefill PDIs>;<decode PDIs>", each a priority
// list such as "9;19,35"; false when malformed.
bool parse_pdis(const std::string &value, std::vector<int32_t> *prefill,
                std::vector<int32_t> *decode) {
  size_t semi = value.find(';');
  return semi != std::string::npos &&
         parse_pdi_list(value.substr(0, semi), prefill) &&
         parse_pdi_list(value.substr(semi + 1), decode);
}

std::string pdi_list_string(const std::vector<int32_t> &pdis) {
  std::string s;
  for (int32_t p : pdis)
    s += (s.empty() ? "" : ",") + std::to_string(p);
  return s;
}

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
  s.memory_mode = ryzenai_corelib_weights_memory_copy;
  return s;
}

//===----------------------------------------------------------------------===//
// Per-session context
//===----------------------------------------------------------------------===//

// One packed op instance, keyed by its constant weight pointer (stable for the
// session), op kind and epsilon. `rows_off_npu` records padded row counts the
// NPU refused, which are routed to the GPU from then on.
struct OpEntry {
  ryzenai_corelib_weights_ptr weights = nullptr;
  bool unsupported = false;
  std::string why; // When unsupported: why the instance runs on the GPU.
  std::set<int64_t> shipped_rows;
  std::set<int64_t> rows_off_npu;
  // Padded rows -> BF16 device tensors, in argument order of the op.
  std::map<int64_t, std::vector<ryzenai_corelib_tensor_ptr>> tensors;
};

// What one routed call did, filled by the wrapper for the report.
struct Call {
  const char *op;
  std::string shape;
  const void *weights = nullptr;
  std::string why; // Set when the call goes to the GPU.
};

// RYZENAICORELIB_REPORT=1: per (op, shape), the instances (distinct weights)
// and the calls that ran on the NPU or the GPU, with why the GPU ones did.
struct ShapeStats {
  std::set<const void *> instances;
  uint64_t npu_calls = 0, gpu_calls = 0;
  std::map<std::string, uint64_t> gpu_why;
};

struct CorelibContext {
  const CorelibApi *api = nullptr;
  ryzenai_corelib_stream_ptr stream = nullptr;
  bool usable = false;
  bool strict = false;
  bool report = false;
  std::string stream_desc; // The NPU stream, or why there is none.
  std::map<std::tuple<const void *, int, float>, OpEntry> ops;
  std::map<std::pair<std::string, std::string>, ShapeStats> stats;
  std::set<std::string> warned; // Messages already printed once.
  std::vector<uint8_t> staging_in, staging_aux, staging_out, staging_out2;
};

// Prints `msg` the first time only: one line per shape, not per instance.
void warn_once(CorelibContext &ctx, const std::string &msg) {
  if (ctx.warned.insert(msg).second)
    fprintf(stderr, "[corelib] %s\n", msg.c_str());
}

void print_report(const CorelibContext &ctx) {
  fprintf(stderr, "[corelib] report: %s\n", ctx.stream_desc.c_str());
  if (ctx.stats.empty()) {
    fprintf(stderr, "[corelib]   no MatMulNBits, RmsNorm or SkipRmsNorm call "
                    "was routed to corelib\n");
    return;
  }
  fprintf(stderr, "[corelib]   %-13s %-38s %5s %9s %9s  %s\n", "op", "shape",
          "inst", "NPU calls", "GPU calls", "GPU because");
  uint64_t inst = 0, npu = 0, gpu = 0, npu_inst = 0;
  for (const auto &[key, s] : ctx.stats) {
    std::string why;
    for (const auto &[w, n] : s.gpu_why)
      why += (why.empty() ? "" : "; ") + w + " (" + std::to_string(n) + ")";
    fprintf(stderr, "[corelib]   %-13s %-38s %5zu %9llu %9llu  %s\n",
            key.first.c_str(), key.second.c_str(), s.instances.size(),
            (unsigned long long)s.npu_calls, (unsigned long long)s.gpu_calls,
            why.c_str());
    inst += s.instances.size();
    npu += s.npu_calls;
    gpu += s.gpu_calls;
    if (s.npu_calls)
      npu_inst += s.instances.size();
  }
  fprintf(stderr,
          "[corelib]   total: %llu op instances, %llu of them ran on the NPU "
          "at least once; %llu calls on the NPU, %llu on the GPU\n",
          (unsigned long long)inst, (unsigned long long)npu_inst,
          (unsigned long long)npu, (unsigned long long)gpu);
}

void destroy_context(void *p) {
  auto *ctx = static_cast<CorelibContext *>(p);
  if (ctx->report)
    print_report(*ctx);
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
    hipdnn_ep::corelib::remove_user(api);
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
  ctx->report = read_env("RYZENAICORELIB_REPORT") == "1";
  ctx->stream_desc = "no NPU stream, every routed op ran on the GPU (see the "
                     "[corelib] line above for why)";

  ctx->api = hipdnn_ep::corelib::load_api();
  if (!ctx->api)
    return ctx;
  const CorelibApi &api = *ctx->api;
  // corelib's ops run only on an AIE4 (AIEM) NPU; AIE2P runs only raw kernels.
  if (!api.ryzenai_corelib_npu_usable()) {
    fprintf(stderr, "[corelib] no usable AIE4 NPU (corelib's ops run only on "
                    "AIE4, with XRT 2.26.0 or later); corelib ops run on the "
                    "GPU\n");
    return ctx;
  }
  std::vector<int32_t> prefill_pdis, decode_pdis;
  std::string pdis = read_env("RYZENAICORELIB_PDI");
  if (pdis.empty())
    pdis = kDefaultPdis;
  if (!parse_pdis(pdis, &prefill_pdis, &decode_pdis)) {
    fprintf(stderr,
            "[corelib] RYZENAICORELIB_PDI='%s' is not '<prefill PDIs>;<decode "
            "PDIs>'; corelib ops run on the GPU\n",
            pdis.c_str());
    return ctx;
  }
  if (api.ryzenai_corelib_create_stream(&ctx->stream) !=
      ryzenai_corelib_status_success) {
    fprintf(stderr,
            "[corelib] create_stream failed: %s; corelib ops run on "
            "the GPU\n",
            api.ryzenai_corelib_get_last_error_message());
    return ctx;
  }
  auto drop_stream = [&] {
    api.ryzenai_corelib_object_release(ctx->stream);
    ctx->stream = nullptr;
  };

  // NPU only: a call the NPU cannot run then fails (unsupported or
  // missing_kernel) instead of running on corelib's own GPU or CPU paths, and
  // hip-ep runs it on its GPU wrapper.
  if (api.ryzenai_corelib_stream_set_enabled_backends(
          ctx->stream, ryzenai_corelib_backend_npu) !=
          ryzenai_corelib_status_success ||
      api.ryzenai_corelib_stream_use_pdis(
          ctx->stream, ryzenai_corelib_model_phase_prefill, prefill_pdis.data(),
          prefill_pdis.size()) != ryzenai_corelib_status_success ||
      api.ryzenai_corelib_stream_use_pdis(
          ctx->stream, ryzenai_corelib_model_phase_decode, decode_pdis.data(),
          decode_pdis.size()) != ryzenai_corelib_status_success) {
    fprintf(stderr,
            "[corelib] configuring the stream failed: %s; corelib ops run on "
            "the GPU\n",
            api.ryzenai_corelib_get_last_error_message());
    drop_stream();
    return ctx;
  }
  if (!api.ryzenai_corelib_stream_npu_usable(ctx->stream)) {
    fprintf(stderr, "[corelib] the stream cannot use the NPU; corelib ops run "
                    "on the GPU\n");
    drop_stream();
    return ctx;
  }

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

  // With neither a bundle nor a kernel store no op could reach the NPU.
  const char *root = api.ryzenai_corelib_stream_get_kernels_root(ctx->stream);
  if (bundle.empty() && (!root || !*root)) {
    fprintf(stderr, "[corelib] no kernels (set RYZENAI_CORELIB_KERNELS_ROOT or "
                    "RYZENAICORELIB_KERNEL_BUNDLE); corelib ops run on the "
                    "GPU\n");
    drop_stream();
    return ctx;
  }
  ctx->stream_desc = "NPU stream: prefill PDIs " +
                     pdi_list_string(prefill_pdis) + ", decode PDIs " +
                     pdi_list_string(decode_pdis) + ", kernel bundle '" +
                     bundle + "', kernels root '" + (root ? root : "") + "'";
  fprintf(stderr, "[corelib] %s\n", ctx->stream_desc.c_str());
  ctx->usable = true;
  hipdnn_ep::corelib::add_user();
  return ctx;
}

// Smallest shipped row count that holds `rows`, or 0 when none does.
int64_t pick_rows(const OpEntry &entry, int64_t rows) {
  auto it = entry.shipped_rows.lower_bound(rows);
  if (it == entry.shipped_rows.end() || entry.rows_off_npu.count(*it))
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

// The NPU refused the op at these padded rows: no kernel ships there on the
// stream's PDIs, or the NPU run failed. corelib ran nothing, so the GPU wrapper
// runs this call and this shape from then on. Under HIPDNN_EP_STRICT=1 it is
// instead an error surfaced through the device error flag (read by
// inference_compute).
int note_off_npu(RuntimeState *state, CorelibContext &ctx, OpEntry &entry,
                 const char *op, int64_t rows, const char *what) {
  entry.rows_off_npu.insert(rows);
  fprintf(stderr, "[corelib] %s at %lld rows: %s failed: %s%s\n", op,
          (long long)rows, what,
          ctx.api->ryzenai_corelib_get_last_error_message(),
          ctx.strict ? " (HIPDNN_EP_STRICT)"
                     : "; using the GPU for that shape from now on");
  if (!ctx.strict)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  int code = HIPDNN_EP_ERR_CORELIB_OFF_NPU;
  if (state->device_error_flag)
    h2d(state, state->device_error_flag, &code, sizeof(code));
  return code;
}

// Creates (once) the BF16 device tensors for `rows` padded rows. A device
// tensor is undefined until written, so the first `inputs` are zeroed: each
// call writes only its live rows, and the pad rows then stay finite.
const std::vector<ryzenai_corelib_tensor_ptr> *
tensors_for(CorelibContext &ctx, OpEntry &entry, int64_t rows,
            std::initializer_list<int64_t> widths, size_t inputs) {
  auto it = entry.tensors.find(rows);
  if (it != entry.tensors.end())
    return &it->second;
  std::vector<ryzenai_corelib_tensor_ptr> ts;
  auto fail = [&](const char *what) {
    fprintf(stderr, "[corelib] %s failed: %s\n", what,
            ctx.api->ryzenai_corelib_get_last_error_message());
    for (ryzenai_corelib_tensor_ptr p : ts)
      ctx.api->ryzenai_corelib_object_release(p);
    return nullptr;
  };
  for (int64_t w : widths) {
    int64_t shape[2] = {rows, w};
    ryzenai_corelib_tensor_ptr t = nullptr;
    if (ctx.api->ryzenai_corelib_create_device_tensor(
            ryzenai_corelib_data_type_bf16, shape, 2, &t) !=
        ryzenai_corelib_status_success)
      return fail("create_device_tensor");
    ts.push_back(t);
    if (ts.size() <= inputs) {
      std::vector<uint16_t> zeros(size_t(rows * w));
      if (ctx.api->ryzenai_corelib_tensor_write(
              t, ryzenai_corelib_data_type_bf16, zeros.data(), zeros.size(),
              0) != ryzenai_corelib_status_success)
        return fail("tensor_write");
    }
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

// corelib derives the kernel group from the model's: divisible by 64 takes the
// g64 layout, otherwise g32.
int64_t kernel_group(int64_t block_size) {
  return block_size % 64 == 0 ? 64 : 32;
}

// Lists only the stream's PDIs' kernels, so every row count found here is one
// the stream can reach.
bool collect_matmul_rows(void *p,
                         const ryzenai_corelib_matmul_bf16_kernel_info *info) {
  auto *q = static_cast<MatMulKernelQuery *>(p);
  // Accept either spelling of the group. The KV-cache writing (`_gm`) kernels
  // are listed too, with gm_heads set; a plain matmul cannot run them.
  if (info->gm_heads == 0 && info->k == q->k && info->n == q->n &&
      info->has_bias == q->has_bias &&
      (info->group_size == q->block_size ||
       info->group_size == kernel_group(q->block_size)))
    q->rows->insert(info->m);
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
  entry.why = "reading the weights back failed";
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

  // corelib takes FP16 or FP32 scales (independent of A's type) and narrows
  // them to BF16 itself, and an FP32 bias whatever the activation type.
  std::vector<float> bias_fp32(bias ? size_t(N) : 0);
  for (size_t i = 0; i < bias_fp32.size(); ++i)
    bias_fp32[i] = to_float(data_type, bias_raw.data(), i);

  std::vector<ryzenai_corelib_weights_source> sources = {
      make_source(ryzenai_corelib_weights_role_qweight, qweight.data(),
                  qweight.size(), ryzenai_corelib_weights_data_type_onnx_uint8,
                  {N, K / 2}),
      make_source(ryzenai_corelib_weights_role_scales, scales_raw.data(),
                  scales_raw.size(),
                  scale_elem_size == 2
                      ? ryzenai_corelib_weights_data_type_onnx_float16
                      : ryzenai_corelib_weights_data_type_onnx_float,
                  {N, groups}),
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
  desc.format = ryzenai_corelib_matmul_bf16_weights_format_mladf_v4;
  entry.why = "packing the weights failed";
  if (!ok(ctx,
          ctx.api->ryzenai_corelib_matmul_bf16_weights_create(
              &desc, sources.data(), sources.size(), nullptr, &entry.weights),
          "matmul_bf16_weights_create"))
    return false;

  MatMulKernelQuery query{K, N, block_size, bias != nullptr,
                          &entry.shipped_rows};
  entry.why = "listing the kernels failed";
  if (!ok(ctx,
          ctx.api->ryzenai_corelib_matmul_bf16_enum_kernels(
              ctx.stream, collect_matmul_rows, &query),
          "matmul_bf16_enum_kernels"))
    return false;
  if (entry.shipped_rows.empty()) {
    entry.why = "no kernel at this shape on the stream's PDIs";
    warn_once(ctx, "no matmul kernel for K=" + std::to_string(K) +
                       " N=" + std::to_string(N) +
                       " group=" + std::to_string(block_size) + " bias=" +
                       std::to_string(bias != nullptr) + "; using the GPU");
    return false;
  }
  entry.why.clear();
  return true;
}

//===----------------------------------------------------------------------===//
// RmsNorm / SkipRmsNorm (shared weights: epsilon + BF16 scale)
//===----------------------------------------------------------------------===//

using NormKernelQuery = std::pair<int64_t, std::set<int64_t> *>;

template <class KernelInfo>
bool collect_norm_rows(void *p, const KernelInfo *info) {
  auto *q = static_cast<NormKernelQuery *>(p);
  if (info->k == q->first)
    q->second->insert(info->m);
  return true;
}

// `fused` selects add_rmsnorm's kernels, which ship at exact (M, k) pairs of
// their own.
bool pack_norm(RuntimeState *state, CorelibContext &ctx, OpEntry &entry,
               const void *scale, int64_t k, float epsilon, int64_t data_type,
               bool fused) {
  size_t elem = 0;
  ryzenai_corelib_data_type unused;
  activation_type(data_type, &unused, &elem);
  std::vector<uint8_t> raw(size_t(k) * elem);
  entry.why = "reading the weights back failed";
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
  entry.why = "packing the weights failed";
  if (!ok(ctx,
          ctx.api->ryzenai_corelib_rmsnorm_bf16_weights_create(
              &desc, &source, 1, nullptr, &entry.weights),
          "rmsnorm_bf16_weights_create"))
    return false;

  NormKernelQuery query{k, &entry.shipped_rows};
  ryzenai_corelib_status st =
      fused
          ? ctx.api->ryzenai_corelib_add_rmsnorm_bf16_enum_kernels(
                ctx.stream,
                collect_norm_rows<ryzenai_corelib_add_rmsnorm_bf16_kernel_info>,
                &query)
          : ctx.api->ryzenai_corelib_rmsnorm_bf16_enum_kernels(
                ctx.stream,
                collect_norm_rows<ryzenai_corelib_rmsnorm_bf16_kernel_info>,
                &query);
  entry.why = "listing the kernels failed";
  if (!ok(ctx, st, "rmsnorm enum_kernels"))
    return false;
  if (entry.shipped_rows.empty()) {
    entry.why = "no kernel at this width on the stream's PDIs";
    warn_once(ctx, std::string("no ") + (fused ? "add_rmsnorm" : "rmsnorm") +
                       " kernel for k=" + std::to_string(k) +
                       "; using the GPU");
    return false;
  }
  entry.why.clear();
  return true;
}

enum OpKind { kMatMulNBits, kRmsNorm, kSkipRmsNorm };

// Returns the entry for these weights, packing it on first use; `unsupported`
// (with `why`) when the op cannot run on corelib.
template <class PackFn>
OpEntry &entry_for(CorelibContext &ctx, const void *weights, OpKind kind,
                   float epsilon, PackFn pack) {
  auto [it, inserted] =
      ctx.ops.try_emplace(std::make_tuple(weights, int(kind), epsilon));
  OpEntry &entry = it->second;
  if (inserted && !pack(entry))
    entry.unsupported = true;
  return entry;
}

// Records `call`'s outcome for the report; `rc` is the wrapper's result.
void record(RuntimeState *state, const Call &call, int rc) {
  // Also counts calls rejected before the wrapper needed the context.
  CorelibContext &ctx = *get_context(state);
  if (!ctx.report)
    return;
  ShapeStats &s = ctx.stats[{call.op, call.shape}];
  if (call.weights)
    s.instances.insert(call.weights);
  if (rc == HIPDNN_EP_SUCCESS) {
    ++s.npu_calls;
    return;
  }
  ++s.gpu_calls;
  s.gpu_why[rc == HIPDNN_EP_CORELIB_NOT_HANDLED
                ? (call.why.empty() ? std::string("unknown") : call.why)
                : "error " + std::to_string(rc)]++;
}

// Ends a wrapper on the GPU path, recording why.
#define CORELIB_USE_GPU(reason)                                                \
  do {                                                                         \
    call.why = (reason);                                                       \
    return HIPDNN_EP_CORELIB_NOT_HANDLED;                                      \
  } while (0)

// The reason a padded row count has no NPU kernel, for the report.
std::string rows_reason(const OpEntry &entry, int64_t rows) {
  auto it = entry.shipped_rows.lower_bound(rows);
  if (it != entry.shipped_rows.end() && entry.rows_off_npu.count(*it))
    return "NPU refused " + std::to_string(*it) + " rows earlier";
  return "no kernel at " + std::to_string(rows) + " rows (largest " +
         std::to_string(*entry.shipped_rows.rbegin()) + ")";
}

// The wrappers. Each fills `call` for the report and returns
// HIPDNN_EP_CORELIB_NOT_HANDLED (through CORELIB_USE_GPU, with the reason) when
// the GPU wrapper is to run the call.

int matmul_nbits(Call &call, RuntimeState *state, const void *A, const void *B,
                 const void *scales, const void *zero_points, const void *g_idx,
                 const void *bias, void *output, int64_t M, int64_t N,
                 int64_t K, int64_t batch_count, int64_t bits,
                 int64_t block_size, int64_t zp_elem_size,
                 int64_t scale_elem_size, int64_t data_type) {
  call.weights = B;
  call.shape = "K=" + std::to_string(K) + " N=" + std::to_string(N) +
               " block=" + std::to_string(block_size) +
               (bits != 4 ? " bits=" + std::to_string(bits) : "") +
               (zero_points ? " zp" : "") + (bias ? " bias" : "");
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  if (bits != 4)
    CORELIB_USE_GPU("corelib takes 4-bit weights only");
  if (g_idx)
    CORELIB_USE_GPU("g_idx is not supported");
  // K must be whole tiles of the layout: 128 at g64, 64 at g32.
  if (block_size % 32 != 0 || K % block_size != 0 ||
      K % (2 * kernel_group(block_size)) != 0)
    CORELIB_USE_GPU("block size or K does not fit corelib's 4-bit layout");
  if ((zero_points && zp_elem_size != 1) ||
      (scale_elem_size != 2 && scale_elem_size != 4))
    CORELIB_USE_GPU("unsupported zero point or scale type");
  if (!activation_type(data_type, &act_type, &elem))
    CORELIB_USE_GPU("unsupported activation type");

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    CORELIB_USE_GPU("no NPU stream");
  OpEntry &entry = entry_for(*ctx, B, kMatMulNBits, 0.0f, [&](OpEntry &e) {
    return pack_matmul(state, *ctx, e, B, scales, zero_points, bias, N, K,
                       block_size, scale_elem_size, data_type);
  });
  if (entry.unsupported)
    CORELIB_USE_GPU(entry.why);

  const int64_t rows = M * batch_count;
  const int64_t padded = pick_rows(entry, rows);
  if (padded == 0)
    CORELIB_USE_GPU(rows_reason(entry, rows));
  const auto *ts = tensors_for(*ctx, entry, padded, {K, N}, 1);
  if (!ts)
    CORELIB_USE_GPU("creating corelib tensors failed");
  const CorelibApi &api = *ctx->api;

  ctx->staging_in.resize(size_t(rows * K) * elem);
  ctx->staging_out.resize(size_t(rows * N) * elem);
  if (!d2h(state, ctx->staging_in.data(), A, ctx->staging_in.size()))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write(
              (*ts)[0], act_type, ctx->staging_in.data(), size_t(rows * K), 0),
          "tensor_write"))
    CORELIB_USE_GPU("tensor_write failed");
  call.why = "NPU refused " + std::to_string(padded) + " rows";
  if (api.ryzenai_corelib_matmul_bf16(ctx->stream, (*ts)[0], entry.weights,
                                      nullptr, (*ts)[1]) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "matmul_nbits", padded,
                        "matmul_bf16");
  if (api.ryzenai_corelib_stream_synchronize(ctx->stream) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "matmul_nbits", padded,
                        "stream_synchronize");
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_read(
              (*ts)[1], act_type, ctx->staging_out.data(), size_t(rows * N), 0),
          "tensor_read"))
    CORELIB_USE_GPU("tensor_read failed");

  if (!h2d(state, output, ctx->staging_out.data(), ctx->staging_out.size()))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  return HIPDNN_EP_SUCCESS;
}

int rms_norm(Call &call, RuntimeState *state, void *input, void *scale,
             void *output, int64_t input_num_elements,
             int64_t scale_num_elements, int64_t norm_num_elements,
             float epsilon, int64_t data_type) {
  const int64_t k = norm_num_elements;
  call.weights = scale;
  call.shape = "k=" + std::to_string(k);
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  // corelib's scale spans exactly the normalized width.
  if (k <= 0 || scale_num_elements != k || input_num_elements % k != 0)
    CORELIB_USE_GPU("the scale does not span the normalized width");
  if (!activation_type(data_type, &act_type, &elem))
    CORELIB_USE_GPU("unsupported activation type");

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    CORELIB_USE_GPU("no NPU stream");
  OpEntry &entry = entry_for(*ctx, scale, kRmsNorm, epsilon, [&](OpEntry &e) {
    return pack_norm(state, *ctx, e, scale, k, epsilon, data_type, false);
  });
  if (entry.unsupported)
    CORELIB_USE_GPU(entry.why);

  const int64_t rows = input_num_elements / k;
  const int64_t padded = pick_rows(entry, rows);
  if (padded == 0)
    CORELIB_USE_GPU(rows_reason(entry, rows));
  // In-place is unsafe at large M, so input and output are separate tensors.
  const auto *ts = tensors_for(*ctx, entry, padded, {k, k}, 1);
  if (!ts)
    CORELIB_USE_GPU("creating corelib tensors failed");
  const CorelibApi &api = *ctx->api;

  const size_t bytes = size_t(input_num_elements) * elem;
  const size_t count = size_t(input_num_elements);
  ctx->staging_in.resize(bytes);
  ctx->staging_out.resize(bytes);
  if (!d2h(state, ctx->staging_in.data(), input, bytes))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[0], act_type,
                                           ctx->staging_in.data(), count, 0),
          "tensor_write"))
    CORELIB_USE_GPU("tensor_write failed");
  call.why = "NPU refused " + std::to_string(padded) + " rows";
  if (api.ryzenai_corelib_rmsnorm_bf16(ctx->stream, (*ts)[0], entry.weights,
                                       (*ts)[1]) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "rms_norm", padded, "rmsnorm_bf16");
  if (api.ryzenai_corelib_stream_synchronize(ctx->stream) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "rms_norm", padded,
                        "stream_synchronize");
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_read((*ts)[1], act_type,
                                          ctx->staging_out.data(), count, 0),
          "tensor_read"))
    CORELIB_USE_GPU("tensor_read failed");

  if (!h2d(state, output, ctx->staging_out.data(), bytes))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  return HIPDNN_EP_SUCCESS;
}

int skip_rms_norm(Call &call, RuntimeState *state, void *input, void *skip,
                  void *gamma, void *bias, void *output,
                  void *input_skip_bias_sum, int64_t input_num_elements,
                  int64_t gamma_num_elements, float epsilon,
                  int64_t data_type) {
  const int64_t k = gamma_num_elements;
  call.weights = gamma;
  call.shape = "k=" + std::to_string(k) + (bias ? " bias" : "");
  ryzenai_corelib_data_type act_type;
  size_t elem = 0;
  // corelib's fused add + RMSNorm has no bias input.
  if (bias)
    CORELIB_USE_GPU("corelib's add_rmsnorm takes no bias");
  if (k <= 0 || input_num_elements % k != 0)
    CORELIB_USE_GPU("the input is not whole rows of the width");
  if (!activation_type(data_type, &act_type, &elem))
    CORELIB_USE_GPU("unsupported activation type");

  CorelibContext *ctx = get_context(state);
  if (!ctx->usable)
    CORELIB_USE_GPU("no NPU stream");
  OpEntry &entry =
      entry_for(*ctx, gamma, kSkipRmsNorm, epsilon, [&](OpEntry &e) {
        return pack_norm(state, *ctx, e, gamma, k, epsilon, data_type, true);
      });
  if (entry.unsupported)
    CORELIB_USE_GPU(entry.why);

  const int64_t rows = input_num_elements / k;
  const int64_t padded = pick_rows(entry, rows);
  if (padded == 0)
    CORELIB_USE_GPU(rows_reason(entry, rows));
  // x, residual, total, normed.
  const auto *ts = tensors_for(*ctx, entry, padded, {k, k, k, k}, 2);
  if (!ts)
    CORELIB_USE_GPU("creating corelib tensors failed");
  const CorelibApi &api = *ctx->api;

  const size_t bytes = size_t(input_num_elements) * elem;
  const size_t count = size_t(input_num_elements);
  ctx->staging_in.resize(bytes);
  ctx->staging_aux.resize(bytes);
  ctx->staging_out.resize(bytes);
  if (!d2h(state, ctx->staging_in.data(), input, bytes) ||
      !d2h(state, ctx->staging_aux.data(), skip, bytes))
    return HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;

  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[0], act_type,
                                           ctx->staging_in.data(), count, 0),
          "tensor_write") ||
      !ok(*ctx,
          api.ryzenai_corelib_tensor_write((*ts)[1], act_type,
                                           ctx->staging_aux.data(), count, 0),
          "tensor_write"))
    CORELIB_USE_GPU("tensor_write failed");
  call.why = "NPU refused " + std::to_string(padded) + " rows";
  if (api.ryzenai_corelib_add_rmsnorm_bf16(ctx->stream, (*ts)[0], (*ts)[1],
                                           entry.weights, (*ts)[2], (*ts)[3]) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "skip_rms_norm", padded,
                        "add_rmsnorm_bf16");
  if (api.ryzenai_corelib_stream_synchronize(ctx->stream) !=
      ryzenai_corelib_status_success)
    return note_off_npu(state, *ctx, entry, "skip_rms_norm", padded,
                        "stream_synchronize");
  if (!ok(*ctx,
          api.ryzenai_corelib_tensor_read((*ts)[3], act_type,
                                          ctx->staging_out.data(), count, 0),
          "tensor_read"))
    CORELIB_USE_GPU("tensor_read failed");

  if (!h2d(state, output, ctx->staging_out.data(), bytes))
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  if (input_skip_bias_sum) {
    ctx->staging_out2.resize(bytes);
    if (!ok(*ctx,
            api.ryzenai_corelib_tensor_read((*ts)[2], act_type,
                                            ctx->staging_out2.data(), count, 0),
            "tensor_read"))
      CORELIB_USE_GPU("tensor_read failed");
    if (!h2d(state, input_skip_bias_sum, ctx->staging_out2.data(), bytes))
      return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  }
  return HIPDNN_EP_SUCCESS;
}

#undef CORELIB_USE_GPU

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
  if (!state)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  Call call{"MatMulNBits"};
  int rc = matmul_nbits(call, state, A, B, scales, zero_points, g_idx, bias,
                        output, M, N, K, batch_count, bits, block_size,
                        zp_elem_size, scale_elem_size, data_type);
  record(state, call, rc);
  return rc;
}

int wrap_corelib_rms_norm(RuntimeState *state, void *input, void *scale,
                          void *output, int64_t input_num_elements,
                          int64_t scale_num_elements, int64_t norm_num_elements,
                          int64_t /*element_size_bytes*/, int64_t /*axis*/,
                          float epsilon, int64_t /*stash_type*/,
                          int64_t data_type) {
  if (!state)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  Call call{"RmsNorm"};
  int rc = rms_norm(call, state, input, scale, output, input_num_elements,
                    scale_num_elements, norm_num_elements, epsilon, data_type);
  record(state, call, rc);
  return rc;
}

int wrap_corelib_skip_rms_norm(RuntimeState *state, void *input, void *skip,
                               void *gamma, void *bias, void *output,
                               void *input_skip_bias_sum,
                               int64_t input_num_elements,
                               int64_t gamma_num_elements,
                               int64_t /*element_size_bytes*/, float epsilon,
                               int64_t data_type) {
  if (!state)
    return HIPDNN_EP_CORELIB_NOT_HANDLED;
  Call call{"SkipRmsNorm"};
  int rc = skip_rms_norm(call, state, input, skip, gamma, bias, output,
                         input_skip_bias_sum, input_num_elements,
                         gamma_num_elements, epsilon, data_type);
  record(state, call, rc);
  return rc;
}

} // extern "C"
