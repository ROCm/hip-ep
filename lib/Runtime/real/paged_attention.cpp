/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../Kernels/hip/autotune/paged_attention/paged_attention_autotune.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "hip_custom_kernels.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <type_traits>
#include <vector>

struct PagedAttentionState : OpStateT<PagedAttentionState> {
  std::map<std::string, hipdnn_ep::PagedAttentionResult> prepared;
};
extern "C" int8_t
hipdnn_ep_op_state_construct_paged_attention(RuntimeState *state,
                                             int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, PagedAttentionState::create().release());
  return 0;
}

struct PagedMropeOptions {
  const int64_t *positions;
  int64_t positionTokens, rotaryDim, sectionT, sectionH, sectionW, layout;
  float scale;
};
struct NoPagedMrope {};
template <bool Mrope>
static int wrapPagedAttentionImpl(
    RuntimeState *state, int32_t op_state_slot, const void *q, const void *k,
    const void *v, void *k_pool, void *v_pool, const int32_t *cum,
    const int32_t *past, const int32_t *table, const void *cos_cache,
    const void *sin_cache, const int32_t *metadata, void *output,
    int64_t tokens, int64_t key_tokens, int64_t value_tokens,
    int64_t output_tokens, int64_t batch, int64_t pages, int64_t value_pages,
    int64_t page_size, int64_t value_page_size, int64_t head_dim,
    int64_t table_batch, int64_t table_columns, int64_t cum_count,
    int64_t metadata_count, int64_t cos_rows, int64_t sin_rows,
    int64_t num_heads, int64_t kv_heads, int64_t window, int64_t do_rotary,
    int64_t rotary_interleaved, float scale, float softcap,
    const PagedMropeOptions *mrope) {
  auto fail = [&](int code) {
    if (state)
      (void)hipdnn_ep_state_set_error_flag(state, code);
    return code;
  };
  if (!state || batch < 1 || batch >= INT32_MAX || tokens < 0 ||
      tokens > INT32_MAX || key_tokens != tokens || value_tokens != tokens ||
      output_tokens != tokens || value_page_size != page_size ||
      value_pages != pages || table_batch != batch || cum_count != batch + 1 ||
      pages < 1 || pages > INT32_MAX || page_size < 1 ||
      page_size > INT32_MAX || table_columns < 1 || table_columns > INT32_MAX ||
      head_dim < 1 || head_dim > INT32_MAX || num_heads < 1 ||
      num_heads > INT32_MAX || kv_heads < 1 || kv_heads > INT32_MAX ||
      window < -1 || window > INT32_MAX || cos_rows < 0 ||
      cos_rows > INT32_MAX || cos_rows != sin_rows ||
      (do_rotary != 0 && do_rotary != 1) || rotary_interleaved != 0 ||
      (metadata ? (metadata_count != 2 && metadata_count != 3)
                : metadata_count != 0))
    return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  if constexpr (Mrope) {
    if (!mrope || mrope->positionTokens != tokens ||
        (tokens && !mrope->positions) || do_rotary || rotary_interleaved ||
        cos_rows < 1 || mrope->rotaryDim <= 0 || (mrope->rotaryDim & 1) ||
        mrope->rotaryDim > head_dim || mrope->sectionT < 0 ||
        mrope->sectionH < 0 || mrope->sectionW < 0 ||
        mrope->sectionT > mrope->rotaryDim / 2 ||
        mrope->sectionH > mrope->rotaryDim / 2 ||
        mrope->sectionW > mrope->rotaryDim / 2 ||
        mrope->sectionT + mrope->sectionH + mrope->sectionW !=
            mrope->rotaryDim / 2 ||
        (mrope->layout != 0 && mrope->layout != 1) ||
        !std::isfinite(mrope->scale))
      return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  }
  if (tokens && (!q || !k || !v || !k_pool || !v_pool || !cum || !past ||
                 !table || !output))
    return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  void *stream = tokens ? hipdnn_ep_state_get_stream(state) : nullptr;
  if (tokens && !stream)
    return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  int32_t bounds[3] = {};
  int32_t *error_flag = nullptr;
  int max_queries = 0;
  int max_kv = 0;
  int min_kv = INT32_MAX;
  if (tokens) {
    error_flag = static_cast<int32_t *>(
        hipdnn_ep_state_get_error_flag_device_ptr(state));
    if (!error_flag)
      return fail(HIP_PAGED_ATTENTION_HIP_ERROR);
    // A preceding checked op can fail asynchronously. Page uses a separate
    // workspace status, so preserve the session error before any cache writes.
    // Request lengths define the phase. Metadata contains sizing bounds,
    // independently of whether its producer left it on host or device.
    std::vector<int32_t> host_cum(batch + 1), host_past(batch);
    int32_t prior_status = 0;
    if (hipMemcpyAsync(&prior_status, error_flag, sizeof(prior_status),
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(stream)) != hipSuccess ||
        (metadata &&
         hipMemcpyAsync(bounds, metadata, metadata_count * sizeof(int32_t),
                        hipMemcpyDefault,
                        static_cast<hipStream_t>(stream)) != hipSuccess) ||
        hipMemcpyAsync(host_cum.data(), cum, (batch + 1) * sizeof(int32_t),
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(stream)) != hipSuccess ||
        hipMemcpyAsync(host_past.data(), past, batch * sizeof(int32_t),
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(stream)) != hipSuccess ||
        hipStreamSynchronize(static_cast<hipStream_t>(stream)) != hipSuccess)
      return fail(HIP_PAGED_ATTENTION_HIP_ERROR);
    if (prior_status)
      return prior_status;
    if (metadata && (bounds[0] < 0 || bounds[1] < 0 || bounds[2] < 0 ||
                     (bounds[1] && bounds[2] > bounds[1])))
      return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
    if (host_cum.front() != 0 || host_cum.back() != tokens)
      return fail(HIP_PAGED_ATTENTION_BAD_CUM);
    for (int64_t b = 0; b < batch; ++b) {
      int64_t queries = int64_t(host_cum[b + 1]) - host_cum[b];
      int64_t kv = int64_t(host_past[b]) + queries;
      if (queries < 0 || host_cum[b] < 0 || host_cum[b + 1] > tokens)
        return fail(HIP_PAGED_ATTENTION_BAD_CUM);
      if (host_past[b] < 0 || kv > INT32_MAX)
        return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
      max_queries = std::max(max_queries, int(queries));
      if (queries) {
        max_kv = std::max(max_kv, int(kv));
        min_kv = std::min(min_kv, int(kv));
      }
    }
    if (metadata && ((bounds[0] && max_queries > bounds[0]) ||
                     (bounds[1] && max_kv > bounds[1]) ||
                     (bounds[2] && min_kv < bounds[2])))
      return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  }
  hip_paged_attention_f16_desc desc{};
  desc.q = q;
  desc.k_new = k;
  desc.v_new = v;
  desc.k_pool = k_pool;
  desc.v_pool = v_pool;
  desc.o = output;
  desc.block_table = table;
  desc.cum = cum;
  desc.past_seqlens = past;
  desc.cos_cache = cos_cache;
  desc.sin_cache = sin_cache;
  desc.batch = batch;
  desc.tokens = tokens;
  desc.num_heads = num_heads;
  desc.kv_num_heads = kv_heads;
  desc.head_dim = head_dim;
  desc.num_pages = pages;
  desc.page_size = page_size;
  desc.table_columns = table_columns;
  desc.max_position = cos_rows;
  desc.phase = max_queries <= 1 ? HIP_PAGED_ATTENTION_DECODE
                                : HIP_PAGED_ATTENTION_PREFILL;
  desc.do_rotary = do_rotary;
  desc.rotary_interleaved = rotary_interleaved;
  desc.rotary_dim = do_rotary ? head_dim : 0;
  desc.local_window_size = window;
  desc.scale = scale;
  desc.softcap = softcap;
  // Separate template instances retain the old Page boundary and avoid
  // adding any MRoPE arguments to the dense GQA or ordinary Page kernels.
  [[maybe_unused]] std::conditional_t<Mrope, hip_paged_attention_mrope_f16_desc,
                                      NoPagedMrope>
      rotary{};
  if constexpr (Mrope) {
    rotary.page = desc;
    rotary.page.q = rotary.page.k_new = nullptr;
    rotary.page.cos_cache = rotary.page.sin_cache = nullptr;
    rotary.q = static_cast<const float *>(q);
    rotary.k_new = static_cast<const float *>(k);
    rotary.cos_cache = static_cast<const float *>(cos_cache);
    rotary.sin_cache = static_cast<const float *>(sin_cache);
    rotary.position_ids = mrope->positions;
    rotary.rotary_dim = mrope->rotaryDim;
    rotary.max_position = cos_rows;
    rotary.section_t = mrope->sectionT;
    rotary.section_h = mrope->sectionH;
    rotary.section_w = mrope->sectionW;
    rotary.mrope_layout = mrope->layout;
    rotary.rotary_scale = mrope->scale;
  }
  size_t bytes = 0;
  int rc;
  if constexpr (Mrope)
    rc = hip_paged_attention_mrope_f16_workspace_size(&rotary, &bytes);
  else
    rc = hip_paged_attention_f16_workspace_size(&desc, &bytes);
  if (rc)
    return fail(rc);
  if (!tokens)
    return 0;
  if (hipdnn_ep_state_ensure_paged_attention_scratch(state, bytes))
    return fail(HIP_PAGED_ATTENTION_HIP_ERROR);
  void *workspace = hipdnn_ep_state_get_paged_attention_scratch(state);
  auto *opState = PagedAttentionState::get_op_state(state, op_state_slot);
  if (!opState)
    return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  const char *mode = std::getenv("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE");
  const bool online = mode && std::strcmp(mode, "online") == 0;
  if (mode && *mode && !online && std::strcmp(mode, "lookup") &&
      std::strcmp(mode, "auto"))
    return fail(HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  std::ostringstream key;
  // LUT points require phase + "_" as their leading discriminator. Keep
  // ordinary Page keys unchanged and distinguish the FP32 MRoPE input path.
  key << (desc.phase == HIP_PAGED_ATTENTION_DECODE ? "decode" : "prefill");
  if constexpr (Mrope)
    key << "_mrope_f32_f16_r" << mrope->rotaryDim << "_sections"
        << mrope->sectionT << '-' << mrope->sectionH << '-' << mrope->sectionW
        << "_layout" << mrope->layout << "_rscale"
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << mrope->scale;
  key << "_model_b" << batch << "_t" << tokens << "_h" << num_heads << "_g"
      << kv_heads << "_d" << head_dim << "_p" << pages << "_s" << page_size
      << "_cols" << table_columns << "_qbound" << bounds[0] << "_kvbound"
      << bounds[1] << "_lower" << bounds[2] << "_maxq" << max_queries
      << "_maxkv" << max_kv << "_window" << window << "_rotary" << do_rotary
      << "_scale" << scale << "_maxpos" << cos_rows << "_mode"
      << (online ? "online" : "lookup");
  OP_PROFILE(
      Mrope ? "paged_mrope_attention" : "paged_attention",
      [&] { return key.str(); }, state);
  // Session preparations belong to this compiler-assigned op slot and are
  // destroyed by the existing OpState cleanup. They never masquerade as LUT
  // hits.
  hipdnn_ep::PagedAttentionResult result;
  auto configured = [&](const GqaPagedConfig &config) {
    if constexpr (Mrope)
      return hip_paged_attention_mrope_f16_configured_v1(
          stream, &rotary, workspace, bytes, &config, sizeof(config));
    else
      return hip_paged_attention_f16_configured_v2(
          stream, &desc, workspace, bytes, &config, sizeof(config));
  };
  auto cached = opState->prepared.find(key.str());
  if (cached != opState->prepared.end()) {
    result = cached->second;
    result.source = "session:prepared:" + result.source;
    rc = configured(result.config);
  } else {
    result = online ? hipdnn_ep::PagedAttentionResult{gqa_paged_default(desc),
                                                      "online"}
                    : hipdnn_ep::resolvePagedAttention(desc, key.str());
    if (result.source.rfind("lookup:", 0) == 0)
      rc = configured(result.config);
    else {
      if constexpr (Mrope) {
        rc = hip_paged_attention_mrope_f16_prepare_v1(
            stream, &rotary, workspace, bytes, &result.config,
            sizeof(result.config));
        result.source += "->autotune:mrope-production-prepare";
      } else {
        rc = hipdnn_ep::preparePagedAttention(stream, desc, workspace, bytes,
                                              key.str(), online, &result);
      }
      if (!rc) {
        int status = 0;
        if (hipMemcpyAsync(&status, workspace, sizeof(status),
                           hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)) != hipSuccess ||
            hipStreamSynchronize(static_cast<hipStream_t>(stream)) !=
                hipSuccess)
          return fail(HIP_PAGED_ATTENTION_HIP_ERROR);
        if (!status) {
          if (opState->prepared.size() >= 256)
            opState->prepared.erase(opState->prepared.begin());
          opState->prepared.emplace(key.str(), result);
        }
      }
    }
  }
  // prepare appends AND computes attention. Do not execute a second append.
  if (rc)
    return fail(rc);
  // Consume the native workspace status before returning the output to the
  // compiled graph. The inference protocol also observes the sticky error.
  int status = 0;
  if (hipMemcpyAsync(&status, workspace, sizeof(status), hipMemcpyDeviceToHost,
                     static_cast<hipStream_t>(stream)) != hipSuccess ||
      hipStreamSynchronize(static_cast<hipStream_t>(stream)) != hipSuccess)
    return fail(HIP_PAGED_ATTENTION_HIP_ERROR);
  if (status)
    return fail(status);
  RUNTIME_DEBUG_LOG("[paged-runtime] key=%s source=%s config=%d,%d,%d,%d,%d,%d "
                    "workspace=%zu\n",
                    key.str().c_str(), result.source.c_str(),
                    result.config.splits, result.config.bkv, result.config.hb,
                    result.config.qw, result.config.mt, result.config.nd,
                    bytes);
  return 0;
}

int wrap_paged_attention(
    RuntimeState *state, int32_t op_state_slot, const void *q, const void *k,
    const void *v, void *k_pool, void *v_pool, const int32_t *cum,
    const int32_t *past, const int32_t *table, const void *cos_cache,
    const void *sin_cache, const int32_t *metadata, void *output,
    int64_t tokens, int64_t key_tokens, int64_t value_tokens,
    int64_t output_tokens, int64_t batch, int64_t pages, int64_t value_pages,
    int64_t page_size, int64_t value_page_size, int64_t head_dim,
    int64_t table_batch, int64_t table_columns, int64_t cum_count,
    int64_t metadata_count, int64_t cos_rows, int64_t sin_rows,
    int64_t num_heads, int64_t kv_heads, int64_t window, int64_t do_rotary,
    int64_t rotary_interleaved, float scale, float softcap) {
  return wrapPagedAttentionImpl<false>(
      state, op_state_slot, q, k, v, k_pool, v_pool, cum, past, table,
      cos_cache, sin_cache, metadata, output, tokens, key_tokens, value_tokens,
      output_tokens, batch, pages, value_pages, page_size, value_page_size,
      head_dim, table_batch, table_columns, cum_count, metadata_count, cos_rows,
      sin_rows, num_heads, kv_heads, window, do_rotary, rotary_interleaved,
      scale, softcap, nullptr);
}

int wrap_paged_attention_mrope(
    RuntimeState *state, int32_t op_state_slot, const void *q, const void *k,
    const void *v, void *k_pool, void *v_pool, const int32_t *cum,
    const int32_t *past, const int32_t *table, const void *cos_cache,
    const void *sin_cache, const int32_t *metadata, void *output,
    int64_t tokens, int64_t key_tokens, int64_t value_tokens,
    int64_t output_tokens, int64_t batch, int64_t pages, int64_t value_pages,
    int64_t page_size, int64_t value_page_size, int64_t head_dim,
    int64_t table_batch, int64_t table_columns, int64_t cum_count,
    int64_t metadata_count, int64_t cos_rows, int64_t sin_rows,
    int64_t num_heads, int64_t kv_heads, int64_t window, int64_t do_rotary,
    int64_t rotary_interleaved, float scale, float softcap,
    const int64_t *position_ids, int64_t position_tokens, int64_t rotary_dim,
    int64_t section_t, int64_t section_h, int64_t section_w,
    int64_t mrope_layout, float mrope_scale) {
  const PagedMropeOptions mrope{position_ids, position_tokens, rotary_dim,
                                section_t,    section_h,       section_w,
                                mrope_layout, mrope_scale};
  return wrapPagedAttentionImpl<true>(
      state, op_state_slot, q, k, v, k_pool, v_pool, cum, past, table,
      cos_cache, sin_cache, metadata, output, tokens, key_tokens, value_tokens,
      output_tokens, batch, pages, value_pages, page_size, value_page_size,
      head_dim, table_batch, table_columns, cum_count, metadata_count, cos_rows,
      sin_rows, num_heads, kv_heads, window, do_rotary, rotary_interleaved,
      scale, softcap, &mrope);
}
