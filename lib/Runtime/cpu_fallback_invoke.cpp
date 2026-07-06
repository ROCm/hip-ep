/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "cpu_fallback_invoke.h"
#include "runtime_state_internal.h"

#include <cstdio>
#include <new>
#include <vector>

namespace {

static constexpr int64_t kMaxTensorBytes = 2LL * 1024 * 1024 * 1024;

static bool shape_dims_ok(const int64_t *shape, int64_t rank) {
  if (rank <= 0)
    return true;
  if (!shape)
    return false;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] < 0)
      return false;
  }
  return true;
}

static int64_t elem_bytes(int64_t hip_dtype) {
  return hipdnn_ep_datatype_size(hip_dtype);
}

} // namespace

int hipdnn_cpu_fallback_try_generic(RuntimeState *state, void *stream,
                                    const char *env_token,
                                    const HipdnnCpuFbGenericDesc *desc) {
  if (!state || !desc || !desc->op_name || !state->cpu_fallback.invoke)
    return 1;
  if (!hipdnn_ep_debug_cpu_fallback_ops_contains(env_token))
    return 1;

  if (desc->op_name && std::strcmp(desc->op_name, "Gather") == 0) {
    int64_t in_bytes = 0;
    for (int32_t i = 0; i < desc->num_inputs; ++i) {
      const int64_t eb = elem_bytes(desc->inputs[i].hip_dtype);
      if (eb > 0 && desc->inputs[i].num_elements > 0) {
        in_bytes += desc->inputs[i].num_elements * eb;
      }
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] cpu_fallback Gather: staging D2H (~%lld bytes input)\n",
        static_cast<long long>(in_bytes));
  }

  if (desc->num_inputs < 0 || desc->num_outputs < 0 ||
      desc->num_inputs + desc->num_outputs > HIPDNN_CPU_FB_MAX_IO ||
      desc->num_attrs < 0 || desc->num_attrs > HIPDNN_CPU_FB_MAX_ATTRS) {
    RUNTIME_DEBUG_LOG("[REAL] cpu_fallback %s: bad io/attr counts\n",
                      desc->op_name);
    return 1;
  }

  for (int32_t i = 0; i < desc->num_inputs; ++i) {
    const auto &t = desc->inputs[i];
    if (!t.device && t.num_elements > 0) {
      RUNTIME_DEBUG_LOG("[REAL] cpu_fallback %s: null input %d\n", desc->op_name,
                        i);
      return 1;
    }
    if (!shape_dims_ok(t.shape, t.rank))
      return 1;
    const int64_t eb = elem_bytes(t.hip_dtype);
    if (eb < 0 || t.num_elements < 0)
      return 1;
    if (t.num_elements > 0 && t.num_elements > kMaxTensorBytes / eb) {
      RUNTIME_DEBUG_LOG(
          "[REAL] cpu_fallback %s: input %d too large for host staging\n",
          desc->op_name, i);
      return 1;
    }
  }
  for (int32_t i = 0; i < desc->num_outputs; ++i) {
    const auto &t = desc->outputs[i];
    if (!t.device && t.num_elements > 0)
      return 1;
    if (!shape_dims_ok(t.shape, t.rank))
      return 1;
    const int64_t eb = elem_bytes(t.hip_dtype);
    if (eb < 0 || t.num_elements < 0)
      return 1;
    if (t.num_elements > 0 && t.num_elements > kMaxTensorBytes / eb) {
      RUNTIME_DEBUG_LOG(
          "[REAL] cpu_fallback %s: output %d too large for host staging\n",
          desc->op_name, i);
      return 1;
    }
  }

  if (wrap_hipStreamSynchronize(stream) != 0)
    return -1;

  std::vector<std::vector<char>> host_inputs(
      static_cast<size_t>(desc->num_inputs));
  std::vector<std::vector<char>> host_outputs(
      static_cast<size_t>(desc->num_outputs));
  try {
    for (int32_t i = 0; i < desc->num_inputs; ++i) {
      const int64_t bytes =
          desc->inputs[i].num_elements * elem_bytes(desc->inputs[i].hip_dtype);
      if (bytes > 0)
        host_inputs[static_cast<size_t>(i)].resize(
            static_cast<size_t>(bytes));
    }
    for (int32_t i = 0; i < desc->num_outputs; ++i) {
      const int64_t bytes =
          desc->outputs[i].num_elements *
          elem_bytes(desc->outputs[i].hip_dtype);
      if (bytes > 0)
        host_outputs[static_cast<size_t>(i)].resize(
            static_cast<size_t>(bytes));
    }
  } catch (const std::bad_alloc &) {
    RUNTIME_DEBUG_LOG("[REAL] cpu_fallback %s: host staging OOM\n",
                      desc->op_name);
    return 1;
  }

  for (int32_t i = 0; i < desc->num_inputs; ++i) {
    const int64_t bytes =
        desc->inputs[i].num_elements * elem_bytes(desc->inputs[i].hip_dtype);
    if (bytes <= 0)
      continue;
    if (wrap_hipMemcpyD2H(host_inputs[static_cast<size_t>(i)].data(),
                          desc->inputs[i].device, bytes, stream) != 0)
      return -1;
  }
  if (wrap_hipStreamSynchronize(stream) != 0)
    return -1;

  HipdnnCpuFbGenericHostDesc host_desc{};
  host_desc.op_name = desc->op_name;
  host_desc.domain = desc->domain ? desc->domain : "";
  host_desc.opset = desc->opset > 0 ? desc->opset : 13;
  host_desc.num_inputs = desc->num_inputs;
  host_desc.num_outputs = desc->num_outputs;
  host_desc.num_attrs = desc->num_attrs;

  for (int32_t i = 0; i < desc->num_inputs; ++i) {
    host_desc.inputs[i].host = host_inputs[static_cast<size_t>(i)].empty()
                                   ? nullptr
                                   : host_inputs[static_cast<size_t>(i)].data();
    host_desc.inputs[i].rank = desc->inputs[i].rank;
    host_desc.inputs[i].shape = desc->inputs[i].shape;
    host_desc.inputs[i].num_elements = desc->inputs[i].num_elements;
    host_desc.inputs[i].hip_dtype = desc->inputs[i].hip_dtype;
  }
  for (int32_t i = 0; i < desc->num_outputs; ++i) {
    host_desc.outputs[i].host_mut =
        host_outputs[static_cast<size_t>(i)].empty()
            ? nullptr
            : host_outputs[static_cast<size_t>(i)].data();
    host_desc.outputs[i].rank = desc->outputs[i].rank;
    host_desc.outputs[i].shape = desc->outputs[i].shape;
    host_desc.outputs[i].num_elements = desc->outputs[i].num_elements;
    host_desc.outputs[i].hip_dtype = desc->outputs[i].hip_dtype;
  }
  for (int32_t i = 0; i < desc->num_attrs; ++i) {
    host_desc.attrs[i] = desc->attrs[i];
  }

  const int fb_rc =
      state->cpu_fallback.invoke(state->cpu_fallback.user, state,
                               HIPDNN_CPU_FB_OP_GENERIC, &host_desc,
                               sizeof(host_desc));
  if (fb_rc != 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] cpu_fallback %s: EP invoke failed rc=%d; using GPU kernel\n",
        desc->op_name, fb_rc);
    return 1;
  }

  for (int32_t i = 0; i < desc->num_outputs; ++i) {
    const int64_t bytes =
        desc->outputs[i].num_elements * elem_bytes(desc->outputs[i].hip_dtype);
    if (bytes <= 0)
      continue;
    if (wrap_hipMemcpyH2D(desc->outputs[i].device,
                          host_outputs[static_cast<size_t>(i)].data(), bytes,
                          stream) != 0)
      return -1;
  }
  return 0;
}
