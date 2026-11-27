/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "qmoe_expert_cache.h"

#include "../hipdnn_ep_runtime.h"
#include "../runtime_state_internal.h"
#include "hip/env.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

constexpr int kSegments = 6;
// hip_qmoe_bucket_tokens_ragged keeps a num_experts-entry LDS histogram and
// rejects more than this many experts.
constexpr int64_t kMaxPoolSlots = 1024;
constexpr size_t kAlign = 256;

size_t alignUp(size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

double secondsSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
      .count();
}

struct Pool {
  int64_t slots = 0;
  void *base = nullptr;
  char *seg[kSegments] = {};
  std::vector<int64_t> owner; // slot -> node * num_experts + expert, or -1
  std::vector<uint64_t> stamp;
  std::vector<int32_t> prev, next; // LRU list, head = most recently used
  int32_t head = -1, tail = -1;

  void unlink(int32_t s) {
    if (prev[s] >= 0) {
      next[prev[s]] = next[s];
    } else {
      head = next[s];
    }
    if (next[s] >= 0) {
      prev[next[s]] = prev[s];
    } else {
      tail = prev[s];
    }
  }
  void pushFront(int32_t s) {
    prev[s] = -1;
    next[s] = head;
    if (head >= 0) {
      prev[head] = s;
    }
    head = s;
    if (tail < 0) {
      tail = s;
    }
  }
};

struct PhaseStats {
  uint64_t calls = 0, hits = 0, misses = 0;
};

struct Cache {
  int64_t num_nodes = 0;
  int64_t num_experts = 0;
  int64_t seg_bytes[kSegments] = {};
  int64_t expert_bytes = 0;
  std::vector<std::array<int64_t, kSegments>> node_constants;
  std::vector<bool> node_registered;
  std::vector<int> node_pool;
  std::vector<Pool> pools;
  std::vector<int32_t> key_slot;
  uint64_t call = 0;

  int32_t *host_idx = nullptr;
  size_t host_idx_cap = 0;
  uint8_t *staging = nullptr;
  size_t staging_cap = 0;
  std::vector<std::pair<int32_t, int32_t>> misses; // (slot, expert)

  bool stats = false;
  PhaseStats decode, prefill;
  double t_d2h = 0, t_lookup = 0, t_read = 0, t_h2d = 0;
};

struct PrepareCtx {
  RuntimeState *state;
  Cache *cache;
  int64_t node;
};

void destroyCache(RuntimeState *state) {
  auto *c = static_cast<Cache *>(state->qmoe_expert_cache);
  if (!c) {
    return;
  }
  if (c->stats) {
    auto rate = [](const PhaseStats &p) {
      const uint64_t n = p.hits + p.misses;
      return n ? static_cast<double>(p.hits) / static_cast<double>(n) : 0.0;
    };
    fprintf(stderr,
            "[QMoE offload] decode calls=%llu hits=%llu misses=%llu "
            "hit_rate=%.4f miss_bytes=%llu\n",
            (unsigned long long)c->decode.calls,
            (unsigned long long)c->decode.hits,
            (unsigned long long)c->decode.misses, rate(c->decode),
            (unsigned long long)(c->decode.misses * c->expert_bytes));
    fprintf(stderr,
            "[QMoE offload] prefill calls=%llu hits=%llu misses=%llu "
            "hit_rate=%.4f miss_bytes=%llu\n",
            (unsigned long long)c->prefill.calls,
            (unsigned long long)c->prefill.hits,
            (unsigned long long)c->prefill.misses, rate(c->prefill),
            (unsigned long long)(c->prefill.misses * c->expert_bytes));
    fprintf(stderr,
            "[QMoE offload] time d2h+sync=%.3fs lookup=%.3fs read=%.3fs "
            "h2d_enqueue=%.3fs\n",
            c->t_d2h, c->t_lookup, c->t_read, c->t_h2d);
  }
  for (Pool &p : c->pools) {
    if (p.base) {
      (void)hipdnn_ep_state_device_free(state, p.base);
    }
  }
  if (c->host_idx) {
    (void)hipHostFree(c->host_idx);
  }
  if (c->staging) {
    (void)hipHostFree(c->staging);
  }
  delete c;
  state->qmoe_expert_cache = nullptr;
  state->qmoe_expert_cache_destroy = nullptr;
}

int growPinned(void **buf, size_t *cap, size_t need) {
  if (need <= *cap) {
    return 0;
  }
  if (*buf) {
    (void)hipHostFree(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  if (hipHostMalloc(buf, need, hipHostMallocDefault) != hipSuccess) {
    *buf = nullptr;
    return -1;
  }
  *cap = need;
  return 0;
}

Cache *createCache(RuntimeState *state, const int64_t constants[kSegments],
                   int64_t num_nodes, int64_t num_experts) {
  auto *c = new Cache();
  c->num_nodes = num_nodes;
  c->num_experts = num_experts;
  for (int j = 0; j < kSegments; ++j) {
    const int64_t idx = constants[j];
    if (idx < 0) {
      continue;
    }
    if (static_cast<size_t>(idx) >= state->num_constants ||
        !state->disk_constants[idx].data ||
        state->disk_constants[idx].size % num_experts != 0) {
      fprintf(stderr,
              "[QMoE offload] constant %lld is not a disk-resident expert "
              "table\n",
              (long long)idx);
      delete c;
      return nullptr;
    }
    c->seg_bytes[j] = state->disk_constants[idx].size / num_experts;
    c->expert_bytes += c->seg_bytes[j];
  }

  const int64_t all = num_nodes * num_experts;
  int64_t total = all;
  const std::string mb =
      hipdnn_ep::env_string("HIPDNN_EP_QMOE_EXPERT_CACHE_MB");
  if (!mb.empty()) {
    const int64_t bytes = static_cast<int64_t>(atoll(mb.c_str())) << 20;
    total = std::min(all, bytes / c->expert_bytes);
  }
  const int64_t n_pools = (total + kMaxPoolSlots - 1) / kMaxPoolSlots;
  if (n_pools <= 0 || n_pools > num_nodes || total / n_pools < num_experts) {
    fprintf(stderr,
            "[QMoE offload] %lld slots cannot give each pool at least %lld "
            "slots\n",
            (long long)total, (long long)num_experts);
    delete c;
    return nullptr;
  }

  c->node_constants.resize(num_nodes);
  c->node_registered.assign(num_nodes, false);
  c->node_pool.resize(num_nodes);
  c->key_slot.assign(all, -1);
  c->pools.resize(n_pools);
  int64_t node = 0;
  for (int64_t p = 0; p < n_pools; ++p) {
    const int64_t nodes_in_pool =
        num_nodes / n_pools + (p < num_nodes % n_pools ? 1 : 0);
    for (int64_t i = 0; i < nodes_in_pool; ++i) {
      c->node_pool[node++] = static_cast<int>(p);
    }
    Pool &pool = c->pools[p];
    pool.slots = total / n_pools + (p < total % n_pools ? 1 : 0);
    size_t bytes = 0;
    for (int j = 0; j < kSegments; ++j) {
      bytes += alignUp(static_cast<size_t>(pool.slots * c->seg_bytes[j]));
    }
    if (hipdnn_ep_state_device_alloc(state, &pool.base, bytes) != hipSuccess) {
      fprintf(stderr, "[QMoE offload] pool alloc of %zu bytes failed\n", bytes);
      pool.base = nullptr;
      state->qmoe_expert_cache = c;
      destroyCache(state);
      return nullptr;
    }
    char *cursor = static_cast<char *>(pool.base);
    for (int j = 0; j < kSegments; ++j) {
      if (c->seg_bytes[j] > 0) {
        pool.seg[j] = cursor;
      }
      cursor += alignUp(static_cast<size_t>(pool.slots * c->seg_bytes[j]));
    }
    pool.owner.assign(pool.slots, -1);
    pool.stamp.assign(pool.slots, 0);
    pool.prev.assign(pool.slots, -1);
    pool.next.assign(pool.slots, -1);
    for (int32_t s = static_cast<int32_t>(pool.slots) - 1; s >= 0; --s) {
      pool.pushFront(s);
    }
  }

  c->stats = hipdnn_ep::env_enabled("HIPDNN_EP_QMOE_EXPERT_STATS");
  if (c->stats) {
    fprintf(stderr,
            "[QMoE offload] nodes=%lld experts=%lld expert_bytes=%lld "
            "slots=%lld of %lld pools=%lld io=mmap\n",
            (long long)num_nodes, (long long)num_experts,
            (long long)c->expert_bytes, (long long)total, (long long)all,
            (long long)n_pools);
  }
  state->qmoe_expert_cache = c;
  state->qmoe_expert_cache_destroy = destroyCache;
  return c;
}

int registerNode(RuntimeState *state, Cache *c, int64_t node,
                 const int64_t constants[kSegments]) {
  if (node < 0 || node >= c->num_nodes) {
    return -1;
  }
  if (c->node_registered[node]) {
    return 0;
  }
  for (int j = 0; j < kSegments; ++j) {
    const int64_t idx = constants[j];
    const bool present = idx >= 0;
    if (present != (c->seg_bytes[j] > 0)) {
      return -1;
    }
    if (!present) {
      continue;
    }
    if (static_cast<size_t>(idx) >= state->num_constants ||
        !state->disk_constants || !state->disk_constants[idx].data ||
        state->disk_constants[idx].size != c->seg_bytes[j] * c->num_experts) {
      return -1;
    }
    c->node_constants[node][j] = idx;
  }
  c->node_registered[node] = true;
  return 0;
}

void prefetch(RuntimeState *state, Cache *c, int64_t node) {
#ifdef _WIN32
  std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
  ranges.reserve(c->misses.size() * kSegments);
  for (const auto &[slot, expert] : c->misses) {
    for (int j = 0; j < kSegments; ++j) {
      if (c->seg_bytes[j] == 0) {
        continue;
      }
      const DiskConstant &dc =
          state->disk_constants[c->node_constants[node][j]];
      ranges.push_back({const_cast<uint8_t *>(dc.data) +
                            static_cast<size_t>(expert) * c->seg_bytes[j],
                        static_cast<SIZE_T>(c->seg_bytes[j])});
    }
  }
  PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
#else
  (void)state;
  (void)c;
  (void)node;
#endif
}

int prepare(void *ctx_ptr, void *d_expert_indices, int64_t num_tokens,
            int64_t k, const void *pools_out[6]) {
  auto *ctx = static_cast<PrepareCtx *>(ctx_ptr);
  RuntimeState *state = ctx->state;
  Cache *c = ctx->cache;
  const int64_t node = ctx->node;
  Pool &pool = c->pools[c->node_pool[node]];
  hipStream_t stream = state->stream;
  const int64_t routes = num_tokens * k;
  const size_t idx_bytes = static_cast<size_t>(routes) * sizeof(int32_t);

  auto t = std::chrono::steady_clock::now();
  if (growPinned(reinterpret_cast<void **>(&c->host_idx), &c->host_idx_cap,
                 idx_bytes) != 0 ||
      hipMemcpyAsync(c->host_idx, d_expert_indices, idx_bytes,
                     hipMemcpyDeviceToHost, stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess) {
    fprintf(stderr, "[QMoE offload] expert index readback failed\n");
    return -1;
  }
  c->t_d2h += secondsSince(t);

  t = std::chrono::steady_clock::now();
  const uint64_t call = ++c->call;
  PhaseStats &phase = num_tokens == 1 ? c->decode : c->prefill;
  ++phase.calls;
  c->misses.clear();
  const int64_t E = c->num_experts;
  for (int64_t r = 0; r < routes; ++r) {
    const int32_t e = c->host_idx[r];
    if (e < 0 || e >= E) {
      if (e >= E) {
        c->host_idx[r] = static_cast<int32_t>(pool.slots);
      }
      continue;
    }
    const int64_t key = node * E + e;
    int32_t s = c->key_slot[key];
    if (s >= 0) {
      if (pool.stamp[s] != call) {
        ++phase.hits;
        pool.stamp[s] = call;
      }
    } else {
      s = pool.tail;
      while (pool.stamp[s] == call) {
        s = pool.prev[s];
      }
      if (pool.owner[s] >= 0) {
        c->key_slot[pool.owner[s]] = -1;
      }
      pool.owner[s] = key;
      pool.stamp[s] = call;
      c->key_slot[key] = s;
      c->misses.push_back({s, e});
      ++phase.misses;
    }
    pool.unlink(s);
    pool.pushFront(s);
    c->host_idx[r] = s;
  }
  c->t_lookup += secondsSince(t);

  const size_t nm = c->misses.size();
  if (nm > 0) {
    t = std::chrono::steady_clock::now();
    size_t seg_off[kSegments] = {};
    size_t need = 0;
    for (int j = 0; j < kSegments; ++j) {
      seg_off[j] = need;
      need += alignUp(nm * static_cast<size_t>(c->seg_bytes[j]));
    }
    if (growPinned(reinterpret_cast<void **>(&c->staging), &c->staging_cap,
                   need) != 0) {
      fprintf(stderr, "[QMoE offload] staging alloc of %zu bytes failed\n",
              need);
      return -1;
    }
    prefetch(state, c, node);
    for (int j = 0; j < kSegments; ++j) {
      const size_t sb = static_cast<size_t>(c->seg_bytes[j]);
      if (sb == 0) {
        continue;
      }
      const uint8_t *src =
          state->disk_constants[c->node_constants[node][j]].data;
      for (size_t m = 0; m < nm; ++m) {
        memcpy(c->staging + seg_off[j] + m * sb,
               src + static_cast<size_t>(c->misses[m].second) * sb, sb);
      }
    }
    c->t_read += secondsSince(t);

    t = std::chrono::steady_clock::now();
    for (size_t m = 0; m < nm;) {
      size_t run = 1;
      while (m + run < nm &&
             c->misses[m + run].first ==
                 c->misses[m].first + static_cast<int32_t>(run)) {
        ++run;
      }
      for (int j = 0; j < kSegments; ++j) {
        const size_t sb = static_cast<size_t>(c->seg_bytes[j]);
        if (sb == 0) {
          continue;
        }
        if (hipMemcpyAsync(pool.seg[j] +
                               static_cast<size_t>(c->misses[m].first) * sb,
                           c->staging + seg_off[j] + m * sb, run * sb,
                           hipMemcpyHostToDevice, stream) != hipSuccess) {
          fprintf(stderr, "[QMoE offload] expert upload failed\n");
          return -1;
        }
      }
      m += run;
    }
    c->t_h2d += secondsSince(t);
  }

  if (hipMemcpyAsync(d_expert_indices, c->host_idx, idx_bytes,
                     hipMemcpyHostToDevice, stream) != hipSuccess) {
    fprintf(stderr, "[QMoE offload] slot index upload failed\n");
    return -1;
  }
  for (int j = 0; j < kSegments; ++j) {
    pools_out[j] = pool.seg[j];
  }
  return 0;
}

} // namespace

int wrap_qmoe_offload(RuntimeState *state, const void *input,
                      const void *router_probs, const void *router_weights,
                      int64_t fc1_weights, int64_t fc1_scales, int64_t fc1_bias,
                      int64_t fc2_weights, int64_t fc2_scales, int64_t fc2_bias,
                      const void *fc3_weights, const void *fc3_scales,
                      const void *fc3_bias, const void *fc1_zero_points,
                      const void *fc2_zero_points, const void *fc3_zero_points,
                      void *output, int64_t num_tokens, int64_t hidden_size,
                      int64_t inter_size, int64_t num_experts, int64_t k,
                      int64_t expert_weight_bits, int64_t block_size,
                      int64_t swiglu_fusion, int64_t activation_type,
                      float activation_alpha, float activation_beta,
                      float swiglu_limit, int64_t normalize_routing_weights,
                      int64_t elem_size, const void *correction_bias,
                      int64_t routing_type, float routed_scaling_factor,
                      float routing_norm_eps, int64_t node, int64_t num_nodes) {
  if (!state || num_experts <= 0 || num_nodes <= 0 || !state->disk_constants) {
    fprintf(stderr, "wrap_qmoe_offload: invalid arguments\n");
    return -1;
  }
  const int64_t constants[kSegments] = {fc1_weights, fc1_scales, fc1_bias,
                                        fc2_weights, fc2_scales, fc2_bias};
  auto *c = static_cast<Cache *>(state->qmoe_expert_cache);
  if (!c) {
    c = createCache(state, constants, num_nodes, num_experts);
    if (!c) {
      return -1;
    }
  }
  if (c->num_nodes != num_nodes || c->num_experts != num_experts ||
      registerNode(state, c, node, constants) != 0) {
    fprintf(stderr,
            "wrap_qmoe_offload: node %lld does not match the expert cache "
            "layout\n",
            (long long)node);
    return -1;
  }
  PrepareCtx ctx{state, c, node};
  const QMoEExpertHook hook{c->pools[c->node_pool[node]].slots, prepare, &ctx};
  return hipdnn_ep_qmoe_run(
      state, input, router_probs, router_weights, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, fc3_weights, fc3_scales, fc3_bias,
      fc1_zero_points, fc2_zero_points, fc3_zero_points, output, num_tokens,
      hidden_size, inter_size, num_experts, k, expert_weight_bits, block_size,
      swiglu_fusion, activation_type, activation_alpha, activation_beta,
      swiglu_limit, normalize_routing_weights, elem_size, correction_bias,
      routing_type, routed_scaling_factor, routing_norm_eps, &hook);
}
