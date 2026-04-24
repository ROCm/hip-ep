/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

// Simple per-op, per-shape profiler for MorphizenEP runtime.
// Always-on. Each wrap_* call is timed (with stream sync for GPU accuracy).
// Raw per-invocation data is written to perf_profile.csv at process exit
// with parsed dimension columns for easy filtering.

#include "debug_log.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct RuntimeState;
typedef struct ihipStream_t *hipStream_t;

extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *state);

// ---------------------------------------------------------------------------
// PerfRecord -- one measurement per wrap_* invocation
// ---------------------------------------------------------------------------

struct PerfRecord {
  char op[32];
  char shape[128];
  double ms;
};

// ---------------------------------------------------------------------------
// Shape parser: extract key=value pairs into named columns
// ---------------------------------------------------------------------------

struct ParsedDims {
  int64_t M, N, K, batch, bits, blk;
  int64_t n, mode, rows, hidden;
  int64_t b, sq, skv, h, d;
  int64_t heads, rot;
  int64_t data, idx, out;
  int64_t out_n, out_c, out_h, out_w;
  int64_t op_code, bytes;
  int64_t src, dst;
  int64_t ch, seq, tok, experts;
  int64_t tA, tB;

  ParsedDims() { memset(this, 0, sizeof(*this)); }

  // Parse "M=128,N=4096,K=4096,bits=4,blk=128" into fields
  void parse(const char *shape) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", shape);
    // Manual tokenizer (no strtok dependency)
    char *p = buf;
    while (*p) {
      char *key_start = p;
      char *eq = nullptr;
      // Find '=' or ',' or end
      while (*p && *p != '=' && *p != ',') p++;
      if (*p == '=') {
        *p = '\0';
        eq = p;
        p++;
        int64_t val = strtoll(p, &p, 10);
        set(key_start, val);
      }
      // Skip comma
      if (*p == ',') p++;
    }
  }

private:
  void set(const char *key, int64_t val) {
    if (strcmp(key, "M") == 0)          M = val;
    else if (strcmp(key, "N") == 0)     N = val;
    else if (strcmp(key, "K") == 0)     K = val;
    else if (strcmp(key, "batch") == 0) batch = val;
    else if (strcmp(key, "bits") == 0)  bits = val;
    else if (strcmp(key, "blk") == 0)   blk = val;
    else if (strcmp(key, "n") == 0)     n = val;
    else if (strcmp(key, "mode") == 0)  mode = val;
    else if (strcmp(key, "rows") == 0)  rows = val;
    else if (strcmp(key, "hidden") == 0) hidden = val;
    else if (strcmp(key, "b") == 0)     b = val;
    else if (strcmp(key, "sq") == 0)    sq = val;
    else if (strcmp(key, "skv") == 0)   skv = val;
    else if (strcmp(key, "h") == 0)     h = val;
    else if (strcmp(key, "d") == 0)     d = val;
    else if (strcmp(key, "heads") == 0) heads = val;
    else if (strcmp(key, "rot") == 0)   rot = val;
    else if (strcmp(key, "data") == 0)  data = val;
    else if (strcmp(key, "idx") == 0)   idx = val;
    else if (strcmp(key, "out") == 0)   out = val;
    else if (strcmp(key, "out_n") == 0) out_n = val;
    else if (strcmp(key, "out_c") == 0) out_c = val;
    else if (strcmp(key, "out_h") == 0) out_h = val;
    else if (strcmp(key, "out_w") == 0) out_w = val;
    else if (strcmp(key, "op") == 0)    op_code = val;
    else if (strcmp(key, "bytes") == 0) bytes = val;
    else if (strcmp(key, "src") == 0)   src = val;
    else if (strcmp(key, "dst") == 0)   dst = val;
    else if (strcmp(key, "ch") == 0)    ch = val;
    else if (strcmp(key, "seq") == 0)   seq = val;
    else if (strcmp(key, "tok") == 0)   tok = val;
    else if (strcmp(key, "experts") == 0) experts = val;
    else if (strcmp(key, "tA") == 0)    tA = val;
    else if (strcmp(key, "tB") == 0)    tB = val;
  }
};

// ---------------------------------------------------------------------------
// PerfState -- collects all records, writes CSV at process exit
// ---------------------------------------------------------------------------

class PerfState {
  std::vector<PerfRecord> records_;

public:
  void record(const char *op, const char *shape, double ms) {
    PerfRecord r;
    snprintf(r.op, sizeof(r.op), "%s", op);
    snprintf(r.shape, sizeof(r.shape), "%s", shape);
    r.ms = ms;
    records_.push_back(r);
  }

  void write_csv() {
    if (records_.empty())
      return;

    FILE *csv = fopen("perf_profile.csv", "w");
    if (!csv)
      return;

    // Header: raw fields + parsed dimension columns
    fprintf(csv, "Index,Op,Shape,Time_ms,"
                 "M,N,K,batch,bits,blk,"
                 "n,mode,rows,hidden,"
                 "b,sq,skv,h,d,"
                 "heads,rot,"
                 "data,idx,out,"
                 "out_n,out_c,out_h,out_w,"
                 "op_code,bytes,"
                 "src,dst,"
                 "ch,seq,tok,experts,"
                 "tA,tB\n");

    for (size_t i = 0; i < records_.size(); i++) {
      auto &r = records_[i];
      ParsedDims p;
      p.parse(r.shape);

      fprintf(csv,
              "%zu,%s,\"%s\",%.4f,"
              "%lld,%lld,%lld,%lld,%lld,%lld,"
              "%lld,%lld,%lld,%lld,"
              "%lld,%lld,%lld,%lld,%lld,"
              "%lld,%lld,"
              "%lld,%lld,%lld,"
              "%lld,%lld,%lld,%lld,"
              "%lld,%lld,"
              "%lld,%lld,"
              "%lld,%lld,%lld,%lld,"
              "%lld,%lld\n",
              i, r.op, r.shape, r.ms,
              (long long)p.M, (long long)p.N, (long long)p.K,
              (long long)p.batch, (long long)p.bits, (long long)p.blk,
              (long long)p.n, (long long)p.mode,
              (long long)p.rows, (long long)p.hidden,
              (long long)p.b, (long long)p.sq, (long long)p.skv,
              (long long)p.h, (long long)p.d,
              (long long)p.heads, (long long)p.rot,
              (long long)p.data, (long long)p.idx, (long long)p.out,
              (long long)p.out_n, (long long)p.out_c,
              (long long)p.out_h, (long long)p.out_w,
              (long long)p.op_code, (long long)p.bytes,
              (long long)p.src, (long long)p.dst,
              (long long)p.ch, (long long)p.seq,
              (long long)p.tok, (long long)p.experts,
              (long long)p.tA, (long long)p.tB);
    }

    fclose(csv);
    fprintf(stderr, "[PERF] CSV written: perf_profile.csv (%zu records)\n",
            records_.size());
  }

  ~PerfState() { write_csv(); }

  static PerfState *get() {
    if (!instance_) {
      instance_ = new PerfState();
      std::atexit(cleanup);
    }
    return instance_;
  }

private:
  static PerfState *instance_;
  static void cleanup() {
    if (instance_) {
      delete instance_;
      instance_ = nullptr;
    }
  }
};

// Defined in profiler.cpp (single definition across all bitcode modules)

// ---------------------------------------------------------------------------
// RAII Timer -- syncs stream when profiling to get accurate GPU time
// ---------------------------------------------------------------------------

class PerfTimer {
  std::chrono::steady_clock::time_point start_;
  hipStream_t stream_;
  char op_[32];
  char shape_[128];
  bool active_;

public:
  PerfTimer(RuntimeState *state, const char *op, const char *shape)
      : active_(true) {
    if (!active_)
      return;
    snprintf(op_, sizeof(op_), "%s", op);
    snprintf(shape_, sizeof(shape_), "%s", shape);
    stream_ = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
    start_ = std::chrono::steady_clock::now();
  }

  ~PerfTimer();
};

// ---------------------------------------------------------------------------
// Macro for easy instrumentation
// ---------------------------------------------------------------------------

#define PERF_TIMER(state, op, shape_fmt, ...)                                  \
  char _perf_shape_[128];                                                      \
  snprintf(_perf_shape_, sizeof(_perf_shape_), shape_fmt, ##__VA_ARGS__);      \
  PerfTimer _perf_timer_(state, op, _perf_shape_)
