/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Emits GQA autotune LUT rows computed by the cost model, as JSON in the same
// shape as etc/gqa_autotune/*.json.
//
// The runtime resolves a config in two data tiers (exact key, then bucket key)
// and nothing else; there is no model on the dispatch path. This tool is how
// the model still earns its keep: it precomputes rows offline for geometries
// nobody measured, so those shapes hit the table instead of the heuristic.
//
//   gqa-lut-rows-from-model --arch gfx1151 --cus 20 \
//       --geometry 32:8:128 --geometry 64:8:64 \
//       --out rows.json
//
// Output is only the `entries` array. Review it, merge it into the arch JSON
// under a distinct model_key, and run flatc as documented in
// etc/gqa_autotune/README.md. Rows are marked Bucket: they are computed, not
// measured, so a measured Exact row always wins over them.
//
// Coverage is the point and the limit. A LUT answers the keys it contains, so
// the geometry list and the length ladder below decide what is covered; a shape
// outside them still falls through to the heuristic.

#include "gqa_cost_model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace {

struct Geometry {
  int num_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
};

struct Row {
  const char *phase = "Decode";
  int batch = 1, num_heads = 0, kv_heads = 0, head_dim = 0;
  int seq_q = 0, seq_kv = 0, max_seq = 0, local_window = 0;
  bool use_wmma = false;
  int splits = 0, m_tiles = 0, bkv = 0, nw = 0, mt = 0, nd = 0;
};

// Power-of-two ceilings, matching how the runtime builds a bucket key.
const int kLengthBuckets[] = {128,  256,  512,   1024,  2048,
                              4096, 8192, 16384, 32768, 65536};

bool parseGeometry(const std::string &text, Geometry *out) {
  // "<num_heads>:<kv_heads>:<head_dim>"
  const size_t a = text.find(':');
  if (a == std::string::npos)
    return false;
  const size_t b = text.find(':', a + 1);
  if (b == std::string::npos)
    return false;
  out->num_heads = std::atoi(text.substr(0, a).c_str());
  out->kv_heads = std::atoi(text.substr(a + 1, b - a - 1).c_str());
  out->head_dim = std::atoi(text.substr(b + 1).c_str());
  if (out->num_heads <= 0 || out->kv_heads <= 0 ||
      out->num_heads % out->kv_heads != 0)
    return false;
  return out->head_dim == 64 || out->head_dim == 128 || out->head_dim == 256;
}

hip_gqa_shape_t makeShape(const Geometry &g, int sq, int skv, int window,
                          int cus) {
  hip_gqa_shape_t shape{};
  shape.batch = 1;
  shape.num_heads = g.num_heads;
  shape.kv_heads = g.kv_heads;
  shape.head_dim = g.head_dim;
  shape.q_len = sq;
  shape.kv_len = skv;
  shape.window = window;
  shape.cu_count = cus;
  return shape;
}

// Pick the highest-scoring candidate, or report that the model declined.
bool bestDecode(const hip_gqa_shape_t &shape, int max_splits, bool *use_wmma,
                int *splits) {
  const int hpg = shape.num_heads / shape.kv_heads;
  const int eff = (shape.window > 0 && shape.window < shape.kv_len)
                      ? shape.window
                      : shape.kv_len;
  const int cap = std::max(1, std::min(max_splits, 64));
  const int max_useful = std::max(1, (eff + 15) / 16);
  const int min_splits = std::min(cap, std::max(1, (eff + 255) / 256));
  (void)hpg;

  static const int kLadder[] = {2, 4, 8, 16, 32, 48, 64};
  double best = 0.0;
  for (int wmma = 0; wmma <= 1; ++wmma) {
    for (const int base : kLadder) {
      const int s = std::min({base, cap, max_useful});
      if (s < min_splits)
        continue;
      hip_gqa_config_t cfg{};
      cfg.path = HIP_GQA_PATH_DECODE;
      cfg.use_wmma = wmma;
      cfg.splits = s;
      const double score = hip_gqa_config_score(&shape, &cfg);
      if (score > best) {
        best = score;
        *use_wmma = wmma != 0;
        *splits = s;
      }
    }
  }
  return best > 0.0;
}

bool bestPrefill(const hip_gqa_shape_t &shape, hip_gqa_config_t *out) {
  double best = 0.0;
  auto consider = [&](const hip_gqa_config_t &cfg) {
    const double score = hip_gqa_config_score(&shape, &cfg);
    if (score > best) {
      best = score;
      *out = cfg;
    }
  };

  if (shape.head_dim == 64) {
    for (const int mt : {1, 2})
      for (const int bkv : {32, 64}) {
        hip_gqa_config_t cfg{};
        cfg.path = HIP_GQA_PATH_PREFILL_V5;
        cfg.m_tiles = mt;
        cfg.bkv = bkv;
        consider(cfg);
      }
  } else if (shape.head_dim == 128) {
    for (const int nw : {1, 2, 4})
      for (const int bkv : {32, 64})
        for (const int mt : {1, 2}) {
          hip_gqa_config_t cfg{};
          cfg.path = HIP_GQA_PATH_PREFILL_V7;
          cfg.num_waves = nw;
          cfg.bkv = bkv;
          cfg.m_tiles = mt;
          consider(cfg);
        }
  } else {
    for (const int nd : {2, 4})
      for (const int mt : {1, 2})
        for (const int bkv : {32, 64}) {
          if (nd == 4 && bkv != 32)
            continue; // not instantiated
          hip_gqa_config_t cfg{};
          cfg.path = HIP_GQA_PATH_PREFILL_V8;
          cfg.num_waves = nd;
          cfg.m_tiles = mt;
          cfg.bkv = bkv;
          consider(cfg);
        }
  }
  return best > 0.0;
}

void writeRow(std::ostream &os, const Row &row, bool last) {
  os << "    {\n"
     << "      \"key\": {\n"
     << "        \"phase\": \"" << row.phase << "\",\n"
     << "        \"match\": \"Bucket\",\n"
     << "        \"kv_dtype\": \"Fp16\",\n"
     << "        \"batch\": " << row.batch << ",\n"
     << "        \"num_heads\": " << row.num_heads << ",\n"
     << "        \"kv_num_heads\": " << row.kv_heads << ",\n"
     << "        \"head_dim\": " << row.head_dim << ",\n"
     << "        \"seq_q\": " << row.seq_q << ",\n"
     << "        \"seq_kv\": " << row.seq_kv << ",\n"
     << "        \"max_seq\": " << row.max_seq << ",\n"
     << "        \"local_window\": " << row.local_window << "\n"
     << "      },\n"
     << "      \"config\": {\n"
     << "        \"use_wmma\": " << (row.use_wmma ? "true" : "false") << ",\n"
     << "        \"splits\": " << row.splits << ",\n"
     << "        \"m_tiles\": " << row.m_tiles << ",\n"
     << "        \"bkv\": " << row.bkv << ",\n"
     << "        \"nw\": " << row.nw << ",\n"
     << "        \"mt\": " << row.mt << ",\n"
     << "        \"nd\": " << row.nd << "\n"
     << "      }\n"
     << "    }" << (last ? "\n" : ",\n");
}

void usage(const char *argv0) {
  std::fprintf(
      stderr,
      "usage: %s --arch <gfx....> --geometry <H:G:d> [--geometry ...]\n"
      "          [--cus N] [--out <rows.json>] [--windows <w,w,...>]\n"
      "\n"
      "  --geometry  num_heads:kv_heads:head_dim, repeatable\n"
      "  --cus       device CU count the model should assume (default 20)\n"
      "  --windows   sliding windows to emit decode rows for; 0 = full\n"
      "              attention (default \"0\")\n",
      argv0);
}

} // namespace

int main(int argc, char **argv) {
  std::vector<Geometry> geometries;
  std::vector<int> windows;
  std::string arch, out_path;
  int cus = 20;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? argv[++i] : std::string();
    };
    if (arg == "--arch") {
      arch = next();
    } else if (arg == "--cus") {
      cus = std::atoi(next().c_str());
    } else if (arg == "--out") {
      out_path = next();
    } else if (arg == "--geometry") {
      Geometry g;
      if (!parseGeometry(next(), &g)) {
        std::fprintf(stderr,
                     "bad --geometry (want H:G:d, H%%G==0, d in 64/128/256)\n");
        return 2;
      }
      geometries.push_back(g);
    } else if (arg == "--windows") {
      std::string list = next();
      size_t pos = 0;
      while (!list.empty()) {
        const size_t comma = list.find(',', pos);
        windows.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos)
          break;
        pos = comma + 1;
      }
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (arch.empty() || geometries.empty()) {
    usage(argv[0]);
    return 2;
  }
  if (windows.empty())
    windows.push_back(0);

  std::vector<Row> rows;
  for (const Geometry &g : geometries) {
    for (const int window : windows) {
      // Decode: one row per KV bucket. seq_kv carries the effective length
      // after the window clamp, which is what the runtime keys on.
      for (const int kv : kLengthBuckets) {
        const int eff = (window > 0) ? std::min(window, kv) : kv;
        const hip_gqa_shape_t shape = makeShape(g, 1, kv, window, cus);
        bool use_wmma = false;
        int splits = 0;
        if (!bestDecode(shape, 64, &use_wmma, &splits))
          continue;
        Row row;
        row.phase = "Decode";
        row.num_heads = g.num_heads;
        row.kv_heads = g.kv_heads;
        row.head_dim = g.head_dim;
        row.seq_q = 1;
        row.seq_kv = eff;
        row.max_seq = kv;
        row.local_window = window > 0 ? window : 0;
        row.use_wmma = use_wmma;
        row.splits = splits;
        rows.push_back(row);
      }

      // Prefill: v5 keys on seq_q and the window only; v7/v8 also on seq_kv.
      if (window > 0 && g.head_dim != 64)
        continue; // the fused prefill declines a window at d != 64
      for (const int sq : kLengthBuckets) {
        for (const int kv : kLengthBuckets) {
          if (kv < sq)
            continue;
          const hip_gqa_shape_t shape = makeShape(g, sq, kv, window, cus);
          hip_gqa_config_t best{};
          if (!bestPrefill(shape, &best))
            continue;
          Row row;
          row.num_heads = g.num_heads;
          row.kv_heads = g.kv_heads;
          row.head_dim = g.head_dim;
          row.seq_q = sq;
          row.max_seq = kv;
          row.use_wmma = false;
          switch (g.head_dim) {
          case 64:
            row.phase = "PrefillV5";
            row.seq_kv = 0; // not part of the v5 key
            row.local_window = window > 0 ? window : 0;
            row.m_tiles = best.m_tiles;
            row.bkv = best.bkv;
            break;
          case 128:
            row.phase = "PrefillV7";
            row.seq_kv = kv;
            row.nw = best.num_waves;
            row.bkv = best.bkv;
            row.mt = best.m_tiles;
            break;
          default:
            row.phase = "PrefillV8";
            row.seq_kv = kv;
            row.nd = best.num_waves;
            row.mt = best.m_tiles;
            row.bkv = best.bkv;
            break;
          }
          rows.push_back(row);
          if (g.head_dim == 64)
            break; // v5 ignores seq_kv, so one row per seq_q is enough
        }
      }
    }
  }

  std::ofstream file;
  std::ostream *os = &std::cout;
  if (!out_path.empty()) {
    file.open(out_path);
    if (!file) {
      std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
      return 1;
    }
    os = &file;
  }

  *os << "{\n  \"_comment\": \"computed by gqa-lut-rows-from-model for " << arch
      << " (cus=" << cus << "); merge into the arch JSON and run flatc\",\n"
      << "  \"entries\": [\n";
  for (size_t i = 0; i < rows.size(); ++i)
    writeRow(*os, rows[i], i + 1 == rows.size());
  *os << "  ]\n}\n";

  std::fprintf(stderr, "%zu rows for %zu geometr%s\n", rows.size(),
               geometries.size(), geometries.size() == 1 ? "y" : "ies");
  return 0;
}
