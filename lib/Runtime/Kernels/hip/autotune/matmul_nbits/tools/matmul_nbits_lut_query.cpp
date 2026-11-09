/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Reports how each shape in a CSV resolves, and the config it lands on.
 *
 * This is how the coverage targets are checked rather than assumed:
 *   - every oga_models shape should answer Exact
 *   - an industry model nobody measured should answer Nearest at a small
 *     distance; the distance histogram is the real coverage metric, because
 *     "Nearest" on its own is satisfied by a point on the far side of the space
 *   - Fallback should be reached only by a categorical key nothing was measured
 *     for (an unusual group size, or bits != 4)
 *
 * GPU-free. Input CSV needs K,N,block_size,has_zp columns (what
 * extract_shapes.py emits); --m selects the M values to query.
 */
#include "matmul_nbits_autotune.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace hipdnn_ep::matmul_nbits_autotune;

static bool acceptWmma(void*, const WmmaAnswer&) { return true; }
static bool acceptGemv(void*, const GemvAnswer&) { return true; }

// Use the same geometry legality as the dense U4 entry when reviewing a
// production nearest match. This GPU-free query checks geometry only; the
// product additionally checks device support and maxGridSize at runtime.
// Generic table dumps retain their existing policy.
static bool acceptProductionWmma(void* context, const WmmaAnswer& answer) {
  const Request& q = *static_cast<const Request*>(context);
  if (answer.bm <= 0 || answer.bn <= 0 || answer.wt_m <= 0 || answer.wt_n <= 0 || answer.bk <= 0)
    return false;
  const int threads = (answer.bm / (16 * answer.wt_m))
                    * (answer.bn / (16 * answer.wt_n)) * 32;
  if (threads <= 0 || (answer.bn * answer.bk) % threads) return false;
  const int nibbles = answer.bn * answer.bk / threads;
  return nibbles > 0 && q.group_size % nibbles == 0 && q.k % answer.bk == 0
      && (answer.ext || answer.bn >= 64 || q.n <= 2 * answer.bn)
      && (q.b_row_bytes == 0 || answer.fused);
}

static const char* name(Source s) {
  switch (s) {
  case Source::Exact: return "Exact";
  case Source::Nearest: return "Nearest";
  case Source::Fallback: return "Fallback";
  default: return "None";
  }
}

struct Row {
  int k, n, gs, zp;
  int m = 0, row_bytes = 0;
  Phase phase = Phase::Prefill;
  std::string model;
};

static std::vector<Row> load(const char* path, bool requests) {
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  std::string line;
  if (!std::getline(in, line)) return {};
  std::vector<std::string> head;
  { std::stringstream ss(line); std::string c;
    while (std::getline(ss, c, ',')) head.push_back(c); }
  auto col = [&](const char* n1, int dflt) {
    for (size_t i = 0; i < head.size(); ++i) if (head[i] == n1) return int(i);
    return dflt;
  };
  const int ck = col("K", -1), cn = col("N", -1), cb = col("block_size", -1),
            cz = col("has_zp", -1), cm = col("model", -1);
  const int c_m = col("M", -1), cp = col("phase", -1), cs = col("row_stride", -1);
  if (ck < 0 || cn < 0 || cb < 0 || cz < 0) {
    std::fprintf(stderr, "csv needs K,N,block_size,has_zp\n"); std::exit(1);
  }
  if (requests && (c_m < 0 || cp < 0 || cs < 0)) {
    std::fprintf(stderr, "request csv also needs M,phase,row_stride\n"); std::exit(1);
  }
  std::vector<Row> out;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> c;
    std::stringstream ss(line); std::string cell;
    while (std::getline(ss, cell, ',')) c.push_back(cell);
    if (int(c.size()) <= std::max(std::max(ck, cn), std::max(cb, cz))) continue;
    Row r;
    r.k = std::atoi(c[ck].c_str()); r.n = std::atoi(c[cn].c_str());
    r.gs = std::atoi(c[cb].c_str()); r.zp = std::atoi(c[cz].c_str());
    r.model = (cm >= 0 && cm < int(c.size())) ? c[cm] : "";
    if ((!requests && r.k % 32) || r.k <= 0 || r.n <= 0 || r.gs <= 0) continue;
    if (requests) {
      if (int(c.size()) <= std::max(c_m, std::max(cp, cs))) {
        std::fprintf(stderr, "incomplete request row\n"); std::exit(1);
      }
      r.m = std::atoi(c[c_m].c_str());
      if (c[cp] == "Prefill") r.phase = Phase::Prefill;
      else if (c[cp] == "Decode") r.phase = Phase::Decode;
      else if (c[cp] == "DecodeDp4a") r.phase = Phase::DecodeDp4a;
      else { std::fprintf(stderr, "invalid request phase\n"); std::exit(1); }
      if (c[cs] == "Padded")
        r.row_bytes = ((r.k + r.gs - 1) / r.gs) * (r.gs / 2) + 128;
      else if (c[cs] != "Arrival" && c[cs] != "Any") {
        std::fprintf(stderr, "invalid request stride\n"); std::exit(1);
      }
      if (r.m <= 0 || (r.phase == Phase::Prefill) == (c[cs] == "Any")) {
        std::fprintf(stderr, "invalid request M/phase/stride\n"); std::exit(1);
      }
    }
    out.push_back(r);
  }
  return out;
}

int main(int argc, char** argv) {
  const char* csv = nullptr;
  std::vector<int> ms = {1, 128, 512, 2048};
  bool verbose = false, per_model = false, dump = false, requests = false;
  bool production_wmma = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--shapes" && i + 1 < argc) csv = argv[++i];
    else if (a == "--requests" && i + 1 < argc) { csv = argv[++i]; requests = true; }
    else if (a == "--m" && i + 1 < argc) {
      ms.clear(); std::stringstream ss(argv[++i]); std::string t;
      while (std::getline(ss, t, ',')) ms.push_back(std::atoi(t.c_str()));
    } else if (a == "--verbose") verbose = true;
    else if (a == "--per-model") per_model = true;
    else if (a == "--production-wmma") production_wmma = true;
    // Machine-readable "shape -> resolved config", one per line. Diffing two
    // of these is how a table change is checked: adding measurements may move
    // which point answers, and this shows exactly which configs that changed.
    else if (a == "--dump") dump = true;
    else { std::fprintf(stderr,
        "usage: %s --shapes f.csv [--m 1,128] [--verbose] [--per-model] "
        "[--dump] [--requests exact-key-requests.csv] [--production-wmma]\n", argv[0]); return 1; }
  }
  if (!csv) { std::fprintf(stderr, "--shapes required\n"); return 1; }

  const Stats st = stats();
  std::printf("table_loaded=%d points=%u invalid=%u\n\n", int(st.table_loaded),
              st.points, st.invalid_points);

  const std::vector<Row> rows = load(csv, requests);
  std::map<std::string, std::map<std::string, int>> by_model;
  std::map<std::string, int> total;
  std::vector<float> distances;   // Nearest hits only; Exact is 0 by definition
  int worst_fallback = 0;

  for (const Row& r : rows) {
    for (int m : requests ? std::vector<int>{r.m} : ms) {
      // Phases a shape of this M would actually reach.
      std::vector<Phase> phases;
      if (requests) { phases = {r.phase}; }
      else if (m == 1) { phases = {Phase::Decode, Phase::DecodeDp4a}; }
      else { phases = {Phase::Prefill}; }
      for (Phase p : phases) {
        Request q;
        q.phase = p; q.bits = 4; q.m = m; q.n = r.n; q.k = r.k;
        q.group_size = r.gs; q.has_zp = r.zp != 0;
        q.b_row_bytes = r.row_bytes;
        if (production_wmma && p == Phase::Prefill) q.k = (q.k + 31) & ~31;
        const Result res = resolve(q,
            production_wmma && p == Phase::Prefill ? acceptProductionWmma : acceptWmma,
            acceptGemv, production_wmma ? &q : nullptr);
        const char* tier = name(res.source);
        ++total[tier];
        if (res.source == Source::Nearest) distances.push_back(res.distance);
        if (dump) {
          if (requests)
            std::printf("request phase=%d row_bytes=%d source=%d distance=%.9g ",
                        int(p), q.b_row_bytes, int(res.source), res.distance);
          // The source is deliberately not printed: adding measurements moves
          // which point answers, and that is allowed. Only the config matters.
          if (p == Phase::Prefill)
            std::printf("P m=%d n=%d k=%d gs=%d zp=%d -> %d %d %d %d %d %d %d ext=%d variant=%d\n",
                        m, r.n, r.k, r.gs, r.zp, res.wmma.bm, res.wmma.bn,
                        res.wmma.swizzle_n, res.wmma.wt_m, res.wmma.wt_n,
                        res.wmma.bk, int(res.wmma.fused), int(res.wmma.ext), res.wmma.variant);
          else
            std::printf("%s m=%d n=%d k=%d gs=%d zp=%d -> %d %d\n",
                        p == Phase::Decode ? "D" : "Q", m, r.n, r.k, r.gs,
                        r.zp, res.gemv.threads, res.gemv.tile_n);
        }
        if (per_model) ++by_model[r.model][tier];
        if (res.source == Source::Fallback || res.source == Source::None) {
          ++worst_fallback;
          if (verbose)
            std::printf("  %-9s M=%-5d N=%-7d K=%-6d gs=%-3d zp=%d  %s\n",
                        tier, m, r.n, r.k, r.gs, r.zp, r.model.c_str());
        } else if (verbose && res.source == Source::Nearest) {
          std::printf("  %-9s M=%-5d N=%-7d K=%-6d gs=%-3d zp=%d  d=%.3f  %s\n",
                      tier, m, r.n, r.k, r.gs, r.zp, res.distance,
                      r.model.c_str());
        }
      }
    }
  }

  if (per_model) {
    std::printf("%-64s %6s %6s %6s %6s\n", "model", "Exact", "Near", "Fallbk",
                "None");
    for (const auto& kv : by_model)
      std::printf("%-64s %6d %6d %6d %6d\n", kv.first.c_str(),
                  kv.second.count("Exact") ? kv.second.at("Exact") : 0,
                  kv.second.count("Nearest") ? kv.second.at("Nearest") : 0,
                  kv.second.count("Fallback") ? kv.second.at("Fallback") : 0,
                  kv.second.count("None") ? kv.second.at("None") : 0);
    std::printf("\n");
  }

  int sum = 0;
  for (const auto& kv : total) sum += kv.second;
  std::printf("%d queries: ", sum);
  for (const char* t : {"Exact", "Nearest", "Fallback", "None"}) {
    const int c = total.count(t) ? total[t] : 0;
    std::printf("%s=%d (%.1f%%) ", t, c, sum ? 100.0 * c / sum : 0.0);
  }
  std::printf("\n");

  // How far the non-exact answers had to reach. This is the number to watch
  // when deciding which shapes are worth adding to the sweep: a long tail here
  // is the table extrapolating, and each entry names a hole in the coverage.
  if (!distances.empty()) {
    std::sort(distances.begin(), distances.end());
    const size_t n = distances.size();
    std::printf("nearest distance (octaves): median %.3f  p90 %.3f  max %.3f\n",
                distances[n / 2], distances[size_t(n * 0.9)], distances[n - 1]);
  }
  return worst_fallback > 0 ? 2 : 0;
}
