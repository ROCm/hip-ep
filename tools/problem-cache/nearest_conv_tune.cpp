/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "nearest_conv_tune.h"

#include "problem_cache_generated.h"

#include "flatbuffers/flatbuffers.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// Keep after windows.h.
#include <aclapi.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mlir {
namespace hip {
namespace {

struct ConvFields {
  std::string arch;
  std::string kind;
  int64_t f = 1;
  std::string filterLayout;
  std::string inputLayout;
  std::string outputLayout;
  int64_t n = 1, c = 1, h = 1, w = 1, k = 1;
  int64_t y = 1, x = 1, p = 0, q = 0, u = 1, v = 1, l = 1, j = 1, g = 1;
  int64_t cu = 0, chiplets = 1;
  std::string inputFusions;
  std::string outputFusions;
  std::string supportsSplitK;
};

struct GemmShape {
  int64_t m = 1;
  int64_t n = 1;
  int64_t k = 1;
};

std::string trimCopy(std::string s) {
  auto isTrim = [](unsigned char c) {
    return std::isspace(c) != 0 || c == '"';
  };
  while (!s.empty() && isTrim(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && isTrim(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

std::optional<int64_t> parseFlag(std::string_view s, std::string_view flag) {
  auto pos = s.find(flag);
  if (pos == std::string_view::npos)
    return std::nullopt;
  pos += flag.size();
  while (pos < s.size() &&
         std::isspace(static_cast<unsigned char>(s[pos])) != 0)
    ++pos;
  if (pos >= s.size())
    return std::nullopt;
  try {
    size_t consumed = 0;
    int64_t value = std::stoll(std::string(s.substr(pos)), &consumed);
    if (consumed == 0)
      return std::nullopt;
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::string> parseWordFlag(std::string_view s,
                                         std::string_view flag) {
  auto pos = s.find(flag);
  if (pos == std::string_view::npos)
    return std::nullopt;
  pos += flag.size();
  while (pos < s.size() &&
         std::isspace(static_cast<unsigned char>(s[pos])) != 0)
    ++pos;
  if (pos >= s.size())
    return std::nullopt;
  auto end = s.find(' ', pos);
  if (end == std::string_view::npos)
    return std::string(s.substr(pos));
  return std::string(s.substr(pos, end - pos));
}

std::optional<std::string> parseEqualsFlag(std::string_view s,
                                           std::string_view flag) {
  auto pos = s.find(flag);
  if (pos == std::string_view::npos)
    return std::nullopt;
  pos += flag.size();
  if (pos >= s.size())
    return std::string();
  auto end = s.find(' ', pos);
  if (end == std::string_view::npos)
    return std::string(s.substr(pos));
  return std::string(s.substr(pos, end - pos));
}

std::optional<ConvFields> parseConvProblem(std::string key) {
  key = trimCopy(std::move(key));
  if (key.empty())
    return std::nullopt;

  auto convPos = key.find("conv");
  if (convPos == std::string::npos)
    return std::nullopt;
  const std::string conv = key.substr(convPos);
  auto kindEnd = conv.find(' ');
  ConvFields fields;
  fields.kind = kindEnd == std::string::npos ? conv : conv.substr(0, kindEnd);
  if (fields.kind.compare(0, 4, "conv") != 0)
    return std::nullopt;

  std::string prefix = key.substr(0, convPos);
  std::vector<std::string> tabs;
  std::string token;
  for (char ch : prefix) {
    if (ch == '\t') {
      if (!token.empty())
        tabs.push_back(token);
      token.clear();
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty())
    tabs.push_back(token);
  if (!tabs.empty() && tabs[0].compare(0, 3, "gfx") == 0)
    fields.arch = tabs[0];
  if (tabs.size() > 1) {
    try {
      fields.cu = std::stoll(tabs[1]);
    } catch (...) {
      fields.cu = 0;
    }
  }
  if (tabs.size() > 2) {
    try {
      fields.chiplets = std::stoll(tabs[2]);
    } catch (...) {
      fields.chiplets = 1;
    }
  }

  fields.f = parseFlag(conv, "-F").value_or(1);
  fields.filterLayout = parseWordFlag(conv, "-f ").value_or("");
  fields.inputLayout = parseWordFlag(conv, "-I ").value_or("");
  fields.outputLayout = parseWordFlag(conv, "-O ").value_or("");
  fields.n = parseFlag(conv, "-n ").value_or(1);
  fields.c = parseFlag(conv, "-c ").value_or(1);
  fields.h = parseFlag(conv, "-H ").value_or(1);
  fields.w = parseFlag(conv, "-W ").value_or(1);
  fields.k = parseFlag(conv, "-k ").value_or(1);
  fields.y = parseFlag(conv, "-y ").value_or(1);
  fields.x = parseFlag(conv, "-x ").value_or(1);
  fields.p = parseFlag(conv, "-p ").value_or(0);
  fields.q = parseFlag(conv, "-q ").value_or(0);
  fields.u = parseFlag(conv, "-u ").value_or(1);
  fields.v = parseFlag(conv, "-v ").value_or(1);
  fields.l = parseFlag(conv, "-l ").value_or(1);
  fields.j = parseFlag(conv, "-j ").value_or(1);
  fields.g = parseFlag(conv, "-g ").value_or(1);
  fields.inputFusions = parseEqualsFlag(conv, "-inputFusions=").value_or("");
  fields.outputFusions = parseEqualsFlag(conv, "-outputFusions=").value_or("");
  fields.supportsSplitK = parseWordFlag(conv, "-supportsSplitK ").value_or("");
  return fields;
}

int64_t outDim(int64_t in, int64_t pad, int64_t ksz, int64_t stride,
               int64_t dil) {
  const int64_t eff = dil * (ksz - 1) + 1;
  return (in + 2 * pad - eff) / std::max<int64_t>(stride, 1) + 1;
}

GemmShape gemmShape(const ConvFields &problem) {
  const int64_t ho = std::max<int64_t>(
      outDim(problem.h, problem.p, problem.y, problem.u, problem.l), 1);
  const int64_t wo = std::max<int64_t>(
      outDim(problem.w, problem.q, problem.x, problem.v, problem.j), 1);
  const int64_t groups = std::max<int64_t>(problem.g, 1);
  GemmShape shape;
  // One group is one implicit GEMM: M is the output channels of that group.
  shape.m = std::max<int64_t>(problem.k / groups, 1);
  shape.n = std::max<int64_t>(problem.n * ho * wo, 1);
  shape.k = std::max<int64_t>((problem.c / groups) * problem.y * problem.x, 1);
  return shape;
}

bool sameGroup(const ConvFields &query, const ConvFields &row) {
  if (query.kind != row.kind || query.f != row.f)
    return false;
  if (!query.arch.empty() && !row.arch.empty() && query.arch != row.arch)
    return false;
  if (query.filterLayout != row.filterLayout ||
      query.inputLayout != row.inputLayout ||
      query.outputLayout != row.outputLayout)
    return false;
  if (query.u != row.u || query.v != row.v || query.l != row.l ||
      query.j != row.j)
    return false;
  return query.inputFusions == row.inputFusions &&
         query.outputFusions == row.outputFusions &&
         query.supportsSplitK == row.supportsSplitK;
}

// The distances are doubles end to end. A weight that fits a float, up to
// about 3.4e38 or down to about 1.4e-45, would overflow or underflow the same
// arithmetic in float: with weight_m = 1e20 every row would be infinitely far
// from the query, and the "keep the earlier row on a tie" rule would then pick
// the first row instead of the nearest.
double gemmDistance(const GemmShape &query, const GemmShape &row,
                    double weightM, double weightN, double weightK) {
  auto term = [](double weight, int64_t a, int64_t b) {
    const double delta = weight * (std::log2(static_cast<double>(a)) -
                                   std::log2(static_cast<double>(b)));
    return delta * delta;
  };
  return std::sqrt(term(weightM, query.m, row.m) +
                   term(weightN, query.n, row.n) +
                   term(weightK, query.k, row.k));
}

bool isConvSolution(std::string_view solution) {
  return solution.compare(0, 5, "gemm:") == 0 ||
         solution.compare(0, 5, "attn:") == 0;
}

struct FusionFields {
  std::string inputFusions;
  std::string outputFusions;
  std::string supportsSplitK;
};

struct Axes {
  int64_t v[4] = {1, 1, 1, 1};
  int n = 3;
};

struct BlockTile {
  bool known = false;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t n1 = 0;
  bool hasN1 = false;
};

struct ProblemPrefix {
  std::string arch;
  std::string body;
};

ProblemPrefix splitPrefix(std::string key) {
  key = trimCopy(std::move(key));
  ProblemPrefix out;
  std::string fields[3];
  std::string token;
  int tabs = 0;
  size_t i = 0;
  for (; i < key.size() && tabs < 3; ++i) {
    if (key[i] == '\t') {
      fields[tabs++] = token;
      token.clear();
    } else {
      token.push_back(key[i]);
    }
  }
  if (tabs < 3) {
    out.body = key;
    return out;
  }
  if (fields[0].compare(0, 3, "gfx") == 0)
    out.arch = fields[0];
  out.body = trimCopy(key.substr(i));
  return out;
}

FusionFields parseFusions(std::string_view body) {
  FusionFields fusions;
  fusions.inputFusions = parseEqualsFlag(body, "-inputFusions=").value_or("");
  fusions.outputFusions = parseEqualsFlag(body, "-outputFusions=").value_or("");
  fusions.supportsSplitK = parseWordFlag(body, "-supportsSplitK ").value_or("");
  return fusions;
}

bool sameArch(const std::string &query, const std::string &row) {
  return query.empty() || row.empty() || query == row;
}

bool sameWord(const std::optional<std::string> &query,
              const std::optional<std::string> &row) {
  if (!query || !row)
    return !query && !row;
  return *query == *row;
}

bool sameOptionalInt(const std::optional<int64_t> &query,
                     const std::optional<int64_t> &row) {
  if (!query || !row)
    return !query && !row;
  return *query == *row;
}

bool sameFusions(const FusionFields &query, const FusionFields &row) {
  return query.inputFusions == row.inputFusions &&
         query.outputFusions == row.outputFusions &&
         query.supportsSplitK == row.supportsSplitK;
}

bool parseIntList(std::string_view text, std::vector<int64_t> &out) {
  size_t i = 0;
  while (i < text.size()) {
    try {
      size_t consumed = 0;
      int64_t value = std::stoll(std::string(text.substr(i)), &consumed);
      if (consumed == 0)
        return !out.empty();
      out.push_back(value);
      i += consumed;
      if (i < text.size() && text[i] == ',')
        ++i;
      else
        break;
    } catch (...) {
      return !out.empty();
    }
  }
  return !out.empty();
}

bool parsePositional(std::string_view solution, std::string_view prefix,
                     int &version, std::vector<int64_t> &nums) {
  auto pos = solution.find(prefix);
  if (pos == std::string_view::npos)
    return false;
  std::string_view rest = solution.substr(pos + prefix.size());
  if (rest.size() < 3 || rest[0] != 'v' ||
      std::isdigit(static_cast<unsigned char>(rest[1])) == 0)
    return false;
  version = 0;
  size_t i = 1;
  while (i < rest.size() &&
         std::isdigit(static_cast<unsigned char>(rest[i])) != 0) {
    version = version * 10 + (rest[i] - '0');
    ++i;
  }
  if (i >= rest.size() || rest[i] != ':')
    return false;
  return parseIntList(rest.substr(i + 1), nums);
}

// Named `mPerBlock=` / `mPerBlockG0=` and positional `gemm:vN:` / `attn:vN:`.
// v1..v5 attention lists are m, n, k with nPerBlockG1 untiled. v6 inserts
// nPerBlockG1 as the third field. An unrecognized spelling is left unknown
// so rocMLIR still judges it.
BlockTile parseBlockTile(std::string_view solution, bool gemmGemm) {
  BlockTile tile;
  if (gemmGemm) {
    auto m = parseFlag(solution, "mPerBlockG0=");
    auto n = parseFlag(solution, "nPerBlockG0=");
    auto k = parseFlag(solution, "kPerBlock=");
    if (m && n && k) {
      tile.known = true;
      tile.m = *m;
      tile.n = *n;
      tile.k = *k;
      if (auto n1 = parseFlag(solution, "nPerBlockG1=")) {
        tile.hasN1 = true;
        tile.n1 = *n1;
      }
      return tile;
    }
  } else {
    auto m = parseFlag(solution, "mPerBlock=");
    auto n = parseFlag(solution, "nPerBlock=");
    auto k = parseFlag(solution, "kPerBlock=");
    if (m && n && k) {
      tile.known = true;
      tile.m = *m;
      tile.n = *n;
      tile.k = *k;
      return tile;
    }
  }

  int version = 0;
  std::vector<int64_t> nums;
  const char *prefix = gemmGemm ? "attn:" : "gemm:";
  if (!parsePositional(solution, prefix, version, nums))
    return tile;
  if (!gemmGemm) {
    if (nums.size() < 3)
      return tile;
    tile.known = true;
    tile.m = nums[0];
    tile.n = nums[1];
    tile.k = nums[2];
    return tile;
  }
  if (version >= 6) {
    if (nums.size() < 4)
      return tile;
    tile.known = true;
    tile.m = nums[0];
    tile.n = nums[1];
    tile.n1 = nums[2];
    tile.k = nums[3];
    tile.hasN1 = true;
    return tile;
  }
  if (nums.size() < 3)
    return tile;
  tile.known = true;
  tile.m = nums[0];
  tile.n = nums[1];
  tile.k = nums[2];
  tile.hasN1 = true;
  tile.n1 = 0;
  return tile;
}

bool tileFits(const BlockTile &tile, int64_t m, int64_t n, int64_t k,
              int64_t n1Limit, bool checkN1) {
  if (!tile.known)
    return true;
  if (tile.m > m || tile.n > n || tile.k > k)
    return false;
  return !checkN1 || !tile.hasN1 || tile.n1 <= 0 || tile.n1 <= n1Limit;
}

// Nearest-neighbor guard only. An exact problem string skips this: rocMLIR
// compiles a covering tile larger than the dimension (the next power of two
// on K, a padded M or N block). A spelling with no block sizes is left to
// rocMLIR. Named and positional gemm: tiles are both checked, and a
// non-positive block does not fit.
bool tileLegal(const ConvFields &problem, std::string_view solution) {
  const BlockTile tile = parseBlockTile(solution, /*gemmGemm=*/false);
  if (!tile.known)
    return true;
  if (tile.m <= 0 || tile.n <= 0 || tile.k <= 0)
    return false;
  const GemmShape shape = gemmShape(problem);
  return tileFits(tile, shape.m, shape.n, shape.k, 0, false);
}

double axesDistance(const Axes &query, const Axes &row, double weightM,
                    double weightN, double weightK) {
  const double weights[4] = {weightM, weightN, weightK, weightN};
  double sum = 0;
  for (int i = 0; i < query.n; ++i) {
    const double delta =
        weights[i] * (std::log2(static_cast<double>(query.v[i])) -
                      std::log2(static_cast<double>(row.v[i])));
    sum += delta * delta;
  }
  return std::sqrt(sum);
}

bool axesEqual(const Axes &query, const Axes &row) {
  for (int i = 0; i < query.n; ++i) {
    if (query.v[i] != row.v[i])
      return false;
  }
  return true;
}

struct GemmFields {
  std::string arch;
  std::string dtype;
  std::optional<std::string> outDatatype;
  std::optional<std::string> transA;
  std::optional<std::string> transB;
  std::optional<std::string> transO;
  bool scaled = false;
  std::optional<std::string> scaleA;
  std::optional<std::string> scaleB;
  std::optional<std::string> transScaleA;
  std::optional<std::string> transScaleB;
  std::optional<int64_t> g;
  int64_t m = 1;
  int64_t n = 1;
  int64_t k = 1;
  FusionFields fusions;
};

struct AttnFields {
  std::string arch;
  std::string dtype;
  std::optional<std::string> transQ;
  std::optional<std::string> transK;
  std::optional<std::string> transV;
  std::optional<std::string> transO;
  std::optional<std::string> causal;
  std::optional<std::string> returnLse;
  std::optional<int64_t> splitKv;
  std::optional<int64_t> sliding;
  std::optional<int64_t> numHeadsQ;
  std::optional<int64_t> numHeadsKv;
  std::optional<int64_t> g;
  int64_t seqQ = 1;
  int64_t seqK = 1;
  int64_t headQk = 1;
  int64_t headV = 1;
  std::optional<std::string> withScale;
  std::optional<std::string> withBias;
  std::optional<std::string> transBias;
  FusionFields fusions;
};

struct GegFields {
  std::string arch;
  std::string dtype;
  std::optional<std::string> transA;
  std::optional<std::string> transB;
  std::optional<std::string> transC;
  std::optional<std::string> transO;
  std::optional<int64_t> g;
  int64_t m = 1;
  int64_t n = 1;
  int64_t k = 1;
  int64_t gemmO = 1;
  FusionFields fusions;
};

std::optional<GemmFields> parseGemmProblem(std::string key) {
  ProblemPrefix split = splitPrefix(std::move(key));
  const std::string &body = split.body;
  if (body.find("-gemmO") != std::string::npos ||
      body.find("-seq_len_q") != std::string::npos)
    return std::nullopt;
  auto dtype = parseWordFlag(body, "-t ");
  auto m = parseFlag(body, "-m ");
  auto n = parseFlag(body, "-n ");
  auto k = parseFlag(body, "-k ");
  if (!dtype || !m || !n || !k)
    return std::nullopt;
  GemmFields fields;
  fields.arch = split.arch;
  fields.dtype = *dtype;
  fields.outDatatype = parseWordFlag(body, "-out_datatype ");
  fields.transA = parseWordFlag(body, "-transA ");
  fields.transB = parseWordFlag(body, "-transB ");
  fields.transO = parseWordFlag(body, "-transO ");
  fields.scaled = body.find("-scaledGemm") != std::string::npos;
  if (fields.scaled) {
    fields.scaleA = parseWordFlag(body, "-scale_a_dtype ");
    fields.scaleB = parseWordFlag(body, "-scale_b_dtype ");
    fields.transScaleA = parseWordFlag(body, "-transScaleA ");
    fields.transScaleB = parseWordFlag(body, "-transScaleB ");
  }
  fields.g = parseFlag(body, "-g ");
  fields.m = std::max<int64_t>(*m, 1);
  fields.n = std::max<int64_t>(*n, 1);
  fields.k = std::max<int64_t>(*k, 1);
  fields.fusions = parseFusions(body);
  return fields;
}

bool sameGemmGroup(const GemmFields &query, const GemmFields &row) {
  return sameArch(query.arch, row.arch) && query.dtype == row.dtype &&
         sameWord(query.outDatatype, row.outDatatype) &&
         sameWord(query.transA, row.transA) &&
         sameWord(query.transB, row.transB) &&
         sameWord(query.transO, row.transO) && query.scaled == row.scaled &&
         sameWord(query.scaleA, row.scaleA) &&
         sameWord(query.scaleB, row.scaleB) &&
         sameWord(query.transScaleA, row.transScaleA) &&
         sameWord(query.transScaleB, row.transScaleB) &&
         sameOptionalInt(query.g, row.g) &&
         sameFusions(query.fusions, row.fusions);
}

Axes gemmAxes(const GemmFields &fields) {
  Axes axes;
  axes.n = 3;
  axes.v[0] = fields.m;
  axes.v[1] = fields.n;
  axes.v[2] = fields.k;
  return axes;
}

bool gemmTileLegal(const GemmFields &query, std::string_view solution) {
  return tileFits(parseBlockTile(solution, false), query.m, query.n, query.k, 0,
                  false);
}

std::optional<AttnFields> parseAttentionProblem(std::string key) {
  ProblemPrefix split = splitPrefix(std::move(key));
  const std::string &body = split.body;
  if (body.find("-seq_len_q") == std::string::npos &&
      body.find("-transQ") == std::string::npos)
    return std::nullopt;
  auto dtype = parseWordFlag(body, "-t ");
  auto seqQ = parseFlag(body, "-seq_len_q ");
  auto seqK = parseFlag(body, "-seq_len_k ");
  auto headQk = parseFlag(body, "-head_dim_qk ");
  auto headV = parseFlag(body, "-head_dim_v ");
  if (!dtype || !seqQ || !seqK || !headQk || !headV)
    return std::nullopt;
  AttnFields fields;
  fields.arch = split.arch;
  fields.dtype = *dtype;
  fields.transQ = parseWordFlag(body, "-transQ ");
  fields.transK = parseWordFlag(body, "-transK ");
  fields.transV = parseWordFlag(body, "-transV ");
  fields.transO = parseWordFlag(body, "-transO ");
  fields.causal = parseWordFlag(body, "-causal ");
  fields.returnLse = parseWordFlag(body, "-return_lse ");
  fields.splitKv = parseFlag(body, "-split_kv ");
  fields.sliding = parseFlag(body, "-sliding_window_look_back ");
  fields.numHeadsQ = parseFlag(body, "-num_heads_q ");
  fields.numHeadsKv = parseFlag(body, "-num_heads_kv ");
  fields.g = parseFlag(body, "-g ");
  fields.seqQ = std::max<int64_t>(*seqQ, 1);
  fields.seqK = std::max<int64_t>(*seqK, 1);
  fields.headQk = std::max<int64_t>(*headQk, 1);
  fields.headV = std::max<int64_t>(*headV, 1);
  fields.withScale = parseWordFlag(body, "-with-attn-scale ");
  fields.withBias = parseWordFlag(body, "-with-attn-bias ");
  fields.transBias = parseWordFlag(body, "-transBias ");
  fields.fusions = parseFusions(body);
  return fields;
}

bool sameAttentionGroup(const AttnFields &query, const AttnFields &row) {
  return sameArch(query.arch, row.arch) && query.dtype == row.dtype &&
         sameWord(query.transQ, row.transQ) &&
         sameWord(query.transK, row.transK) &&
         sameWord(query.transV, row.transV) &&
         sameWord(query.transO, row.transO) &&
         sameWord(query.causal, row.causal) &&
         sameWord(query.returnLse, row.returnLse) &&
         sameOptionalInt(query.splitKv, row.splitKv) &&
         sameOptionalInt(query.sliding, row.sliding) &&
         sameOptionalInt(query.numHeadsQ, row.numHeadsQ) &&
         sameOptionalInt(query.numHeadsKv, row.numHeadsKv) &&
         sameOptionalInt(query.g, row.g) && query.headQk == row.headQk &&
         query.headV == row.headV && sameWord(query.withScale, row.withScale) &&
         sameWord(query.withBias, row.withBias) &&
         sameWord(query.transBias, row.transBias) &&
         sameFusions(query.fusions, row.fusions);
}

Axes attentionAxes(const AttnFields &fields) {
  Axes axes;
  axes.n = 2;
  axes.v[0] = fields.seqQ;
  axes.v[1] = fields.seqK;
  return axes;
}

bool attentionTileLegal(const AttnFields &query, std::string_view solution) {
  return tileFits(parseBlockTile(solution, true), query.seqQ, query.seqK,
                  query.headQk, query.headV, true);
}

std::optional<GegFields> parseGegProblem(std::string key) {
  ProblemPrefix split = splitPrefix(std::move(key));
  const std::string &body = split.body;
  if (body.find("-gemmO") == std::string::npos ||
      body.find("-transA") == std::string::npos)
    return std::nullopt;
  auto dtype = parseWordFlag(body, "-t ");
  auto m = parseFlag(body, "-m ");
  auto n = parseFlag(body, "-n ");
  auto k = parseFlag(body, "-k ");
  auto gemmO = parseFlag(body, "-gemmO ");
  if (!dtype || !m || !n || !k || !gemmO)
    return std::nullopt;
  GegFields fields;
  fields.arch = split.arch;
  fields.dtype = *dtype;
  fields.transA = parseWordFlag(body, "-transA ");
  fields.transB = parseWordFlag(body, "-transB ");
  fields.transC = parseWordFlag(body, "-transC ");
  fields.transO = parseWordFlag(body, "-transO ");
  fields.g = parseFlag(body, "-g ");
  fields.m = std::max<int64_t>(*m, 1);
  fields.n = std::max<int64_t>(*n, 1);
  fields.k = std::max<int64_t>(*k, 1);
  fields.gemmO = std::max<int64_t>(*gemmO, 1);
  fields.fusions = parseFusions(body);
  return fields;
}

bool sameGegGroup(const GegFields &query, const GegFields &row) {
  return sameArch(query.arch, row.arch) && query.dtype == row.dtype &&
         sameWord(query.transA, row.transA) &&
         sameWord(query.transB, row.transB) &&
         sameWord(query.transC, row.transC) &&
         sameWord(query.transO, row.transO) &&
         sameOptionalInt(query.g, row.g) &&
         sameFusions(query.fusions, row.fusions);
}

Axes gegAxes(const GegFields &fields) {
  Axes axes;
  axes.n = 4;
  axes.v[0] = fields.m;
  axes.v[1] = fields.n;
  axes.v[2] = fields.k;
  axes.v[3] = fields.gemmO;
  return axes;
}

bool gegTileLegal(const GegFields &query, std::string_view solution) {
  return tileFits(parseBlockTile(solution, true), query.m, query.n, query.k,
                  query.gemmO, true);
}

template <typename Fields>
std::optional<NearestConvTuneHit>
searchNearest(std::string_view problemKey, const Fields &query,
              const std::vector<ProblemCacheConvEntry> &entries,
              const std::vector<std::string> &rejectedSolutions, double weightM,
              double weightN, double weightK,
              std::optional<Fields> (*parse)(std::string),
              bool (*sameGroupOf)(const Fields &, const Fields &),
              Axes (*axesOf)(const Fields &),
              bool (*legal)(const Fields &, std::string_view)) {
  const Axes queryAxes = axesOf(query);
  const ProblemCacheConvEntry *best = nullptr;
  Fields bestFields{};
  double bestDistance = 0;
  for (const auto &entry : entries) {
    if (!isConvSolution(entry.solution) || entry.problem == problemKey)
      continue;
    if (std::find(rejectedSolutions.begin(), rejectedSolutions.end(),
                  entry.solution) != rejectedSolutions.end())
      continue;
    auto fields = parse(entry.problem);
    if (!fields || !sameGroupOf(query, *fields) ||
        !legal(query, entry.solution))
      continue;
    const double distance =
        axesDistance(queryAxes, axesOf(*fields), weightM, weightN, weightK);
    if (best != nullptr && distance >= bestDistance)
      continue;
    best = &entry;
    bestFields = *fields;
    bestDistance = distance;
  }
  if (best == nullptr)
    return std::nullopt;
  NearestConvTuneHit hit;
  hit.solution = best->solution;
  hit.matchedProblem = best->problem;
  hit.distance = bestDistance;
  hit.exact = axesEqual(queryAxes, axesOf(bestFields));
  return hit;
}

} // namespace

bool loadProblemCacheLut(const uint8_t *data, size_t size, ProblemCacheLut &lut,
                         std::string &error) {
  lut = {};
  if (data == nullptr || size == 0) {
    error = "problem cache is empty";
    return false;
  }
  flatbuffers::Verifier verifier(data, size);
  namespace fbs = hipdnn_ep::rocmlir_autotune::fbs;
  if (!fbs::VerifyRocmlirAutotuneLutBuffer(verifier)) {
    error = "problem cache is not an RCAL flatbuffer";
    return false;
  }
  const auto *table = fbs::GetRocmlirAutotuneLut(data);
  if (table->schema_version() != 1) {
    error = "unsupported rocmlir LUT schema version " +
            std::to_string(table->schema_version());
    return false;
  }
  if (table->kernel_abi() == nullptr ||
      table->kernel_abi()->str() != "rocmlir-conv-v1") {
    error = "unsupported rocmlir LUT kernel_abi";
    return false;
  }
  // A weight has to be a finite positive float: an infinite one makes the
  // distance NaN (infinity * 0) for a row that shares the axis, and then the
  // comparison that picks the nearest row accepts every row in turn.
  const auto usable = [](float weight) {
    return std::isfinite(weight) && weight > 0.0f;
  };
  if (!usable(table->weight_m()) || !usable(table->weight_n()) ||
      !usable(table->weight_k())) {
    error = "rocmlir LUT has a non-positive or non-finite distance weight";
    return false;
  }
  if (table->gpu_arch() != nullptr)
    lut.gpuArch = table->gpu_arch()->str();
  lut.weightM = table->weight_m();
  lut.weightN = table->weight_n();
  lut.weightK = table->weight_k();
  const auto *configs = table->configs();
  const auto *points = table->points();
  if (configs == nullptr || points == nullptr)
    return true;
  for (const auto *point : *points) {
    if (point == nullptr || point->problem() == nullptr)
      continue;
    if (point->config() >= configs->size())
      continue;
    const auto *config = configs->Get(point->config());
    if (config == nullptr || config->perf_config() == nullptr)
      continue;
    std::string solution = config->perf_config()->str();
    if (!isConvSolution(solution))
      continue;
    lut.entries.push_back({point->problem()->str(), std::move(solution)});
  }
  return true;
}

namespace {

// Parses \p text as a LUT JSON object. On success \p root points into
// \p document, which must outlive it.
bool parseLutObject(std::string_view text, llvm::json::Value &document,
                    const llvm::json::Object *&root, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(llvm::StringRef(text.data(), text.size()));
  if (!parsed) {
    error = "problem cache is not valid JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  document = std::move(*parsed);
  root = document.getAsObject();
  if (root == nullptr) {
    error = "problem cache JSON must be an object";
    return false;
  }
  return true;
}

// Reads the rest of \p in into \p text with stream-level calls only. On Linux
// a directory opens as an ifstream but cannot be read; istream::read turns that
// into badbit, whereas an istreambuf_iterator over the same stream throws
// std::ios_base::failure. Returns false if the read failed.
bool readStreamText(std::ifstream &in, std::string &text) {
  char buffer[1 << 14];
  while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0)
    text.append(buffer, static_cast<size_t>(in.gcount()));
  return !in.bad();
}

// Same checks as the FlatBuffer overload, on the fields every LUT carries. An
// absent schema_version or weight takes the problem_cache.fbs default; a
// present one of the wrong type is an error rather than a silent default.
// Fills the header fields of \p lut (arch and weights), not its entries.
bool readLutHeaderJson(const llvm::json::Object &root, ProblemCacheLut &lut,
                       std::string &error) {
  int64_t schemaVersion = 1;
  if (root.get("schema_version") != nullptr) {
    std::optional<int64_t> value = root.getInteger("schema_version");
    if (!value) {
      error = "rocmlir LUT schema_version is not an integer";
      return false;
    }
    schemaVersion = *value;
  }
  if (schemaVersion != 1) {
    error = "unsupported rocmlir LUT schema version " +
            std::to_string(schemaVersion);
    return false;
  }
  std::optional<llvm::StringRef> kernelAbi = root.getString("kernel_abi");
  if (!kernelAbi || *kernelAbi != "rocmlir-conv-v1") {
    error = "unsupported rocmlir LUT kernel_abi";
    return false;
  }
  const auto readWeight = [&](const char *name, float &out) {
    if (root.get(name) == nullptr)
      return true;
    std::optional<double> value = root.getNumber(name);
    if (!value)
      return false;
    // The weight is stored as a float, so it has to be a finite positive
    // float. 1e50 is finite as a double but outside the float range, and
    // converting it is undefined (in practice infinity). 1e-50 is positive as a
    // double but rounds to zero, which the FlatBuffer loader rejects. Either
    // would corrupt the distance (an infinite or a dropped axis), so check the
    // range before converting and the converted value after.
    // The lower end matters as much as the upper one: -1e300 is finite and
    // outside the float range, and converting it is just as undefined.
    if (!std::isfinite(*value) || !(*value > 0.0) ||
        *value > static_cast<double>(std::numeric_limits<float>::max()))
      return false;
    const float narrowed = static_cast<float>(*value);
    if (!std::isfinite(narrowed) || !(narrowed > 0.0f))
      return false;
    out = narrowed;
    return true;
  };
  float weightM = 1.0f, weightN = 1.0f, weightK = 1.0f;
  if (!readWeight("weight_m", weightM) || !readWeight("weight_n", weightN) ||
      !readWeight("weight_k", weightK)) {
    error = "rocmlir LUT has a non-positive or non-finite distance weight";
    return false;
  }
  // The `.fbs` types: rocm_version is an int32 and model_key a string. A
  // document that holds anything else could not be compiled to a `.fb`.
  if (root.get("rocm_version") != nullptr) {
    std::optional<int64_t> version = root.getInteger("rocm_version");
    if (!version || *version < std::numeric_limits<int32_t>::min() ||
        *version > std::numeric_limits<int32_t>::max()) {
      error = "rocmlir LUT rocm_version is not a 32-bit integer";
      return false;
    }
  }
  if (root.get("model_key") != nullptr && !root.getString("model_key")) {
    error = "rocmlir LUT model_key is not a string";
    return false;
  }
  if (root.get("gpu_arch") != nullptr) {
    // A gpu_arch that is not a string would otherwise read as absent, and an
    // absent arch matches every GPU.
    std::optional<llvm::StringRef> arch = root.getString("gpu_arch");
    if (!arch) {
      error = "rocmlir LUT gpu_arch is not a string";
      return false;
    }
    lut.gpuArch = arch->str();
  }
  lut.weightM = weightM;
  lut.weightN = weightN;
  lut.weightK = weightK;
  return true;
}

// Limits of the `.fbs` types the document is compiled to.
constexpr size_t kMaxConfigs = 65536;
constexpr int64_t kMaxDimension = std::numeric_limits<uint32_t>::max();

// A dimension worked out from a problem string. One that does not fit the
// table's uint32 is stored as 0, which the `.fbs` already uses for "not
// recognized", rather than written out as a value the converter rejects.
int64_t storableDimension(int64_t value) {
  return value >= 0 && value <= kMaxDimension ? value : 0;
}

// The rows of a LUT document. The objects point into the parsed document,
// which must outlive them.
struct LutPointRecord {
  const llvm::json::Object *object = nullptr;
  std::string problem;
  size_t config = 0;
};
struct LutFallbackRecord {
  const llvm::json::Object *object = nullptr;
  size_t config = 0;
};
struct LutRows {
  std::vector<std::string> perfConfigs;
  std::vector<LutPointRecord> points;
  std::vector<LutFallbackRecord> fallbacks;
  bool hasFallbacks = false;
};

// Reads `configs`, `points` and `fallbacks` and checks every record. The
// loader and the rewrite both read the rows through here, so they accept and
// refuse the same documents: a record that cannot be read is an error, never
// skipped. A point whose perfConfig is not gemm:/attn: is still a valid row;
// only the lookup ignores it.
bool readLutRows(const llvm::json::Object &root, LutRows &rows,
                 std::string &error) {
  const auto malformed = [&error](const std::string &what) {
    error = "problem cache is malformed: " + what;
    return false;
  };
  const llvm::json::Array *configs = root.getArray("configs");
  const llvm::json::Array *points = root.getArray("points");
  if (configs == nullptr)
    return malformed("`configs` is missing or not an array");
  if (points == nullptr)
    return malformed("`points` is missing or not an array");

  // `config` is a uint16 in the `.fbs`, so a table can name 65536 configs.
  if (configs->size() > kMaxConfigs)
    return malformed("`configs` has " + std::to_string(configs->size()) +
                     " entries; at most " + std::to_string(kMaxConfigs) +
                     " fit the table format");
  rows.perfConfigs.reserve(configs->size());
  for (size_t i = 0; i < configs->size(); ++i) {
    const llvm::json::Object *object = (*configs)[i].getAsObject();
    std::optional<llvm::StringRef> perfConfig =
        object != nullptr ? object->getString("perf_config") : std::nullopt;
    if (!perfConfig)
      return malformed("configs[" + std::to_string(i) +
                       "] is not an object with a string `perf_config`");
    rows.perfConfigs.push_back(perfConfig->str());
  }

  const auto configIndexOf = [&](const llvm::json::Object &object,
                                 const std::string &where, size_t &config) {
    std::optional<int64_t> index = object.getInteger("config");
    if (!index)
      return malformed(where + " has no integer `config`");
    if (*index < 0 || static_cast<size_t>(*index) >= rows.perfConfigs.size())
      return malformed(where + " has `config` " + std::to_string(*index) +
                       ", outside the " +
                       std::to_string(rows.perfConfigs.size()) + " configs");
    config = static_cast<size_t>(*index);
    return true;
  };
  rows.points.reserve(points->size());
  for (size_t i = 0; i < points->size(); ++i) {
    const std::string where = "points[" + std::to_string(i) + "]";
    LutPointRecord record;
    record.object = (*points)[i].getAsObject();
    if (record.object == nullptr)
      return malformed(where + " is not an object");
    std::optional<llvm::StringRef> problem =
        record.object->getString("problem");
    if (!problem)
      return malformed(where + " has no string `problem`");
    record.problem = problem->str();
    if (!configIndexOf(*record.object, where, record.config))
      return false;
    // m, n and k are uint32 in the `.fbs`.
    for (const char *dim : {"m", "n", "k"}) {
      const llvm::json::Value *value = record.object->get(dim);
      if (value == nullptr)
        continue;
      std::optional<int64_t> number = value->getAsInteger();
      if (!number || *number < 0 || *number > kMaxDimension)
        return malformed(where + " has an `m`, `n` or `k` that is not an "
                                 "integer from 0 to 4294967295");
    }
    rows.points.push_back(std::move(record));
  }

  // `fallbacks` is optional, but one that is present is index-linked to
  // `configs` like the points.
  if (const llvm::json::Value *value = root.get("fallbacks")) {
    const llvm::json::Array *array = value->getAsArray();
    if (array == nullptr)
      return malformed("`fallbacks` is not an array");
    rows.hasFallbacks = true;
    for (size_t i = 0; i < array->size(); ++i) {
      const std::string where = "fallbacks[" + std::to_string(i) + "]";
      LutFallbackRecord record;
      record.object = (*array)[i].getAsObject();
      if (record.object == nullptr)
        return malformed(where + " is not an object");
      if (!configIndexOf(*record.object, where, record.config))
        return false;
      rows.fallbacks.push_back(record);
    }
  }
  return true;
}

} // namespace

bool loadProblemCacheLutJson(std::string_view text, ProblemCacheLut &lut,
                             std::string &error) {
  lut = {};
  if (text.empty()) {
    error = "problem cache is empty";
    return false;
  }
  llvm::json::Value document(nullptr);
  const llvm::json::Object *root = nullptr;
  if (!parseLutObject(text, document, root, error))
    return false;
  if (!readLutHeaderJson(*root, lut, error)) {
    lut = {};
    return false;
  }

  LutRows rows;
  if (!readLutRows(*root, rows, error)) {
    lut = {};
    return false;
  }
  for (const LutPointRecord &point : rows.points) {
    const std::string &solution = rows.perfConfigs[point.config];
    if (isConvSolution(solution))
      lut.entries.push_back({point.problem, solution});
  }
  return true;
}

namespace {

// Serializes rewrites within one process. The file is read, merged and
// renamed over, so two compiles that overlap would otherwise each start from
// the same old table and the second rename would drop the first one's
// winners. Every read of an external table takes it too: on Windows an open
// std::ifstream does not share delete access, so a read that overlaps the
// rename would make the replacement fail and lose that compile's winners.
// Separate processes are not covered.
std::mutex &rewriteMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

bool loadProblemCacheLutJsonFile(const std::string &path, ProblemCacheLut &lut,
                                 std::string &error) {
  std::lock_guard<std::mutex> lock(rewriteMutex());
  {
    std::error_code ec;
    const std::filesystem::file_status status =
        std::filesystem::status(path, ec);
    if (!ec && std::filesystem::exists(status) &&
        !std::filesystem::is_regular_file(status)) {
      lut = {};
      error = "problem cache " + path + " is not a regular file";
      return false;
    }
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    lut = {};
    error = "problem cache not found: " + path;
    return false;
  }
  std::string text;
  if (!readStreamText(in, text)) {
    lut = {};
    error = "failed reading problem cache: " + path;
    return false;
  }
  if (text.size() >= 8 &&
      hipdnn_ep::rocmlir_autotune::fbs::RocmlirAutotuneLutBufferHasIdentifier(
          text.data())) {
    lut = {};
    error = path + " is an RCAL FlatBuffer (.fb), not the LUT JSON";
    return false;
  }
  if (!loadProblemCacheLutJson(text, lut, error)) {
    error = path + ": " + error;
    return false;
  }
  return true;
}

namespace {

// One row of the table being rewritten.
struct StoredPoint {
  std::string problem;
  std::string perfConfig;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

// One fallback record of the document being rewritten. `record` is the whole
// object, so fields this code does not know survive; only its `config` index is
// replaced when the configs are renumbered.
struct StoredFallback {
  llvm::json::Object record;
  std::string perfConfig;
};

bool isBlank(std::string_view text) {
  return std::all_of(text.begin(), text.end(),
                     [](unsigned char c) { return std::isspace(c) != 0; });
}

} // namespace

bool mergeProblemCacheJson(std::string_view existing, std::string_view gpuArch,
                           const std::vector<ProblemCacheWinner> &winners,
                           std::string &out, std::string &error) {
  out.clear();
  // Every row, in file order, which is also the output order. Lookup reads the
  // rows in order: an exact match takes the first row it can use, so a later
  // row of the same problem is the alternative when the first is filtered out
  // or does not compile, and a nearest-neighbour tie keeps the earlier row.
  // Reordering the rows could therefore change what a lookup returns.
  std::vector<StoredPoint> rows;
  std::vector<StoredFallback> fallbacks;
  bool hadFallbacks = false;
  llvm::json::Object root;

  if (isBlank(existing)) {
    root["schema_version"] = static_cast<int64_t>(1);
    root["gpu_arch"] = std::string(gpuArch);
    root["rocm_version"] = static_cast<int64_t>(0);
    root["kernel_abi"] = "rocmlir-conv-v1";
    root["model_key"] = "";
    root["weight_m"] = 1.0;
    root["weight_n"] = 1.0;
    root["weight_k"] = 1.0;
    hadFallbacks = true;
  } else {
    llvm::json::Value document(nullptr);
    const llvm::json::Object *parsedRoot = nullptr;
    if (!parseLutObject(existing, document, parsedRoot, error))
      return false;
    ProblemCacheLut header;
    if (!readLutHeaderJson(*parsedRoot, header, error))
      return false;
    if (!header.gpuArch.empty() && !gpuArch.empty() &&
        header.gpuArch != gpuArch) {
      error = "problem cache is for " + header.gpuArch + ", not " +
              std::string(gpuArch) + "; refusing to rewrite it";
      return false;
    }
    // An existing document is checked in full before anything is built from
    // it. The rewrite replaces the file, so a record it cannot represent has
    // to stop the rewrite, not be dropped from the result. The loader reads
    // the same rows, so a document with no `configs` or `points` is refused by
    // both instead of being rebuilt from the winners alone.
    LutRows parsed;
    if (!readLutRows(*parsedRoot, parsed, error)) {
      error += "; refusing to rewrite it";
      return false;
    }
    root = *parsedRoot;
    // The version the header check accepted, written out. A document without
    // the field reads as version 1, but problem_cache_to_fb.py takes a missing
    // one as 0 and refuses it, so the rewrite must not carry the absence over.
    root["schema_version"] = static_cast<int64_t>(1);

    // Every point is kept, whatever kind of perfConfig it names: the loader
    // skips points whose perfConfig is not gemm:/attn:, but a rewrite must not
    // delete them.
    rows.reserve(parsed.points.size());
    for (const LutPointRecord &point : parsed.points) {
      StoredPoint stored;
      stored.problem = point.problem;
      stored.perfConfig = parsed.perfConfigs[point.config];
      // m, n and k are advisory. Each one that is present is kept as it is.
      // Each one that is absent is worked out from the problem string, so a
      // point with only some of them keeps the ones it has.
      implicitGemmDims(point.problem, stored.m, stored.n, stored.k);
      stored.m = storableDimension(stored.m);
      stored.n = storableDimension(stored.n);
      stored.k = storableDimension(stored.k);
      const char *names[3] = {"m", "n", "k"};
      int64_t *outDims[3] = {&stored.m, &stored.n, &stored.k};
      for (int i = 0; i < 3; ++i)
        if (std::optional<int64_t> dim = point.object->getInteger(names[i]))
          *outDims[i] = *dim;
      rows.push_back(std::move(stored));
    }

    // A fallback is index-linked to `configs` and is renumbered with it.
    hadFallbacks = parsed.hasFallbacks;
    for (const LutFallbackRecord &record : parsed.fallbacks) {
      StoredFallback fallback;
      fallback.record = *record.object;
      fallback.perfConfig = parsed.perfConfigs[record.config];
      fallbacks.push_back(std::move(fallback));
    }
  }

  // A winner is a fresh measurement of its problem, so it replaces every row
  // the file had for it: it takes the place of the first such row, and the
  // others go. A later winner for the same problem wins over an earlier one.
  // Problems with no winner keep all their rows where they were. New problems
  // go at the end, sorted by problem, so the output does not depend on the
  // order the winners came in.
  std::map<std::string, StoredPoint> fresh;
  for (const ProblemCacheWinner &winner : winners) {
    if (winner.problem.empty() || winner.perfConfig.empty())
      continue;
    StoredPoint stored;
    stored.problem = winner.problem;
    stored.perfConfig = winner.perfConfig;
    implicitGemmDims(winner.problem, stored.m, stored.n, stored.k);
    stored.m = storableDimension(stored.m);
    stored.n = storableDimension(stored.n);
    stored.k = storableDimension(stored.k);
    fresh[winner.problem] = std::move(stored);
  }
  std::vector<StoredPoint> merged;
  merged.reserve(rows.size() + fresh.size());
  for (StoredPoint &row : rows) {
    auto replacement = fresh.find(row.problem);
    if (replacement == fresh.end()) {
      merged.push_back(std::move(row));
    } else if (!replacement->second.problem.empty()) {
      merged.push_back(std::move(replacement->second));
      // Moved from: marks the winner as placed, so later rows of the same
      // problem are dropped rather than replaced again.
      replacement->second.problem.clear();
    }
  }
  for (auto &[problem, stored] : fresh)
    if (!stored.problem.empty())
      merged.push_back(std::move(stored));

  // The configs that remain are the ones a point or a fallback uses, each once.
  // A point and a fallback that name the same perfConfig share an entry, and
  // both are renumbered from it.
  llvm::json::Array outConfigs;
  std::map<std::string, int64_t> configIndex;
  const auto indexOf = [&](const std::string &perfConfig) {
    auto found = configIndex.find(perfConfig);
    if (found == configIndex.end()) {
      found = configIndex
                  .emplace(perfConfig, static_cast<int64_t>(outConfigs.size()))
                  .first;
      outConfigs.push_back(llvm::json::Object{{"perf_config", perfConfig}});
    }
    return found->second;
  };
  llvm::json::Array outPoints;
  for (const StoredPoint &stored : merged)
    outPoints.push_back(
        llvm::json::Object{{"problem", stored.problem},
                           {"config", indexOf(stored.perfConfig)},
                           {"m", stored.m},
                           {"n", stored.n},
                           {"k", stored.k}});
  llvm::json::Array outFallbacks;
  for (StoredFallback &fallback : fallbacks) {
    fallback.record["config"] = indexOf(fallback.perfConfig);
    outFallbacks.push_back(std::move(fallback.record));
  }
  if (outConfigs.size() > kMaxConfigs) {
    error = "problem cache would hold " + std::to_string(outConfigs.size()) +
            " distinct configs; at most " + std::to_string(kMaxConfigs) +
            " fit the table format";
    return false;
  }
  root["configs"] = std::move(outConfigs);
  root["points"] = std::move(outPoints);
  if (hadFallbacks)
    root["fallbacks"] = std::move(outFallbacks);

  out = llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
  out += "\n";
  return true;
}

bool mayReplaceInDirectory(uint32_t fileOwner, uint32_t directoryOwner,
                           bool directoryIsSticky, uint32_t callerUid) {
  return callerUid == 0 || !directoryIsSticky || fileOwner == callerUid ||
         directoryOwner == callerUid;
}

bool mayFollowDanglingLink(uint32_t linkOwner, uint32_t directoryOwner,
                           bool directoryIsStickyAndWritable,
                           uint32_t callerUid) {
  return !directoryIsStickyAndWritable || linkOwner == callerUid ||
         linkOwner == directoryOwner;
}

namespace {

#ifndef _WIN32
std::string errnoMessage() { return std::generic_category().message(errno); }
#endif

// Finds the file to rewrite for \p path: where a symlink points, not the link
// itself. The path is walked one component at a time, so a symlink anywhere in
// it is followed, not only the last component. A link whose target exists is
// resolved to it, as the kernel would. A dangling one is followed by hand,
// since weakly_canonical would leave it in place and the rewrite could then
// neither rename over it nor create directories through it: the directories
// and the file are created at its target and the link stays. A dangling link
// is only followed under the kernel's rule for sticky, world-writable
// directories (see mayFollowDanglingLink), which the by-hand follow would
// otherwise bypass. Whatever follows the first component that does not exist
// is kept as it is. A loop, or a link that cannot be read, is refused rather
// than overwritten.
bool resolveWriteTarget(const std::string &path, std::filesystem::path &target,
                        std::string &error) {
  namespace fs = std::filesystem;
  constexpr int kMaxLinkHops = 40;
  std::error_code ec;
  const fs::path absolute = fs::absolute(path, ec);
  if (ec) {
    error = "cannot resolve problem cache " + path + ": " + ec.message();
    return false;
  }
  fs::path current = absolute.root_path();
  std::deque<fs::path> pending;
  for (const fs::path &part : absolute.relative_path())
    if (!part.empty())
      pending.push_back(part);
  int hops = 0;
  while (!pending.empty()) {
    const fs::path part = pending.front();
    pending.pop_front();
    if (part == ".")
      continue;
    if (part == "..") {
      current = current.parent_path();
      continue;
    }
    const fs::path next = current / part;
    const fs::file_status link = fs::symlink_status(next, ec);
    if (link.type() == fs::file_type::not_found) {
      // Nothing here, so nothing below it exists either.
      current = next;
      while (!pending.empty()) {
        if (pending.front() == "..")
          current = current.parent_path();
        else if (pending.front() != ".")
          current /= pending.front();
        pending.pop_front();
      }
      break;
    }
    if (ec) {
      error = "cannot resolve problem cache " + path + ": " + ec.message();
      return false;
    }
    if (!fs::is_symlink(link)) {
      current = next;
      continue;
    }
    if (++hops > kMaxLinkHops) {
      error = "cannot resolve problem cache " + path +
              ": too many levels of symbolic links";
      return false;
    }
    if (fs::exists(next, ec)) {
      // The kernel follows this one, so do as it does.
      const fs::path resolved = fs::canonical(next, ec);
      if (ec) {
        error = "cannot resolve problem cache " + path + ": " + ec.message();
        return false;
      }
      current = resolved;
      continue;
    }
    // Dangling, or a loop: follow it by hand.
#ifndef _WIN32
    {
      struct stat linkStat {};
      struct stat dirStat {};
      if (::lstat(next.c_str(), &linkStat) != 0 ||
          ::stat(current.empty() ? "." : current.c_str(), &dirStat) != 0) {
        error = "cannot inspect symbolic link " + next.string() + ": " +
                errnoMessage();
        return false;
      }
      const bool sticky =
          (dirStat.st_mode & S_ISVTX) != 0 && (dirStat.st_mode & S_IWOTH) != 0;
      if (!mayFollowDanglingLink(linkStat.st_uid, dirStat.st_uid, sticky,
                                 ::geteuid())) {
        error = "refusing to follow symbolic link " + next.string() +
                " in a shared directory: it belongs to another user";
        return false;
      }
    }
#endif
    const fs::path destination = fs::read_symlink(next, ec);
    if (ec) {
      error =
          "cannot read symbolic link " + next.string() + ": " + ec.message();
      return false;
    }
    if (destination.has_root_path())
      current = destination.root_path();
    std::deque<fs::path> inserted;
    for (const fs::path &piece : destination.relative_path())
      if (!piece.empty())
        inserted.push_back(piece);
    pending.insert(pending.begin(), inserted.begin(), inserted.end());
  }
  target = std::move(current);
  return true;
}

} // namespace

namespace {

// An RCAL FlatBuffer, the format of the shipped tables, by its file identifier.
bool looksLikeFlatBuffer(std::string_view bytes) {
  namespace fbs = hipdnn_ep::rocmlir_autotune::fbs;
  return bytes.size() >= 8 &&
         fbs::RocmlirAutotuneLutBufferHasIdentifier(bytes.data());
}

std::string flatBufferRewriteError(const std::string &path) {
  return path +
         " is an RCAL FlatBuffer (.fb). --enable-problem-cache-rw reads and "
         "writes the LUT JSON; decode the .fb with flatc --json --strict-json "
         "--defaults-json --raw-binary problem_cache.fbs -- <file>, or start "
         "from a new file";
}

// Reads \p target, the resolved path of \p path, for a rewrite. \p present is
// false when nothing is there. A stat that fails for a reason other than "no
// such file" is an error: the file may be there, and writing a new table over
// it would lose it.
bool readRewriteTarget(const std::string &path,
                       const std::filesystem::path &target, std::string &text,
                       bool &present, std::string &error) {
  namespace fs = std::filesystem;
  std::error_code ec;
  text.clear();
  present = fs::exists(target, ec);
  if (ec) {
    error = "cannot inspect problem cache " + path + ": " + ec.message();
    return false;
  }
  if (!present)
    return true;
  // A pipe would block the read for good, and a device would be read as an
  // empty table and then renamed over. Only a regular file is a cache.
  if (!fs::is_regular_file(target, ec)) {
    error = "problem cache " + path + " is not a regular file";
    return false;
  }
  std::ifstream in(target, std::ios::binary);
  if (!in) {
    error = "cannot read problem cache: " + path;
    return false;
  }
  if (!readStreamText(in, text)) {
    error = "failed reading problem cache: " + path;
    return false;
  }
  return true;
}

// The part of a temporary's name after `.tmp`: `<pid>-<counter>`, or the
// digits alone that earlier builds wrote.
bool isTemporarySuffix(std::string_view suffix) {
  const auto digits = [](std::string_view text) {
    return !text.empty() &&
           std::all_of(text.begin(), text.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
  };
  const size_t dash = suffix.find('-');
  if (dash == std::string_view::npos)
    return digits(suffix);
  return digits(suffix.substr(0, dash)) && digits(suffix.substr(dash + 1));
}

// Removes temporary files that an earlier rewrite of \p target left behind
// when it was killed between writing and renaming. Only old ones: a recent
// one may belong to a rewrite that another process is doing right now.
void removeStaleTemporaries(const std::filesystem::path &target) {
  namespace fs = std::filesystem;
  constexpr auto kStaleAfter = std::chrono::hours(1);
  std::error_code ec;
  const fs::path dir =
      target.has_parent_path() ? target.parent_path() : fs::path(".");
  const std::string prefix = target.filename().string() + ".tmp";
  const auto now = fs::file_time_type::clock::now();
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix))
      continue;
    if (!isTemporarySuffix(std::string_view(name).substr(prefix.size())))
      continue;
    std::error_code fileEc;
    const auto written = fs::last_write_time(it->path(), fileEc);
    if (!fileEc && now - written > kStaleAfter)
      fs::remove(it->path(), fileEc);
  }
}

#ifdef _WIN32
std::string windowsMessage(DWORD code) {
  return std::error_code(static_cast<int>(code), std::system_category())
      .message();
}

// Whether the access control list of \p path grants the calling user all of
// \p rights. This is the check the system itself makes when the rewrite
// creates or replaces a file, so it also covers inherited and deny entries,
// which the read-only attribute does not. A probe that cannot be completed is
// not a refusal: the rewrite will then report the real error.
bool windowsGrants(const std::filesystem::path &path, DWORD rights) {
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  if (GetNamedSecurityInfoW(
          const_cast<wchar_t *>(path.c_str()), SE_FILE_OBJECT,
          OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION |
              DACL_SECURITY_INFORMATION,
          nullptr, nullptr, nullptr, nullptr, &descriptor) != ERROR_SUCCESS)
    return true;
  HANDLE process = nullptr;
  HANDLE token = nullptr;
  bool granted = true;
  if (OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_QUERY,
                       &process) &&
      DuplicateToken(process, SecurityImpersonation, &token)) {
    GENERIC_MAPPING mapping = {FILE_GENERIC_READ, FILE_GENERIC_WRITE,
                               FILE_GENERIC_EXECUTE, FILE_ALL_ACCESS};
    PRIVILEGE_SET privileges = {};
    DWORD privilegesSize = sizeof(privileges);
    DWORD grantedRights = 0;
    BOOL allowed = FALSE;
    DWORD desired = rights;
    MapGenericMask(&desired, &mapping);
    if (AccessCheck(descriptor, token, desired, &mapping, &privileges,
                    &privilegesSize, &grantedRights, &allowed))
      granted = allowed != FALSE;
  }
  if (token)
    CloseHandle(token);
  if (process)
    CloseHandle(process);
  LocalFree(descriptor);
  return granted;
}
#endif

// Whether a rewrite can put its temporary file beside \p target and rename it
// over \p target: that needs write access to the directory, not just to the
// file. A directory that does not exist yet is created by the rewrite, so the
// nearest one that does exist is the one that has to allow it. On POSIX, in a
// sticky directory (/tmp) the rename also needs the file to be yours, or the
// directory, which the kernel enforces. On Windows the rewrite needs the right
// to add files (and, for a directory that is created, subdirectories) there,
// and the right to delete the file it replaces, either on the file or on its
// directory. Without this the compile would run in full and only then fail to
// write its winners.
bool canWriteBeside(const std::string &path,
                    const std::filesystem::path &target, bool present,
                    std::string &error) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path parent =
      target.has_parent_path() ? target.parent_path() : fs::path(".");
  fs::path dir = parent;
  while (!fs::exists(dir, ec) && dir.has_parent_path() &&
         dir.parent_path() != dir)
    dir = dir.parent_path();
  const bool created = dir != parent;
#ifdef _WIN32
  const DWORD needed =
      created ? (FILE_ADD_FILE | FILE_ADD_SUBDIRECTORY) : FILE_ADD_FILE;
  if (!windowsGrants(dir, needed)) {
    error = "cannot write problem cache " + path + ": " + dir.string() +
            (created ? ", where its directory would be created, does not "
                       "allow it"
                     : " does not allow new files");
    return false;
  }
  if (present && !windowsGrants(target, DELETE) &&
      !windowsGrants(dir, FILE_DELETE_CHILD)) {
    error = "cannot replace problem cache " + path + ": neither the file nor " +
            dir.string() + " allows deleting it";
    return false;
  }
  return true;
#else
  if (::access(dir.c_str(), W_OK | X_OK) != 0) {
    error = "cannot write problem cache " + path + ": " + dir.string() +
            (created ? ", where its directory would be created, is not "
                       "writable"
                     : " is not writable");
    return false;
  }
  if (present) {
    struct stat dirStat {};
    struct stat fileStat {};
    if (::stat(dir.c_str(), &dirStat) == 0 &&
        ::stat(target.c_str(), &fileStat) == 0 &&
        !mayReplaceInDirectory(fileStat.st_uid, dirStat.st_uid,
                               (dirStat.st_mode & S_ISVTX) != 0, ::geteuid())) {
      error = "cannot replace problem cache " + path + ": " + dir.string() +
              " is sticky and the file belongs to another user";
      return false;
    }
  }
  return true;
#endif
}

// Whether this process may write \p target itself. The rewrite renames a new
// file over it, which needs write access to the directory only, so a file that
// was made read-only would otherwise be replaced anyway.
bool isWritable(const std::filesystem::path &target) {
#ifdef _WIN32
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::perms perms = fs::status(target, ec).permissions();
  return !ec && (perms & fs::perms::owner_write) != fs::perms::none;
#else
  return ::access(target.c_str(), W_OK) == 0;
#endif
}

#ifdef _WIN32
// A security descriptor that grants the calling user, and nobody else, full
// access, and that does not inherit entries from the directory. A file created
// with it is closed to every other account from the moment it exists; the
// read-only/read-write attribute that _S_IREAD | _S_IWRITE sets would say
// nothing about that, and the file would take the ACL of the directory, which
// commonly lets other users read it. Not copyable or movable, because the
// attributes point into the object.
class OwnerOnlyAttributes {
public:
  OwnerOnlyAttributes() = default;
  OwnerOnlyAttributes(const OwnerOnlyAttributes &) = delete;
  OwnerOnlyAttributes &operator=(const OwnerOnlyAttributes &) = delete;

  bool init(std::string &error) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
      error = "cannot read the user of this process: " +
              windowsMessage(GetLastError());
      return false;
    }
    DWORD size = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    user_.resize(size);
    if (size == 0 ||
        !GetTokenInformation(token, TokenUser, user_.data(), size, &size)) {
      error = "cannot read the user of this process: " +
              windowsMessage(GetLastError());
      CloseHandle(token);
      return false;
    }
    CloseHandle(token);
    PSID sid = reinterpret_cast<TOKEN_USER *>(user_.data())->User.Sid;
    DWORD aclSize = sizeof(ACL) + sizeof(ACCESS_ALLOWED_ACE) - sizeof(DWORD) +
                    GetLengthSid(sid);
    aclSize = (aclSize + 3) & ~static_cast<DWORD>(3);
    acl_.resize(aclSize);
    PACL dacl = reinterpret_cast<PACL>(acl_.data());
    if (!InitializeAcl(dacl, aclSize, ACL_REVISION) ||
        !AddAccessAllowedAce(dacl, ACL_REVISION, FILE_ALL_ACCESS, sid) ||
        !InitializeSecurityDescriptor(&descriptor_,
                                      SECURITY_DESCRIPTOR_REVISION) ||
        !SetSecurityDescriptorDacl(&descriptor_, TRUE, dacl, FALSE) ||
        !SetSecurityDescriptorControl(&descriptor_, SE_DACL_PROTECTED,
                                      SE_DACL_PROTECTED)) {
      error = "cannot build an owner-only access list: " +
              windowsMessage(GetLastError());
      return false;
    }
    attributes_.nLength = sizeof(attributes_);
    attributes_.lpSecurityDescriptor = &descriptor_;
    attributes_.bInheritHandle = FALSE;
    return true;
  }

  SECURITY_ATTRIBUTES *get() { return &attributes_; }

private:
  std::vector<unsigned char> user_;
  std::vector<unsigned char> acl_;
  SECURITY_DESCRIPTOR descriptor_ = {};
  SECURITY_ATTRIBUTES attributes_ = {};
};
#endif

// Creates \p path, which must not exist yet, writes \p text to it and flushes
// it to the disk, so that a crash after the rename cannot leave an empty or
// partial table in its place. \p exists is set when the name was taken. On any
// other failure the caller removes the file.
//
// The file is created readable by its owner only, whatever the umask is (mode
// 0600; on Windows an explicit access list that names only the calling user),
// so the table (which carries model metadata) is never readable by others
// while it is written, or if the process dies before the rename. On POSIX,
// \p finalPerms and \p groupId, the mode and the group of the file being
// replaced, are applied once the contents are written, so that a shared cache
// keeps its group access. Windows has no such mode or group: the file stays
// private to its owner, and anyone else who needs it has to be granted access
// after the fact.
bool writeNewFile(const std::filesystem::path &path, std::string_view text,
                  std::optional<std::filesystem::perms> finalPerms,
                  std::optional<uint32_t> groupId, bool &exists,
                  std::string &error) {
  exists = false;
#ifdef _WIN32
  (void)finalPerms;
  (void)groupId;
  OwnerOnlyAttributes attributes;
  if (!attributes.init(error))
    return false;
  HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, attributes.get(),
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const DWORD code = GetLastError();
    exists = code == ERROR_FILE_EXISTS || code == ERROR_ALREADY_EXISTS;
    error = windowsMessage(code);
    return false;
  }
  const auto writeSome = [handle](const char *data, size_t size) {
    DWORD written = 0;
    if (!WriteFile(handle, data,
                   static_cast<DWORD>(std::min<size_t>(size, 1u << 30)),
                   &written, nullptr))
      return static_cast<long long>(-1);
    return static_cast<long long>(written);
  };
  const auto sync = [handle] { return FlushFileBuffers(handle) != FALSE; };
  const auto closeFile = [handle] { return CloseHandle(handle) != FALSE; };
  const auto lastError = [] { return windowsMessage(GetLastError()); };
  const auto interrupted = [] { return false; };
#else
  const int fd =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) {
    exists = errno == EEXIST;
    error = errnoMessage();
    return false;
  }
  const auto writeSome = [fd](const char *data, size_t size) {
    return static_cast<long long>(
        ::write(fd, data, std::min<size_t>(size, INT_MAX)));
  };
  const auto sync = [fd] { return ::fsync(fd) == 0; };
  const auto closeFile = [fd] { return ::close(fd) == 0; };
  const auto lastError = [] { return errnoMessage(); };
  const auto interrupted = [] { return errno == EINTR; };
#endif
  const char *data = text.data();
  size_t left = text.size();
  while (left > 0) {
    const long long written = writeSome(data, left);
    if (written < 0) {
      if (interrupted())
        continue;
      error = lastError();
      closeFile();
      return false;
    }
    data += written;
    left -= static_cast<size_t>(written);
  }
#ifndef _WIN32
  // Best effort: a failure here is not worth losing the winners over, and the
  // file is then only more private than the one it replaces. The group goes
  // first, because changing it can clear the set-group-ID bit of the mode.
  if (groupId)
    (void)::fchown(fd, static_cast<uid_t>(-1), static_cast<gid_t>(*groupId));
  if (finalPerms)
    (void)::fchmod(fd, static_cast<mode_t>(*finalPerms) & 0777);
#endif
  if (!sync()) {
    error = lastError();
    closeFile();
    return false;
  }
  if (!closeFile()) {
    error = lastError();
    return false;
  }
  return true;
}

// Flushes the directory entry that the rename changed. Best effort: the table
// itself is already on the disk, and some file systems refuse this.
void syncDirectory(const std::filesystem::path &dir) {
#ifndef _WIN32
  const int fd = ::open(dir.empty() ? "." : dir.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0)
    return;
  ::fsync(fd);
  ::close(fd);
#else
  (void)dir;
#endif
}

int processId() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(::getpid());
#endif
}

} // namespace

bool checkProblemCacheJsonFile(const std::string &path,
                               std::string_view gpuArch, ProblemCacheLut &lut,
                               std::string &error) {
  lut = {};
  std::lock_guard<std::mutex> lock(rewriteMutex());
  std::filesystem::path target;
  if (!resolveWriteTarget(path, target, error))
    return false;
  std::string existing;
  bool present = false;
  if (!readRewriteTarget(path, target, existing, present, error))
    return false;
  if (!canWriteBeside(path, target, present, error))
    return false;
  if (!present)
    return true;
  if (!isWritable(target)) {
    error = "problem cache " + path + " is not writable; refusing to use it";
    return false;
  }
  if (isBlank(existing))
    return true;
  if (looksLikeFlatBuffer(existing)) {
    error = flatBufferRewriteError(path);
    return false;
  }
  // The same checks the rewrite makes, so a file it would refuse is refused
  // before the compile spends its time on a search.
  std::string merged;
  if (!mergeProblemCacheJson(existing, gpuArch, {}, merged, error)) {
    error = path + ": " + error;
    return false;
  }
  if (!loadProblemCacheLutJson(existing, lut, error)) {
    error = path + ": " + error;
    return false;
  }
  return true;
}

bool updateProblemCacheJsonFile(const std::string &path,
                                std::string_view gpuArch,
                                const std::vector<ProblemCacheWinner> &winners,
                                std::string &error) {
  namespace fs = std::filesystem;
  std::lock_guard<std::mutex> lock(rewriteMutex());
  std::error_code ec;
  fs::path target;
  if (!resolveWriteTarget(path, target, error))
    return false;

  std::string existing;
  bool present = false;
  if (!readRewriteTarget(path, target, existing, present, error))
    return false;
  if (present && !isWritable(target)) {
    error =
        "problem cache " + path + " is not writable; refusing to replace it";
    return false;
  }
  if (looksLikeFlatBuffer(existing)) {
    error = flatBufferRewriteError(path);
    return false;
  }

  std::string merged;
  if (!mergeProblemCacheJson(existing, gpuArch, winners, merged, error)) {
    error = path + ": " + error;
    return false;
  }
  // Nothing new: leave an existing file as it is rather than rewrite it with
  // the same contents. A missing one is still created, as a new table.
  const bool anyWinner =
      std::any_of(winners.begin(), winners.end(), [](const auto &winner) {
        return !winner.problem.empty() && !winner.perfConfig.empty();
      });
  if (present && !anyWinner)
    return true;
  if (!canWriteBeside(path, target, present, error))
    return false;

  if (target.has_parent_path())
    fs::create_directories(target.parent_path(), ec);
  removeStaleTemporaries(target);

  // Write beside the target and rename over it, so the old file survives a
  // failed or interrupted write. The name carries the process id and is
  // created exclusively, so two processes never write the same temporary.
  std::optional<uint32_t> oldGroup;
#ifndef _WIN32
  {
    struct stat st {};
    if (present && ::stat(target.c_str(), &st) == 0)
      oldGroup = static_cast<uint32_t>(st.st_gid);
  }
#endif
  std::optional<fs::perms> oldPerms;
  if (present) {
    const fs::file_status old = fs::status(target, ec);
    if (!ec)
      oldPerms = old.permissions();
  }
  static std::atomic<unsigned> counter{0};
  constexpr int kMaxNameAttempts = 100;
  fs::path temporary;
  for (int attempt = 0;; ++attempt) {
    temporary = target;
    temporary += ".tmp" + std::to_string(processId()) + "-" +
                 std::to_string(counter.fetch_add(1));
    bool exists = false;
    std::string reason;
    if (writeNewFile(temporary, merged, oldPerms, oldGroup, exists, reason))
      break;
    if (exists && attempt + 1 < kMaxNameAttempts)
      continue;
    if (!exists)
      fs::remove(temporary, ec);
    error =
        "failed writing problem cache: " + temporary.string() + ": " + reason;
    return false;
  }
  fs::rename(temporary, target, ec);
  if (ec) {
    const std::string reason = ec.message();
    fs::remove(temporary, ec);
    error = "cannot replace problem cache " + path + ": " + reason;
    return false;
  }
  syncDirectory(target.parent_path());
  return true;
}

bool implicitGemmDims(std::string_view problem, int64_t &m, int64_t &n,
                      int64_t &k) {
  m = n = k = 0;
  const std::string key(problem);
  if (auto fields = parseConvProblem(key)) {
    const GemmShape shape = gemmShape(*fields);
    m = shape.m;
    n = shape.n;
    k = shape.k;
    return true;
  }
  if (auto fields = parseAttentionProblem(key)) {
    m = fields->seqQ;
    n = fields->seqK;
    k = fields->headQk;
    return true;
  }
  if (auto fields = parseGegProblem(key)) {
    m = fields->m;
    n = fields->n;
    k = fields->k;
    return true;
  }
  if (auto fields = parseGemmProblem(key)) {
    m = fields->m;
    n = fields->n;
    k = fields->k;
    return true;
  }
  return false;
}

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions,
                    double weightM, double weightN, double weightK) {
  // The stored perfConfig was benchmarked for this exact kernel. Do not
  // re-check the block against the derived GEMM; rocMLIR pads past it.
  for (const auto &entry : entries) {
    if (entry.problem != problemKey || !isConvSolution(entry.solution))
      continue;
    if (std::find(rejectedSolutions.begin(), rejectedSolutions.end(),
                  entry.solution) != rejectedSolutions.end())
      continue;
    NearestConvTuneHit hit;
    hit.solution = entry.solution;
    hit.matchedProblem = entry.problem;
    hit.distance = 0;
    hit.exact = true;
    return hit;
  }

  if (entries.empty())
    return std::nullopt;
  const std::string key(problemKey);
  if (auto query = parseConvProblem(key)) {
    const GemmShape queryGemm = gemmShape(*query);

    const ProblemCacheConvEntry *best = nullptr;
    ConvFields bestFields;
    double bestDistance = 0;
    for (const auto &entry : entries) {
      if (!isConvSolution(entry.solution) || entry.problem == problemKey)
        continue;
      if (std::find(rejectedSolutions.begin(), rejectedSolutions.end(),
                    entry.solution) != rejectedSolutions.end())
        continue;
      auto fields = parseConvProblem(entry.problem);
      if (!fields || !sameGroup(*query, *fields) ||
          !tileLegal(*query, entry.solution))
        continue;
      const double distance = gemmDistance(queryGemm, gemmShape(*fields),
                                           weightM, weightN, weightK);
      // Ties keep the earlier row, matching the MatMulNBits probe.
      if (best != nullptr && distance >= bestDistance)
        continue;
      best = &entry;
      bestFields = *fields;
      bestDistance = distance;
    }
    if (best == nullptr)
      return std::nullopt;

    const GemmShape rowGemm = gemmShape(bestFields);
    NearestConvTuneHit hit;
    hit.solution = best->solution;
    hit.matchedProblem = best->problem;
    hit.distance = bestDistance;
    hit.exact = queryGemm.m == rowGemm.m && queryGemm.n == rowGemm.n &&
                queryGemm.k == rowGemm.k;
    return hit;
  }

  if (auto query = parseAttentionProblem(key))
    return searchNearest(problemKey, *query, entries, rejectedSolutions,
                         weightM, weightN, weightK, parseAttentionProblem,
                         sameAttentionGroup, attentionAxes, attentionTileLegal);
  if (auto query = parseGegProblem(key))
    return searchNearest(problemKey, *query, entries, rejectedSolutions,
                         weightM, weightN, weightK, parseGegProblem,
                         sameGegGroup, gegAxes, gegTileLegal);
  if (auto query = parseGemmProblem(key))
    return searchNearest(problemKey, *query, entries, rejectedSolutions,
                         weightM, weightN, weightK, parseGemmProblem,
                         sameGemmGroup, gemmAxes, gemmTileLegal);
  return std::nullopt;
}

} // namespace hip
} // namespace mlir
