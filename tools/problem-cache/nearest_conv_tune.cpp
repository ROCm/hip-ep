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
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

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
  if (std::optional<llvm::StringRef> arch = root.getString("gpu_arch"))
    lut.gpuArch = arch->str();
  lut.weightM = weightM;
  lut.weightN = weightN;
  lut.weightK = weightK;
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

  const llvm::json::Array *configs = root->getArray("configs");
  const llvm::json::Array *points = root->getArray("points");
  if (configs == nullptr || points == nullptr)
    return true;

  // perf_config per config index. A config with no string stays empty, and the
  // points that name it are skipped, as in the FlatBuffer overload.
  std::vector<std::string> perfConfigs;
  perfConfigs.reserve(configs->size());
  for (const llvm::json::Value &config : *configs) {
    const llvm::json::Object *object = config.getAsObject();
    std::optional<llvm::StringRef> perfConfig =
        object != nullptr ? object->getString("perf_config") : std::nullopt;
    perfConfigs.push_back(perfConfig ? perfConfig->str() : std::string());
  }

  for (const llvm::json::Value &point : *points) {
    const llvm::json::Object *object = point.getAsObject();
    if (object == nullptr)
      continue;
    std::optional<llvm::StringRef> problem = object->getString("problem");
    std::optional<int64_t> index = object->getInteger("config");
    if (!problem || !index || *index < 0 ||
        static_cast<size_t>(*index) >= perfConfigs.size())
      continue;
    const std::string &solution = perfConfigs[static_cast<size_t>(*index)];
    if (solution.empty() || !isConvSolution(solution))
      continue;
    lut.entries.push_back({problem->str(), solution});
  }
  return true;
}

bool loadProblemCacheLutJsonFile(const std::string &path, ProblemCacheLut &lut,
                                 std::string &error) {
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

  // An existing document is checked in full before anything is built from it.
  // The rewrite replaces the file, so a record it cannot represent has to stop
  // the rewrite, not be dropped from the result.
  const auto malformed = [&error](const std::string &what) {
    error = "problem cache is malformed: " + what + "; refusing to rewrite it";
    return false;
  };

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
    root = *parsedRoot;

    // The loader reads a document with no `configs` or `points` as an empty
    // table. A rewrite would then build both from the winners alone and
    // replace whatever those keys held, so they must be real arrays here.
    const llvm::json::Array *configs = parsedRoot->getArray("configs");
    const llvm::json::Array *points = parsedRoot->getArray("points");
    if (configs == nullptr)
      return malformed("`configs` is missing or not an array");
    if (points == nullptr)
      return malformed("`points` is missing or not an array");

    std::vector<std::string> perfConfigs;
    perfConfigs.reserve(configs->size());
    for (size_t i = 0; i < configs->size(); ++i) {
      const llvm::json::Object *object = (*configs)[i].getAsObject();
      std::optional<llvm::StringRef> perfConfig =
          object != nullptr ? object->getString("perf_config") : std::nullopt;
      if (!perfConfig)
        return malformed("configs[" + std::to_string(i) +
                         "] is not an object with a string `perf_config`");
      perfConfigs.push_back(perfConfig->str());
    }

    // Every point is kept, whatever kind of perfConfig it names: the loader
    // skips points whose perfConfig is not gemm:/attn:, but a rewrite must not
    // delete them.
    const auto configIndexOf = [&](const llvm::json::Object &object,
                                   const std::string &where,
                                   std::string &perfConfig) {
      std::optional<int64_t> index = object.getInteger("config");
      if (!index)
        return malformed(where + " has no integer `config`");
      if (*index < 0 || static_cast<size_t>(*index) >= perfConfigs.size())
        return malformed(where + " has `config` " + std::to_string(*index) +
                         ", outside the " + std::to_string(perfConfigs.size()) +
                         " configs");
      perfConfig = perfConfigs[static_cast<size_t>(*index)];
      return true;
    };
    for (size_t i = 0; i < points->size(); ++i) {
      const std::string where = "points[" + std::to_string(i) + "]";
      const llvm::json::Object *object = (*points)[i].getAsObject();
      if (object == nullptr)
        return malformed(where + " is not an object");
      std::optional<llvm::StringRef> problem = object->getString("problem");
      if (!problem)
        return malformed(where + " has no string `problem`");
      StoredPoint stored;
      stored.problem = problem->str();
      if (!configIndexOf(*object, where, stored.perfConfig))
        return false;
      // m, n and k are advisory. Each one that is present must be an integer
      // and is kept as it is. Each one that is absent is worked out from the
      // problem string, so a point with only some of them keeps the ones it
      // has.
      const llvm::json::Value *dims[3] = {object->get("m"), object->get("n"),
                                          object->get("k")};
      for (const llvm::json::Value *dim : dims)
        if (dim != nullptr && !dim->getAsInteger())
          return malformed(where + " has a non-integer `m`, `n` or `k`");
      implicitGemmDims(problem->str(), stored.m, stored.n, stored.k);
      int64_t *outDims[3] = {&stored.m, &stored.n, &stored.k};
      for (int i = 0; i < 3; ++i)
        if (dims[i] != nullptr)
          *outDims[i] = *dims[i]->getAsInteger();
      rows.push_back(std::move(stored));
    }

    // `fallbacks` is optional, but one that is present is index-linked to
    // `configs` and has to be renumbered with it.
    if (const llvm::json::Value *value = parsedRoot->get("fallbacks")) {
      const llvm::json::Array *array = value->getAsArray();
      if (array == nullptr)
        return malformed("`fallbacks` is not an array");
      hadFallbacks = true;
      for (size_t i = 0; i < array->size(); ++i) {
        const std::string where = "fallbacks[" + std::to_string(i) + "]";
        const llvm::json::Object *object = (*array)[i].getAsObject();
        if (object == nullptr)
          return malformed(where + " is not an object");
        StoredFallback fallback;
        if (!configIndexOf(*object, where, fallback.perfConfig))
          return false;
        fallback.record = *object;
        fallbacks.push_back(std::move(fallback));
      }
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
  root["configs"] = std::move(outConfigs);
  root["points"] = std::move(outPoints);
  if (hadFallbacks)
    root["fallbacks"] = std::move(outFallbacks);

  out = llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
  out += "\n";
  return true;
}

namespace {

// Finds the file to rewrite for \p path: where a symlink points, not the link
// itself. weakly_canonical resolves every link that leads to something that
// exists, but leaves a dangling one in place, and renaming over that would
// replace the link. A dangling link is followed by hand, so the file is
// created at its target and the link stays. A loop, or a link that cannot be
// read, is refused rather than overwritten.
bool resolveWriteTarget(const std::string &path, std::filesystem::path &target,
                        std::string &error) {
  namespace fs = std::filesystem;
  constexpr int kMaxLinkHops = 40;
  std::error_code ec;
  fs::path current = path;
  for (int hop = 0; hop < kMaxLinkHops; ++hop) {
    fs::path resolved = fs::weakly_canonical(current, ec);
    if (ec) {
      // Not resolvable (for example a link loop). A plain path is written
      // as it is, but a link is never replaced.
      std::error_code statusEc;
      if (fs::is_symlink(fs::symlink_status(current, statusEc)) || statusEc) {
        error = "cannot resolve problem cache " + path + ": " + ec.message();
        return false;
      }
      target = current;
      return true;
    }
    if (!fs::is_symlink(fs::symlink_status(resolved, ec)) || ec) {
      target = std::move(resolved);
      return true;
    }
    // Only a dangling link is still a link here.
    fs::path link = fs::read_symlink(resolved, ec);
    if (ec) {
      error = "cannot read symbolic link " + resolved.string() + ": " +
              ec.message();
      return false;
    }
    current = link.is_absolute() ? link : resolved.parent_path() / link;
  }
  error = "cannot resolve problem cache " + path +
          ": too many levels of symbolic links";
  return false;
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
         "writes the LUT JSON; decode the .fb with flatc --json --raw-binary "
         "problem_cache.fbs -- <file>, or start from a new file";
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
    if (!std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                     name.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; }))
      continue;
    std::error_code fileEc;
    const auto written = fs::last_write_time(it->path(), fileEc);
    if (!fileEc && now - written > kStaleAfter)
      fs::remove(it->path(), fileEc);
  }
}

// Serializes rewrites within one process. The file is read, merged and
// renamed over, so two compiles that overlap would otherwise each start from
// the same old table and the second rename would drop the first one's
// winners. Separate processes are not covered.
std::mutex &rewriteMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

bool loadProblemCacheLutFile(const std::string &path, ProblemCacheLut &lut,
                             std::string &error) {
  lut = {};
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "problem cache not found: " + path;
    return false;
  }
  std::string bytes;
  if (!readStreamText(in, bytes)) {
    error = "failed reading problem cache: " + path;
    return false;
  }
  const bool ok =
      looksLikeFlatBuffer(bytes)
          ? loadProblemCacheLut(reinterpret_cast<const uint8_t *>(bytes.data()),
                                bytes.size(), lut, error)
          : loadProblemCacheLutJson(bytes, lut, error);
  if (!ok)
    error = path + ": " + error;
  return ok;
}

bool checkProblemCacheJsonFile(const std::string &path,
                               std::string_view gpuArch, ProblemCacheLut &lut,
                               std::string &error) {
  lut = {};
  std::filesystem::path target;
  if (!resolveWriteTarget(path, target, error))
    return false;
  std::string existing;
  bool present = false;
  if (!readRewriteTarget(path, target, existing, present, error))
    return false;
  if (!present || isBlank(existing))
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

  if (target.has_parent_path())
    fs::create_directories(target.parent_path(), ec);
  removeStaleTemporaries(target);

  // Write beside the target and rename over it, so the old file survives a
  // failed or interrupted write.
  fs::path temporary = target;
  temporary +=
      ".tmp" + std::to_string(
                   std::chrono::steady_clock::now().time_since_epoch().count());
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(merged.data(), static_cast<std::streamsize>(merged.size()));
    out.flush();
    if (!out) {
      out.close();
      fs::remove(temporary, ec);
      error = "failed writing problem cache: " + temporary.string();
      return false;
    }
  }
  // The rename swaps in a new file, so give it the mode the old one had rather
  // than the default one. Best effort: a failure here is not worth losing the
  // winners over.
  if (present) {
    const fs::file_status old = fs::status(target, ec);
    if (!ec)
      fs::permissions(temporary, old.permissions(), ec);
  }
  fs::rename(temporary, target, ec);
  if (ec) {
    const std::string reason = ec.message();
    fs::remove(temporary, ec);
    error = "cannot replace problem cache " + path + ": " + reason;
    return false;
  }
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
