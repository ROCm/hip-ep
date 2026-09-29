/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "nearest_conv_tune.h"

#include "problem_cache_generated.h"

#include "flatbuffers/flatbuffers.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
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
  const int64_t ho =
      std::max<int64_t>(outDim(problem.h, problem.p, problem.y, problem.u,
                               problem.l),
                        1);
  const int64_t wo =
      std::max<int64_t>(outDim(problem.w, problem.q, problem.x, problem.v,
                               problem.j),
                        1);
  GemmShape shape;
  shape.m = std::max<int64_t>(problem.k, 1);
  shape.n = std::max<int64_t>(problem.n * ho * wo, 1);
  shape.k = std::max<int64_t>(
      (problem.c / std::max<int64_t>(problem.g, 1)) * problem.y * problem.x,
      1);
  return shape;
}

// Nearest-neighbor guard only. An exact problem string skips this: rocMLIR
// compiles a covering tile larger than the dimension (the next power of two
// on K, a padded M or N block). Missing block sizes are left to rocMLIR.
bool tileLegal(const ConvFields &problem, std::string_view solution) {
  auto m = parseFlag(solution, "mPerBlock=");
  auto n = parseFlag(solution, "nPerBlock=");
  auto k = parseFlag(solution, "kPerBlock=");
  if (!m || !n || !k)
    return true;
  const GemmShape shape = gemmShape(problem);
  return *m <= shape.m && *n <= shape.n && *k <= shape.k;
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

float gemmDistance(const GemmShape &query, const GemmShape &row, float weightM,
                  float weightN, float weightK) {
  auto term = [](float weight, int64_t a, int64_t b) {
    const float delta = weight * (std::log2(static_cast<float>(a)) -
                                  std::log2(static_cast<float>(b)));
    return delta * delta;
  };
  return std::sqrt(term(weightM, query.m, row.m) +
                   term(weightN, query.n, row.n) + term(weightK, query.k, row.k));
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
  fusions.outputFusions =
      parseEqualsFlag(body, "-outputFusions=").value_or("");
  fusions.supportsSplitK =
      parseWordFlag(body, "-supportsSplitK ").value_or("");
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

float axesDistance(const Axes &query, const Axes &row, float weightM,
                   float weightN, float weightK) {
  const float weights[4] = {weightM, weightN, weightK, weightN};
  float sum = 0;
  for (int i = 0; i < query.n; ++i) {
    const float delta = weights[i] * (std::log2(static_cast<float>(query.v[i])) -
                                      std::log2(static_cast<float>(row.v[i])));
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
  return tileFits(parseBlockTile(solution, false), query.m, query.n, query.k,
                  0, false);
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
         query.headV == row.headV &&
         sameWord(query.withScale, row.withScale) &&
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
std::optional<NearestConvTuneHit> searchNearest(
    std::string_view problemKey, const Fields &query,
    const std::vector<ProblemCacheConvEntry> &entries,
    const std::vector<std::string> &rejectedSolutions, float weightM,
    float weightN, float weightK,
    std::optional<Fields> (*parse)(std::string),
    bool (*sameGroupOf)(const Fields &, const Fields &),
    Axes (*axesOf)(const Fields &),
    bool (*legal)(const Fields &, std::string_view)) {
  const Axes queryAxes = axesOf(query);
  const ProblemCacheConvEntry *best = nullptr;
  Fields bestFields{};
  float bestDistance = 0;
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
    const float distance =
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
  if (!(table->weight_m() > 0.0f) || !(table->weight_n() > 0.0f) ||
      !(table->weight_k() > 0.0f)) {
    error = "rocmlir LUT has a non-positive distance weight";
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

bool loadProblemCacheLut(const std::string &path, ProblemCacheLut &lut,
                         std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "problem cache not found: " + path;
    return false;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  if (!in && !in.eof()) {
    error = "failed reading problem cache: " + path;
    return false;
  }
  return loadProblemCacheLut(bytes.data(), bytes.size(), lut, error);
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
                    float weightM, float weightN, float weightK) {
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
  float bestDistance = 0;
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
    const float distance =
        gemmDistance(queryGemm, gemmShape(*fields), weightM, weightN, weightK);
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
