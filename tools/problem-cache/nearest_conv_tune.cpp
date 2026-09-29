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

// matmul_nbits_autotune.fbs defaults. One octave of M, N, or K is the same
// distance. This dump has not been fit, so the weights stay at 1.
constexpr float kWeightM = 1.0f;
constexpr float kWeightN = 1.0f;
constexpr float kWeightK = 1.0f;

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

float gemmDistance(const GemmShape &query, const GemmShape &row) {
  auto term = [](float weight, int64_t a, int64_t b) {
    const float delta = weight * (std::log2(static_cast<float>(a)) -
                                  std::log2(static_cast<float>(b)));
    return delta * delta;
  };
  return std::sqrt(term(kWeightM, query.m, row.m) +
                   term(kWeightN, query.n, row.n) +
                   term(kWeightK, query.k, row.k));
}

bool isConvSolution(std::string_view solution) {
  return solution.compare(0, 5, "gemm:") == 0 ||
         solution.compare(0, 5, "attn:") == 0;
}

} // namespace

bool loadProblemCacheConvEntries(const uint8_t *data, size_t size,
                                 std::vector<ProblemCacheConvEntry> &entries,
                                 std::string &error) {
  entries.clear();
  if (data == nullptr || size == 0) {
    error = "problem cache is empty";
    return false;
  }
  flatbuffers::Verifier verifier(data, size);
  if (!hipdnn_ep::problem_cache::VerifyProblemCacheBuffer(verifier)) {
    error = "problem cache is not an MXPC flatbuffer";
    return false;
  }
  const auto *cache = hipdnn_ep::problem_cache::GetProblemCache(data);
  if (cache->schema_version() != 1) {
    error = "unsupported problem-cache schema version " +
            std::to_string(cache->schema_version());
    return false;
  }
  const auto *rows = cache->entries();
  if (rows == nullptr)
    return true;
  for (const auto *row : *rows) {
    if (row == nullptr || row->problem() == nullptr || row->solution() == nullptr)
      continue;
    std::string solution = row->solution()->str();
    if (!isConvSolution(solution))
      continue;
    entries.push_back({row->problem()->str(), std::move(solution)});
  }
  return true;
}

bool loadProblemCacheConvEntries(const std::string &path,
                                 std::vector<ProblemCacheConvEntry> &entries,
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
  return loadProblemCacheConvEntries(bytes.data(), bytes.size(), entries,
                                     error);
}

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions) {
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

  auto query = parseConvProblem(std::string(problemKey));
  if (!query || entries.empty())
    return std::nullopt;
  const GemmShape queryGemm = gemmShape(*query);

  const ProblemCacheConvEntry *best = nullptr;
  ConvFields bestFields;
  float bestDistance = 0;
  for (const auto &entry : entries) {
    if (!isConvSolution(entry.solution))
      continue;
    if (std::find(rejectedSolutions.begin(), rejectedSolutions.end(),
                  entry.solution) != rejectedSolutions.end())
      continue;
    auto fields = parseConvProblem(entry.problem);
    if (!fields || !sameGroup(*query, *fields) ||
        !tileLegal(*query, entry.solution))
      continue;
    const float distance = gemmDistance(queryGemm, gemmShape(*fields));
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

} // namespace hip
} // namespace mlir
