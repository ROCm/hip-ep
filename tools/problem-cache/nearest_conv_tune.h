/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Nearest rocMLIR tile from a problem-cache table.
//
// Same shape as the MatMulNBits LUT: a hard group, then one weighted log2
// distance. The weights live in the table (`weight_m/n/k`), defaulting to 1.
// A miss only searches rows of the same kernel kind.
//
// Conv. Arch, conv kind, -F, layouts, stride, dilation, and the fusion
// suffix must match. The distance is the implicit GEMM:
//   M = K_out / groups, N = batch * out_h * out_w,
//   K = (C / groups) * filter_h * filter_w.
//
// Gemm. Arch, datatype, output datatype, transposes, scaled-gemm flags,
// -g, and the fusion suffix must match. The distance is -m, -n, -k.
//
// Attention. Arch, datatype, transposes, causal, LSE, split-KV, sliding
// window, head counts, head dims, scale/bias, and the fusion suffix must
// match. The distance is -seq_len_q and -seq_len_k.
//
// GemmElementwiseGemm. Arch, datatype, the four transposes, -g, and the
// fusion suffix must match. The distance is -m, -n, -k, and -gemmO. The
// fourth axis uses weight_n.
//
// The neighbor supplies only its gemm:/attn: perfConfig. An identical
// problem string is returned as stored. The block-size check applies only
// when borrowing a neighbor. `rejectedSolutions` are tiles that already
// failed to compile; the next nearest legal row is returned.

#ifndef HIP_PROBLEM_CACHE_NEAREST_CONV_TUNE_H
#define HIP_PROBLEM_CACHE_NEAREST_CONV_TUNE_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mlir {
namespace hip {

struct ProblemCacheConvEntry {
  std::string problem;
  std::string solution;
};

// One loaded table, from a shipped .fb or an external JSON file. `entries` are
// the gemm:/attn: points; other perfConfigs are dropped. Weights are the
// table's log-space metric. `gpuArch` is the table's gpu_arch; empty means the
// table is not restricted to one device.
struct ProblemCacheLut {
  std::string gpuArch;
  float weightM = 1.0f;
  float weightN = 1.0f;
  float weightK = 1.0f;
  std::vector<ProblemCacheConvEntry> entries;
};

struct NearestConvTuneHit {
  std::string solution;
  std::string matchedProblem;
  // Weighted log2 distance. Zero when every distance axis matches.
  float distance = 0;
  bool exact = false;
};

// Primary M, N, and K of a conv, gemm, attention, or gemm-elementwise-gemm
// problem string. False, and m/n/k left at 0, for any other string. The dump
// writes these into each LUT point. The lookup re-parses the problem string
// and does not read the stored values.
bool implicitGemmDims(std::string_view problem, int64_t &m, int64_t &n,
                      int64_t &k);

// Rows whose perfConfig is gemm:/attn:. Anything else is skipped.
//
// The shipped tables: a verified RCAL FlatBuffer (`tools/problem-cache/lut/
// <arch>.fb`) embedded in the library.
bool loadProblemCacheLut(const uint8_t *data, size_t size, ProblemCacheLut &lut,
                         std::string &error);

// An external table: the LUT JSON document that `hip-rocmlir-compiler
// --enable-problem-cache-rw` writes (`schema_version`, `gpu_arch`,
// `kernel_abi`, `weight_m/n/k`, `configs`, `points`, `fallbacks`). It applies
// the same checks and skips the same rows as the FlatBuffer overload, so the
// file loads without a `.fb` conversion. Absent `schema_version` and weights
// take the `.fbs` defaults (1, 1.0). `kernel_abi` is required.
bool loadProblemCacheLutJson(std::string_view text, ProblemCacheLut &lut,
                             std::string &error);

// Reads \p path and parses it with loadProblemCacheLutJson. A `.fb` file is
// not accepted here and reports a JSON parse error.
bool loadProblemCacheLutJsonFile(const std::string &path, ProblemCacheLut &lut,
                                 std::string &error);

// A problem string and the perfConfig the search picked for it.
struct ProblemCacheWinner {
  std::string problem;
  std::string perfConfig;
};

// Merges \p winners into the LUT JSON document \p existing and returns the new
// document in \p out. Blank \p existing starts a new document for \p gpuArch.
//
// Every point already in \p existing is kept, whatever kind of perfConfig it
// names, so a rewrite never drops data the loader would skip. A winner replaces
// the point of the same problem string, and a new problem is added. If a
// problem appears twice, its first row is kept, as a lookup returns that one.
//
// `configs` is rebuilt without duplicates, so its indices change. It holds
// every perfConfig a point or a fallback uses. The `config` of each point and
// each fallback is renumbered together, so a fallback keeps naming the same
// perfConfig, even one that no point uses. `points` is sorted by problem, so
// the output is deterministic. `fallbacks` is written only if \p existing had
// it, and its other fields are kept. Top-level fields besides `configs`,
// `points` and `fallbacks` (weights, `model_key`, ...) are carried over.
//
// Fails, leaving \p out empty, without producing a document when \p existing is
// not a supported LUT (invalid JSON, wrong `schema_version` or `kernel_abi`, a
// bad weight), names a different non-empty `gpu_arch` than \p gpuArch, or has a
// structure that cannot be rewritten without losing data: `configs` or
// `points` missing or not an array, a config without a string `perf_config`, a
// point that is not an object or lacks a string `problem` or an in-range
// integer `config` (or has a non-integer `m`, `n` or `k`), or `fallbacks`
// that is not an array of objects with an in-range integer `config`.
bool mergeProblemCacheJson(std::string_view existing, std::string_view gpuArch,
                           const std::vector<ProblemCacheWinner> &winners,
                           std::string &out, std::string &error);

// Reads \p path (a missing or empty file starts a new table), merges
// \p winners with mergeProblemCacheJson, and replaces \p path with the result.
// The new contents go to a sibling temporary file that is renamed over \p path,
// so a failed or interrupted write leaves the old file intact. The file is read
// again at write time, not taken from an earlier load, so entries another
// compile added in between are kept. Missing parent directories are created. A
// symlinked \p path is rewritten at its target, which is created if the link
// is dangling, and the link stays. A symlink loop is refused.
bool updateProblemCacheJsonFile(const std::string &path,
                                std::string_view gpuArch,
                                const std::vector<ProblemCacheWinner> &winners,
                                std::string &error);

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions = {},
                    float weightM = 1.0f, float weightN = 1.0f,
                    float weightK = 1.0f);

} // namespace hip
} // namespace mlir

#endif
