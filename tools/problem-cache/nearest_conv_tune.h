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
//
// Borrowing a neighbor is the part that `allowNearest` turns off. With it
// false only an identical problem string matches, and every other problem is a
// miss. The compiler passes false unless nearest-config lookup was enabled
// (see AutotuneOptions::nearestConfig); the default here is true so that the
// table logic itself stays testable on its own.

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
  // Weighted log2 distance. Zero when every distance axis matches. A double,
  // like the weights it is computed with, so a large or small table weight
  // cannot overflow or underflow it.
  double distance = 0;
  bool exact = false;
};

// Primary M, N, and K of a conv, gemm, attention, or gemm-elementwise-gemm
// problem string. False, and m/n/k left at 0, for any other string. The
// rewrite writes these into each LUT point. The lookup re-parses the problem
// string and does not read the stored values.
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
// `kernel_abi`, `weight_m/n/k`, `configs`, `points`, `fallbacks`). It checks
// the document as mergeProblemCacheJson does, so the loader and the rewrite
// accept and refuse the same files: a missing `configs` or `points`, or a
// record that cannot be read, is an error rather than a skipped row. Rows whose
// perfConfig is not gemm:/attn: are valid and skipped, as in the FlatBuffer
// overload. Absent `schema_version`, `gpu_arch` and weights take the `.fbs`
// defaults (1, any GPU, 1.0); present ones must have the right type.
// `kernel_abi` is required.
bool loadProblemCacheLutJson(std::string_view text, ProblemCacheLut &lut,
                             std::string &error);

// Reads \p path and parses it with loadProblemCacheLutJson. A `.fb` file is
// not accepted here and is reported as a FlatBuffer.
bool loadProblemCacheLutJsonFile(const std::string &path, ProblemCacheLut &lut,
                                 std::string &error);

// Whether the kernel lets the caller rename a new file over \p fileOwner's
// file in a directory owned by \p directoryOwner. In a sticky directory (/tmp)
// that takes owning the file or the directory; root may always.
bool mayReplaceInDirectory(uint32_t fileOwner, uint32_t directoryOwner,
                           bool directoryIsSticky, uint32_t callerUid);

// Whether a rewrite may follow a dangling symbolic link that it finds on the
// way to the cache file. It applies the rule the kernel uses for
// protected_symlinks, which does not cover a link followed by hand: in a
// sticky, world-writable directory (/tmp) a link is followed only when its
// owner is the caller or the directory's owner. Elsewhere any link is.
bool mayFollowDanglingLink(uint32_t linkOwner, uint32_t directoryOwner,
                           bool directoryIsStickyAndWritable,
                           uint32_t callerUid);

// A problem string and the perfConfig the search picked for it.
struct ProblemCacheWinner {
  std::string problem;
  std::string perfConfig;
};

// Merges \p winners into the LUT JSON document \p existing and returns the new
// document in \p out. Blank \p existing starts a new document for \p gpuArch.
//
// Every point already in \p existing is kept, whatever kind of perfConfig it
// names, so a rewrite never drops data the loader would skip. A problem the
// winners do not touch keeps all its rows, in file order, even when it repeats:
// a lookup returns the first row it can use, so a later row is the alternative
// when the first is filtered out or rejected. A winner is a fresh measurement,
// so it takes the place of the first row of its problem string and the other
// rows of that problem go. Rows keep their order, because a nearest-neighbour
// tie keeps the earlier row; new problems go at the end, sorted by problem.
//
// `configs` is rebuilt without duplicates, so its indices change. It holds
// every perfConfig a point or a fallback uses. The `config` of each point and
// each fallback is renumbered together, so a fallback keeps naming the same
// perfConfig, even one that no point uses. `fallbacks` is written only if
// \p existing had it, and its other fields are kept. Top-level fields besides
// `configs`, `points` and `fallbacks` (weights, `model_key`, ...) are carried
// over. `schema_version` is the exception: it is always written out, as the
// version that was checked, so that a document which left it out (read as
// version 1) can still be compiled to a `.fb`. A dimension worked out from a
// problem string that does not fit the table's uint32 is stored as 0, the
// schema's "not recognized".
//
// Fails, leaving \p out empty, without producing a document when \p existing is
// not a supported LUT (invalid JSON, wrong `schema_version` or `kernel_abi`, a
// bad weight, a `gpu_arch` or `model_key` that is not a string, a
// `rocm_version` that is not an int32), names a different non-empty `gpu_arch`
// than \p gpuArch, or has a structure that cannot be rewritten without losing
// data or compiled to the table format: `configs` or `points` missing or not an
// array, more than 65536 configs (also when the rewrite would need that
// many), a config without a string `perf_config`, a point that is not an object
// or lacks a string `problem` or an in-range integer `config` (or has an `m`,
// `n` or `k` that is not an integer from 0 to 2^32-1), or `fallbacks` that is
// not an array of objects with an in-range integer `config`.
bool mergeProblemCacheJson(std::string_view existing, std::string_view gpuArch,
                           const std::vector<ProblemCacheWinner> &winners,
                           std::string &out, std::string &error);

// Checks, before a compile, that updateProblemCacheJsonFile can rewrite \p path
// for \p gpuArch, and loads it into \p lut for lookup. A missing or blank file
// is an empty table. Fails, with \p lut empty, for a file that the rewrite
// would refuse (see mergeProblemCacheJson), for an RCAL FlatBuffer, for a path
// that is not a regular file (a pipe would block, a device would be replaced),
// for a file that is read-only, for a directory the rewrite could not put its
// temporary file in (or create, when it does not exist yet; in a sticky
// directory also when the file is another user's; on Windows judged from the
// access list of the nearest existing directory, for adding files and for
// deleting the file it replaces), and for a file that cannot be read. Without
// these a compile would run in full and then fail to write.
bool checkProblemCacheJsonFile(const std::string &path,
                               std::string_view gpuArch, ProblemCacheLut &lut,
                               std::string &error);

// Reads \p path (a missing or empty file starts a new table), merges
// \p winners with mergeProblemCacheJson, and replaces \p path with the result.
// With no winners an existing file is left as it is. An RCAL FlatBuffer is
// refused.
// The new contents go to a sibling temporary file that is renamed over \p path,
// so a failed or interrupted write leaves the old file intact. The temporary
// is created exclusively and readable by its owner only, whatever the umask is
// (on Windows with an explicit, non-inherited access list that names only the
// calling user), written and flushed to disk before the rename. On POSIX the
// old file's mode and group are then put on it, so a shared cache keeps its
// group access; on Windows the file stays private to its owner. The owner is
// the caller's, which cannot be kept without root, and ACLs, extended
// attributes and hard links of the old file are not carried over. The file is
// read again at write time, not taken from an earlier load, so entries another
// compile added in between are kept. Missing parent directories are created. A
// symlinked \p path, or a path that goes through a symlinked directory, is
// rewritten at its target, which (with its directories) is created if the link
// is dangling, and the links stay; a dangling link is followed only under the
// kernel's rule for sticky, world-writable directories (see
// mayFollowDanglingLink). A symlink loop, in any component, is refused, and so
// is anything but a regular file, a read-only file, and a directory that does
// not allow the rename (see checkProblemCacheJsonFile). Rewrites in one process
// are serialized, so overlapping compiles there do not lose each other's
// winners; separate processes can, and the last one to finish wins. Temporary
// files older than an hour, left by a rewrite that was killed, are removed.
bool updateProblemCacheJsonFile(const std::string &path,
                                std::string_view gpuArch,
                                const std::vector<ProblemCacheWinner> &winners,
                                std::string &error);

std::optional<NearestConvTuneHit>
findNearestConvTune(std::string_view problemKey,
                    const std::vector<ProblemCacheConvEntry> &entries,
                    const std::vector<std::string> &rejectedSolutions = {},
                    double weightM = 1.0, double weightN = 1.0,
                    double weightK = 1.0, bool allowNearest = true);

} // namespace hip
} // namespace mlir

#endif
