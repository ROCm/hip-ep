/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlEmitter.h - Emit the registered GGML dialect ---------------===//
//
// Turning a captured split into the `ggml` dialect, in the process that
// captured it. Nothing about the graph needs to cross a file for that to
// happen.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_GGMLEMITTER_H
#define HIP_FRONTEND_GGML_GGMLEMITTER_H

#include "hip/Frontend/Ggml/CapturedGraph.h"

#include <string>
#include <vector>

namespace hip::ggml::emit {

/// A symbol the graph's caller says its extents are made of (a ggml
/// backend's split: SplitGraph.h's generalize): `scale` times it at extent
/// `dimension` of each value named in `places` - an input, or a view -, or
/// the product of two such symbols.
struct CallerSymbol {
  /// Where it enters: extent `dimension` of `value` is `scale` times it plus
  /// `offset` (dimension 4: where the view `value` begins, in bytes).
  struct Place {
    std::string value;
    unsigned dimension = 0;
    int64_t scale = 1;
    int64_t offset = 0;
  };
  std::string name;
  int64_t minimum = 1;
  int64_t maximum = 1;
  std::vector<Place> places;
  std::vector<std::string> product;
};

struct Options {
  /// The symbols the graph's extents are made of.
  std::vector<CallerSymbol> callerSymbols;
};

/// Emits what the capture observed as a verified `ggml` dialect module and
/// returns its text. The returned buffer belongs to the caller; emission does
/// not read or write files.
///
/// This is the whole front end: the graph is given its runtime names, its
/// in-place writes are given results, and it becomes SSA, in one walk. Nothing
/// stands between the observation and the module.
std::string serializeModule(const capture::CapturedGraph &graph,
                            const Options &options);

} // namespace hip::ggml::emit

#endif // HIP_FRONTEND_GGML_GGMLEMITTER_H
