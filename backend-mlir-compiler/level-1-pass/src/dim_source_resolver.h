/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

namespace hipdnn::level1pass {

/// Outcome of resolving one dynamic (-1) output dimension to a DimSource entry.
enum class DimSourceKind {
  /// The dim's symbolic name maps to a graph-input dim. The EP resolves the
  /// extent from that input before inference; carry (input_idx, dim_idx).
  ResolvedToInput,
  /// Output-allocator ABI with no input mapping. The DLL allocates the output
  /// in-graph (sized from `hip.alloc_output`'s operands) and the EP never reads
  /// this entry, so emit an unresolved sentinel (-1/-1/resolved=false).
  UnresolvedSentinel,
  /// Classic ABI with no input mapping. The EP pre-allocates the output and
  /// cannot size this dim, so the caller must fail the compile.
  Unresolvable,
};

/// A resolved DimSource: `input_idx`/`dim_idx` are meaningful only for
/// `ResolvedToInput` and are sentinel (-1) otherwise.
struct DimSourceResolution {
  DimSourceKind kind;
  int input_idx = -1;
  int dim_idx = -1;
};

/// Decides how to emit the DimSource for one dynamic output dimension.
///
/// The dim is resolved by looking `paramName` (its symbolic name, or null if it
/// has none) up in `dimParamMap`, which maps each symbolic name to the input
/// (index, dim) that declares it. Two cases have no such input: the dim carries
/// no symbolic name, or its name is declared on no input (e.g. a data-dependent
/// extent computed inside the graph). These are fatal under the classic ABI
/// (the EP pre-allocates outputs and has no other way to size the dim) but
/// benign when `useOutputAllocator` is set (the DLL sizes the output in-graph),
/// where they become a sentinel.
inline DimSourceResolution resolveDynamicOutputDim(
    const std::string *paramName,
    const std::unordered_map<std::string, std::pair<int, int>> &dimParamMap,
    bool useOutputAllocator) {
  if (paramName) {
    auto it = dimParamMap.find(*paramName);
    if (it != dimParamMap.end())
      return {DimSourceKind::ResolvedToInput, it->second.first,
              it->second.second};
  }
  return useOutputAllocator
             ? DimSourceResolution{DimSourceKind::UnresolvedSentinel}
             : DimSourceResolution{DimSourceKind::Unresolvable};
}

} // namespace hipdnn::level1pass
