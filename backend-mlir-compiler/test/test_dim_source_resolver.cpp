/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file test_dim_source_resolver.cpp
 * @brief Unit tests for resolveDynamicOutputDim (DimSource emission policy).
 *
 * Pure-logic tests: no ORT, no GPU, no protobuf. Exercises every branch of the
 * dynamic-output-dim resolution used by build_metadata_json.
 */

#include "dim_source_resolver.h"

#include <gtest/gtest.h>

using hipdnn::level1pass::DimSourceKind;
using hipdnn::level1pass::resolveDynamicOutputDim;

namespace {

using DimParamMap = std::unordered_map<std::string, std::pair<int, int>>;

// A symbolic name that maps to an input dim resolves to that input regardless
// of ABI mode.
TEST(DimSourceResolver, NameMappedToInputResolves) {
  DimParamMap m{{"seq_len", {2, 1}}};
  const std::string name = "seq_len";

  for (bool allocator : {false, true}) {
    auto r = resolveDynamicOutputDim(&name, m, allocator);
    EXPECT_EQ(r.kind, DimSourceKind::ResolvedToInput);
    EXPECT_EQ(r.input_idx, 2);
    EXPECT_EQ(r.dim_idx, 1);
  }
}

// No symbolic name + allocator ABI -> sentinel (DLL sizes the output in-graph).
TEST(DimSourceResolver, NoNameAllocatorEmitsSentinel) {
  DimParamMap m;
  auto r = resolveDynamicOutputDim(/*paramName=*/nullptr, m,
                                   /*useOutputAllocator=*/true);
  EXPECT_EQ(r.kind, DimSourceKind::UnresolvedSentinel);
  EXPECT_EQ(r.input_idx, -1);
  EXPECT_EQ(r.dim_idx, -1);
}

// No symbolic name + classic ABI -> unresolvable (caller fails the compile).
TEST(DimSourceResolver, NoNameClassicIsUnresolvable) {
  DimParamMap m;
  auto r = resolveDynamicOutputDim(/*paramName=*/nullptr, m,
                                   /*useOutputAllocator=*/false);
  EXPECT_EQ(r.kind, DimSourceKind::Unresolvable);
}

// Symbolic name present but on no input + allocator ABI -> sentinel. This is
// the data-dependent-extent case (e.g. NonZero) that the relaxation unblocks.
TEST(DimSourceResolver, UnmappedNameAllocatorEmitsSentinel) {
  DimParamMap m{{"batch", {0, 0}}};
  const std::string name = "num_nonzero"; // not declared by any input
  auto r = resolveDynamicOutputDim(&name, m, /*useOutputAllocator=*/true);
  EXPECT_EQ(r.kind, DimSourceKind::UnresolvedSentinel);
}

// Symbolic name present but on no input + classic ABI -> unresolvable.
TEST(DimSourceResolver, UnmappedNameClassicIsUnresolvable) {
  DimParamMap m{{"batch", {0, 0}}};
  const std::string name = "num_nonzero";
  auto r = resolveDynamicOutputDim(&name, m, /*useOutputAllocator=*/false);
  EXPECT_EQ(r.kind, DimSourceKind::Unresolvable);
}

} // namespace
