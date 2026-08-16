/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace morphizen {

inline constexpr char kOnnxDimParamsMetadataKey[] =
    "com.amd.morphizen.onnx_dim_params.v1";
inline constexpr char kOnnxDimParamsModuleAttr[] = "hipdnn.onnx_dim_params_v1";
inline constexpr char kOnnxDimParamsEncodingVersion[] = "HSDI1";
inline constexpr char kOnnxDimParamsResolverPolicyVersion[] =
    "onnx-symbolic-extent-v1-main-graph-single-input-binding";
inline constexpr char kInitializerDataDigestMetadataKey[] =
    "com.amd.morphizen.initializer_data_sha256.v1";
inline constexpr char kCompilerGraphDigestMetadataKey[] =
    "com.amd.morphizen.compiler_graph_sha256.v1";

struct SymbolicDimRecord {
  std::string scope;
  std::string value_name;
  std::vector<std::string> dimensions;
};

/// Encode records using the deterministic HSDI1 wire format.
/// Throws std::invalid_argument for malformed or duplicate input records.
std::string encode_symbolic_dim_records(std::vector<SymbolicDimRecord> records);

/// Decode and validate canonical HSDI1 bytes.
/// Returns nullopt and fills `error` when the input is malformed or
/// non-canonical.
std::optional<std::vector<SymbolicDimRecord>>
decode_symbolic_dim_records(std::string_view encoded, std::string &error);

} // namespace morphizen
