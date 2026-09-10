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

/// Normative HSDI1 symbolic-dimension metadata contract.
///
/// The wire grammar is ASCII framing around opaque byte strings:
///
///   file     := "HSDI1\n" record-count "\n" record*
///   record   := bytes(scope) bytes(value-name) rank ":" bytes(dimension)* "\n"
///   bytes(x) := byte-count ":" lowercase-hex(x)
///
/// Decimal counts contain no leading zero unless the value is zero. A bytes
/// field has exactly twice `byte-count` hexadecimal characters. Records are
/// strictly sorted by the bytewise lexicographic `(scope, value-name)` pair;
/// duplicate pairs, trailing bytes, and non-canonical spellings are invalid.
///
/// HSDI1 permits at most 1,000,000 records, 1,024 dimensions per record,
/// 16 MiB of decoded bytes per scope, value name, or dimension, and 64 MiB in
/// the complete encoding. Scope and value name are non-empty. An empty
/// dimension (`0:`) means that axis has no symbolic identity. A missing
/// metadata property and the canonical zero-record value `HSDI1\n0\n` both
/// provide no symbolic proofs; cache identity normalizes missing metadata to
/// that zero-record value.
///
/// HSDI1 is immutable. An incompatible grammar, field meaning, ordering, or
/// bound requires a new encoding magic and new versioned metadata key/module
/// attribute. A resolver-only semantic change instead updates
/// kOnnxDimParamsResolverPolicyVersion. Decoders reject unknown magic.
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
