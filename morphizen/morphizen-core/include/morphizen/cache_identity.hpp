/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen/export.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace google {
namespace protobuf {
class MessageLite;
} // namespace protobuf
} // namespace google

namespace onnxruntime {
class Graph;
} // namespace onnxruntime

namespace morphizen {

struct CacheKeyInputs {
  std::string_view base_key;
  std::string_view graph_digest;
  std::string_view compiler_contract;
  std::string_view symbolic_metadata;
  std::string_view initializer_digest;
  std::string_view compiler_graph_digest;
};

enum class CacheLoadKind {
  Fresh,
  Prebuilt,
  EpContext,
};

enum class CacheLoadAction {
  Use,
  Recompile,
  Reject,
};

struct InitializerDigest {
  std::string name;
  std::string sha256;
};

struct ExternalInitializerSlice {
  std::filesystem::path path;
  int64_t offset = 0;
  size_t size = 0;
  std::string sha256;
};

class MORPHIZEN_DLL_SPEC CacheIntegrityError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

MORPHIZEN_DLL_SPEC bool is_lowercase_sha256(std::string_view value);

MORPHIZEN_DLL_SPEC std::string
compute_framed_sha256(const std::vector<std::string> &fields);

MORPHIZEN_DLL_SPEC std::string finalize_cache_key(const CacheKeyInputs &inputs);

MORPHIZEN_DLL_SPEC bool is_finalized_cache_key(std::string_view key);

MORPHIZEN_DLL_SPEC CacheLoadAction select_cache_load_action(
    CacheLoadKind kind, bool initializer_digest_finalized, bool cache_available,
    std::string_view expected_key, std::string_view loaded_key);

MORPHIZEN_DLL_SPEC std::string
serialize_deterministically(const google::protobuf::MessageLite &message);

MORPHIZEN_DLL_SPEC std::string
combine_initializer_digests(std::vector<InitializerDigest> digests);

MORPHIZEN_DLL_SPEC std::filesystem::path
normalize_external_initializer_path(const std::filesystem::path &model_path,
                                    const std::filesystem::path &external_path);

MORPHIZEN_DLL_SPEC bool
verify_external_initializer_slice(const ExternalInitializerSlice &slice,
                                  std::string &error);

MORPHIZEN_DLL_SPEC std::string
compute_graph_initializer_digest(const onnxruntime::Graph &graph);

MORPHIZEN_DLL_SPEC bool
verify_graph_initializer_digest(const onnxruntime::Graph &graph,
                                std::string &error);

} // namespace morphizen
