/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/cache_identity.hpp"
#include "morphizen/config.pb.h"
#include "morphizen/morphizen.hpp"
#include "morphizen/symbolic_dims.hpp"
#include "morphizen/util.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAILED: " << message << "\n";
  return condition;
}

} // namespace

int main() {
  using morphizen::CacheKeyInputs;
  using morphizen::CacheLoadAction;
  using morphizen::CacheLoadKind;

  const std::string empty_symbols = morphizen::encode_symbolic_dim_records({});
  const std::string partition_a = morphizen::encode_symbolic_dim_records(
      {{"main_graph", "lhs", {"N"}}, {"main_graph", "rhs", {"N"}}});
  const std::string partition_b = morphizen::encode_symbolic_dim_records(
      {{"main_graph", "lhs", {"N"}}, {"main_graph", "rhs", {"M"}}});
  const std::string digest_a(64, 'a');
  const std::string digest_b(64, 'b');

  CacheKeyInputs inputs{"base",        "graph",  "contract-a",
                        empty_symbols, digest_a, digest_b};
  const std::string first = morphizen::finalize_cache_key(inputs);
  bool ok = check(first == morphizen::finalize_cache_key(inputs),
                  "cache key must be deterministic");
  ok &= check(morphizen::is_finalized_cache_key(first),
              "final key must use canonical lowercase SHA-256 form");

  inputs.symbolic_metadata = partition_a;
  const std::string symbolic_a = morphizen::finalize_cache_key(inputs);
  inputs.symbolic_metadata = partition_b;
  ok &= check(symbolic_a != morphizen::finalize_cache_key(inputs),
              "symbolic partition must change the cache key");
  inputs.symbolic_metadata = partition_a;
  inputs.compiler_contract = "contract-b";
  ok &= check(symbolic_a != morphizen::finalize_cache_key(inputs),
              "compiler contract must change the cache key");

  ok &= check(morphizen::select_cache_load_action(CacheLoadKind::Fresh, true,
                                                  true, first, first) ==
                  CacheLoadAction::Use,
              "matching fresh context must be accepted");
  ok &= check(morphizen::select_cache_load_action(CacheLoadKind::Fresh, true,
                                                  true, first, symbolic_a) ==
                  CacheLoadAction::Reject,
              "fresh context mismatch must be rejected");
  ok &= check(morphizen::select_cache_load_action(CacheLoadKind::Prebuilt, true,
                                                  true, first, symbolic_a) ==
                  CacheLoadAction::Recompile,
              "prebuilt mismatch must trigger a fresh compile");
  ok &= check(morphizen::select_cache_load_action(CacheLoadKind::Prebuilt,
                                                  false, true, first, first) ==
                  CacheLoadAction::Recompile,
              "prebuilt cache without initializer identity must be bypassed");
  ok &= check(morphizen::select_cache_load_action(CacheLoadKind::Prebuilt, true,
                                                  false, first, first) ==
                  CacheLoadAction::Recompile,
              "unavailable prebuilt cache must be bypassed");
  ok &= check(morphizen::select_cache_load_action(
                  CacheLoadKind::EpContext, true, true, first, symbolic_a) ==
                  CacheLoadAction::Reject,
              "EPContext mismatch must be rejected");

  morphizen::ConfigProto config;
  (*config.mutable_provider_options())["z"] = "last";
  (*config.mutable_provider_options())["a"] = "first";
  const std::string serialized = morphizen::serialize_deterministically(config);
  morphizen::ConfigProto parsed;
  ok &= check(!serialized.empty() && parsed.ParseFromString(serialized),
              "deterministic protobuf bytes must be finalized before return");
  ok &= check(serialized == morphizen::serialize_deterministically(config),
              "protobuf serialization must be deterministic");

  std::vector<morphizen::InitializerDigest> digests{{"z", digest_a},
                                                    {"a", digest_b}};
  std::vector<morphizen::InitializerDigest> reversed{{"a", digest_b},
                                                     {"z", digest_a}};
  ok &= check(morphizen::combine_initializer_digests(digests) ==
                  morphizen::combine_initializer_digests(reversed),
              "initializer digest must not depend on traversal order");

  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("hipdnn_initializer_snapshot_" + std::to_string(nonce) + ".bin");
  {
    std::ofstream output(path, std::ios::binary);
    output << "abcdef";
  }
  morphizen::ExternalInitializerSlice slice{
      path, 1, 3, morphizen::get_sha256_of_buffer("bcd", 3)};
  std::string error;
  ok &= check(morphizen::verify_external_initializer_slice(slice, error),
              "unchanged external initializer slice must verify");
  {
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(2);
    output.put('X');
  }
  ok &= check(!morphizen::verify_external_initializer_slice(slice, error),
              "initializer mutation must be rejected");
  std::error_code ignored;
  std::filesystem::remove(path, ignored);

  const std::filesystem::path normalized =
      morphizen::normalize_external_initializer_path(
          std::filesystem::temp_directory_path() / "model" / "." /
              "network.onnx",
          "weights/../weights.bin");
  ok &= check(normalized == (std::filesystem::temp_directory_path() / "model" /
                             "weights.bin")
                                .lexically_normal(),
              "equivalent logical initializer paths must normalize");

  return ok ? 0 : 1;
}
