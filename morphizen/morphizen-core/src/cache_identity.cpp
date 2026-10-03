/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/cache_identity.hpp"

#include "morphizen/graph.hpp"
#include "morphizen/model.hpp"
#include "morphizen/morphizen_ort_api.h"
#include "morphizen/symbolic_dims.hpp"
#include "morphizen/util.hpp"
#include "sha256.h"

#include <algorithm>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message_lite.h>
#include <sstream>

namespace morphizen {
namespace {

constexpr std::string_view kFinalizedCacheKeyPrefix = "hsdi1-";
constexpr std::string_view kOrtMemoryAddressLocation = "*/_ORT_MEM_ADDR_/*";

void add_framed(SHA256 &sha256, std::string_view value) {
  uint64_t size = value.size();
  unsigned char big_endian_size[sizeof(size)];
  for (size_t i = 0; i < sizeof(size); ++i) {
    unsigned shift = static_cast<unsigned>((sizeof(size) - i - 1) * 8);
    big_endian_size[i] = static_cast<unsigned char>((size >> shift) & 0xff);
  }
  sha256.add(big_endian_size, sizeof(big_endian_size));
  if (!value.empty())
    sha256.add(value.data(), value.size());
}

InitializerDigest
digest_initializer(const Graph &graph,
                   morphizen_cxx::NodeArgConstRef initializer) {
  std::string location;
  size_t offset = 0;
  size_t size = 0;
  size_t checksum = 0;
  (void)MORPHIZEN_ORT_API(node_arg_external_location)(
      graph, *initializer.ptr(), location, offset, size, checksum);

  std::string digest;
  if (location == kOrtMemoryAddressLocation) {
    digest = get_sha256_of_buffer(
        reinterpret_cast<const void *>(static_cast<uintptr_t>(offset)), size);
  } else if (!location.empty() && location.front() != '<') {
    digest =
        get_sha256_of_file_slice(location, static_cast<int64_t>(offset), size);
  } else {
    gsl::span<const char> data = initializer.const_data_as_raw();
    digest = get_sha256_of_buffer(data.data(), data.size());
  }
  return {initializer.name(), std::move(digest)};
}

} // namespace

bool is_lowercase_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

std::string compute_framed_sha256(const std::vector<std::string> &fields) {
  SHA256 sha256;
  for (const std::string &field : fields)
    add_framed(sha256, field);
  return sha256.getHash();
}

std::string finalize_cache_key(const CacheKeyInputs &inputs) {
  std::string symbolic_error;
  if (!decode_symbolic_dim_records(inputs.symbolic_metadata, symbolic_error))
    throw std::invalid_argument("invalid symbolic dimension metadata: " +
                                symbolic_error);
  if (!inputs.initializer_digest.empty() &&
      !is_lowercase_sha256(inputs.initializer_digest))
    throw std::invalid_argument("invalid initializer data SHA-256 metadata");
  if (!inputs.compiler_graph_digest.empty() &&
      !is_lowercase_sha256(inputs.compiler_graph_digest))
    throw std::invalid_argument("invalid compiler graph SHA-256 metadata");

  SHA256 sha256;
  add_framed(sha256, "hipdnn.symbolic.extent.cache.v1");
  add_framed(sha256, inputs.graph_digest);
  add_framed(sha256, inputs.base_key);
  add_framed(sha256, inputs.compiler_contract);
  add_framed(sha256, inputs.symbolic_metadata);
  add_framed(sha256, inputs.initializer_digest);
  add_framed(sha256, inputs.compiler_graph_digest);
  add_framed(sha256, kOnnxDimParamsEncodingVersion);
  add_framed(sha256, kOnnxDimParamsResolverPolicyVersion);
  return std::string(kFinalizedCacheKeyPrefix) + sha256.getHash();
}

bool is_finalized_cache_key(std::string_view key) {
  return key.size() == kFinalizedCacheKeyPrefix.size() + 64 &&
         key.substr(0, kFinalizedCacheKeyPrefix.size()) ==
             kFinalizedCacheKeyPrefix &&
         is_lowercase_sha256(key.substr(kFinalizedCacheKeyPrefix.size()));
}

CacheLoadAction select_cache_load_action(CacheLoadKind kind,
                                         bool initializer_digest_finalized,
                                         bool cache_available,
                                         std::string_view expected_key,
                                         std::string_view loaded_key) {
  if (kind == CacheLoadKind::Prebuilt &&
      (!initializer_digest_finalized || !cache_available))
    return CacheLoadAction::Recompile;
  if (!loaded_key.empty() && loaded_key == expected_key)
    return CacheLoadAction::Use;
  return kind == CacheLoadKind::Prebuilt ? CacheLoadAction::Recompile
                                         : CacheLoadAction::Reject;
}

std::string
serialize_deterministically(const google::protobuf::MessageLite &message) {
  std::string result;
  bool success = false;
  {
    google::protobuf::io::StringOutputStream output(&result);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    success = message.SerializeToCodedStream(&coded) && !coded.HadError();
  }
  if (!success)
    throw std::runtime_error(
        "cannot serialize compiler contract for cache identity");
  return result;
}

std::string
combine_initializer_digests(std::vector<InitializerDigest> digests) {
  std::sort(digests.begin(), digests.end(),
            [](const InitializerDigest &lhs, const InitializerDigest &rhs) {
              return lhs.name < rhs.name;
            });
  std::string framed = "HIDI1\n";
  for (const InitializerDigest &digest : digests) {
    if (!is_lowercase_sha256(digest.sha256))
      throw std::invalid_argument("invalid initializer SHA-256 digest");
    framed += std::to_string(digest.name.size()) + ":" + digest.name;
    framed += std::to_string(digest.sha256.size()) + ":" + digest.sha256;
  }
  return get_sha256_of_buffer(framed.data(), framed.size());
}

std::filesystem::path normalize_external_initializer_path(
    const std::filesystem::path &model_path,
    const std::filesystem::path &external_path) {
  std::filesystem::path resolved = external_path;
  if (!external_path.is_absolute()) {
    std::filesystem::path model_dir = model_path.has_parent_path()
                                          ? model_path.parent_path()
                                          : std::filesystem::path();
    resolved = model_dir / external_path;
  }
  return std::filesystem::absolute(resolved).lexically_normal();
}

bool verify_external_initializer_slice(const ExternalInitializerSlice &slice,
                                       std::string &error) {
  error.clear();
  if (!is_lowercase_sha256(slice.sha256)) {
    error = "invalid expected initializer SHA-256";
    return false;
  }
  try {
    std::string actual =
        get_sha256_of_file_slice(slice.path, slice.offset, slice.size);
    if (actual != slice.sha256) {
      error = "external initializer changed: " + slice.path.string();
      return false;
    }
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
  return true;
}

std::string compute_graph_initializer_digest(const Graph &graph) {
  std::vector<InitializerDigest> digests;
  for (morphizen_cxx::NodeArgConstRef initializer :
       morphizen_cxx::GraphConstRef(graph).constant_initializers())
    digests.push_back(digest_initializer(graph, initializer));
  return combine_initializer_digests(std::move(digests));
}

bool verify_graph_initializer_digest(const Graph &graph, std::string &error) {
  error.clear();
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  const auto &model = graph_ref.model();
  auto model_ref = morphizen_cxx::ModelConstRef(model);
  if (!model_ref.has_metadata(kInitializerDataDigestMetadataKey))
    return true;

  std::string expected =
      model_ref.get_metadata(kInitializerDataDigestMetadataKey);
  try {
    std::string actual = compute_graph_initializer_digest(graph);
    if (actual != expected) {
      error = "initializer data changed after cache identity finalization";
      return false;
    }
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
  return true;
}

} // namespace morphizen
