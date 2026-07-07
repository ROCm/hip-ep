/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#  define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include "morphizen-foundation/mem_binary.hpp"
#include "morphizen-foundation/env_config.hpp"
#include <glog/logging.h>
#include <iostream>
#include <unordered_map>
#ifdef ENABLE_COMPRESSION
#  include <zlib.h>
#endif
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MEM_XCLBIN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MEM_XCLBIN) >= n)

namespace morphizen {
struct CompressionInfo {
  const uint8_t* data;
#ifdef ENABLE_COMPRESSION
  size_t compressed_size;
#endif
  size_t origin_size;
#ifdef ENABLE_COMPRESSION
  CompressionInfo(const uint8_t* d, size_t c, size_t o)
      : data(d), compressed_size(c), origin_size(o) {}
#else
  CompressionInfo(const uint8_t* d, size_t o) : data(d), origin_size(o) {}
#endif
};

#ifdef ENABLE_COMPRESSION
std::vector<char> uncompress(const uint8_t* byte, size_t compressed_size,
                             size_t origin_size) {
  std::vector<char> ret;
  ret.resize(static_cast<size_t>(origin_size));
  z_stream infstream;
  infstream.zalloc = Z_NULL;
  infstream.zfree = Z_NULL;
  infstream.opaque = Z_NULL;
  infstream.avail_in = static_cast<unsigned int>(compressed_size);
  infstream.next_in = reinterpret_cast<Bytef*>(
      const_cast<char*>(reinterpret_cast<const char*>(byte)));
  infstream.avail_out = static_cast<unsigned int>(origin_size);
  infstream.next_out = reinterpret_cast<Bytef*>(ret.data());
  inflateInit(&infstream);
  inflate(&infstream, Z_NO_FLUSH);
  inflateEnd(&infstream);
  return ret;
}
#endif
#include "mem_binary_file.hpp.inc"
std::vector<char> get_mem_binary(const std::string& filename) {
  auto span = get_mem_binary_span(filename);
  std::vector<char> ret(span->data(), span->data() + span->size());
  return ret;
}

bool has_mem_binary(const std::string& filename) {
  return get_mem_binary_span(filename) != std::nullopt;
}

std::optional<gsl::span<const char>>
get_mem_binary_span(const std::string& filename) {
  thread_local std::unordered_map<std::string, std::vector<char>> store;
  auto iter = binary_map.find(filename);
  if (iter == binary_map.end()) {
    MY_LOG(1) << " -- mem_binary not found: " << filename;
    return std::nullopt;
  }
  MY_LOG(1) << "  -- found mem_binary: " << filename
            << " from builtin inside morphizen";
  auto info = iter->second;
#ifdef ENABLE_COMPRESSION
  if (info.compressed_size == 0) {
    return gsl::span<const char>(reinterpret_cast<const char*>(info.data),
                                 info.origin_size);
  }
  if (store.find(filename) == store.end()) {
    store[filename] =
        uncompress(info.data, info.compressed_size, info.origin_size);
  }
  const auto& data = store.find(filename)->second;

  return gsl::span<const char>(data.data(), data.size());
#else
  return gsl::span<const char>(reinterpret_cast<const char*>(info.data),
                               info.origin_size);
#endif
}
} // namespace morphizen
