/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "morphizen/mem_xclbin.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip_plugin.hpp"
#include <glog/logging.h>
#include <iostream>
#include <zlib.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MEM_XCLBIN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MEM_XCLBIN) >= n)
DEF_ENV_PARAM_2(VAIP_XCLBIN_BACKEND, "onnxruntime_vitisai_ep", std::string)

namespace vaip_core {
struct CompressionInfo {
  const uint8_t* data;
  size_t compressed_size;
  size_t origin_size;
  CompressionInfo(const uint8_t* d, size_t c, size_t o)
      : data(d), compressed_size(c), origin_size(o) {}
};

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
#include "mem_xclbin_file.hpp.inc"
std::vector<char> get_mem_xclbin_builtin(const std::string& filename) {
  auto iter = xclbin_map.find(filename);
  auto info = iter->second;
  return uncompress(info.data, info.compressed_size, info.origin_size);
}

std::vector<char> get_mem_xclbin(const std::string& filename) {
  std::vector<char> mem_xclbin;
  auto vaip_get_mem_xclbin_plugin = Plugin::get(ENV_PARAM(VAIP_XCLBIN_BACKEND));
  auto loaded_from_backend = false;
  auto has_mem_xclbin = xclbin_map.find(filename) != xclbin_map.end();
  if (has_mem_xclbin) {
    MY_LOG(1) << "  -- found mem_xclbin: " << filename
              << " from builtin inside morphizen";
    mem_xclbin = get_mem_xclbin_builtin(filename);
  } else {
    if (vaip_get_mem_xclbin_plugin) {
      auto vaip_has_mem_xclbin =
          vaip_get_mem_xclbin_plugin->get_method<bool, const char*>(
              "vaip_has_mem_xclbin");
      if (vaip_has_mem_xclbin) {
        if (vaip_has_mem_xclbin(filename.c_str())) {
          auto vaip_get_mem_xclbin = vaip_get_mem_xclbin_plugin->get_method<
              void, const char*, void*, void (*)(void*, void*, size_t)>(
              "vaip_get_mem_xclbin");
          if (vaip_get_mem_xclbin) {
            vaip_get_mem_xclbin(
                filename.data(), reinterpret_cast<void*>(&mem_xclbin),
                [](void* env, void* data, size_t size) {
                  auto* ret = static_cast<std::vector<char>*>(env);
                  std::swap(*ret,
                            std::vector<char>(static_cast<char*>(data),
                                              static_cast<char*>(data) + size));
                });
            loaded_from_backend = true;
            MY_LOG(1) << "  -- found mem_xclbin: " << filename
                      << " from backend " << ENV_PARAM(VAIP_XCLBIN_BACKEND);
          } else {
            MY_LOG(1) << "  -- cannot found symbol: vaip_get_mem_xclbin "
                      << " from backend " << ENV_PARAM(VAIP_XCLBIN_BACKEND);
          }
        } else {
          MY_LOG(1) << "  -- not found mem_xclbin: " << filename
                    << " from backend " << ENV_PARAM(VAIP_XCLBIN_BACKEND);
        }
      } else {
        MY_LOG(1) << "  -- cannot found symbol: vaip_has_mem_xclbin "
                  << " from backend " << ENV_PARAM(VAIP_XCLBIN_BACKEND);
      }
    } else {
      MY_LOG(1) << "  -- cannot found plugin: "
                << ENV_PARAM(VAIP_XCLBIN_BACKEND);
    }
  }
  return mem_xclbin;
}

bool has_mem_xclbin(const std::string& filename) {
  auto vaip_get_mem_xclbin_plugin = Plugin::get(ENV_PARAM(VAIP_XCLBIN_BACKEND));
  auto ret = xclbin_map.find(filename) != xclbin_map.end();
  if (ret) {
    MY_LOG(1) << "  -- found mem_xclbin: " << filename
              << " from builtin inside morphizen";
  } else {
    if (vaip_get_mem_xclbin_plugin) {
      auto vaip_has_mem_xclbin =
          vaip_get_mem_xclbin_plugin->get_method<bool, const char*>(
              "vaip_get_mem_xclbin");
      if (vaip_has_mem_xclbin) {
        if (vaip_has_mem_xclbin(filename.c_str())) {
          MY_LOG(1) << "  -- found mem_xclbin: " << filename << " from backend "
                    << ENV_PARAM(VAIP_XCLBIN_BACKEND);
          ret = true;
        } else {
          MY_LOG(1) << "  -- not found mem_xclbin: " << filename
                    << " from backend " << ENV_PARAM(VAIP_XCLBIN_BACKEND);
        }
      } else {
        MY_LOG(1) << "  -- cannot found symbol: vaip_get_mem_xclbin "
                  << " from backend ";
      }
    } else {
      MY_LOG(1) << "  -- cannot found plugin: "
                << ENV_PARAM(VAIP_XCLBIN_BACKEND);
    }
    if (ret == false) {
      MY_LOG(1) << "  -- not found mem_xclbin: " << filename
                << " from backend, trying builtin inside morphizen";
      ret = xclbin_map.find(filename) != xclbin_map.end();
      if (ret) {
        MY_LOG(1) << "  -- found mem_xclbin: " << filename
                  << " from builtin inside morphizen";
      } else {
        MY_LOG(1) << "  -- not found mem_xclbin: " << filename
                  << " from builtin inside morphizen";
      }
    }
  }
  return ret;
}
} // namespace vaip_core
