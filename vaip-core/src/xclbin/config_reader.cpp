/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "vaip/config_reader.hpp"
#include "nlohmann/json.hpp"
#include "core/session/onnxruntime_cxx_api.h"
#include "vitis/ai/env_config.hpp"
#include "vaip/xclbin_file.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
DEF_ENV_PARAM_2(DEBUG_LOG_LEVEL, "error", std::string)
DEF_ENV_PARAM(XLNX_ENABLE_OLD_QDQ, "1")
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0")
DEF_ENV_PARAM(XLNX_ENABLE_PY3_ROUND, "0")

namespace vaip_core {
static const std::string get_default_config() {
#include "config_json_binary.hpp"
  std::string ret(reinterpret_cast<const char*>(config), sizeof(config));
  return ret;
}

// FIXME: The name of this function is misleading.
static nlohmann::json
get_config_json_str_from_config_file(const std::string& filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::string err_msg =
        std::string{"failed to open config file: "} + filename;
    throw std::runtime_error(err_msg);
  }
  return nlohmann::json::parse(f);
}

static void
update_enable_batch(const onnxruntime::ProviderOptions& session_option) {
  if (session_option.find("xlnx_enable_batch") != session_option.end()) {
    std::string enable_batch = session_option.at("xlnx_enable_batch");
    if (enable_batch == "1") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    } else if (enable_batch == "0") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 0;
    } else {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    }
  }
}

static void
update_enable_old_qdq(const onnxruntime::ProviderOptions& session_option) {
  if (session_option.find("xlnx_enable_old_qdq") != session_option.end()) {
    std::string enable_old_qdq = session_option.at("xlnx_enable_old_qdq");
    if (enable_old_qdq == "1") {
      ENV_PARAM(XLNX_ENABLE_OLD_QDQ) = 1;
    } else if (enable_old_qdq == "0") {
      ENV_PARAM(XLNX_ENABLE_OLD_QDQ) = 0;
    } else {
      ENV_PARAM(XLNX_ENABLE_OLD_QDQ) = 1;
    }
#ifdef _WIN32
    _putenv_s("XLNX_ENABLE_OLD_QDQ", enable_old_qdq.c_str());
#else
    setenv("XLNX_ENABLE_OLD_QDQ", enable_old_qdq.c_str(), 1);
#endif
  }
}

static void
update_py3_round(const onnxruntime::ProviderOptions& session_option) {
  if (session_option.find("xlnx_enable_py3_round") != session_option.end()) {
    std::string py3_round = session_option.at("xlnx_enable_py3_round");
    if (py3_round == "1") {
      ENV_PARAM(XLNX_ENABLE_PY3_ROUND) = 1;
    } else {
      py3_round = std::string("0");
      ENV_PARAM(XLNX_ENABLE_PY3_ROUND) = 0;
    }
#ifdef _WIN32
    _putenv_s("XLNX_ENABLE_PY3_ROUND", py3_round.c_str());
#else
    setenv("XLNX_ENABLE_PY3_ROUND", py3_round.c_str(), 1);
#endif
  }
}

static void
update_log_level(const onnxruntime::ProviderOptions& session_option) {
  std::string log_level = ENV_PARAM(DEBUG_LOG_LEVEL);
  if (session_option.find("log_level") != session_option.end()) {
    log_level = session_option.at("log_level");
  }

  if (log_level == "info") {
    FLAGS_minloglevel = google::GLOG_INFO;
  } else if (log_level == "warning") {
    FLAGS_minloglevel = google::GLOG_WARNING;
  } else if (log_level == "error") {
    FLAGS_minloglevel = google::GLOG_ERROR;
  } else if (log_level == "fatal") {
    FLAGS_minloglevel = google::GLOG_FATAL;
  } else {
    FLAGS_minloglevel = google::GLOG_ERROR;
  }
}
static void restore_session_options(nlohmann::json& ret,
                                    std::string entry_second) {
  std::map<std::string, std::string> session_config_options_entry_list = {};
#if VAIP_ORT_API_MAJOR >= 10
  auto options = reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(entry_second));
  VAIP_ORT_API(session_option_configuration)
  (&session_config_options_entry_list, options,
   [](void* mmap, const char* name, const char* value) {
     auto* map_ptr =
         reinterpret_cast<std::map<std::string, std::string>*>(mmap);
     map_ptr->insert({name, value});
   });
#else
  LOG(WARNING) << "ORT API version is less than 10, now used is "
               << VAIP_ORT_API_MAJOR
               << ", session options will not be restored.";
#endif
  for (const auto& option : session_config_options_entry_list) {
    std::string key = option.first;
    // Here are the actual session_configs.
    ret["ort_session_config"][key] = option.second;
  }
}

static nlohmann::json
get_config_json(const onnxruntime::ProviderOptions& options) {
  nlohmann::json ret;
  update_log_level(options);
  update_enable_batch(options);
  update_enable_old_qdq(options);
  update_py3_round(options);
  bool config_set = options.find("config_file") != options.end();
  if (config_set) {
    ret = get_config_json_str_from_config_file(options.at("config_file"));
  } else {
    ret = nlohmann::json::parse(get_default_config());
  }
  bool xclbin_in_config_file = ret.count("xclbin") > 0;
  const std::string session_prefix = "ort_session_config.";
  for (const auto& entry : options) {
    // In the case where xclbin are (mistakenly) specified both in
    // `ProviderOptions["config_file"]` and in `ProvidersOptions["xclbin"]`.
    if (entry.first == "xclbin" && xclbin_in_config_file) {
      continue;
    }
    if (entry.first == "session_options") {
      // The key here is "session_options," and the value is a string that
      // points to the session_options object.
      restore_session_options(ret, entry.second);
      continue;
    }
    // this is actually provider options, for backward compatibility, we keep
    // this key in the root of the json.
    ret["sessionOptions"][entry.first] = entry.second;
  }
  return ret;
}

std::string get_config_json_str(const onnxruntime::ProviderOptions& options) {
  try {
    auto data = vaip_core::get_config_json(options);
    return data.dump();
  } catch (const std::exception& e) {
    LOG(FATAL) << "Error: " << e.what() << std::endl;
    return "";
  }
}

Ort::SessionOptions*
get_session_option(const onnxruntime::ProviderOptions& options) {
  auto iter = options.find("session_options");
  if (iter == options.end()) {
    return nullptr;
  }
  return reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(iter->second));
}
} // namespace vaip_core
