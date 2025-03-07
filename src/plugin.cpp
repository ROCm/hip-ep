/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include "vaip/plugin.hpp"

#include <glog/logging.h>

#include <iostream>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_VART_PLUGIN, "0")
#include <sstream>
#include <string>
#include <unordered_map>
namespace vaip_core {

#if _WIN32
// clang-format off
#  include <windows.h>
// clang-format on
#  include <libloaderapi.h>

static std::unordered_map<std::string, std::unordered_map<std::string, void*>>&
get_store() {
  static std::unordered_map<std::string, std::unordered_map<std::string, void*>>
      store_;
  return store_;
}

static std::wstring s2ws(const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}

vitis::ai::plugin_t open_plugin(const std::string& name,
                                vitis::ai::scope_t scope) {
  static_assert(sizeof(vitis::ai::plugin_t) == sizeof(HMODULE));
  return LoadLibraryW(s2ws(name).c_str());
}

void* plugin_sym(vitis::ai::plugin_t plugin, const std::string& name) {
  return GetProcAddress(static_cast<HMODULE>(plugin), name.c_str());
}

void close_plugin(vitis::ai::plugin_t plugin) {
  FreeLibrary(static_cast<HMODULE>(plugin));
}
#else
#  include <dlfcn.h>
vitis::ai::plugin_t open_plugin(const std::string& name,
                                vitis::ai::scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  return dlopen(name.c_str(), scope == vitis::ai::scope_t::PUBLIC
                                  ? flag_public
                                  : flag_private);
}
void* plugin_sym(vitis::ai::plugin_t plugin, const std::string& name) {
  dlerror(); // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error(vitis::ai::plugin_t plugin) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin(vitis::ai::plugin_t plugin) { dlclose(plugin); }
#endif

} // namespace vaip_core