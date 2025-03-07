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

#include <glog/logging.h>
// include glog/logging.h to define CHECK before include vaip_plugin.hpp
#include "vaip/plugin.hpp"
#include "vaip/vaip_plugin.hpp"
#include "vitis/ai/weak.hpp"

namespace vaip_core {
std::string Plugin::guess_name(const char* name) {
#ifdef _WIN32
  return std::string("") + name + ".dll";
#else
  return std::string("lib") + name + ".so";
#endif
}
struct Tag_Plugin_Func_Set {
  vitis::ai::plugin_t (*open_plugin)(const std::string& name,
                                     vitis::ai::scope_t scope);
  void* (*plugin_sym)(vitis::ai::plugin_t plugin, const std::string& name);
  void (*close_plugin)(vitis::ai::plugin_t plugin);
};

Plugin_Func_Set g_static_plugin_func_set = {
    vitis::ai::open_plugin, vitis::ai::plugin_sym, vitis::ai::close_plugin};

Plugin_Func_Set g_dynamic_plugin_func_set = {
    vaip_core::open_plugin, vaip_core::plugin_sym, vaip_core::close_plugin};

Plugin_Func_Set* g_static_plugin_func_set_ptr = &g_static_plugin_func_set;
Plugin_Func_Set* g_dynamic_plugin_func_set_ptr = &g_dynamic_plugin_func_set;

Plugin::Plugin(const char* name, Plugin_Func_Set* func_set)
    : name_{name}, so_name_{guess_name(name)}, func_set_{func_set},
      plugin_{func_set->open_plugin(so_name_, vitis::ai::scope_t::PUBLIC)} {
  CHECK(plugin_ != nullptr) << "cannot open plugin: "
                            << "name_ " << name_ << " "       //
                            << "so_name_ " << so_name_ << " " //
      ;
}

Plugin::~Plugin() { func_set_->close_plugin((vitis::ai::plugin_t)plugin_); }

std::unordered_map<std::string, std::shared_ptr<Plugin>> Plugin::store_;

Plugin* Plugin::get(const std::string& plugin_name, Plugin_Func_Set* func_set) {
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] = vitis::ai::WeakStore<std::string, Plugin>::create(
        plugin_name, plugin_name.c_str(), func_set);
  }
  it = store_.find(plugin_name);
  CHECK(it != store_.end())
      << "cannot load plugin. plugin_name=" << plugin_name;
  return it->second.get();
}
void* Plugin::my_plugin_sym(void* handle, const char* name) {
  return func_set_->plugin_sym((vitis::ai::plugin_t)handle, name);
}
} // namespace vaip_core
