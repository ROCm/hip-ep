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
#include <sstream>
// clang-format off
// we must include glog before morphizen headers
#include <glog/logging.h>
// clang-format on
#include "morphizen/env_config.hpp"
#include "morphizen/vaip_plugin.hpp"
#include "morphizen/weak.hpp"
// include local headers
#include "./cleanup.hpp"
DEF_ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN) >= n)
namespace vaip_core {
struct Tag_Plugin_Func_Set {
  plugin_t (*open_plugin)(const std::string& name, scope_t scope);
  void* (*plugin_sym)(plugin_t plugin, const std::string& name);
  void (*close_plugin)(plugin_t plugin);
};

std::string Plugin::guess_name(const char* name) {
#ifdef _WIN32
  return std::string("") + name + ".dll";
#else
  return std::string("lib") + name + ".so";
#endif
}

Plugin::Plugin(const char* name, Plugin_Func_Set* func_set)
    : name_{name}, so_name_{guess_name(name)}, func_set_{func_set},
      plugin_{func_set->open_plugin(so_name_, scope_t::PUBLIC)} {
  CHECK(plugin_ != nullptr) << "cannot open plugin: "
                            << "name_ " << name_ << " "       //
                            << "so_name_ " << so_name_ << " " //
      ;
  MY_LOG(1) << "  -- open plugin: " << name_ << " " << so_name_
            << " this=" << (void*)this;
}

Plugin::~Plugin() {
  func_set_->close_plugin((plugin_t)plugin_);
  MY_LOG(1) << "  -- close plugin: " << name_ << " " << so_name_
            << " this=" << (void*)this;
}
// TODO: for now all Plugin::get store loaded plugin in a global
// store. it is not a good solution, it is possible to put all plugin
// into pass_context?
static std::unordered_map<std::string, std::shared_ptr<Plugin>>&
get_global_plugin_store() {
  static std::unordered_map<std::string, std::shared_ptr<Plugin>> store_;
  static bool init = false;
  if (init == false) {
    init = true;
    add_cleanup_function("cleanup global plugin store",
                         []() { store_.clear(); });
  }
  return store_;
}

Plugin* Plugin::get(const std::string& plugin_name, Plugin_Func_Set* func_set) {
  auto& store_ = get_global_plugin_store();
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] = morphizen::WeakStore<std::string, Plugin>::create(
        plugin_name, plugin_name.c_str(), func_set);
  }
  it = store_.find(plugin_name);
  CHECK(it != store_.end())
      << "cannot load plugin. plugin_name=" << plugin_name;
  return it->second.get();
}
void* Plugin::my_plugin_sym(void* handle, const char* name) const {
  return func_set_->plugin_sym((plugin_t)handle, name);
}

static std::unordered_map<std::string, std::unordered_map<std::string, void*>>&
get_store() {
  static std::unordered_map<std::string, std::unordered_map<std::string, void*>>
      store_;
  return store_;
}
plugin_t open_plugin_static(const std::string& name, scope_t scope) {
  return reinterpret_cast<plugin_t>(new std::string(name));
}

void* plugin_sym_static(plugin_t plugin, const std::string& symbol) {
  auto& name = *reinterpret_cast<std::string*>(plugin);
  auto& store = get_store();
  auto it_lib = store.find(name);
  if (it_lib == store.end()) {
    std::cerr << "cannot find lib:" << name << std::endl;
    std::cerr << "valid libs are: " << std::endl;
    for (auto& x : store) {
      std::cerr << "  libs=" << x.first << "\n";
    }
    return nullptr;
  }
  auto it_sym = it_lib->second.find(symbol);
  if (it_sym == it_lib->second.end()) {
    std::cerr << "cannot find symbol " << symbol << " in " << name << std::endl;
    std::cerr << "valid symbols are: " << std::endl;
    for (auto& x : it_lib->second) {
      std::cerr << "  symbols=" << x.first << "\n";
    }
    return nullptr;
  }
  return it_sym->second;
}
void close_plugin_static(plugin_t plugin) {
  delete reinterpret_cast<std::string*>(plugin);
}

std::string plugin_error_static(plugin_t plugin) { return "N/A"; }

void register_plugin_static(const std::string& name, const std::string& symbol,
                            void* addr) {
  MY_LOG(1) << "register: " << name << " " << symbol << " " << addr;
  get_store()[name][symbol] = addr;
}

void unregister_plugin_static(const std::string& name,
                              const std::string& symbol, void* addr) {
  MY_LOG(1) << "unregister: " << name << " " << symbol << " " << addr;
  auto it = get_store().find(name);
  if (it == get_store().end()) {
    MY_LOG(1) << "cannot find lib: " << name;
    return;
  }
  auto it_sym = it->second.find(symbol);
  if (it_sym == it->second.end()) {
    MY_LOG(1) << "cannot find symbol: " << symbol;
    return;
  }
  it->second.erase(it_sym);
  return;
}

StaticPluginRegister::StaticPluginRegister(const char* name, const char* symbol,
                                           void* addr)
    : name_(name), symbol_(symbol), addr_(addr) {
#if _WIN32
  std::string lib_name = std::string(name) + ".dll";
#else
  std::string lib_name = std::string("lib") + name + ".so";
#endif
  register_plugin_static(lib_name, symbol, addr);
  std::ostringstream str;
  str << "unload " << symbol << " @ " << name;
  add_cleanup_function(str.str(), [lib_name, symbol, addr]() {
    register_plugin_static(lib_name, symbol, addr);
  });
}
StaticPluginRegister::~StaticPluginRegister() {}

Plugin_Func_Set g_static_plugin_func_set = {
    open_plugin_static, plugin_sym_static, close_plugin_static};

Plugin_Func_Set g_dynamic_plugin_func_set = {vaip_core::open_plugin_dyn,
                                             vaip_core::plugin_sym_dyn,
                                             vaip_core::close_plugin_dyn};

Plugin_Func_Set* g_static_plugin_func_set_ptr = &g_static_plugin_func_set;
Plugin_Func_Set* g_dynamic_plugin_func_set_ptr = &g_dynamic_plugin_func_set;
#if _WIN32

#else

#endif
} // namespace vaip_core
