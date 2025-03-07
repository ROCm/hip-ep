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

#pragma once
#include "./_sanity_check.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vaip/export.h>
namespace vaip_core {
struct Tag_Plugin_Func_Set;
typedef Tag_Plugin_Func_Set Plugin_Func_Set;
extern Plugin_Func_Set* g_static_plugin_func_set_ptr;
extern Plugin_Func_Set* g_dynamic_plugin_func_set_ptr;
struct Plugin {
  VAIP_DLL_SPEC
  Plugin(const char* name,
         Plugin_Func_Set* func_set = g_static_plugin_func_set_ptr);
  VAIP_DLL_SPEC ~Plugin();
  template <typename R, typename... Args>
  R invoke(const char* name, Args... args) {
    auto sym = my_plugin_sym(plugin_, name);
#if defined(CHECK)
    CHECK(sym != nullptr) << "no such function: " << name << "; " //
                          << "libname " << name_ << " "           //
                          << "so_name " << so_name_ << " "        //
        ;
#else
#  ifndef USE_VITISAI
#    warning "it would be better to include <glog/logging.h>"
#  endif
#endif
    typedef R (*fun_type_t)(Args...);
    fun_type_t f = reinterpret_cast<fun_type_t>(sym);
    return f(std::forward<Args>(args)...);
  }
  static Plugin* get(const std::string& name,
                     Plugin_Func_Set* func_set = g_static_plugin_func_set_ptr);

private:
  std::string name_;
  std::string so_name_;
  Plugin_Func_Set* func_set_;
  void* plugin_;

private:
  static std::string guess_name(const char* name);
  static std::unordered_map<std::string, std::shared_ptr<Plugin>> store_;
  void* my_plugin_sym(void*, const char*);
};

template <typename T, typename... Args> class WithPlugin {
public:
  static std::unique_ptr<T> create(const std::string& plugin_name,
                                   Plugin_Func_Set* func_set, Args... args) {
    auto plugin = Plugin::get(plugin_name, func_set);
    auto ret = plugin->invoke<T*>(T::entry_point, args...);
    return std::unique_ptr<T>(ret);
  }
};
} // namespace vaip_core
