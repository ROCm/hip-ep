/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sstream>
#include <string>
#include <utility>

namespace morphizen {
using plugin_t = void*;
enum class scope_t { PUBLIC, PRIVATE };

std::pair<plugin_t, bool> open_plugin_dyn(const std::string& name,
                                          scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  void* h = dlopen(name.c_str(),
                   scope == scope_t::PUBLIC ? flag_public : flag_private);
  if (!h) {
    // Capture dlerror() before any other libc call: dlfcn state is per-thread
    // but any other dlfcn-touching code in the callback chain would clobber it.
    const char* err = dlerror();
    // Gate the print behind MORPHIZEN_DEBUG_PLUGIN to match the rest of the
    // plugin diagnostics (see MY_LOG in morphizen_plugin.cpp). Callers probe
    // optional plugins in a static-then-dynamic fallback chain, so the
    // dynamic-leg failure is expected on the happy path and would otherwise
    // be log spam. We use std::getenv directly rather than DEF_ENV_PARAM to
    // avoid ODR-redefining ENV_PARAM_MORPHIZEN_DEBUG_PLUGIN (already defined
    // in morphizen_plugin.cpp's TU).
    const char* dbg = std::getenv("MORPHIZEN_DEBUG_PLUGIN");
    if (dbg && dbg[0] != '\0' && dbg[0] != '0')
      std::fprintf(stderr, "[morphizen] dlopen(\"%s\") failed: %s\n",
                   name.c_str(), err ? err : "(no dlerror)");
  }
  return {h, true};
}
void* plugin_sym_dyn(plugin_t plugin, const std::string& name) {
  dlerror(); // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error_dyn(plugin_t /*plugin*/) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin_dyn(plugin_t plugin) { dlclose(plugin); }
} // namespace morphizen
