/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// NOTE: it would be better that platform speicif codes go to a single
// cpp file, because "windows.h" does not work well with other header
// files. it raises some strange errors.
#include <windows.h>

//
#include <libloaderapi.h>
#include <sstream>
#include <string>
namespace morphizen {
using plugin_t = void *;
enum class scope_t { PUBLIC, PRIVATE };
static std::wstring s2ws(const std::string &s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t *buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}

std::pair<plugin_t, bool> open_plugin_dyn(const std::string &name,
                                          scope_t /*scope*/) {
  static_assert(sizeof(plugin_t) == sizeof(HMODULE));
  auto handle = reinterpret_cast<HMODULE>(GetModuleHandleW(s2ws(name).c_str()));
  if (handle) {
    return {handle, false};
  }
  return {LoadLibraryW(s2ws(name).c_str()), true};
}

void *plugin_sym_dyn(plugin_t plugin, const std::string &name) {
  return GetProcAddress((HMODULE)plugin, name.c_str());
}
std::string plugin_error_dyn(plugin_t /*plugin*/) {
  std::ostringstream str;
  str << "ERROR CODE: " << GetLastError();
  return str.str();
}
void close_plugin_dyn(plugin_t plugin) { FreeLibrary((HMODULE)plugin); }
} // namespace morphizen
