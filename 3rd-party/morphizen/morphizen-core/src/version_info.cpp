/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "version_info.hpp"
#include "morphizen/morphizen_ort_api.h"
#include <glog/logging.h>
#include <sstream>
#include <vector>

#ifndef PROJECT_GIT_COMMIT_ID
#  define PROJECT_GIT_COMMIT_ID "N/A"
#endif

// Include generated version header (for Linux and as fallback for Windows)
#ifdef HAVE_VERSION_INFO_CONFIG
#  include "version_info_config.h"
#endif

#ifdef _WIN32
#  include <windows.h>
#  pragma comment(lib, "version.lib")
#endif
namespace morphizen {
const std::string get_lib_name() {
  const auto ret = std::string{"morphizen"} + "." +
                   std::to_string(get_morphizen_version_major()) + "." +
                   std::to_string(get_morphizen_version_minor()) + "." +
                   std::to_string(get_morphizen_version_patch());
  return ret;
}

const std::string get_lib_id() {
  const auto ret = std::string{PROJECT_GIT_COMMIT_ID};
  return ret;
}

// NOTE: Version functions are defined in morphizen-ort-api-ext to avoid
// duplicate symbols when linking both morphizen-ort-api-ext.lib and
// morphizen-core-static.lib
#if 0
unsigned int get_morphizen_version_major() {
#  ifdef MORPHIZEN_ORT_API_MAJOR
  return MORPHIZEN_ORT_API_MAJOR;
#  else
  return 1;
#  endif
}

unsigned int get_morphizen_version_minor() {
#  ifdef MORPHIZEN_ORT_API_MINOR
  return MORPHIZEN_ORT_API_MINOR;
#  else
  return 0;
#  endif
}

unsigned int get_morphizen_version_patch() {
#  ifdef MORPHIZEN_ORT_API_PATCH
  return MORPHIZEN_ORT_API_PATCH;
#  else
  return 0;
#  endif
}
#endif

extern "C" uint32_t morphizen_get_version() {
  return (get_morphizen_version_major() << 24) |
         (get_morphizen_version_minor() << 16) |
         (get_morphizen_version_patch() << 8);
}

#ifdef _WIN32
// Helper function to query version info string from DLL resource
static std::string query_version_info(const char* value_name) {
  // Get the path of the current DLL
  HMODULE hModule = NULL;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCSTR)&query_version_info, &hModule)) {
    return "N/A";
  }

  char module_path[MAX_PATH];
  if (GetModuleFileNameA(hModule, module_path, MAX_PATH) == 0) {
    return "N/A";
  }

  // Get version info size
  DWORD handle;
  DWORD size = GetFileVersionInfoSizeA(module_path, &handle);
  if (size == 0) {
    return "N/A";
  }

  // Allocate buffer and get version info
  std::vector<BYTE> buffer(size);
  if (!GetFileVersionInfoA(module_path, handle, size, buffer.data())) {
    return "N/A";
  }

  // Query the value
  struct LANGANDCODEPAGE {
    WORD wLanguage;
    WORD wCodePage;
  }* lpTranslate;

  UINT cbTranslate;
  if (!VerQueryValueA(buffer.data(), "\\VarFileInfo\\Translation",
                      (LPVOID*)&lpTranslate, &cbTranslate)) {
    return "N/A";
  }

  // Use the first language/codepage pair
  char sub_block[256];
  sprintf_s(sub_block, sizeof(sub_block), "\\StringFileInfo\\%04x%04x\\%s",
            lpTranslate[0].wLanguage, lpTranslate[0].wCodePage, value_name);

  LPVOID value;
  UINT value_len;
  if (VerQueryValueA(buffer.data(), sub_block, &value, &value_len)) {
    return std::string((char*)value);
  }

  return "N/A";
}
#endif

// Version resource information from DLL
const std::string get_dll_company_name() {
#ifdef _WIN32
  return query_version_info("CompanyName");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_COMPANY_NAME;
#else
  return "AMD Inc";
#endif
}

const std::string get_dll_product_name() {
#ifdef _WIN32
  return query_version_info("ProductName");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_PRODUCT_NAME;
#else
  return "AMD Ryzen AI";
#endif
}

const std::string get_dll_legal_copyright() {
#ifdef _WIN32
  return query_version_info("LegalCopyright");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_LEGAL_COPYRIGHT;
#else
  return "\u00A9 AMD Inc. All rights reserved.";
#endif
}

const std::string get_dll_file_version() {
#ifdef _WIN32
  return query_version_info("FileVersion");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_DOTTED_VERSION;
#else
  return "N/A";
#endif
}

const std::string get_dll_product_version() {
#ifdef _WIN32
  return query_version_info("ProductVersion");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_PRODUCT_VERSION;
#else
  return "N/A";
#endif
}

const std::string get_dll_file_description() {
#ifdef _WIN32
  return query_version_info("FileDescription");
#elif defined(HAVE_VERSION_INFO_CONFIG)
  return RAI_FILE_DESCRIPTION;
#else
  return "N/A";
#endif
}

} // namespace morphizen
extern "C" const char* morphizen_get_build_info() {
  static char ret[2048] = {'\0'};
  if (ret[0] == '\0') {
    std::ostringstream str;
    using version_vec_tuple =
        std::vector<std::tuple<std::string, std::string, std::string>>;
    str << "\t"
        << "MORPHIZEN_ORT_API: " << morphizen::get_morphizen_version_major()
        << "." << morphizen::get_morphizen_version_minor() << "."
        << morphizen::get_morphizen_version_patch() << "\n";
    str << "\tBUILD: " << PROJECT_GIT_COMMIT_ID << "\n";
    for (auto& info : version_vec_tuple{
#include "morphizen_version_info.hpp.inc"
         }) {
      str << "\t" << std::get<0>(info) << ";" << std::get<1>(info) << ";"
          << std::get<2>(info) << "\n";
    }
    auto c = str.str();
    CHECK_LE(c.size(), sizeof(ret)) << " buffer overflow";
    std::memcpy(ret, c.data(), c.size());
  }
  return &ret[0];
}
// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
