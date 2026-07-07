/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <string>
template <typename T> struct PathToString;
template <> struct PathToString<char> {
  std::string operator()(const std::filesystem::path &path) const {
    return path.u8string();
  }
};
template <> struct PathToString<wchar_t> {
  std::wstring operator()(const std::filesystem::path &path) const {
    return path.wstring();
  }
};

template <typename T> struct ToOrtString;
template <> struct ToOrtString<char> {
  std::string operator()(const std::string &str) const { return str; }
};
template <> struct ToOrtString<wchar_t> {
  std::wstring operator()(const std::string &str) const {
    return std::wstring(str.begin(), str.end());
  }
};
