/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cassert>
#include <cstdint>
#include <sstream>
namespace morphizen {

template <typename T> void parse_value(const std::string &text, T &value) {
  std::istringstream is(text);
  if (!(is >> value)) {
    assert(false);
  }

  if (is.rdbuf()->in_avail() != 0) {
    assert(false);
  }
}

inline void parse_value(const std::string &text, long long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoll(text.substr(2), 0, 16);
  } else {
    value = stoll(text, 0, 10);
  }
}

inline void parse_value(const std::string &text, uint32_t &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = static_cast<uint32_t>(stoul(text.substr(2), 0, 16));
  } else {
    value = static_cast<uint32_t>(stoul(text, 0, 10));
  }
}

inline void parse_value(const std::string &text, uint64_t &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoull(text.substr(2), 0, 16);
  } else {
    value = stoull(text, 0, 10);
  }
}

inline void parse_value(const std::string &text, long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stol(text.substr(2), 0, 16);
  } else {
    value = stol(text, 0, 10);
  }
}

inline void parse_value(const std::string &text, bool &value) {
  if (text == "yes" || text == "on" || text == "enable" || text == "true") {
    value = true;
  } else {
    value = false;
  }
}
} // namespace morphizen
