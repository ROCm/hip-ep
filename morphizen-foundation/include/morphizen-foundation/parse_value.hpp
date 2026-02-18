/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>

namespace morphizen::foundation {

/**
 * @brief Parse string value to various types with error checking
 *
 * This template function provides type-safe parsing of string values
 * with assertion-based error checking. Specialized versions exist for
 * numeric types with hex support and boolean types with multiple formats.
 *
 * @tparam T Target type for parsing
 * @param text String to parse
 * @param value Reference to store the parsed value
 */
template <typename T> void parse_value(const std::string& text, T& value) {
  std::istringstream is(text);
  if (!(is >> value)) {
    assert(false && "Failed to parse value");
  }

  if (is.rdbuf()->in_avail() != 0) {
    assert(false && "Extra characters after parsed value");
  }
}

/**
 * @brief Specialization for long long with hex support
 */
inline void parse_value(const std::string& text, long long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = std::stoll(text.substr(2), 0, 16);
  } else {
    value = std::stoll(text, 0, 10);
  }
}

/**
 * @brief Specialization for uint32_t with hex support
 */
inline void parse_value(const std::string& text, uint32_t& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = static_cast<uint32_t>(std::stoul(text.substr(2), 0, 16));
  } else {
    value = static_cast<uint32_t>(std::stoul(text, 0, 10));
  }
}

/**
 * @brief Specialization for uint64_t with hex support
 */
inline void parse_value(const std::string& text, uint64_t& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = std::stoull(text.substr(2), 0, 16);
  } else {
    value = std::stoull(text, 0, 10);
  }
}

/**
 * @brief Specialization for long with hex support
 */
inline void parse_value(const std::string& text, long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = std::stol(text.substr(2), 0, 16);
  } else {
    value = std::stol(text, 0, 10);
  }
}

/**
 * @brief Specialization for int with hex support
 */
inline void parse_value(const std::string& text, int& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = std::stoi(text.substr(2), 0, 16);
  } else {
    value = std::stoi(text, 0, 10);
  }
}

/**
 * @brief Specialization for bool with multiple text formats
 *
 * Accepts: "yes", "on", "enable", "true", "1" as true
 * Everything else is considered false
 */
inline void parse_value(const std::string& text, bool& value) {
  if (text == "yes" || text == "on" || text == "enable" || text == "true" ||
      text == "1") {
    value = true;
  } else {
    value = false;
  }
}

} // namespace morphizen::foundation
