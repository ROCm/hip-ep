/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace hipdnn_ep_test {
namespace input {

inline int64_t integer(const std::string& text, const char* name,
                       int64_t minimum = 1,
                       int64_t maximum = std::numeric_limits<int>::max()) {
  if (text.empty() || text.find_first_not_of("-0123456789") != std::string::npos)
    throw std::invalid_argument(std::string(name) + " must be an integer");
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno == ERANGE || end != text.c_str() + text.size() ||
      value < minimum || value > maximum)
    throw std::invalid_argument(std::string(name) + " is outside its supported range");
  return static_cast<int64_t>(value);
}

inline std::string value(int argc, char** argv, int& index) {
  if (index + 1 >= argc)
    throw std::invalid_argument(std::string("missing value for ") + argv[index]);
  return argv[++index];
}

inline bool boolean(const std::string& text, const char* name) {
  return integer(text, name, 0, 1) != 0;
}

inline size_t bytes(std::initializer_list<uint64_t> dimensions, size_t element_size) {
  size_t result = element_size;
  for (uint64_t dim : dimensions) {
    if (dim == 0 || dim > std::numeric_limits<size_t>::max() / result)
      throw std::invalid_argument("tensor size is zero or overflows size_t");
    result *= static_cast<size_t>(dim);
  }
  return result;
}

inline void shape3(const std::string& text, int& first, int& second, int& third) {
  const size_t x = text.find('x');
  const size_t y = x == std::string::npos ? x : text.find('x', x + 1);
  if (x == std::string::npos || y == std::string::npos ||
      text.find('x', y + 1) != std::string::npos)
    throw std::invalid_argument("shape must be MxKxN");
  first = static_cast<int>(integer(text.substr(0, x), "M"));
  second = static_cast<int>(integer(text.substr(x + 1, y - x - 1), "K"));
  third = static_cast<int>(integer(text.substr(y + 1), "N"));
}

}  // namespace input
}  // namespace hipdnn_ep_test
