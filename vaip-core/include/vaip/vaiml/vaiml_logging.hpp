#pragma once

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <string>
#include <vector>

#ifdef _WIN32
#  pragma warning(disable : 4244)
#  pragma warning(disable : 4996)
// #pragma warning(disable : 4840)
#endif

DEF_ENV_PARAM(DEBUG_VAIML_PARTITION, "0")
// n: DEBUG verbosity level
// 1: Key DEBUG messages
// 2: Verbose DEBUG messages
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIML_PARTITION) >= n)

template <typename... Args>
inline void vaiml_print_func(std::ostream& os, Args&&... args) {
  (os << ... << args);
}

#define WRITELINE_TO_LOG(ofs, ...)                                             \
  vaiml_print_func(std::cout, __VA_ARGS__);                                    \
  vaiml_print_func(ofs, __VA_ARGS__);                                          \
  std::cout << std::endl;                                                      \
  ofs << std::endl;

// This is to work around an issue that lit deos not work with sterr
#define VAIML_INFO_PRINT(...)                                                  \
  std::cout << "INFO: [VAIP-VAIML-PASS] ";                                     \
  vaiml_print_func(std::cout, __VA_ARGS__);                                    \
  std::cout << std::endl;

#define VAIML_WARNING_PRINT(...)                                               \
  std::cout << "WARNING: [VAIP-VAIML-PASS] ";                                  \
  vaiml_print_func(std::cout, __VA_ARGS__);                                    \
  std::cout << std::endl;

#define VAIML_ERROR_PRINT(...)                                                 \
  std::cout << "ERROR: [VAIP-VAIML-PASS] ";                                    \
  vaiml_print_func(std::cout, __VA_ARGS__);                                    \
  std::cout << std::endl;

#define VAIML_DEBUG_PRINT(verbosity, ...)                                      \
  if (ENV_PARAM(DEBUG_VAIML_PARTITION) >= verbosity) {                         \
    std::cout << "DEBUG" << verbosity << ": [VAIP-VAIML-PASS] ";               \
    vaiml_print_func(std::cout, __VA_ARGS__);                                  \
    std::cout << std::endl;                                                    \
  }

namespace vaip_vaiml {

using VaimlStrVec = std::vector<std::string>;
using VaimlTensorShape = std::vector<int64_t>;
using VaimlShapeVec = std::vector<VaimlTensorShape>;
using VaimlShapeDict = std::map<std::string, VaimlTensorShape>;

using StrVectors = std::vector<VaimlStrVec>;

/**
 * @brief Convert string vector to string for print
 */
std::string stringVectorToString(std::vector<std::string>& string_vec);

/**
 * @brief Format a number in 000,000 format for easier reading
 * The code is to make sure that it works even the locale is not set correctly
 * on the running machine.
 */
std::string formatNumberWithCommas(int64_t number);

std::vector<std::string> splitString(const std::string& str, char delimiter);
std::string shapeToString(std::vector<int64_t> shape);
std::vector<int64_t> shapeFromString(const std::string& shape_str,
                                     char delimiter);

} // namespace vaip_vaiml