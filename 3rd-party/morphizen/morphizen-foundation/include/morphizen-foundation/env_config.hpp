/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "./parse_value.hpp"
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

// C function declaration
extern "C" const char *vitis_ai_getenv_s(const char *name);

namespace morphizen::foundation {

/**
 * @brief Helper template for converting strings to various types
 *
 * This template provides type-specific conversion logic for environment
 * configuration values. Specializations exist for common types.
 */
template <typename T> struct env_config_helper {
  /**
   * @brief Convert string to type T
   * @param s String to convert
   * @return Converted value of type T
   */
  static inline T from_string(const std::string &s);
};

/**
 * @brief Get environment variable value with default fallback
 * @param name Environment variable name
 * @param default_value Default value if environment variable is not set
 * @return Environment variable value or default
 */
std::string get_env_string(const char *name,
                           const std::string &default_value = "");

/**
 * @brief Alternative function name for compatibility
 * @param name Environment variable name
 * @param default_value Default value if environment variable is not set
 * @return Environment variable value or default
 */
std::string my_getenv_s(const char *name,
                        const std::string &default_value = "");

/**
 * @brief Template for type-safe environment configuration
 *
 * This template provides compile-time environment variable access with
 * type conversion and default values. The configuration is cached after
 * first access for performance.
 *
 * @tparam T Type of the configuration value
 * @tparam env_name Type providing environment variable name and default value
 */
template <typename T, typename env_name> struct env_config {
  /**
   * @brief Initialize configuration value from environment
   * @return Initialized value
   */
  static T init() {
    const char *name = env_name::get_name();
    const char *defvalue = env_name::get_default_value();
    auto env_value = get_env_string(name, defvalue);
    return env_config_helper<T>::from_string(env_value);
  }

  /// Cached configuration value (initialized once)
  static T value;
};

// Static member definition
template <typename T, typename env_name>
T env_config<T, env_name>::value = env_config<T, env_name>::init();

// Template specializations for common types

template <typename T>
inline T env_config_helper<T>::from_string(const std::string &s) {
  T ret = T();
  parse_value(s, ret); // Uses ADL to find parse_value in morphizen::utils
  return ret;
}

template <>
inline std::string
env_config_helper<std::string>::from_string(const std::string &s) {
  return s;
}

/**
 * @brief Specialization for vector types
 *
 * Supports comma-separated values in environment variables that get
 * parsed into std::vector<T>.
 */
template <typename T> struct env_config_helper<std::vector<T>> {
  static inline std::vector<T> from_string(const std::string &s);
};

template <typename T>
inline std::vector<T>
env_config_helper<std::vector<T>>::from_string(const std::string &s) {
  constexpr char delim = ',';
  std::vector<T> list;
  std::istringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim)) {
    list.push_back(env_config_helper<T>::from_string(item));
  }
  return list;
}

} // namespace morphizen::foundation

/**
 * @brief Define an environment parameter with type and default value
 * @param param_name Name of the parameter (also used as env var name)
 * @param defvalue Default value as string literal
 * @param type C++ type for the parameter
 *
 * Usage: DEF_ENV_PARAM_2(DEBUG_LEVEL, "0", int)
 */
#define DEF_ENV_PARAM_2(param_name, defvalue, type)                            \
  struct ENV_PARAM_##param_name                                                \
      : public ::morphizen::foundation::env_config<type,                       \
                                                   ENV_PARAM_##param_name> {   \
    static const char *get_name() { return #param_name; }                      \
    static const char *get_default_value() { return defvalue; }                \
  };

/**
 * @brief Access the value of an environment parameter
 * @param param_name Name of the parameter defined with DEF_ENV_PARAM_*
 *
 * Usage: int level = ENV_PARAM(DEBUG_LEVEL);
 */
#define ENV_PARAM(param_name) (ENV_PARAM_##param_name::value)

/**
 * @brief Define an integer environment parameter
 * @param param_name Name of the parameter
 * @param defvalue Default value as string literal
 *
 * Usage: DEF_ENV_PARAM(DEBUG_LEVEL, "0")
 */
#define DEF_ENV_PARAM(param_name, defvalue)                                    \
  DEF_ENV_PARAM_2(param_name, defvalue, int)
