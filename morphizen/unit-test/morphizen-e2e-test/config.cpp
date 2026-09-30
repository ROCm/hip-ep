/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./config.hpp"

#include <filesystem>
#include <fstream>

std::vector<std::unique_ptr<E2ETestConfig>>
E2ETestConfig::create(const std::filesystem::path &config_path) {

  // Read the JSON file into a string
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    // must not throw exception, make CI more stable. --gtest_list_tests can
    // return nothing.
    std::cerr << "Could not open config file: "
              << std::filesystem::absolute(config_path).string() << std::endl;
    return {};
  }
  std::string file_str((std::istreambuf_iterator<char>(config_file)),
                       std::istreambuf_iterator<char>());
  config_file.close();
  const std::string json_str_prefix = "{ \"test_configs\": ";
  const std::string json_str_suffix = "}";
  std::string json_str;
  json_str.reserve(json_str_prefix.size() + file_str.size() +
                   json_str_suffix.size());
  json_str.append(json_str_prefix);
  json_str.append(file_str.begin(), file_str.end());
  json_str.append(json_str_suffix);

  MorphizenE2ETestsProto root;
  try {
    root = MorphizenE2ETestsProto::FromJsonString(json_str);
  } catch (const morphizen::json::ParseError &error) {
    // must not throw exception, make CI more stable. --gtest_list_tests can
    // return nothing.
    std::cerr << "Failed to parse JSON: " << json_str
              << " Status : " << error.what() << std::endl;
    return {};
  }
  auto ret = std::vector<std::unique_ptr<E2ETestConfig>>();

  for (const auto &proto : root.test_configs()) {
    ret.push_back(std::make_unique<E2ETestConfig>(proto));
  }
  return ret;
}

E2ETestConfig::E2ETestConfig(const E2ETestConfigProto &proto)
    : config_proto_(std::move(proto)) {}
