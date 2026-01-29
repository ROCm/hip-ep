/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
// clang-format off
#include "morphizen/morphizen.hpp"
#include "morphizen/encryption.hpp"
#include "../morphizen-core/src/pass_context_imp.hpp"
#include <ctime>
#include "../morphizen-core/src/tar_ball.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
// clang-format on

using namespace morphizen;

class TarBallTest : public ::testing::Test {};
TEST_F(TarBallTest, TarTest) {
  std::vector<std::string> test_strings = {"I am ss1", "I am ss2 ,123456",
                                           "I am ss3, 1234567890abcdef"};
  std::map<std::string, std::stringstream> ss_map;
  ss_map["ss0"] << test_strings[0];
  ss_map["ss1"] << test_strings[1];
  ss_map["ss2"] << test_strings[2];
  std::map<std::string, std::stringstream> ss_map_out;

  // 1. create tarball
  std::stringstream tar_sstream;
  {
    TarWriter tar_writer(tar_sstream);
    for (auto it = ss_map.begin(); it != ss_map.end(); ++it) {
      tar_writer.write(it->second, it->first);
    }
    std::cout << "write tar file finish : " << tar_sstream.str() << std::endl;
  }
  // 2. untar
  {
    tar_sstream.seekg(0); // Reset to start
    TarReader tar_reader(tar_sstream);

    auto builder = [&ss_map_out](const std::string& name) -> std::ostream& {
      return ss_map_out[name];
    };

    for (;;) {
      bool iscontinue = tar_reader.read(builder);
      if (!iscontinue) {
        break;
      }
    }
    std::cout << "read tar file finish : " << std::endl;
  }
  bool check_ok = true;
  int i = 0;
  for (auto it = ss_map_out.begin(); it != ss_map_out.end(); ++it) {
    std::cout << it->first << ": " << it->second.str() << std::endl;
    check_ok = check_ok && (it->second.str() == test_strings[i]);
    i++;
  }

  ASSERT_TRUE(check_ok);
}
static std::string generateRandomString(size_t length) {
  const std::string characters = "abcdefghijklmnopqrstuvwxyz"
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "0123456789";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<size_t> distribution(0u, characters.size() - 1);
  std::string randomString;
  for (size_t i = 0; i < length; ++i) {
    randomString += characters[distribution(generator)];
  }
  return randomString;
}

TEST_F(TarBallTest, Encrypt_Test) {
  auto data = generateRandomString(65536);
  auto key = generateRandomString(32);
  std::stringstream data_ss;
  data_ss << data;
  std::stringstream encrypted_str;
  morphizen_encryption::aes_encryption(data_ss, encrypted_str, key);
  // debug info
  std::cout << data.substr(0, 10) << " has been encrypt to "
            << encrypted_str.str().substr(0, 10) << std::endl;

  encrypted_str.seekg(0); // Reset to start
  std::stringstream decrypted_str;
  morphizen_encryption::aes_decryption(encrypted_str, decrypted_str, key);
  std::cout << encrypted_str.str().substr(0, 10) << " has been decrypt to "
            << decrypted_str.str().substr(0, 10) << std::endl;

  ASSERT_TRUE(data == decrypted_str.str());
}
// todo long filename test
