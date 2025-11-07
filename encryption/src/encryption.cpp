/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/encryption.hpp"
#include <memory>
#ifdef WITH_OPENSSL
#  include <openssl/aes.h>
#  include <openssl/conf.h>
#  include <openssl/err.h>
#  include <openssl/evp.h>
#endif
#include "morphizen/vaip_io.hpp"
#include <algorithm>
#include <array>
#include <glog/logging.h>
namespace vaip_encryption {
int has_encryption_support() {
#ifdef WITH_OPENSSL
  return 1;
#else
  return 0;
#endif
}

void aes_encryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst,
                    [[maybe_unused]] const std::string& key) {
#ifdef WITH_OPENSSL
  // key
  std::array<char, 32> aes_256_key = {'\0'};
  if (key.size() > 32) {
    LOG(WARNING) << "The encryption key length exceeds 32 bytes, only the "
                    "first 32 bytes "
                    "will be used, and the excess will be ignored.";
    auto sub_key = key.substr(0, 32);
    std::copy(sub_key.begin(), sub_key.end(), aes_256_key.begin());
  } else {
    std::copy(key.begin(), key.end(), aes_256_key.begin());
  }

  const size_t READ_SIZE = 1024;
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    throw EncryptionError("encryption creating context failed");
  }

  if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char*)aes_256_key.data(), NULL)) {
    throw EncryptionError("encryption initialization failed");
  }
  int len = 0;
  char buffer_out[READ_SIZE] = {0};
  auto data_in = src.read(READ_SIZE);
  while (data_in.has_value() && data_in->size() != 0) {
    if (1 != EVP_EncryptUpdate(ctx, (unsigned char*)buffer_out, &len,
                               (const unsigned char*)data_in.value().data(),
                               (int)data_in->size())) {
      throw EncryptionError("encryption update failed");
    }
    dst.write(buffer_out, len);
    data_in = src.read(READ_SIZE);
  }
  char evp_cipher_block[AES_BLOCK_SIZE];
  if (1 != EVP_EncryptFinal_ex(ctx, (unsigned char*)evp_cipher_block, &len)) {
    throw EncryptionError("encryption finalization failed");
  }
  dst.write(evp_cipher_block, len);
  EVP_CIPHER_CTX_free(ctx);
  return;
#else
  LOG(WARNING) << "Since OpenSSL was not enabled during compilation, the "
                  "encryption feature will be disabled.";
  for (auto buf = src.read(1024); buf; buf = src.read(1024)) {
    auto write_size = dst.write(buf->data(), buf->size());
    CHECK_EQ(write_size, buf->size());
  }
  return;
#endif
}
void aes_decryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst,
                    [[maybe_unused]] const std::string& key) {
#ifdef WITH_OPENSSL
  // key
  std::array<char, 32> aes_256_key = {'\0'};
  if (key.size() > 32) {
    LOG(WARNING) << "The encryption key length exceeds 32 bytes, only the "
                    "first 32 bytes "
                    "will be used, and the excess will be ignored.";
    auto sub_key = key.substr(0, 32);
    std::copy(sub_key.begin(), sub_key.end(), aes_256_key.begin());
  } else {
    std::copy(key.begin(), key.end(), aes_256_key.begin());
  }

  const size_t READ_SIZE = 1024;
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    throw EncryptionError("decryption creating context failed");
  }

  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char*)aes_256_key.data(), NULL)) {
    throw EncryptionError("decryption initialization failed");
  }
  char buffer_out[READ_SIZE] = {0};
  int len = 0;
  auto data_in = src.read(READ_SIZE);
  while (data_in.has_value() && data_in->size() != 0) {
    if (1 != EVP_DecryptUpdate(ctx, (unsigned char*)buffer_out, &len,
                               (const unsigned char*)data_in.value().data(),
                               (int)data_in->size())) {
      throw EncryptionError("decryption update failed");
    }
    dst.write(buffer_out, len);
    data_in = src.read(READ_SIZE);
  }
  char evp_cipher_block[AES_BLOCK_SIZE];

  if (1 != EVP_DecryptFinal_ex(ctx, (unsigned char*)evp_cipher_block, &len)) {
    throw EncryptionError("decryption finalization failed, maybe the "
                          "encryption key is incorrect.");
  }
  dst.write(evp_cipher_block, len);
  EVP_CIPHER_CTX_free(ctx);
  return;
#else
  LOG(WARNING) << "Since OpenSSL was not enabled during compilation, the "
                  "encryption feature will be disabled.";
  for (auto buf = src.read(1024); buf; buf = src.read(1024)) {
    auto write_size = dst.write(buf->data(), buf->size());
    CHECK_EQ(write_size, buf->size());
  }
  return;
#endif
}
} // namespace vaip_encryption
