/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-foundation/encryption.hpp"
#include <algorithm>
#include <array>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <vector>
#ifdef WITH_OPENSSL
#include <openssl/aes.h>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#endif
namespace morphizen_encryption {
int has_encryption_support() {
#ifdef WITH_OPENSSL
  return 1;
#else
  return 0;
#endif
}

void aes_encryption(std::istream &src, std::ostream &dst,
                    [[maybe_unused]] const std::string &key) {
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
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    throw EncryptionError("encryption creating context failed");
  }

  if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char *)aes_256_key.data(),
                              NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    throw EncryptionError("encryption initialization failed");
  }

  int len = 0;
  char buffer_out[READ_SIZE] = {0};
  std::vector<char> buffer_in(READ_SIZE);

  while (src.read(buffer_in.data(), READ_SIZE) || src.gcount() > 0) {
    size_t read_size = src.gcount();
    if (read_size == 0)
      break;

    if (1 != EVP_EncryptUpdate(ctx, (unsigned char *)buffer_out, &len,
                               (const unsigned char *)buffer_in.data(),
                               (int)read_size)) {
      EVP_CIPHER_CTX_free(ctx);
      throw EncryptionError("encryption update failed");
    }
    dst.write(buffer_out, len);
    if (!dst.good()) {
      EVP_CIPHER_CTX_free(ctx);
      throw EncryptionError("write failed");
    }
  }

  char evp_cipher_block[AES_BLOCK_SIZE];
  if (1 != EVP_EncryptFinal_ex(ctx, (unsigned char *)evp_cipher_block, &len)) {
    EVP_CIPHER_CTX_free(ctx);
    throw EncryptionError("encryption finalization failed");
  }
  dst.write(evp_cipher_block, len);
  EVP_CIPHER_CTX_free(ctx);
#else
  LOG(WARNING) << "Since OpenSSL was not enabled during compilation, the "
                  "encryption feature will be disabled.";
  char buffer[1024];
  while (src.read(buffer, 1024) || src.gcount() > 0) {
    dst.write(buffer, src.gcount());
  }
#endif
}
void aes_decryption(std::istream &src, std::ostream &dst,
                    [[maybe_unused]] const std::string &key) {
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
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    throw EncryptionError("decryption creating context failed");
  }

  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char *)aes_256_key.data(),
                              NULL)) {
    EVP_CIPHER_CTX_free(ctx);
    throw EncryptionError("decryption initialization failed");
  }

  char buffer_out[READ_SIZE] = {0};
  int len = 0;
  std::vector<char> buffer_in(READ_SIZE);

  while (src.read(buffer_in.data(), READ_SIZE) || src.gcount() > 0) {
    size_t read_size = src.gcount();
    if (read_size == 0)
      break;

    if (1 != EVP_DecryptUpdate(ctx, (unsigned char *)buffer_out, &len,
                               (const unsigned char *)buffer_in.data(),
                               (int)read_size)) {
      EVP_CIPHER_CTX_free(ctx);
      throw EncryptionError("decryption update failed");
    }
    dst.write(buffer_out, len);
    if (!dst.good()) {
      EVP_CIPHER_CTX_free(ctx);
      throw EncryptionError("write failed");
    }
  }

  char evp_cipher_block[AES_BLOCK_SIZE];
  if (1 != EVP_DecryptFinal_ex(ctx, (unsigned char *)evp_cipher_block, &len)) {
    EVP_CIPHER_CTX_free(ctx);
    throw EncryptionError("decryption finalization failed, maybe the "
                          "encryption key is incorrect.");
  }
  dst.write(evp_cipher_block, len);
  EVP_CIPHER_CTX_free(ctx);
#else
  LOG(WARNING) << "Since OpenSSL was not enabled during compilation, the "
                  "encryption feature will be disabled.";
  char buffer[1024];
  while (src.read(buffer, 1024) || src.gcount() > 0) {
    dst.write(buffer, src.gcount());
  }
#endif
}
} // namespace morphizen_encryption
