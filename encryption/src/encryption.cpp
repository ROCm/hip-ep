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
#include <sodium.h>
#include <stdexcept>

namespace vaip_encryption {
void aes_encryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst, const std::string& key);
void aes_decryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst, const std::string& key);
void XChaCha20_Poly1305_encryption(const vaip_core::IStreamReader& src,
                                   vaip_core::IStreamWriter& dst,
                                   const std::string& key);
void XChaCha20_Poly1305_decryption(const vaip_core::IStreamReader& src,
                                   vaip_core::IStreamWriter& dst,
                                   const std::string& key);
void encryption(const vaip_core::IStreamReader& src,
                vaip_core::IStreamWriter& dst, const std::string& key,
                CryptoAlgorithm algo) {
  switch (algo) {
  case CryptoAlgorithm::AES:
    aes_encryption(src, dst, key);
    break;
  case CryptoAlgorithm::XChaCha20_Poly1305:
    XChaCha20_Poly1305_encryption(src, dst, key);
    break;
  default:
    LOG(WARNING) << "The model will not be encrypted due to algo = None";
    break;
  }
}
void decryption(const vaip_core::IStreamReader& src,
                vaip_core::IStreamWriter& dst, const std::string& key,
                CryptoAlgorithm algo) {
  switch (algo) {
  case CryptoAlgorithm::AES:
    aes_decryption(src, dst, key);
    break;
  case CryptoAlgorithm::XChaCha20_Poly1305:
    XChaCha20_Poly1305_decryption(src, dst, key);
    break;
  default:
    LOG(WARNING) << "The model will not be decryption due to algo = None";
    break;
  }
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
    throw std::runtime_error("encryption creating context failed");
  }

  if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char*)aes_256_key.data(), NULL)) {
    throw std::runtime_error("encryption initialization failed");
  }
  int len = 0;
  char buffer_out[READ_SIZE] = {0};
  auto data_in = src.read(READ_SIZE);
  while (data_in.has_value() && data_in->size() != 0) {
    if (1 != EVP_EncryptUpdate(ctx, (unsigned char*)buffer_out, &len,
                               (const unsigned char*)data_in.value().data(),
                               (int)data_in->size())) {
      throw std::runtime_error("encryption update failed");
    }
    dst.write(buffer_out, len);
    data_in = src.read(READ_SIZE);
  }
  char evp_cipher_block[AES_BLOCK_SIZE];
  if (1 != EVP_EncryptFinal_ex(ctx, (unsigned char*)evp_cipher_block, &len)) {
    throw std::runtime_error("encryption finalization failed");
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
    throw std::runtime_error("decryption creating context failed");
  }

  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_ecb(), NULL,
                              (const unsigned char*)aes_256_key.data(), NULL)) {
    throw std::runtime_error("decryption initialization failed");
  }
  char buffer_out[READ_SIZE] = {0};
  int len = 0;
  auto data_in = src.read(READ_SIZE);
  while (data_in.has_value() && data_in->size() != 0) {
    if (1 != EVP_DecryptUpdate(ctx, (unsigned char*)buffer_out, &len,
                               (const unsigned char*)data_in.value().data(),
                               (int)data_in->size())) {
      throw std::runtime_error("encryption update failed");
    }
    dst.write(buffer_out, len);
    data_in = src.read(READ_SIZE);
  }
  char evp_cipher_block[AES_BLOCK_SIZE];

  if (1 != EVP_DecryptFinal_ex(ctx, (unsigned char*)evp_cipher_block, &len)) {
    throw std::runtime_error("decryption finalization failed");
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

void XChaCha20_Poly1305_encryption(const vaip_core::IStreamReader& src,
                                   vaip_core::IStreamWriter& dst,
                                   const std::string& key) {
  constexpr size_t CHUNK_SIZE = 4096;
  // ensure k length = 32 bytes
  unsigned char k[crypto_secretstream_xchacha20poly1305_KEYBYTES];
  crypto_generichash(k, sizeof k,
                     reinterpret_cast<const unsigned char*>(key.data()),
                     key.size(), nullptr, 0);

  crypto_secretstream_xchacha20poly1305_state state;
  unsigned char header[crypto_secretstream_xchacha20poly1305_HEADERBYTES];
  crypto_secretstream_xchacha20poly1305_init_push(&state, header, k);

  dst.write(reinterpret_cast<const char*>(header), sizeof header);

  while (true) {
    auto chunk_opt = src.read(CHUNK_SIZE);
    if (!chunk_opt || chunk_opt->empty())
      break;
    const unsigned char* chunk =
        reinterpret_cast<const unsigned char*>(chunk_opt->data());
    size_t len = chunk_opt->size();

    std::vector<unsigned char> out(
        len + crypto_secretstream_xchacha20poly1305_ABYTES);
    unsigned long long outlen;
    unsigned char tag = (len < CHUNK_SIZE)
                            ? crypto_secretstream_xchacha20poly1305_TAG_FINAL
                            : 0;

    crypto_secretstream_xchacha20poly1305_push(&state, out.data(), &outlen,
                                               chunk, len, nullptr, 0, tag);
    dst.write(reinterpret_cast<const char*>(out.data()),
              static_cast<size_t>(outlen));

    if (tag == crypto_secretstream_xchacha20poly1305_TAG_FINAL)
      break;
  }
}

void XChaCha20_Poly1305_decryption(const vaip_core::IStreamReader& src,
                                   vaip_core::IStreamWriter& dst,
                                   const std::string& key) {
  constexpr size_t CHUNK_SIZE = 4096;

  unsigned char k[crypto_secretstream_xchacha20poly1305_KEYBYTES];
  crypto_generichash(k, sizeof k,
                     reinterpret_cast<const unsigned char*>(key.data()),
                     key.size(), nullptr, 0);

  auto header_opt = src.read(crypto_secretstream_xchacha20poly1305_HEADERBYTES);
  if (!header_opt ||
      header_opt->size() != crypto_secretstream_xchacha20poly1305_HEADERBYTES)
    throw std::runtime_error("Failed to read header");

  crypto_secretstream_xchacha20poly1305_state state;
  crypto_secretstream_xchacha20poly1305_init_pull(
      &state, reinterpret_cast<const unsigned char*>(header_opt->data()), k);

  while (true) {
    auto chunk_opt =
        src.read(CHUNK_SIZE + crypto_secretstream_xchacha20poly1305_ABYTES);
    if (!chunk_opt || chunk_opt->empty())
      break;

    const unsigned char* chunk =
        reinterpret_cast<const unsigned char*>(chunk_opt->data());
    size_t chunk_len = chunk_opt->size();

    std::vector<unsigned char> out(chunk_len);
    unsigned long long outlen;
    unsigned char tag;

    if (crypto_secretstream_xchacha20poly1305_pull(&state, out.data(), &outlen,
                                                   &tag, chunk, chunk_len,
                                                   nullptr, 0) != 0)
      throw std::runtime_error("Decryption failed");

    dst.write(reinterpret_cast<const char*>(out.data()),
              static_cast<size_t>(outlen));

    if (tag & crypto_secretstream_xchacha20poly1305_TAG_FINAL)
      break;
  }
}
} // namespace vaip_encryption
