/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
#include <stdexcept>
namespace vaip_encryption {

void aes_encryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst, const std::string& key) {
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
  EVP_CIPHER_CTX* ctx;
  if (!(ctx = EVP_CIPHER_CTX_new())) {
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
                    vaip_core::IStreamWriter& dst, const std::string& key) {
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
  EVP_CIPHER_CTX* ctx;
  if (!(ctx = EVP_CIPHER_CTX_new())) {
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
} // namespace vaip_encryption
