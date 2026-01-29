/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <stdexcept>
#include <string>

namespace morphizen {
class IStreamReader;
class IStreamWriter;
} // namespace morphizen

namespace morphizen_encryption {

// Exception class for encryption/decryption errors
class EncryptionError : public std::runtime_error {
public:
  explicit EncryptionError(const std::string& message)
      : std::runtime_error(message) {}
};

int has_encryption_support();
void aes_encryption(const morphizen::IStreamReader& src,
                    morphizen::IStreamWriter& dst, const std::string& key);
void aes_decryption(const morphizen::IStreamReader& src,
                    morphizen::IStreamWriter& dst, const std::string& key);
} // namespace morphizen_encryption
