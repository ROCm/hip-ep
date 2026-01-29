/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <iosfwd>
#include <stdexcept>
#include <string>

namespace morphizen_encryption {

// Exception class for encryption/decryption errors
class EncryptionError : public std::runtime_error {
public:
  explicit EncryptionError(const std::string& message)
      : std::runtime_error(message) {}
};

int has_encryption_support();
void aes_encryption(std::istream& src, std::ostream& dst,
                    const std::string& key);
void aes_decryption(std::istream& src, std::ostream& dst,
                    const std::string& key);
} // namespace morphizen_encryption
