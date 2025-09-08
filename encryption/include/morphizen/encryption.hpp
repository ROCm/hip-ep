/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <string>
namespace vaip_core {
class IStreamReader;
class IStreamWriter;
} // namespace vaip_core
namespace vaip_encryption {
enum class CryptoAlgorithm : int {
  None = 0,
  AES = 1,                // openssl
  XChaCha20_Poly1305 = 2, // sodium
};
void encryption(const vaip_core::IStreamReader& src,
                vaip_core::IStreamWriter& dst, const std::string& key,
                CryptoAlgorithm algo = CryptoAlgorithm::XChaCha20_Poly1305);
void decryption(const vaip_core::IStreamReader& src,
                vaip_core::IStreamWriter& dst, const std::string& key,
                CryptoAlgorithm algo);

} // namespace vaip_encryption
