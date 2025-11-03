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
void aes_encryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst, const std::string& key);
void aes_decryption(const vaip_core::IStreamReader& src,
                    vaip_core::IStreamWriter& dst, const std::string& key);
} // namespace vaip_encryption
