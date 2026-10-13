/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace morphizen {
namespace device_selection {

// Identity ORT attaches to a hardware device. Registration records it and
// session creation matches it to a HIP ordinal. No HIP call belongs here.
//
// Windows reports a LUID. Native Linux reports a PCI bus id. HIP on Linux
// has no LUID, and under WSL hipDeviceGetPCIBusId returns the host address,
// so a WSL guest cannot match ORT's adapter LUID with these keys.
enum class PlatformIdKind { None, PciBusId, Luid, Malformed };

struct PlatformId {
  PlatformIdKind kind = PlatformIdKind::None;
  std::string value;
};

struct PciBusId {
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t device = 0;
  uint32_t function = 0;
};

enum class DeviceSelectionDecision {
  UseDefaultOrdinal,
  MatchIdentity,
  RejectMalformed,
  RejectTooMany,
};

inline const char *PlatformIdMetadataKey() {
#ifdef _WIN32
  return "LUID";
#else
  return "pci_bus_id";
#endif
}

inline std::string_view Trim(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return text;
}

inline bool ParseHexPrefix(std::string_view *text, uint32_t *value) {
  if (text->empty()) {
    return false;
  }
  uint32_t parsed = 0;
  size_t count = 0;
  for (; count < text->size(); ++count) {
    const unsigned char ch = static_cast<unsigned char>((*text)[count]);
    int digit = -1;
    if (ch >= '0' && ch <= '9') {
      digit = ch - '0';
    } else if (ch >= 'a' && ch <= 'f') {
      digit = ch - 'a' + 10;
    } else if (ch >= 'A' && ch <= 'F') {
      digit = ch - 'A' + 10;
    } else {
      break;
    }
    if (parsed > 0x0fffffffu) {
      return false;
    }
    parsed = (parsed << 4) | static_cast<uint32_t>(digit);
  }
  if (count == 0) {
    return false;
  }
  text->remove_prefix(count);
  *value = parsed;
  return true;
}

inline bool Consume(std::string_view *text, char expected) {
  if (text->empty() || text->front() != expected) {
    return false;
  }
  text->remove_prefix(1);
  return true;
}

// Accepts domain:bus:device.function in hex, ignoring case and leading zeros.
inline bool ParsePciBusId(std::string_view text, PciBusId *out) {
  text = Trim(text);
  PciBusId parsed;
  if (!ParseHexPrefix(&text, &parsed.domain) || !Consume(&text, ':') ||
      !ParseHexPrefix(&text, &parsed.bus) || !Consume(&text, ':') ||
      !ParseHexPrefix(&text, &parsed.device) || !Consume(&text, '.') ||
      !ParseHexPrefix(&text, &parsed.function) || !text.empty()) {
    return false;
  }
  *out = parsed;
  return true;
}

inline std::string CanonicalPciBusId(const PciBusId &id) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%04x:%02x:%02x.%x", id.domain, id.bus,
                id.device, id.function);
  return buffer;
}

// ORT publishes a Windows LUID as the decimal form of its 8 little-endian
// bytes. Rejects a sign, whitespace inside the number, and overflow.
inline bool ParseLuid(std::string_view text, uint64_t *out) {
  text = Trim(text);
  if (text.empty()) {
    return false;
  }
  uint64_t value = 0;
  for (char ch : text) {
    if (ch < '0' || ch > '9') {
      return false;
    }
    const uint64_t digit = static_cast<uint64_t>(ch - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      return false;
    }
    value = value * 10 + digit;
  }
  *out = value;
  return true;
}

inline std::string CanonicalLuid(uint64_t value) {
  return std::to_string(value);
}

inline PlatformId PlatformIdFromMetadata(const char *key, const char *value) {
  PlatformId id;
  if (key == nullptr || value == nullptr || value[0] == '\0') {
    return id;
  }
  if (std::strcmp(key, "pci_bus_id") == 0) {
    PciBusId pci;
    if (!ParsePciBusId(value, &pci)) {
      id.kind = PlatformIdKind::Malformed;
      return id;
    }
    id.kind = PlatformIdKind::PciBusId;
    id.value = CanonicalPciBusId(pci);
    return id;
  }
  if (std::strcmp(key, "LUID") == 0) {
    uint64_t luid = 0;
    if (!ParseLuid(value, &luid)) {
      id.kind = PlatformIdKind::Malformed;
      return id;
    }
    id.kind = PlatformIdKind::Luid;
    id.value = CanonicalLuid(luid);
    return id;
  }
  return id;
}

// Zero forwarded devices, and one device with no platform identity, keep
// HIP ordinal 0. One identifiable device is matched. More than one device
// is rejected because a session runs on a single GPU.
inline DeviceSelectionDecision Decide(size_t num_devices, PlatformIdKind kind) {
  if (num_devices > 1) {
    return DeviceSelectionDecision::RejectTooMany;
  }
  if (num_devices == 0 || kind == PlatformIdKind::None) {
    return DeviceSelectionDecision::UseDefaultOrdinal;
  }
  if (kind == PlatformIdKind::Malformed) {
    return DeviceSelectionDecision::RejectMalformed;
  }
  return DeviceSelectionDecision::MatchIdentity;
}

} // namespace device_selection
} // namespace morphizen
