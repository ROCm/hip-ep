/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../src/device-selection.hpp"
#include <gtest/gtest.h>

namespace {

using morphizen::device_selection::CanonicalLuid;
using morphizen::device_selection::CanonicalPciBusId;
using morphizen::device_selection::Decide;
using morphizen::device_selection::DeviceSelectionDecision;
using morphizen::device_selection::ParseLuid;
using morphizen::device_selection::ParsePciBusId;
using morphizen::device_selection::PciBusId;
using morphizen::device_selection::PlatformIdFromMetadata;
using morphizen::device_selection::PlatformIdKind;
using morphizen::device_selection::PlatformIdMetadataKey;

} // namespace

TEST(DeviceSelection, PciBusIdIgnoresCasePaddingAndWhitespace) {
  PciBusId parsed;
  ASSERT_TRUE(ParsePciBusId("  0000:03:00.0  ", &parsed));
  EXPECT_EQ(CanonicalPciBusId(parsed), "0000:03:00.0");

  PciBusId short_form;
  ASSERT_TRUE(ParsePciBusId("0:3:0.0", &short_form));
  EXPECT_EQ(CanonicalPciBusId(short_form), CanonicalPciBusId(parsed));

  PciBusId upper;
  ASSERT_TRUE(ParsePciBusId("0000:0A:1B.C", &upper));
  EXPECT_EQ(CanonicalPciBusId(upper), "0000:0a:1b.c");
}

TEST(DeviceSelection, PciBusIdRejectsMalformedText) {
  PciBusId parsed;
  EXPECT_FALSE(ParsePciBusId("", &parsed));
  EXPECT_FALSE(ParsePciBusId("0000:03:00", &parsed));
  EXPECT_FALSE(ParsePciBusId("0000:03:00.0 extra", &parsed));
  EXPECT_FALSE(ParsePciBusId("gggg:03:00.0", &parsed));
}

TEST(DeviceSelection, LuidIsDecimalAndCanonical) {
  uint64_t value = 0;
  ASSERT_TRUE(ParseLuid(" 00012 ", &value));
  EXPECT_EQ(value, 12u);
  EXPECT_EQ(CanonicalLuid(value), "12");

  EXPECT_FALSE(ParseLuid("", &value));
  EXPECT_FALSE(ParseLuid("-1", &value));
  EXPECT_FALSE(ParseLuid("12abc", &value));
  EXPECT_FALSE(ParseLuid("18446744073709551616", &value));
}

TEST(DeviceSelection, MetadataKeyMatchesTheHostPlatform) {
#ifdef _WIN32
  EXPECT_STREQ(PlatformIdMetadataKey(), "LUID");
#else
  EXPECT_STREQ(PlatformIdMetadataKey(), "pci_bus_id");
#endif
}

TEST(DeviceSelection, MetadataKeySelectsThePlatformIdentity) {
  const auto pci =
      PlatformIdFromMetadata("pci_bus_id", "0000:03:00.0");
  EXPECT_EQ(pci.kind, PlatformIdKind::PciBusId);
  EXPECT_EQ(pci.value, "0000:03:00.0");

  const auto luid = PlatformIdFromMetadata("LUID", "42");
  EXPECT_EQ(luid.kind, PlatformIdKind::Luid);
  EXPECT_EQ(luid.value, "42");

  EXPECT_EQ(PlatformIdFromMetadata("pci_bus_id", "not-a-bus").kind,
            PlatformIdKind::Malformed);
  EXPECT_EQ(PlatformIdFromMetadata("LUID", "nope").kind,
            PlatformIdKind::Malformed);
  EXPECT_EQ(PlatformIdFromMetadata("pci_bus_id", nullptr).kind,
            PlatformIdKind::None);
  EXPECT_EQ(PlatformIdFromMetadata("other", "0000:03:00.0").kind,
            PlatformIdKind::None);
}

TEST(DeviceSelection, DecidesZeroOneAndManyDevices) {
  EXPECT_EQ(Decide(0, PlatformIdKind::None),
            DeviceSelectionDecision::UseDefaultOrdinal);
  EXPECT_EQ(Decide(1, PlatformIdKind::None),
            DeviceSelectionDecision::UseDefaultOrdinal);
  EXPECT_EQ(Decide(1, PlatformIdKind::PciBusId),
            DeviceSelectionDecision::MatchIdentity);
  EXPECT_EQ(Decide(1, PlatformIdKind::Luid),
            DeviceSelectionDecision::MatchIdentity);
  EXPECT_EQ(Decide(1, PlatformIdKind::Malformed),
            DeviceSelectionDecision::RejectMalformed);
  EXPECT_EQ(Decide(2, PlatformIdKind::PciBusId),
            DeviceSelectionDecision::RejectTooMany);
  EXPECT_EQ(Decide(2, PlatformIdKind::None),
            DeviceSelectionDecision::RejectTooMany);
}
