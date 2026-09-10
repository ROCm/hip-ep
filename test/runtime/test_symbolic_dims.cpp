/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/symbolic_dims.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAILED: " << message << "\n";
  return condition;
}

} // namespace

int main() {
  using morphizen::SymbolicDimRecord;

  std::vector<SymbolicDimRecord> unsorted{
      {"main_graph", "z:value;with,delimiters", {"", "u1*u2"}},
      {"main_graph", "pixel_values", {"num_patches", ""}},
      {"main_graph", "unicode", {u8"序列", ""}},
  };
  std::vector<SymbolicDimRecord> sorted{unsorted[1], unsorted[2], unsorted[0]};
  std::string encoded = morphizen::encode_symbolic_dim_records(unsorted);
  std::string sortedEncoded = morphizen::encode_symbolic_dim_records(sorted);
  bool ok = check(encoded == sortedEncoded,
                  "encoding must not depend on insertion order");

  std::string error;
  auto decoded = morphizen::decode_symbolic_dim_records(encoded, error);
  ok &= check(decoded.has_value(), "canonical encoding must decode");
  ok &= check(error.empty(), "successful decode must not report an error");
  ok &= check(decoded && decoded->size() == 3, "decoded record count");
  ok &= check(decoded && (*decoded)[0].value_name == "pixel_values",
              "records must be canonically sorted");
  ok &= check(decoded && (*decoded)[0].dimensions[0] == "num_patches",
              "symbolic dimension must round-trip");
  ok &= check(decoded && (*decoded)[1].dimensions[0] == u8"序列",
              "UTF-8 dimension must round-trip");
  ok &= check(decoded && (*decoded)[2].dimensions[1] == "u1*u2",
              "expression-like dimension must remain opaque");
  ok &= check(decoded &&
                  morphizen::encode_symbolic_dim_records(*decoded) == encoded,
              "decode/re-encode must preserve canonical cache bytes");

  std::string zero =
      morphizen::encode_symbolic_dim_records(std::vector<SymbolicDimRecord>{});
  ok &= check(zero == "HSDI1\n0\n", "empty map has canonical encoding");
  ok &= check(morphizen::encode_symbolic_dim_records({{"g", "x", {"N", ""}}}) ==
                  "HSDI1\n1\n1:671:782:1:4e0:\n",
              "non-empty encoding must match the HSDI1 golden bytes");

  for (const std::string &bad : {
           std::string("HSDI2\n0\n"),
           std::string("HSDI1\n1\n4:6D61696E0:0:\n"),
           std::string("HSDI1\n00\n"),
           std::string("HSDI1\n1\n1:6"),
           std::string("HSDI1\n1\n1:zz1:781:1:4e\n"),
           std::string("HSDI1\n1\n0:1:781:1:4e\n"),
           std::string("HSDI1\n1\n1:671:781025:\n"),
           std::string("HSDI1\n2\n1:671:7a0:\n1:671:610:\n"),
           encoded + "trailing",
       }) {
    error.clear();
    ok &= check(!morphizen::decode_symbolic_dim_records(bad, error),
                "malformed encoding must fail");
    ok &= check(!error.empty(), "malformed encoding must report an error");
  }

  try {
    (void)morphizen::encode_symbolic_dim_records(
        {{"main_graph", "x", {"N"}}, {"main_graph", "x", {"N"}}});
    ok &= check(false, "duplicate records must be rejected");
  } catch (const std::invalid_argument &) {
  }

  return ok ? 0 : 1;
}
