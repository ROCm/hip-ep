/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/symbolic_dims.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>

namespace morphizen {
namespace {

constexpr size_t kMaxRecords = 1'000'000;
constexpr size_t kMaxRank = 1'024;
constexpr size_t kMaxFieldBytes = 16 * 1024 * 1024;
constexpr size_t kMaxEncodedBytes = 64 * 1024 * 1024;

char hex_digit(unsigned value) {
  return value < 10 ? static_cast<char>('0' + value)
                    : static_cast<char>('a' + value - 10);
}

std::string hex_encode(std::string_view value) {
  std::string result;
  result.reserve(value.size() * 2);
  for (unsigned char byte : value) {
    result.push_back(hex_digit(byte >> 4));
    result.push_back(hex_digit(byte & 0x0f));
  }
  return result;
}

std::optional<unsigned> hex_value(char value) {
  if (value >= '0' && value <= '9')
    return static_cast<unsigned>(value - '0');
  if (value >= 'a' && value <= 'f')
    return static_cast<unsigned>(value - 'a' + 10);
  return std::nullopt;
}

bool record_less(const SymbolicDimRecord &lhs, const SymbolicDimRecord &rhs) {
  if (lhs.scope != rhs.scope)
    return lhs.scope < rhs.scope;
  return lhs.value_name < rhs.value_name;
}

void append_bytes(std::string &out, std::string_view value) {
  if (value.size() > kMaxFieldBytes)
    throw std::invalid_argument("symbolic dimension field exceeds limit");
  out += std::to_string(value.size());
  out.push_back(':');
  out += hex_encode(value);
}

class Decoder {
public:
  explicit Decoder(std::string_view input) : input(input) {}

  std::optional<size_t> decimal(char terminator, size_t maximum,
                                std::string_view field) {
    if (position >= input.size() ||
        !std::isdigit(static_cast<unsigned char>(input[position]))) {
      fail(std::string(field) + " has no decimal value");
      return std::nullopt;
    }
    size_t value = 0;
    while (position < input.size() &&
           std::isdigit(static_cast<unsigned char>(input[position]))) {
      unsigned digit = static_cast<unsigned>(input[position++] - '0');
      if (value > (maximum - digit) / 10) {
        fail(std::string(field) + " exceeds limit");
        return std::nullopt;
      }
      value = value * 10 + digit;
    }
    if (!consume(terminator)) {
      fail(std::string(field) + " is not terminated");
      return std::nullopt;
    }
    return value;
  }

  std::optional<std::string> bytes(std::string_view field) {
    auto size = decimal(':', kMaxFieldBytes, field);
    if (!size)
      return std::nullopt;
    if (*size > (input.size() - position) / 2) {
      fail(std::string(field) + " is truncated");
      return std::nullopt;
    }
    std::string result;
    result.reserve(*size);
    for (size_t i = 0; i < *size; ++i) {
      auto high = hex_value(input[position++]);
      auto low = hex_value(input[position++]);
      if (!high || !low) {
        fail(std::string(field) + " contains non-lowercase hex");
        return std::nullopt;
      }
      result.push_back(static_cast<char>((*high << 4) | *low));
    }
    return result;
  }

  bool consume(char expected) {
    if (position >= input.size() || input[position] != expected)
      return false;
    ++position;
    return true;
  }

  bool consume(std::string_view expected) {
    if (input.substr(position, expected.size()) != expected)
      return false;
    position += expected.size();
    return true;
  }

  bool done() const { return position == input.size(); }
  const std::string &error_message() const { return error; }

  void fail(std::string message) {
    if (error.empty())
      error = std::move(message);
  }

private:
  std::string_view input;
  size_t position = 0;
  std::string error;
};

} // namespace

std::string
encode_symbolic_dim_records(std::vector<SymbolicDimRecord> records) {
  if (records.size() > kMaxRecords)
    throw std::invalid_argument("too many symbolic dimension records");
  std::sort(records.begin(), records.end(), record_less);

  std::set<std::pair<std::string, std::string>> keys;
  std::string result = std::string(kOnnxDimParamsEncodingVersion) + "\n";
  result += std::to_string(records.size());
  result.push_back('\n');
  for (const SymbolicDimRecord &record : records) {
    if (record.scope.empty() || record.value_name.empty())
      throw std::invalid_argument(
          "symbolic dimension record has empty scope or value name");
    if (record.dimensions.size() > kMaxRank)
      throw std::invalid_argument(
          "symbolic dimension record rank exceeds limit");
    if (!keys.emplace(record.scope, record.value_name).second)
      throw std::invalid_argument("duplicate symbolic dimension record");

    append_bytes(result, record.scope);
    append_bytes(result, record.value_name);
    result += std::to_string(record.dimensions.size());
    result.push_back(':');
    for (const std::string &dimension : record.dimensions)
      append_bytes(result, dimension);
    result.push_back('\n');
    if (result.size() > kMaxEncodedBytes)
      throw std::invalid_argument("symbolic dimension encoding exceeds limit");
  }
  return result;
}

std::optional<std::vector<SymbolicDimRecord>>
decode_symbolic_dim_records(std::string_view encoded, std::string &error) {
  error.clear();
  if (encoded.size() > kMaxEncodedBytes) {
    error = "symbolic dimension encoding exceeds limit";
    return std::nullopt;
  }

  Decoder decoder(encoded);
  if (!decoder.consume(kOnnxDimParamsEncodingVersion) ||
      !decoder.consume('\n')) {
    error = "unsupported symbolic dimension encoding version";
    return std::nullopt;
  }
  auto record_count = decoder.decimal('\n', kMaxRecords, "record count");
  if (!record_count) {
    error = decoder.error_message();
    return std::nullopt;
  }

  std::vector<SymbolicDimRecord> records;
  records.reserve(*record_count);
  std::pair<std::string, std::string> previous;
  bool have_previous = false;
  for (size_t record_index = 0; record_index < *record_count; ++record_index) {
    auto scope = decoder.bytes("scope");
    auto value_name = decoder.bytes("value name");
    auto rank = decoder.decimal(':', kMaxRank, "rank");
    if (!scope || !value_name || !rank || scope->empty() ||
        value_name->empty()) {
      error = decoder.error_message().empty()
                  ? "record has empty scope or value name"
                  : decoder.error_message();
      return std::nullopt;
    }
    SymbolicDimRecord record{std::move(*scope), std::move(*value_name), {}};
    record.dimensions.reserve(*rank);
    for (size_t axis = 0; axis < *rank; ++axis) {
      auto dimension = decoder.bytes("dimension");
      if (!dimension) {
        error = decoder.error_message();
        return std::nullopt;
      }
      record.dimensions.push_back(std::move(*dimension));
    }
    if (!decoder.consume('\n')) {
      error = "record is not newline terminated";
      return std::nullopt;
    }

    std::pair<std::string, std::string> key{record.scope, record.value_name};
    if (have_previous && !(previous < key)) {
      error = "records are duplicate or not canonically sorted";
      return std::nullopt;
    }
    previous = std::move(key);
    have_previous = true;
    records.push_back(std::move(record));
  }
  if (!decoder.done()) {
    error = "trailing bytes after symbolic dimension records";
    return std::nullopt;
  }
  if (encode_symbolic_dim_records(records) != encoded) {
    error = "symbolic dimension encoding is not canonical";
    return std::nullopt;
  }
  return records;
}

} // namespace morphizen
