// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace morphizen {

// Repeated field with stable element addresses. dump_to_proto holds a pointer
// to an earlier element while appending later ones; std::vector would
// invalidate that pointer.
template <typename T> class ProtoList {
public:
  using container = std::deque<T>;
  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

  T *Add() {
    items_.emplace_back();
    return &items_.back();
  }
  void Add(const T &value) { items_.push_back(value); }
  void Add(T &&value) { items_.push_back(std::move(value)); }
  void Clear() { items_.clear(); }
  int size() const { return static_cast<int>(items_.size()); }
  bool empty() const { return items_.empty(); }
  void Reserve(int n) { (void)n; }
  T &operator[](int index) { return items_.at(static_cast<size_t>(index)); }
  const T &operator[](int index) const {
    return items_.at(static_cast<size_t>(index));
  }
  iterator begin() { return items_.begin(); }
  iterator end() { return items_.end(); }
  const_iterator begin() const { return items_.begin(); }
  const_iterator end() const { return items_.end(); }

private:
  container items_;
};

namespace json {

class ParseError : public std::runtime_error {
public:
  explicit ParseError(const std::string &message)
      : std::runtime_error(message) {}
};

class Json {
public:
  enum class Kind { Null, Bool, Int, Float, String, Array, Object };
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;

  Json() = default;
  static Json null() { return Json(); }
  static Json boolean(bool value) {
    Json json;
    json.value_ = value;
    return json;
  }
  static Json integer(int64_t value) {
    Json json;
    json.value_ = value;
    return json;
  }
  static Json number(double value) {
    Json json;
    json.value_ = value;
    return json;
  }
  static Json str(std::string value) {
    Json json;
    json.value_ = std::move(value);
    return json;
  }
  static Json array(Array value = {}) {
    Json json;
    json.value_ = std::move(value);
    return json;
  }
  static Json object(Object value = {}) {
    Json json;
    json.value_ = std::move(value);
    return json;
  }

  Kind kind() const {
    switch (value_.index()) {
    case 1:
      return Kind::Bool;
    case 2:
      return Kind::Int;
    case 3:
      return Kind::Float;
    case 4:
      return Kind::String;
    case 5:
      return Kind::Array;
    case 6:
      return Kind::Object;
    default:
      return Kind::Null;
    }
  }
  bool is_null() const { return kind() == Kind::Null; }
  bool is_object() const { return kind() == Kind::Object; }
  bool is_array() const { return kind() == Kind::Array; }

  bool as_bool() const {
    if (const auto *value = std::get_if<bool>(&value_)) {
      return *value;
    }
    throw ParseError("expected bool");
  }
  int64_t as_int() const {
    if (const auto *value = std::get_if<int64_t>(&value_)) {
      return *value;
    }
    if (const auto *value = std::get_if<double>(&value_)) {
      return static_cast<int64_t>(*value);
    }
    throw ParseError("expected number");
  }
  int as_int32() const { return static_cast<int>(as_int()); }
  double as_float() const {
    if (const auto *value = std::get_if<double>(&value_)) {
      return *value;
    }
    if (const auto *value = std::get_if<int64_t>(&value_)) {
      return static_cast<double>(*value);
    }
    throw ParseError("expected number");
  }
  const std::string &as_string() const {
    if (const auto *value = std::get_if<std::string>(&value_)) {
      return *value;
    }
    throw ParseError("expected string");
  }
  const Array &as_array() const {
    if (const auto *value = std::get_if<Array>(&value_)) {
      return *value;
    }
    throw ParseError("expected array");
  }
  Array &as_array() {
    if (auto *value = std::get_if<Array>(&value_)) {
      return *value;
    }
    throw ParseError("expected array");
  }
  const Object &as_object() const {
    if (const auto *value = std::get_if<Object>(&value_)) {
      return *value;
    }
    throw ParseError("expected object");
  }
  Object &as_object() {
    if (auto *value = std::get_if<Object>(&value_)) {
      return *value;
    }
    throw ParseError("expected object");
  }
  const Json *find(const std::string &key) const {
    if (!is_object()) {
      return nullptr;
    }
    const auto &object = as_object();
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }

private:
  std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array,
               Object>
      value_{nullptr};
};

inline void append_utf8(std::string &out, uint32_t code_point) {
  if (code_point <= 0x7F) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

class Parser {
public:
  explicit Parser(std::string_view text) : text_(text) {}

  Json parse() {
    skip();
    Json value = parse_value();
    skip();
    if (index_ != text_.size()) {
      throw ParseError("trailing data after JSON value");
    }
    return value;
  }

private:
  std::string_view text_;
  size_t index_ = 0;

  void skip() {
    while (index_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[index_]))) {
      ++index_;
    }
  }
  char peek() const {
    if (index_ >= text_.size()) {
      throw ParseError("unexpected end of JSON");
    }
    return text_[index_];
  }
  char get() {
    char ch = peek();
    ++index_;
    return ch;
  }
  bool consume(char expected) {
    skip();
    if (index_ < text_.size() && text_[index_] == expected) {
      ++index_;
      return true;
    }
    return false;
  }
  void expect(char expected) {
    if (!consume(expected)) {
      throw ParseError(std::string("expected '") + expected + "'");
    }
  }
  bool starts_with(std::string_view literal) const {
    return text_.substr(index_, literal.size()) == literal;
  }

  Json parse_value() {
    skip();
    char ch = peek();
    if (ch == '{') {
      return parse_object();
    }
    if (ch == '[') {
      return parse_array();
    }
    if (ch == '"') {
      return Json::str(parse_string());
    }
    if (ch == 't') {
      expect_literal("true");
      return Json::boolean(true);
    }
    if (ch == 'f') {
      expect_literal("false");
      return Json::boolean(false);
    }
    if (ch == 'n') {
      expect_literal("null");
      return Json::null();
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return parse_number();
    }
    throw ParseError("invalid JSON value");
  }

  void expect_literal(std::string_view literal) {
    if (!starts_with(literal)) {
      throw ParseError("invalid JSON literal");
    }
    index_ += literal.size();
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (index_ >= text_.size()) {
        throw ParseError("unterminated string");
      }
      char ch = get();
      if (ch == '"') {
        break;
      }
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (index_ >= text_.size()) {
        throw ParseError("unterminated escape");
      }
      char escaped = get();
      switch (escaped) {
      case '"':
      case '\\':
      case '/':
        out.push_back(escaped);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u':
        out.append(parse_unicode_escape());
        break;
      default:
        throw ParseError("invalid string escape");
      }
    }
    return out;
  }

  std::string parse_unicode_escape() {
    auto hex = [this]() {
      if (index_ + 4 > text_.size()) {
        throw ParseError("short unicode escape");
      }
      uint32_t value = 0;
      for (int n = 0; n < 4; ++n) {
        char ch = get();
        value <<= 4;
        if (ch >= '0' && ch <= '9') {
          value += static_cast<uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
          value += static_cast<uint32_t>(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
          value += static_cast<uint32_t>(ch - 'A' + 10);
        } else {
          throw ParseError("invalid unicode escape");
        }
      }
      return value;
    };
    uint32_t code = hex();
    if (code >= 0xD800 && code <= 0xDBFF) {
      if (!(index_ + 2 <= text_.size() && text_[index_] == '\\' &&
            text_[index_ + 1] == 'u')) {
        throw ParseError("lonely unicode surrogate");
      }
      index_ += 2;
      uint32_t low = hex();
      if (low < 0xDC00 || low > 0xDFFF) {
        throw ParseError("invalid unicode surrogate pair");
      }
      code = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00));
    }
    std::string encoded;
    append_utf8(encoded, code);
    return encoded;
  }

  Json parse_number() {
    const size_t start = index_;
    if (peek() == '-') {
      get();
    }
    if (index_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[index_]))) {
      throw ParseError("invalid number");
    }
    if (text_[index_] == '0') {
      ++index_;
    } else {
      while (index_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        ++index_;
      }
    }
    bool is_float = false;
    if (index_ < text_.size() && text_[index_] == '.') {
      is_float = true;
      ++index_;
      if (index_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        throw ParseError("invalid fraction");
      }
      while (index_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        ++index_;
      }
    }
    if (index_ < text_.size() && (text_[index_] == 'e' || text_[index_] == 'E')) {
      is_float = true;
      ++index_;
      if (index_ < text_.size() && (text_[index_] == '+' || text_[index_] == '-')) {
        ++index_;
      }
      if (index_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        throw ParseError("invalid exponent");
      }
      while (index_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        ++index_;
      }
    }
    const std::string token(text_.substr(start, index_ - start));
    if (!is_float) {
      try {
        size_t used = 0;
        long long value = std::stoll(token, &used, 10);
        if (used != token.size()) {
          throw ParseError("invalid integer");
        }
        return Json::integer(static_cast<int64_t>(value));
      } catch (const std::out_of_range &) {
        throw ParseError("integer out of range");
      }
    }
    try {
      size_t used = 0;
      double value = std::stod(token, &used);
      if (used != token.size()) {
        throw ParseError("invalid number");
      }
      return Json::number(value);
    } catch (const std::out_of_range &) {
      throw ParseError("number out of range");
    }
  }

  Json parse_array() {
    expect('[');
    Json::Array values;
    skip();
    if (consume(']')) {
      return Json::array(std::move(values));
    }
    while (true) {
      values.push_back(parse_value());
      skip();
      if (consume(']')) {
        break;
      }
      expect(',');
    }
    return Json::array(std::move(values));
  }

  Json parse_object() {
    expect('{');
    Json::Object values;
    skip();
    if (consume('}')) {
      return Json::object(std::move(values));
    }
    while (true) {
      skip();
      if (peek() != '"') {
        throw ParseError("expected object key");
      }
      std::string key = parse_string();
      skip();
      expect(':');
      values.insert_or_assign(std::move(key), parse_value());
      skip();
      if (consume('}')) {
        break;
      }
      expect(',');
    }
    return Json::object(std::move(values));
  }
};

inline Json parse(std::string_view text) { return Parser(text).parse(); }

inline void dump_string(std::ostream &out, const std::string &value) {
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20) {
        const char *hex = "0123456789abcdef";
        out << "\\u00" << hex[ch >> 4] << hex[ch & 0xF];
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  out << '"';
}

inline void dump_into(std::ostream &out, const Json &value, bool pretty,
                      int indent) {
  auto newline = [&]() {
    if (pretty) {
      out << '\n' << std::string(static_cast<size_t>(indent), ' ');
    }
  };
  switch (value.kind()) {
  case Json::Kind::Null:
    out << "null";
    break;
  case Json::Kind::Bool:
    out << (value.as_bool() ? "true" : "false");
    break;
  case Json::Kind::Int:
    out << value.as_int();
    break;
  case Json::Kind::Float: {
    std::ostringstream number;
    number.precision(17);
    number << value.as_float();
    out << number.str();
    break;
  }
  case Json::Kind::String:
    dump_string(out, value.as_string());
    break;
  case Json::Kind::Array: {
    const auto &array = value.as_array();
    out << '[';
    if (!array.empty()) {
      if (pretty) {
        indent += 2;
      }
      for (size_t i = 0; i < array.size(); ++i) {
        newline();
        dump_into(out, array[i], pretty, indent);
        if (i + 1 != array.size()) {
          out << ',';
        }
      }
      if (pretty) {
        indent -= 2;
        newline();
      }
    }
    out << ']';
    break;
  }
  case Json::Kind::Object: {
    const auto &object = value.as_object();
    out << '{';
    if (!object.empty()) {
      if (pretty) {
        indent += 2;
      }
      size_t i = 0;
      for (const auto &entry : object) {
        newline();
        dump_string(out, entry.first);
        out << (pretty ? ": " : ":");
        dump_into(out, entry.second, pretty, indent);
        if (++i != object.size()) {
          out << ',';
        }
      }
      if (pretty) {
        indent -= 2;
        newline();
      }
    }
    out << '}';
    break;
  }
  }
}

inline std::string dump(const Json &value, bool pretty = false) {
  std::ostringstream out;
  dump_into(out, value, pretty, 0);
  return out.str();
}

inline const Json *object_field(const Json &value, const char *key) {
  if (!value.is_object()) {
    throw ParseError(std::string("expected object when reading ") + key);
  }
  return value.find(key);
}

} // namespace json
} // namespace morphizen
