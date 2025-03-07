/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include "./pattern_constant.hpp"
#include "./pattern_node.hpp"
#include "./pattern_or.hpp"
#include "./pattern_wildcard.hpp"
#include "vaip/pattern.hpp"
#include <cassert>
#include <iostream>
#include <optional>
#include <type_traits>
namespace vaip_core {
using pos_t = std::string::size_type;

template <typename T> class Parser {
public:
  Parser(PatternBuilder& builder, const std::string& pattern, pos_t pos,
         T&& result)
      : builder_{builder}, pattern_{pattern}, pos_{pos},
        result_{std::forward<T>(result)} {}
  Parser(PatternBuilder& builder, const std::string& pattern, pos_t pos)
      : builder_{builder}, pattern_{pattern}, pos_{pos}, result_{std::nullopt} {
  }
  void error(const std::string msg);
  Parser<char> peek();
  Parser<size_t> parse_skip_whitespace();
  Parser<std::string> parse_op_type();
  Parser<std::vector<std::shared_ptr<Pattern>>> parse_args();
  Parser<std::unique_ptr<Pattern>> parse_pattern();
  void check_eof();

protected:
  template <typename U, typename V> Parser<U> ok(V&& result, pos_t next_pos) {
    return Parser<U>(builder_, pattern_, next_pos, std::forward<V>(result));
  }
  template <typename U> Parser<U> fail(pos_t next_pos = std::string::npos) {
    return Parser<U>(builder_, pattern_, next_pos);
  }

public:
  PatternBuilder& builder_;
  const std::string& pattern_;
  const pos_t pos_;
  std::optional<T> result_;
};

template <typename T> void Parser<T>::error(const std::string msg) {
  if (pos_ != std::string::npos) {
    assert(pos_ < pattern_.size());
    std::cerr << "parse error " << msg << " at " << pos_ << " pattern=["
              << pattern_.substr(0, pos_) << "<- ERROR MARK ->"
              << pattern_.substr(pos_) << "]" << std::endl;
  } else {
    std::cerr << "parse error " << msg << " at end of string, pattern=["
              << pattern_ << std::endl;
  }
  std::abort();
}

template <typename T> Parser<size_t> Parser<T>::parse_skip_whitespace() {
  auto pos = pattern_.find_first_not_of(" \t\r\n", pos_);
  size_t ret = 0u;
  if (pos != std::string::npos) {
    ret = pos - pos_;
  }
  return ok<size_t>(std::forward<size_t>(ret), pos);
}

template <typename T> Parser<char> Parser<T>::peek() {
  if (pos_ == std::string::npos || pos_ >= pattern_.size()) {
    error("unexpected eof when parsing pattern");
  }
  char c = pattern_[pos_];
  return ok<char>(std::move(c), pos_ + 1);
}

template <typename T> Parser<std::string> Parser<T>::parse_op_type() {
  auto c = peek();
  switch (c.result_.value()) {
  case '"': {
    auto end_pos = pattern_.find('\"', pos_ + 1);
    if (end_pos == std::string::npos) {
      error("cannot end quote");
    }
    return ok<std::string>(pattern_.substr(pos_ + 1, end_pos - pos_ - 1),
                           end_pos + 1);
  }; break;
  default:
    error("cannot begin quote");
  }
  return fail<std::string>();
}

template <typename T>
Parser<std::vector<std::shared_ptr<Pattern>>> Parser<T>::parse_args() {
  auto c = parse_skip_whitespace().peek();

  auto p1 = parse_pattern();
}
template <typename T>
Parser<std::unique_ptr<Pattern>> Parser<T>::parse_pattern() {
  auto c = parse_skip_whitespace().peek();
  switch (c.result_.value()) {
  case '*': {
    return ok<std::unique_ptr<Pattern>>(builder_.wildcard(), pos_ + 1);
  }
  case '(': {
    auto op_type = c.parse_op_type();
    auto args = parse_args(builder_, pattern_, pos_);
    return ok<std::unique_ptr<Pattern>>(
        builder_.node2(op_type.result_.value(), {}), std::string::npos);
  }
  default:
    error("unknown character");
  }
  return fail<std::unique_ptr<Pattern>>();
}

VAIP_DLL_SPEC std::unique_ptr<Pattern>
Pattern::parse(const std::string& pattern) {
  auto builder = PatternBuilder();
  auto parser = Parser<char>(builder, pattern, 0u, ' ');
  auto root_pat = parser.parse_pattern();
  return std::move(root_pat.result_.value());
}
} // namespace vaip_core
