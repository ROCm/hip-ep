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

#include "vaip/guess_reshape.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <glog/logging.h>
#include <vector>
DEF_ENV_PARAM(DEBUG_GUESS_RESHAPE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GUESS_RESHAPE) >= n)

namespace vaip_core {

struct ShapeConsumer {
  ShapeConsumer(const std::vector<int64_t>& shape)
      : shape_{shape}, value_{1}, idx_{0u}, indice_{} {}
  void maybe_consume() {
    if (idx_ != shape_.size()) {
      auto i = shape_.size() - idx_ - 1;
      indice_.emplace_back(i);
      value_ = value_ * shape_[i];
      idx_ = idx_ + 1;
    }
  }
  void reset() {
    value_ = 1;
    indice_ = {};
  }
  bool should_continue() { return idx_ < shape_.size(); }
  std::vector<int64_t> shape_;
  int64_t value_;
  size_t idx_;
  std::vector<size_t> indice_;
};

static std::string
show_mapping(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>&
                 mappings) {
  std::ostringstream str;
  str << "{";
  auto i = 0;
  for (auto& map : mappings) {
    str << (i ? "," : "");
    i++;
    str << " {first:[";
    auto j = 0;
    for (auto& f : map.first) {
      str << (j ? "," : "");
      j++;
      str << f;
    }
    str << "], second:[";
    j = 0;
    for (auto& s : map.second) {
      str << (j ? "," : "");
      j++;
      str << s;
    }
    str << "]}";
  }
  str << "}" << std::endl;
  return str.str();
}

VAIP_DLL_SPEC
std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
guess_reshape(const std::vector<int64_t>& shape_1,
              const std::vector<int64_t>& shape_2) {
  assert(!shape_1.empty());
  assert(!shape_2.empty());
  auto check_shape = [](const std::vector<int64_t>& shape) {
    return std::any_of(shape.begin(), shape.end(),
                       [](int64_t dim) { return dim < 0; });
  };
  auto ret = std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{};
  if (check_shape(shape_1) || check_shape(shape_2)) {
    return ret;
  }
  auto c1 = ShapeConsumer(shape_1);
  auto c2 = ShapeConsumer(shape_2);
  do {
    c1.maybe_consume();
    c2.maybe_consume();
    while (c1.value_ != c2.value_) {
      if (c1.value_ > c2.value_) {
        c2.maybe_consume();
      } else if (c1.value_ < c2.value_) {
        c1.maybe_consume();
      }
    }
    ret.emplace_back(
        std::make_pair(std::move(c1.indice_), std::move(c2.indice_)));
    c1.reset();
    c2.reset();
  } while (c1.should_continue() && c2.should_continue());
  while (c1.should_continue()) {
    c1.maybe_consume();
  }
  while (c2.should_continue()) {
    c2.maybe_consume();
  }
  assert(c1.value_ == 1);
  assert(c2.value_ == 1);
  ret.back().first.insert(ret.back().first.end(), c1.indice_.begin(),
                          c1.indice_.end());
  ret.back().second.insert(ret.back().second.end(), c2.indice_.begin(),
                           c2.indice_.end());
  std::reverse(ret.begin(), ret.end());
  MY_LOG(1) << show_mapping(ret);
  return ret;
}

} // namespace vaip_core
