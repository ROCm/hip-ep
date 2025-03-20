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

#pragma once
#include "./_sanity_check.hpp"
#include "./pattern.hpp"

namespace vaip_core {
class BaseRule {
public:
  VAIP_DLL_SPEC static std::unique_ptr<BaseRule>
  create_rule_chain(std::vector<std::unique_ptr<BaseRule>>&& chain);
  VAIP_DLL_SPEC void apply(onnxruntime::Graph* graph);
  virtual bool apply_once(onnxruntime::Graph* graph,
                          const onnxruntime::Node* node) = 0;
  VAIP_DLL_SPEC virtual ~BaseRule();
};

class Rule : public BaseRule {
public:
  VAIP_DLL_SPEC static std::unique_ptr<Rule> create_rule(
      std::shared_ptr<Pattern> pattern,
      const std::function<bool(onnxruntime::Graph* graph, binder_t& binder)>&
          action);

public:
  explicit Rule() = default;
  virtual ~Rule() = default;

private:
  /// return true if graph is modified, false otherwise.
  virtual bool action(onnxruntime::Graph* graph, binder_t& binder) const = 0;
  virtual const Pattern* pattern() const = 0;

private:
  // it must be VAIP_DLL_SPEC because all derived classes in other DLLs need put
  // this function into vtable.
  VAIP_DLL_SPEC virtual bool
  apply_once(onnxruntime::Graph* graph,
             const onnxruntime::Node* node) override final;
};

} // namespace vaip_core
