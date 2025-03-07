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
#include "./pass_context.hpp"
#include "./vaip_plugin.hpp"
#include "onnxruntime_api.hpp"
#include "vaip/capability.pb.h"
#include <filesystem>
#include <vaip/custom_op.h>
struct OrtSession;
typedef struct OrtSession OrtSession;

namespace vaip_core {

class ExecutionProviderConcrete
    : public ExecutionProvider,
      public WithPlugin<ExecutionProviderConcrete,
                        std::shared_ptr<const PassContext>,
                        const MetaDefProto&> {
public:
  static constexpr char entry_point[] = "create_execution_provider";

public:
  VAIP_DLL_SPEC
  ExecutionProviderConcrete(std::shared_ptr<const PassContext> context,
                            const MetaDefProto& meta_def);
  virtual ~ExecutionProviderConcrete();

  //
  std::shared_ptr<PassContext> get_context() {
    return std::const_pointer_cast<PassContext>(context_);
  }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
};

template <typename T, typename CustomOpImp, typename = void>
struct CustomOp_compile_t {
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    return std::make_unique<CustomOpImp>(context, meta_def, nullptr);
  }
};

template <typename T, typename CustomOpImp>
struct CustomOp_compile_t<
    T, CustomOpImp, std::void_t<decltype(std::declval<T&>().get_model())>> {
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    auto ret =
        std::make_unique<CustomOpImp>(context, meta_def, self->get_model());
    const_cast<T*>(self)->set_model(nullptr);
    return ret;
  }
};

template <class CustomOpImp>
class ExecutionProviderImp : public ExecutionProviderConcrete {
public:
  ExecutionProviderImp(std::shared_ptr<const PassContext> context,
                       const MetaDefProto& meta_def)
      : ExecutionProviderConcrete{context, meta_def} {}
  virtual ~ExecutionProviderImp() {}
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_inputs() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->inputs().begin(), meta_def_->inputs().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_outputs() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->outputs().begin(), meta_def_->outputs().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_nodes() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->nodes().begin(), meta_def_->nodes().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_constant_initializer() const override final {
    return DllSafe<std::vector<std::string>>(
        std::vector<std::string>{meta_def_->constant_initializers().begin(),
                                 meta_def_->constant_initializers().end()});
  }
  // VAIP_ORT_API_MAJOR >= 11
  virtual bool get_meta_def_fallback_CPU() const final {
    return meta_def_->fallback_cpu();
  }

  virtual std::unique_ptr<CustomOp> compile() const override final {
    return CustomOp_compile_t<ExecutionProviderImp,
                              CustomOpImp>::CustomOp_compile(this, context_,
                                                             meta_def_);
  };
};

class CustomOpImp : public CustomOp {
public:
  VAIP_DLL_SPEC CustomOpImp(std::shared_ptr<const PassContext> context,
                            const std::shared_ptr<MetaDefProto>& meta_def,
                            onnxruntime::Model* model);
  VAIP_DLL_SPEC virtual ~CustomOpImp();

public:
  virtual void Compute(const OrtApi* api, OrtKernelContext* context) const = 0;
  /*
  How to use?
  1. After try_fuse, in meta_def, set_fallback_cpu(true)
  auto [meta_def, fuse_error] = self_.try_fuse(onnx_graph_,
  subgraph->get_name(), inputs, outputs, {}, "DPU");
  meta_def->set_fallback_cpu(true);
  2. In custom_op.cpp, if you need to fall back to CPU, call ComputeCpu(api,
  context);
  */
  VAIP_DLL_SPEC void ComputeCpu(const OrtApi* api,
                                OrtKernelContext* context) const;
  std::shared_ptr<PassContext> get_context() const {
    return std::const_pointer_cast<PassContext>(context_);
  }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
  Ort::Session session_;
};

} // namespace vaip_core
