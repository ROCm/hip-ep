/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4946) // reinterpret_cast between related classes in
                                  // protobuf
#endif
#include "morphizen/capability.pb.h"
#ifdef _WIN32
#  pragma warning(pop)
#endif
#include "morphizen/pass_context.hpp"
#include "morphizen/plugin.hpp"
#include "onnxruntime_api.hpp"
#include <filesystem>
#include <morphizen/custom_op.h>
struct OrtSession;
typedef struct OrtSession OrtSession;

namespace morphizen {

class ExecutionProviderConcrete
    : public ExecutionProvider,
      public WithPlugin<ExecutionProviderConcrete,
                        std::shared_ptr<const PassContext>,
                        const MetaDefProto&> {
public:
  static constexpr char entry_point[] = "create_execution_provider";

public:
  MORPHIZEN_DLL_SPEC
  ExecutionProviderConcrete(std::shared_ptr<const PassContext> context,
                            const MetaDefProto& meta_def);
  virtual ~ExecutionProviderConcrete();

  //
  std::shared_ptr<PassContext> get_context() {
    return std::const_pointer_cast<PassContext>(context_);
  }

  // do not use this function, it is a workaround for ORT new EP ABI.
  MetaDefProto& get_meta_def() { return *meta_def_; }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
};

template <typename T, typename CustomOpImp, typename = void>
struct CustomOp_compile_t {
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    auto ret = std::make_unique<CustomOpImp>(context, meta_def, nullptr);
    const_cast<PassContext*>(context.get())->on_custom_op_create_end();
    return ret;
  }
};

template <typename T, typename CustomOpImp>
struct CustomOp_compile_t<
    T, CustomOpImp, std::void_t<decltype(std::declval<T&>().get_model())>> {
  // this code is activated when MORPHIZEN_ORT_API_MAJOR >= 10 see
  // MorphiZen/morphizen#3504 for more details. T is ExecutionProviderImp
  // always. ExecutionProviderImp::get_mode() is added in
  // MORPHIZEN_ORT_API_MAJOR >= 10 see [MorphiZen] Cache node subgraph when
  // necessary (Onnxruntime#22073) for details.
  //
  // in MorphiZenExecutionProvider::Compile(), we create the model object is
  // fall back on CPU is enabled. as below.

  // clang-format off
  /*
  if (ep->get_meta_def_fallback_CPU()) {
      auto& subgraph = fused_node_graph.filtered_graph.get();
      auto& logger = logging::LoggingManager::DefaultLogger();
      auto model_proto = subgraph.CreateModel(logger)->ToProto();
      subgraph.ToProto(*model_proto->mutable_graph(), true, true);
      auto local_registries = IOnnxRuntimeOpSchemaRegistryList{subgraph.GetSchemaRegistry()};
      auto model = Model::Create(std::move(*model_proto), subgraph.ModelPath(), &local_registries, logger);
      ep->set_model(model.release());
    }
*/
  // clang-format on
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    // transfer the model from ExecutionProviderImp to CustomOpImp, so that
    // CustomOpImp<XXX> can access the model
    // then in CustomOpImp constructor, we create a session from the model

    // clang-format off
    /*
     session_{CustomOp_InitSession_t<
        morphizen::OrtApiForMorphizen>::CustomOp_InitSession(morphizen::api(),
                                                     model, meta_def)} {}
    */

    // the next step is to create a ORT::Session from the model
    // NOTE THAT: the model object is deleted after the session object is created
    // so that there is no memory leak for the model object
    /*
    CustomOp_InitSession(const T* api, onnxruntime::Model* model,
                     const std::shared_ptr<MetaDefProto>& meta_def) {
    if (meta_def->fallback_cpu()) {
      CHECK(model);
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MorphiZen_CustomOp");
      auto model_proto = api->model_to_proto(*model);
      auto mproto_string = api->model_proto_serialize_as_string(*model_proto);
      auto session =
          Ort::Session(env, mproto_string.get()->c_str(),
                      mproto_string.get()->size(), Ort::SessionOptions());
      api->model_proto_delete(model_proto);
      api->model_delete(model);
      return session;
    }
    return Ort::Session(nullptr);
  }
  */
    // clang-format on

    // the Ort::Session is very much like a unique_ptr, it transfer the
    // ownership to CustomOpImp::session_. now CustomOpXXX is potentially fall
    // back on CPU, by invoking CustomOpImp::ComputeCpu. it needs some resources
    // to create the session object. So if CustomOp<XXX> don't need this
    // feature, please turn it off by setting meta_def->set_fallback_cpu(false).

    auto ret =
        std::make_unique<CustomOpImp>(context, meta_def, self->get_model());
    const_cast<T*>(self)->set_model(nullptr);
    const_cast<PassContext*>(context.get())->on_custom_op_create_end();
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
  // MORPHIZEN_ORT_API_MAJOR >= 11
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
  MORPHIZEN_DLL_SPEC CustomOpImp(std::shared_ptr<const PassContext> context,
                                 const std::shared_ptr<MetaDefProto>& meta_def,
                                 onnxruntime::Model* model);
  MORPHIZEN_DLL_SPEC virtual ~CustomOpImp();

protected:
  Ort::ConstValue ctxGetInput(Ort::KernelContext& ctx, int index) const;
  Ort::UnownedValue ctxGetOutput(Ort::KernelContext& ctx, int index,
                                 const int64_t* dim_values,
                                 size_t dim_count) const;

public:
  virtual void Compute(const OrtApi* api, OrtKernelContext* context) const = 0;
  /*
  How to use?
  1. After try_fuse, in meta_def, set_fallback_cpu(true)
  auto [meta_def, fuse_error] = self_.try_fuse(onnx_graph_,
  subgraph->get_name(), inputs, outputs, {}, "CUSTOM");
  meta_def->set_fallback_cpu(true);
  2. In custom_op.cpp, if you need to fall back to CPU, call ComputeCpu(api,
  context);
  */
  MORPHIZEN_DLL_SPEC void ComputeCpu(const OrtApi* api,
                                     OrtKernelContext* context) const;
  std::shared_ptr<PassContext> get_context() const {
    return std::const_pointer_cast<PassContext>(context_);
  }

  std::string get_meta_def_param() const {
    return context_->get_meta_def_param(*meta_def_);
  }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
  Ort::Session session_;
};

} // namespace morphizen
