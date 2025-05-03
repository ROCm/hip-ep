/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <deque>
#include <filesystem>
#include <map>

#include "morphizen/graph.hpp"
#include "morphizen/model.hpp"
#include "morphizen/pass.hpp"
// clang-format off
// TODO include order matters
#include "morphizen/custom_op_imp.hpp"
// clang-format on
#include "pass_context_imp.hpp"

namespace vaip_core {
class Pass : public IPass {
public:
  friend class IPass;

public:
  Pass(std::shared_ptr<PassContextImp> context, const PassProto& pass_proto,
       const PassInfo& pass_info);
  Pass(const Pass&) = delete;
  virtual ~Pass();

public:
  static void run_all_passes(std::vector<std::shared_ptr<IPass>>& all_pass,
                             Graph& graph);
  void add_action(action_t action);

private:
  void apply(Graph& graph);
  void maybe_dump_txt(int index, const Graph& graph) const;
  void maybe_dump_onnx(int index, const Graph& graph) const;

  void maybe_gc(Graph& graph) const;
  virtual const std::string& name() const override final;
  virtual void* get_state() override final;
  virtual std::filesystem::path
  get_cache_file_name(const std::string& filename) const override final;
  virtual const ConfigProto& get_config_proto() const override final;
  virtual const std::filesystem::path& get_log_path() const override final;
  virtual void add_subgraph_device_count(const std::string& device,
                                         int count) override final;
  virtual void set_fix_info(const char* name, int fix_pos) override final;
  virtual int get_fix_info(const char* name) const override final;
  virtual bool has_fix_info(const char* name) const override final;
  virtual void dump_fix_info(const char* filename) const override final;
  virtual void dump_const_info(const char* name) const override final;
  virtual void dump_const_data(const char* name) const override final;
  virtual void create_const(const char* name, gsl::span<const char> data,
                            const std::vector<int64_t>& shape,
                            int type) override final;
  virtual void create_empty_const(const char* name, size_t size,
                                  const std::vector<int64_t>& shape,
                                  int type) override final;
  virtual void create_lazy_const(
      const char* name, size_t size, const std::vector<int64_t>& shape,
      int type,
      const std::function<void(gsl::span<char>)>& lazy) override final;

  virtual void create_const_alias(const char* alias_name,
                                  const char* name) override final;
  virtual bool has_const(const char* name) const override final;
  virtual ConstDataInfo get_const_info(const char* name) const override final;
  virtual void* get_const_data_ptr(const char* name,
                                   bool force) const override final;

  virtual const PassProto& get_pass_proto() const override;
  virtual std::vector<AttributeProtoPtr>&
  node_extra_attrs(const char* name) override;

  virtual const Node& fuse(Graph& graph, MetaDefProto&& meta_def) override;
  virtual const Node& level_2_fuse(Graph& graph,
                                   const MetaDefProto& meta_def) override;
  virtual MetaDefProto&
  fuse(Graph& graph, const std::string& name, const std::string& op_type,
       const std::vector<size_t>& nodes, const std::vector<std::string>& inputs,
       const std::vector<std::string>& outputs,
       const std::vector<std::string>& constant_initializers,
       const std::string& device) override;
  virtual const std::shared_ptr<PassContext> get_context() const override;
  virtual std::shared_ptr<PassContext> get_context() override;
  virtual void
  add_context_resource(const std::string& name,
                       std::shared_ptr<void> resource) override final;
  std::string seq_num_as_string() const;
  std::filesystem::path get_dump_file_name(size_t action_index,
                                           const std::string& ext) const;

private:
  std::shared_ptr<PassContextImp> context_;
  std::vector<action_t> action_;
  const PassProto& pass_proto_;
  const int sequence_no_;
  const PassInfo& pass_info_;
  std::shared_ptr<void> state_;
};
} // namespace vaip_core
