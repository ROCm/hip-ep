/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// glog must be included very beginning.
#include <deque>
#include <fstream>
#include <glog/logging.h>
///

#include "./config.hpp"
#include "./profile_utils.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/graph.hpp"
#include "morphizen/plugin.hpp"
#include "morphizen/util.hpp"
#include "pass_imp.hpp"
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <ios>
#include <string>
#include <thread>
static int g_sequence_no = 0;
DEF_ENV_PARAM(ENABLE_SAVE_GRAPH_TXT, "0")
DEF_ENV_PARAM(ENABLE_SAVE_GRAPH_MLIR, "0")
DEF_ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT, "0")
DEF_ENV_PARAM(ENABLE_SAVE_ONNX_MODEL, "0")
DEF_ENV_PARAM(DEBUG_MORPHIZEN_PASS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_PASS) >= n)

namespace morphizen {

void IPass::attach_meta_def_param(MetaDefProto& meta_def,
                                  const char* json_param) const {
  if (json_param == nullptr) {
    return;
  }
  auto json_str = std::string(json_param);
  auto struct_proto = google::protobuf::Struct();
  auto status =
      google::protobuf::util::JsonStringToMessage(json_str, &struct_proto);
  if (!status.ok()) {
    LOG(FATAL) << "failed to attach meta_def param: " << status.ToString();
  }
  meta_def.mutable_param()->CopyFrom(struct_proto);
}

static bool can_be_dumped(const std::shared_ptr<PassContext>& proto) {
  static bool warned = false;
  bool can_be_dumped = proto->get_provider_option("encryption_key", "") == "";
  if (!can_be_dumped && warned == false) {
    LOG(WARNING) << "dumping is not allowed when encryption enabled";
    warned = true;
  }
  return can_be_dumped;
}

struct BreakOnModifed {
  int isModifed = 0;
};
IPass::action_t
create_action_from_node_action(IPass::node_action_t node_action) {
  return [node_action](IPass& self, Graph& graph) {
    int modified = 0;
    auto counter = 0;
    auto last_match_idx = -1;
    auto match_idx = -1;
    do {
      modified = false;
      match_idx = -1;
#if MORPHIZEN_ORT_API_MAJOR >= 14
      auto leaf_nodes_cxx = morphizen_cxx::GraphConstRef(graph).output_nodes();
      // Convert to raw pointers for the API
      auto leaf_nodes = std::vector<const Node*>();
      leaf_nodes.reserve(leaf_nodes_cxx.size());
      for (auto& n : leaf_nodes_cxx) {
        leaf_nodes.push_back(n.ptr());
      }
      MORPHIZEN_ORT_API(graph_reverse_dfs_from_preemp)
      (
          graph, leaf_nodes, nullptr,
          [&](const Node* node) {
            auto node_ref =
                morphizen_cxx::NodeConstRef::from_node(graph, *node);
            auto node_idx = node_ref.index();
            modified = node_action(self, graph, *node);
            if (modified) {
              match_idx = (int)node_idx;
            }
            return modified;
          },
          nullptr,
          [&modified](const Node* /*from*/, const Node* /*to*/) {
            return modified;
          });
#else
      try {
        auto leaf_nodes_cxx =
            morphizen_cxx::GraphConstRef(graph).output_nodes();
        morphizen_cxx::GraphConstRef(graph).reverse_dfs_from_multi(
            gsl::make_span(leaf_nodes_cxx),
            nullptr, //
            [&](morphizen_cxx::NodeConstRef node) {
              auto node_idx = node.index();
              modified = node_action(self, graph, *node.ptr());
              if (modified) {
                match_idx = (int)node_idx;
              }
              if (modified) {
                throw BreakOnModifed{1};
              }
              return false; // leave callback return value
            },              //
            nullptr,        // comp
            [&modified](morphizen_cxx::NodeConstRef /*from*/,
                        morphizen_cxx::NodeConstRef /*to*/) {
              return modified;
            });
      } catch ([[maybe_unused]] BreakOnModifed break_on_modifed) {
      }
#endif
      if (last_match_idx == match_idx) {
        counter++;
      }
      last_match_idx = match_idx;
    } while (modified && counter < 100);
    if (modified) {
      LOG(FATAL) << "endless loop occurs. last_match_idx=" << last_match_idx
                 << " match_idx=" << match_idx;
    }
  };
} // namespace morphizen

Pass::Pass(std::shared_ptr<PassContextImp> context, const PassProto& pass_proto,
           const PassInfo& pass_info)
    : context_(context), pass_proto_{pass_proto}, sequence_no_{g_sequence_no++},
      pass_info_{pass_info}, state_{} {
  if (pass_info.init) {
    auto self = pass_info.init(*this);
    if (pass_info.deinit) {
      state_ = std::shared_ptr<void>(self, pass_info.deinit);
    } else {
      state_ = std::shared_ptr<void>(self, [](void*) {});
    }
  }
  MY_LOG(1) << "create pass: " << name() << " " << pass_info.size
            << " actions in total";
  for (auto i = 0u; i < pass_info.size; ++i) {
    this->add_action(pass_info.get_action(i));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_PASS))
      << "pass is created: " << (void*)this << " name=" << this->name();
}
Pass::~Pass() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_PASS))
      << "pass is decontructed: " << (void*)this << " name=" << this->name();
}
void Pass::apply(Graph& graph_old) {
  Graph* graph = &graph_old;
  int action_index = 0;
  if (pass_info_.preprocess) {
    pass_info_.preprocess(this->get_state(), *this, *graph);
  }
  for (auto& action : action_) {
    action(*this, *graph);
    graph =
        (Graph*)get_context()->get_context_resource("__current_graph").get();
    maybe_dump_txt(action_index, *graph);
    maybe_dump_mlir(action_index, *graph);
    morphizen_cxx::GraphRef(*graph).resolve();
    maybe_dump_txt(action_index + 100, *graph);
    maybe_dump_mlir(action_index + 100, *graph);
    maybe_gc(*graph);
    morphizen_cxx::GraphRef(*graph).resolve();
    maybe_dump_onnx(action_index, *graph);
    action_index = action_index + 1;
  }
  if (pass_info_.postprocess) {
    pass_info_.postprocess(this->get_state(), *this, *graph);
  }
}

const std::string& Pass::name() const { return get_pass_proto().name(); }

void Pass::run_all_passes(std::vector<std::shared_ptr<IPass>>& all_pass,
                          Graph& graph) {
  MY_LOG(1) << "start to run passes, " << all_pass.size() << " in total";
  auto __all_pass_start_time = std::chrono::steady_clock::now();
  PassContextImp* ctx = nullptr;
  for (auto& pass_interface : all_pass) {
    auto pass = dynamic_cast<Pass*>(pass_interface.get());
    CHECK(pass != nullptr) << "dynamic_cast failed";
    if (ctx == nullptr) {
      ctx = pass->context_.get();
      pass->add_context_resource(
          "__current_graph",
          std::shared_ptr<void>((void*)&graph, [](void*) {}));
    }
    auto label = std::to_string(pass->sequence_no_) + "-" + pass->name() + "@" +
                 pass->get_pass_proto().plugin();
    auto measure = ctx->measure(label);
    auto current_graph = (Graph*)pass->get_context()
                             ->get_context_resource("__current_graph")
                             .get();
    auto __pass1_start_time = std::chrono::steady_clock::now();
    MY_LOG(1) << "begin pass :"
              << "run pass [" << pass->seq_num_as_string()
              << "]: " << pass->name()                                       //
              << " plugin=" << pass->get_pass_proto().plugin()               //
              << " enable_log="
              << (pass->get_pass_proto().enable_log() ? "true" : "false")    //
              << " log_verbosity=" << pass->get_pass_proto().log_verbosity() //
        ;

    {
      auto with_pass = pass->context_->with_current_pass(
          *pass); // save and restore current pass.
      pass->apply(*current_graph);
    }
    auto __pass2_start_time = std::chrono::steady_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       __pass2_start_time - __pass1_start_time)
                       .count();
    MY_LOG(1) << "run pass [" << pass->seq_num_as_string()
              << "]: " << pass->name() << " " << ((float)time_us) / 1000.0f
              << " ms elapse. ";
  }
  auto __all_pass_end_time = std::chrono::steady_clock::now();
  auto all_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         __all_pass_end_time - __all_pass_start_time)
                         .count();
  MY_LOG(1) << "run all passes done. " << ((float)all_time_us) / 1000.0f
            << " ms elapse. ";
}

void Pass::maybe_dump_txt(int action_index, const Graph& graph) const {
  if ((!ENV_PARAM(ENABLE_SAVE_GRAPH_TXT)) || (!can_be_dumped(context_))) {
    return;
  }
  auto filepath = get_dump_file_name(action_index, ".txt");
  LOG(INFO) << "pass=" << name()
            << " save txt file to: " << filepath.u8string();
  auto basedir = filepath.parent_path();
  if (!std::filesystem::exists(basedir)) {
    std::filesystem::create_directories(basedir);
  }
  dump_graph(graph, filepath.u8string());
}

void Pass::maybe_dump_mlir(int action_index, const Graph& graph) const {
  if ((!ENV_PARAM(ENABLE_SAVE_GRAPH_MLIR)) || (!can_be_dumped(context_))) {
    return;
  }
#if MORPHIZEN_ORT_API_MAJOR >= 18
  // Use .mlir for text format, .mlirbc for bytecode format
  auto ext = ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT) ? ".mlir" : ".mlirbc";
  auto filepath = get_dump_file_name(action_index, ext);
  LOG(INFO) << "pass=" << name()
            << " save mlir file to: " << filepath.u8string() << " format="
            << (ENV_PARAM(MORPHIZEN_SAVE_MLIR_AS_TEXT) ? "text" : "bytecode");
  auto basedir = filepath.parent_path();
  if (!std::filesystem::exists(basedir)) {
    std::filesystem::create_directories(basedir);
  }
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  auto mlir_string = graph_ref.save_string();
  std::ofstream out(filepath, std::ios::binary);
  if (out.is_open()) {
    out << *mlir_string;
    out.close();
  } else {
    LOG(WARNING) << "Failed to open file for writing: " << filepath.u8string();
  }
#else
  (void)action_index;
  (void)graph;
  LOG(WARNING)
      << "ENABLE_SAVE_GRAPH_MLIR requires MORPHIZEN_ORT_API_MAJOR >= 18";
#endif
}

// onnx graph_save to onnx model maybe has bugs
void Pass::maybe_dump_onnx(int action_index, const Graph& graph) const {
  if ((!ENV_PARAM(ENABLE_SAVE_ONNX_MODEL)) || (!can_be_dumped(context_))) {
    return;
  }
  auto filepath = get_dump_file_name(action_index, ".onnx");
  // get_dump_file_name(action_index, ".dat");
#if _WIN32
  auto dat_filepath = std::string("NUL");
#else
  auto dat_filepath = std::filesystem::relative("/dev/null", filepath);
#endif
  LOG(INFO) << "pass=" << name()
            << " save onnx model file to: " << filepath.u8string()
            << ", data file to " << dat_filepath;
  auto basedir = filepath.parent_path();
  if (!std::filesystem::exists(basedir)) {
    std::filesystem::create_directories(basedir);
  }
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  graph_ref.save(filepath, dat_filepath,
#if MORPHIZEN_ORT_API_MAJOR >= 7
                 // `mode_clone` is already optimized so that constant
                 // intializers are shared with the original graph.
                 std::numeric_limits<size_t>::max()
#else
                 128u
#endif
  );
}

void Pass::maybe_gc(Graph& graph) const {
  if (pass_proto_.enable_gc()) {
    morphizen_cxx::GraphRef(graph).gc();
  }
}

void* Pass::get_state() { return state_.get(); }

const ConfigProto& Pass::get_config_proto() const { return context_->config_; }
std::map<std::string, std::string> Pass::get_all_provider_options() const {
  return context_->get_all_provider_options();
}

void Pass::add_subgraph_device_count(const std::string& device, int count) {
  context_->context_proto.mutable_device_subgraph_count()->insert(
      google::protobuf::MapPair<std::string, int>{device, count});
}

const PassProto& Pass::get_pass_proto() const { return pass_proto_; }

std::string Pass::get_pass_generic_param() const {
  auto json_str = std::string();
  auto status = google::protobuf::util::MessageToJsonString(
      pass_proto_.pass_generic_param(), &json_str);
  if (!status.ok()) {
    LOG(FATAL) << "failed to get pass_generic_param: " << status.ToString();
  }
  return json_str;
}

std::vector<AttributeProtoPtr>& Pass::node_extra_attrs(const char* name) {
  auto& node_extra_attrs = context_->node_extra_attrs;
  auto it = node_extra_attrs.find(std::string(name));
  if (it == node_extra_attrs.end()) {
    std::tie(it, std::ignore) = node_extra_attrs.emplace(
        std::piecewise_construct, std::forward_as_tuple(name),
        std::forward_as_tuple());
  }
  // coverity issue
  CHECK(it != node_extra_attrs.end()) << "iterator is node_extra_attrs.end() ";
  return it->second;
}

const Node& Pass::level_2_fuse(Graph& graph, const MetaDefProto& meta_def) {
  auto name = meta_def.id();
  auto op_type = std::string("not_used_op");
  auto inputs = std::vector<std::string>{meta_def.inputs().begin(),
                                         meta_def.inputs().end()};
  auto outputs = std::vector<std::string>{meta_def.outputs().begin(),
                                          meta_def.outputs().end()};
  auto constant_initializers =
      std::vector<std::string>{meta_def.constant_initializers().begin(),
                               meta_def.constant_initializers().end()};
  auto nodes = std::vector<size_t>();
  nodes.reserve(meta_def.nodes_size());
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  for (auto& first_node_arg_name : meta_def.nodes()) {
    auto node_arg_opt = graph_ref.find_node_arg(first_node_arg_name);
    CHECK(node_arg_opt.has_value())
        << "cannot find node arg: " << first_node_arg_name;
    auto node_opt = node_arg_opt.value().find_producer();
    CHECK(node_opt.has_value())
        << "cannot find producer node: " << first_node_arg_name;
    nodes.push_back(node_opt.value().index());
  }
  const Node& ret = morphizen::graph_fuse(graph, name, op_type, nodes, inputs,
                                          outputs, constant_initializers);
  morphizen_cxx::GraphRef(graph).resolve();
  return ret;
}

const Node& Pass::fuse(Graph& graph, MetaDefProto&& meta_def) {
  auto context = this->context_;
  auto new_meta_def = context->context_proto.mutable_meta_def()->Add();
  *new_meta_def = std::move(meta_def);
  return level_2_fuse(graph, *new_meta_def);
}
MetaDefProto& Pass::fuse(Graph& graph, const std::string& name,
                         const std::string& op_type,
                         const std::vector<size_t>& nodes,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const std::vector<std::string>& constant_initializers,
                         const std::string& device) {
  auto context = this->context_;
  auto meta_def = context->context_proto.mutable_meta_def()->Add();
  meta_def->set_id(name);
  for (auto& input : inputs) {
    meta_def->add_inputs(input);
  }
  for (auto& output : outputs) {
    meta_def->add_outputs(output);
  }
  for (auto& constant_initializer : constant_initializers) {
    meta_def->add_constant_initializers(constant_initializer);
  }
  auto graph_ref = morphizen_cxx::GraphConstRef(graph);
  for (auto n : nodes) {
    auto node_ref = graph_ref.node(n);
    meta_def->add_nodes(node_get_first_output_name(*node_ref.ptr()));
  }
  meta_def->set_device(device);
  morphizen::graph_fuse(graph, name, op_type, nodes, inputs, outputs,
                        constant_initializers);
  return *meta_def;
}

const std::shared_ptr<PassContext> Pass::get_context() const {
  return context_;
}
std::shared_ptr<PassContext> Pass::get_context() { return context_; }

void Pass::add_context_resource(const std::string& name,
                                std::shared_ptr<void> resource) {
  context_->add_context_resource(name, resource);
}

PassContextTimer::PassContextTimer() {}
PassContextTimer::~PassContextTimer() {}

void Pass::add_action(action_t action) { action_.push_back(action); }

MORPHIZEN_DLL_SPEC std::unique_ptr<IPass>
IPass::create_pass(std::shared_ptr<PassContext> context,
                   const PassProto& pass_proto) {
  auto& plugin = pass_proto.plugin();
  auto plugin_holder = Plugin::get(plugin);
  if (plugin_holder == nullptr) {
    LOG(FATAL) << "cannot find plugin: " << plugin
               << " enable env MORPHIZEN_DEBUG_PLUGIN=1 to see more details";
  }
  auto& pass_info = *plugin_holder->invoke<PassInfo*>("morphizen_pass_info");
  auto context_ptr =
      std::dynamic_pointer_cast<morphizen::PassContextImp>(context);
  CHECK(context_ptr != nullptr);
  return std::make_unique<Pass>(context_ptr, pass_proto, pass_info);
}

MORPHIZEN_DLL_SPEC std::unique_ptr<IPass>
IPass::create_pass(std::shared_ptr<PassContext> context,
                   const struct PassInfo& pass_info) {
  auto context_ptr =
      std::dynamic_pointer_cast<morphizen::PassContextImp>(context);
  CHECK(context_ptr != nullptr);
  // Build PassProto locally without mutating ConfigProto
  PassProto pass_proto;
  pass_proto.set_name("annonymous_pass");
  pass_proto.set_plugin("<annonymous_plugin>");
  return std::make_unique<Pass>(context_ptr, pass_proto, pass_info);
}

std::vector<std::shared_ptr<IPass>>
IPass::create_passes(std::shared_ptr<PassContext> context,
                     const std::vector<PassProto>& passes) {
  auto ret = std::vector<std::shared_ptr<IPass>>();
  ret.reserve(passes.size());
  for (const auto& pass_proto : passes) {
    if (pass_proto.disabled()) {
      continue;
    }
    ret.emplace_back(create_pass(context, pass_proto));
  }
  return ret;
}

MORPHIZEN_DLL_SPEC void
IPass::run_passes(std::vector<std::shared_ptr<IPass>> passes, Graph& graph) {
  Pass::run_all_passes(passes, graph);
}

std::string Pass::seq_num_as_string() const {
  auto index_s = std::to_string(sequence_no_);
  std::string::size_type n_zero = 4u;
  index_s =
      std::string(n_zero - std::min(n_zero, index_s.length()), '0') + index_s;
  return index_s;
}
std::filesystem::path Pass::get_dump_file_name(size_t action_index,
                                               const std::string& ext) const {
  auto index_s = seq_num_as_string();
  // TODO: Dump directory (temp/morphizen_dumps/cache_key) is used ONLY for
  // debugging/troubleshooting output files. It is NOT a cache directory.
  // Cache persistence uses EP context tar-based system (tar_file_).
  // See Issue #006 for historical context about cache_dir removal.
  return context_->get_dump_directory() /
         (std::string("morphizen.") + index_s + "." + name() + //
          ".action_" + std::to_string(action_index) +          //
          ext);
}
} // namespace morphizen
