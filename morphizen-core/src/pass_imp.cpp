/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// glog must be included very beginning.
#include <deque>
#include <fstream>
#include <glog/logging.h>
///

#include "./cache_dir.hpp"
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
DEF_ENV_PARAM(ENABLE_SAVE_ONNX_MODEL, "0")
DEF_ENV_PARAM(DEBUG_VAIP_PASS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_PASS) >= n)

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
  bool can_be_dumped = proto->get_config_proto().encryption_key() == "";
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
      auto leaf_nodes = graph_get_output_nodes(graph);
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
        auto leaf_nodes = graph_get_output_nodes(graph);
        morphizen::graph_reverse_dfs_from_multi(
            graph,   //
            gsl::make_span(leaf_nodes),
            nullptr, //
            [&](const Node* node) {
              auto node_ref =
                  morphizen_cxx::NodeConstRef::from_node(graph, *node);
              auto node_idx = node_ref.index();
              modified = node_action(self, graph, *node);
              if (modified) {
                match_idx = (int)node_idx;
              }
              if (modified) {
                throw BreakOnModifed{1};
              }
            }, //
            [&modified](const Node* /*from*/, const Node* /*to*/) {
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
  LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_PASS))
      << "pass is created: " << (void*)this << " name=" << this->name();
}
Pass::~Pass() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_PASS))
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
    graph_resolve(*graph);
    maybe_dump_txt(action_index + 100, *graph);
    maybe_gc(*graph);
    graph_resolve(*graph);
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
    graph_gc(graph);
  }
}

void* Pass::get_state() { return state_.get(); }

std::filesystem::path
Pass::get_cache_file_name(const std::string& filename) const {
  return morphizen::get_cache_file_name(*context_, filename);
}

const ConfigProto& Pass::get_config_proto() const {
  return context_->context_proto.config();
}
std::map<std::string, std::string> Pass::get_all_provider_options() const {
  return context_->get_all_provider_options();
}

const std::filesystem::path& Pass::get_log_path() const {
  return context_->get_log_dir();
}

void Pass::add_subgraph_device_count(const std::string& device, int count) {
  context_->context_proto.mutable_device_subgraph_count()->insert(
      google::protobuf::MapPair<std::string, int>{device, count});
}

void Pass::set_fix_info(const char* name, int fix_pos) {
  context_->context_proto.mutable_fix_info()->insert(
      google::protobuf::MapPair<std::string, int>{std::string(name), fix_pos});
}

int Pass::get_fix_info(const char* name) const {
  auto it = context_->context_proto.fix_info().find(name);
  CHECK(it != context_->context_proto.fix_info().end())
      << "cannot find fix info";
  return it->second;
}

bool Pass::has_fix_info(const char* name) const {
  auto it = context_->context_proto.fix_info().find(name);
  return it != context_->context_proto.fix_info().end();
}

void Pass::dump_fix_info(const char* filename) const {
  auto fullname = get_log_path() / std::string(filename);
  auto stream = std::ofstream(fullname, std::ios_base::trunc);
  LOG(INFO) << "save fix info to " << fullname.u8string();
  for (auto& i : context_->context_proto.fix_info()) {
    stream << i.second << "\t" << i.first << "\n";
  }
}

void Pass::create_const(const char* name, gsl::span<const char> data,
                        const std::vector<int64_t>& shape, int type) {
  size_t offset = 0u;
  if (data.empty()) {
    // see model 1, it is strange that a Resize(x, roi "1985" , ...)
    // where roi has zero data size.
    offset = 0u;
  } else {
    offset = context_->const_data_.size();
  }
  context_->const_data_.insert(context_->const_data_.end(), data.begin(),
                               data.end());
  auto const_data = ConstDataInfo();
  const_data.set_offset(offset);
  const_data.set_size(data.size());
  const_data.mutable_shape()->Assign(shape.begin(), shape.end());
  const_data.set_type(type);
  context_->context_proto.mutable_const_data_info()->insert(
      google::protobuf::MapPair<std::string, ConstDataInfo>{std::string(name),
                                                            const_data});
}

void Pass::create_empty_const(const char* name, size_t size,
                              const std::vector<int64_t>& shape, int type) {
  CHECK_NE(size, 0u);
  auto const_data = ConstDataInfo();
  const_data.set_offset(context_->const_data_.size());
  const_data.set_size(size);
  const_data.mutable_shape()->Assign(shape.begin(), shape.end());
  const_data.set_type(type);
  context_->const_data_.resize(context_->const_data_.size() + size);
  context_->context_proto.mutable_const_data_info()->insert(
      google::protobuf::MapPair<std::string, ConstDataInfo>{std::string(name),
                                                            const_data});
}

void Pass::create_lazy_const(const char* name, size_t size,
                             const std::vector<int64_t>& shape, int type,
                             const std::function<void(gsl::span<char>)>& lazy) {
  CHECK_NE(size, 0u);
  create_empty_const(name, size, shape, type);
  context_->const_lazy_[name] =
      std::make_shared<std::function<void(gsl::span<char>)>>(lazy);
}

void Pass::create_const_alias(const char* alias_name, const char* name) {
  auto it = context_->context_proto.const_data_info().find(name);
  CHECK(it != context_->context_proto.const_data_info().end())
      << "cannot find const info " << name;
  context_->context_proto.mutable_const_data_info()->insert(
      google::protobuf::MapPair<std::string, ConstDataInfo>{
          std::string(alias_name), it->second});
  auto it_lazy = context_->const_lazy_.find(name);
  if (it_lazy != context_->const_lazy_.end()) {
    context_->const_lazy_[alias_name] = it_lazy->second;
  }
}

bool Pass::has_const(const char* name) const {
  auto it = context_->context_proto.const_data_info().find(name);
  return it != context_->context_proto.const_data_info().end();
}

ConstDataInfo Pass::get_const_info(const char* name) const {
  auto it = context_->context_proto.const_data_info().find(name);
  CHECK(it != context_->context_proto.const_data_info().end())
      << "cannot find const info " << name;
  return it->second;
}

void* Pass::get_const_data_ptr(const char* name, bool force) const {
  auto data_info = get_const_info(name);
  auto ret = &context_->const_data_[data_info.offset()];
  if (force) {
    auto lazy_it = context_->const_lazy_.find(name);
    if (lazy_it != context_->const_lazy_.end()) {
      if (*lazy_it->second) {
        (*lazy_it->second)(gsl::span<char>(ret, data_info.size()));
        (*lazy_it->second) = nullptr;
      }
    }
  }
  return ret;
}

void Pass::dump_const_info(const char* filename) const {
  auto fullname = get_log_path() / std::string(filename);
  auto stream = std::ofstream(fullname, std::ios_base::trunc);
  LOG(INFO) << "save const info to " << fullname.u8string();
  auto is_lazy = [this](const std::string& name) {
    auto it = context_->const_lazy_.find(name);
    auto ret = std::string();
    if (it == context_->const_lazy_.end()) {
      ret = "eager";
    } else {
      if (*it->second) {
        ret = "lazy";
      } else {
        ret = "evaluated";
      }
    }
    return ret;
  };
  for (auto& i : context_->context_proto.const_data_info()) {
    stream << i.first << "\t" << i.second.offset() << "\t" << i.second.size()
           << "\t" << i.second.type() << "\t" << is_lazy(i.first) << "\n";
  }
}
void Pass::dump_const_data(const char* name) const {
  auto fullname = get_log_path() / std::string(name);
  if (context_->const_data_.empty()) {
    LOG(INFO) << "no constant data. cancel saving const info to "
              << fullname.u8string();
  } else {
    auto stream =
        std::ofstream(fullname, std::ios_base::trunc | std::ios_base::binary);
    LOG(INFO) << "save const info to " << fullname.u8string();
    CHECK(stream.write(&context_->const_data_[0], context_->const_data_.size())
              .good())
        << " write failure";
  }
  return;
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
  graph_resolve(graph);
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
  auto pass_proto = context_ptr->context_proto.mutable_config()->add_passes();
  pass_proto->set_name("annonymous_pass");
  pass_proto->set_plugin("<annonymous_plugin>");
  return std::make_unique<Pass>(context_ptr, *pass_proto, pass_info);
}

std::vector<std::shared_ptr<IPass>> IPass::create_passes(
    std::shared_ptr<PassContext> context,
    const google::protobuf::RepeatedPtrField<PassProto>& passes) {
  auto ret = std::vector<std::shared_ptr<IPass>>();
  ret.reserve(passes.size());
  for (auto& pass_proto : passes) {
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

static void load_protobuf_message(const fs::path& filename,
                                  google::protobuf::Message& msg) {
  auto json_str = slurp(filename);
  if (json_str.empty()) {
    return;
  }
  auto status = google::protobuf::util::JsonStringToMessage(json_str, &msg);
  CHECK(status.ok()) << "cannot parse json string: filename=" << filename
                     << " json=\"" << json_str << "\"";
}

static void load_context_json(PassContextImp& context) {
  context.context_proto.Clear();
  load_protobuf_message(get_cache_file_name(context, "context.json"),
                        context.context_proto);
}
static void load_context_const_bin(PassContextImp& context) {
  auto const_data_file = get_cache_file_name(context, "const.bin");
  std::ifstream const_data_stream(const_data_file, std::ios::binary);
  if (!const_data_stream.good()) {
    return;
  }
  const_data_stream.seekg(0, std::ios_base::end);
  CHECK(const_data_stream.good()) << "cannot seek " << const_data_file;
  auto size = const_data_stream.tellg();
  const_data_stream.seekg(0, std::ios_base::beg);
  CHECK(const_data_stream.good()) << "cannot rewind " << const_data_file;
  context.const_data_.resize(size);
  CHECK(const_data_stream.read(&context.const_data_[0], size).good())
      << "read fail " << const_data_file << " size=" << size;
}

MORPHIZEN_DLL_SPEC std::shared_ptr<PassContext>
load_context(const std::filesystem::path& cache_dir) {
  auto context = std::make_shared<morphizen::PassContextImp>();
  context->pass_context_log_dir_ = cache_dir;
  load_context_json(*context);
  load_context_const_bin(*context);
  return context;
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
  return context_->get_log_dir() /
         (std::string("vaip.") + index_s + "." + name() + //
          ".action_" + std::to_string(action_index) +     //
          ext);
}
} // namespace morphizen
