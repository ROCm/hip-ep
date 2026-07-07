/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include "morphizen/node_attr.hpp"
#include <morphizen/morphizen_ort_api.h>
#include <morphizen/my_ort.h>
#include <type_traits>
#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include "morphizen/pass_context.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#include <filesystem>
#include <memory>
#include <string>

namespace morphizen {
#define PASS_LOG(self_, n)                                                     \
  LOG_IF(INFO, (self_.get_pass_proto().enable_log() &&                         \
                self_.get_pass_proto().log_verbosity() >= n))
using namespace onnxruntime;

/** @brief create a Context object from a cache directory.
 *
 * it is mainly used by some internal tests and tools.
 *
 */

/** @brief For troubleshooting Pass:fuse / Pass::try_fuse error.
 *
 *  @sa Pass::fuse Pass::try_fuse
 */
struct TryFuseError {
  std::string comments;
  std::vector<std::string> path;
  std::vector<const Node *> body;
  std::vector<std::string> arguments;
  std::vector<std::string> return_values;
};

struct MorphizenOnnxError {
  int err_code;
  std::string err_msg;
};

/** @brief the base class for all passes.
 *
 */
class IPass {
public:
  /** @brief create a concrete pass object from `PassProto`
   *
   *  @param context the Context object shared among all passes.
   *  @param pass_proto pass configuration, pass_proto.plugin is the
   *  name of the shared library, e.g "morphizen-pass_merge_fix", see
   *  `morphizen_config.json` for more examples.
   *
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<IPass>
  create_pass(std::shared_ptr<PassContext> context,
              const PassProto &pass_proto);

  /** @brief create concrete passes object from `PassProto`
   *
   *  @param context the Context object shared among all passes.
   *  @param pass_proto
   *
   *  repeatedly invoke Pass::create_pass to create a vector of IPass objects.
   */
  static std::vector<std::shared_ptr<IPass>>
  create_passes(std::shared_ptr<PassContext> context,
                const std::vector<PassProto> &passes);
  /** @brief create a concrete pass object from `PassInfo`
   *
   *  @param context the Context object shared among all passes.
   *  @param pass_info
   *
   * @NOTE only used for internal test purpuse.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<IPass>
  create_pass(std::shared_ptr<PassContext> context,
              const struct PassInfo &pass_info);

  /** @brief apply all passes
   *
   * @param passes
   * @param graph
   *
   * `graph` is modified in place by all these passes in sequence.
   *
   */
  MORPHIZEN_DLL_SPEC static void
  run_passes(std::vector<std::shared_ptr<IPass>> passes, Graph &graph);

  using action_t = std::function<void(IPass &self, Graph &graph)>;
  using node_action_t =
      std::function<bool(IPass &self, Graph &graph, const Node &node)>;

  IPass() = default;
  virtual ~IPass() = default;
  /**
   * @brief Attaches a JSON parameter to a MetaDefProto object.
   *
   * This function takes a MetaDefProto object and a JSON-formatted string,
   * and associates the JSON parameter with the given MetaDefProto object.
   *
   * @param meta_def Reference to the MetaDefProto object to which the parameter
   * will be attached.
   * @param json_param A C-style string containing the JSON-formatted parameter.
   */
  MORPHIZEN_DLL_SPEC void attach_meta_def_param(MetaDefProto &meta_def,
                                                const char *json_param) const;

public:
  virtual const std::string &name() const = 0;
  /** @brief do not use this function. internal use only
   */
  virtual void *get_state() = 0;
  virtual const ConfigProto &get_config_proto() const = 0;
  virtual std::map<std::string, std::string>
  get_all_provider_options() const = 0;

  /** @brief do not use this function. internal use only
   */
  virtual void add_subgraph_device_count(const std::string &device,
                                         int count) = 0;
  /** @brief do not use this function. internal use only
   */
  virtual const PassProto &get_pass_proto() const = 0;

  /**
   * @brief Retrieves the generic parameter for this pass as a JSON string.
   *
   * This function returns the pass_generic_param field from the PassProto
   * as a JSON-formatted string. The pass_generic_param allows arbitrary
   * parameters to be passed to any pass.
   *
   * @return A string representing the generic parameter in JSON format.
   */
  virtual std::string get_pass_generic_param() const = 0;

  /** @brief do not use this function. internal use only
   */
  virtual std::vector<AttributeProtoPtr> &
  node_extra_attrs(const char *name) = 0;
  /** @brief do not use this function. internal use only
   */
  inline void node_add_extra_attr(const char *name, const NodeAttr &attr) {
    node_extra_attrs(name).push_back(attr_proto_clone(attr.get()));
  }

  /** @brief extract a subgraph into an onnx Node.
   *
   * @note graph is modified in place.
   */
  virtual const Node &fuse(Graph &graph, MetaDefProto &&meta_def) = 0;

  /** @brief extract a subgraph into an onnx Node.
   *
   * @note graph is modified in place.
   */
  virtual MetaDefProto &
  fuse(Graph &graph, const std::string &name, const std::string &op_type,
       const std::vector<size_t> &nodes, const std::vector<std::string> &inputs,
       const std::vector<std::string> &outputs,
       const std::vector<std::string> &constant_initializers,
       const std::string &device) = 0;

  /** @brief extract a subgraph into an onnx Node. level 2 fuse
   *
   * @note graph is modified in place. context.json is not updated.
   */

  virtual const Node &level_2_fuse(Graph &graph,
                                   const MetaDefProto &meta_def) = 0;

  /** @brief atempt extract a subgraph into an onnx Node
   *
   * @note no modification is made
   * @todo change `Graph& graph` to `const Graph& graph`
   */
  MORPHIZEN_DLL_SPEC std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
  try_fuse(const Graph &graph, const std::string &name,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs,
           const std::vector<std::string> &constant_initializers,
           const std::string &device) const;

  /** @brief access the shared Context Object. readonly
   *
   */
  virtual const std::shared_ptr<PassContext> get_context() const = 0;

  /** @brief access the shared Context Object. readwrite.
   *
   * @note: try not to invoke this function as much as possible,
   * i.e. not to update the Context object.
   */
  virtual std::shared_ptr<PassContext> get_context() = 0;
  /** @brief do not use this function. internal use only
   */
  virtual void add_context_resource(const std::string &name,
                                    std::shared_ptr<void> resource) = 0;
};
MORPHIZEN_DLL_SPEC std::pair<std::unique_ptr<MetaDefProto>, TryFuseError>
IPass_try_fuse(const Graph &graph, const std::string &name,
               const std::vector<std::string> &inputs,
               const std::vector<std::string> &outputs,
               const std::vector<std::string> &constant_initializers1,
               const std::string &device);

struct PassInfo {
  typedef union {
    void *__p; // ease of union initialization;
    bool (*process_node)(IPass &self, Graph &graph, const Node &node);
    void (*process_graph)(IPass &self, Graph &graph);
  } process_t;
  typedef struct {
    int type;
    process_t proc;
  } process_def;
  void *(*init)(IPass &self);
  void (*deinit)(void *);
  typedef void (*preprocess_t)(void *state, IPass &self, Graph &graph);
  preprocess_t preprocess;
  typedef void (*postprocess_t)(void *state, IPass &self, Graph &graph);
  postprocess_t postprocess;
  size_t size;
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4200)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  process_def processes[];
#ifdef _WIN32
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

  IPass::action_t get_action(size_t index) const;
};
template <typename T, class = void> struct has_preprocess_t {
  static constexpr PassInfo::preprocess_t preprocess = nullptr;
};

template <typename T>
struct has_preprocess_t<T,
                        std::void_t<decltype(std::declval<T *>()->preprocess(
                            std::declval<IPass &>(), std::declval<Graph &>()))>>
    : public std::true_type {
  static constexpr PassInfo::preprocess_t preprocess =
      [](void *self, IPass &pass, Graph &graph) {
        static_cast<T *>(self)->preprocess(pass, graph);
      };
};

template <typename T, class = void> struct has_postprocess_t {
  static constexpr PassInfo::postprocess_t postprocess = nullptr;
};

template <typename T>
struct has_postprocess_t<
    T, std::void_t<decltype(std::declval<T *>()->postprocess(
           std::declval<IPass &>(), std::declval<Graph &>()))>>
    : public std::true_type {
  static constexpr PassInfo::postprocess_t postprocess =
      [](void *self, IPass &pass, Graph &graph) {
        static_cast<T *>(self)->postprocess(pass, graph);
      };
};
template <typename Pass> struct CommonInitAndDeinit {
  static void *init(IPass &self) {
    auto state = new Pass(self);
    return (void *)state;
  }
  static void deinit(void *state) { delete static_cast<Pass *>(state); }
};

template <typename T, class = void> struct has_process_t {
  // not defined
  // static constexpr int type = -1;
  // static constexpr void* process = nullptr;
};

template <typename T>
struct has_process_t<T, std::void_t<decltype(std::declval<T *>()->process(
                            std::declval<IPass &>(), std::declval<Graph &>()))>>
    : public std::true_type {
  static constexpr int type = 0;
  static void process(IPass &self, Graph &graph) {
    static_cast<T *>(self.get_state())->process(self, graph);
  }
};

template <typename T>
struct has_process_t<
    T, std::enable_if_t<std::is_same_v<
           bool, decltype(std::declval<T *>()->process(
                     std::declval<IPass &>(), std::declval<Graph &>(),
                     std::declval<const Node &>()))>>> : public std::true_type {
  static constexpr int type = 1;
  static bool process(IPass &self, Graph &graph, const Node &node) {
    return static_cast<T *>(self.get_state())->process(self, graph, node);
  }
};

template <typename Pass> struct ProcessorPassInfo {
  static morphizen::PassInfo *pass_info() { return &info; }
  static PassInfo info;
};

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
template <typename Pass>
PassInfo ProcessorPassInfo<Pass>::info = {
    CommonInitAndDeinit<Pass>::init,
    CommonInitAndDeinit<Pass>::deinit,
    has_preprocess_t<Pass>::preprocess,
    has_postprocess_t<Pass>::postprocess,
    1,
    {
        {has_process_t<Pass>::type, {(void *)has_process_t<Pass>::process}},
    }};
#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

namespace fs = std::filesystem;
using namespace onnxruntime;

IPass::action_t
create_action_from_node_action(IPass::node_action_t node_action);
IPass::action_t create_xmodel_process_graph(IPass::action_t action);
} // namespace morphizen

#define DEFINE_MORPHIZEN_PASS(cls, id)                                         \
  static ::morphizen::PassInfo *morphizen_pass_info() {                        \
    return ProcessorPassInfo<cls>::pass_info();                                \
  }                                                                            \
  namespace {                                                                  \
  static ::morphizen::StaticPluginRegister                                     \
      __register(OUTPUT_NAME, "morphizen_pass_info",                           \
                 (void *)&morphizen_pass_info);                                \
  }
