/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#ifdef _WIN32
#  pragma warning(push, 0)
#endif
#include "morphizen/anchor_point.pb.h"
#ifdef _WIN32
#  pragma warning(pop)
#endif
#include "morphizen/pass.hpp"
#include <memory>
namespace morphizen {
class AnchorPoint {
public:
  struct Description {
    Description(const std::string& op);
    Description(const AnchorPointProto& anchor_point);
    Description(const std::string& op, const AnchorPointTransposeOpAttr& order);
    Description(const std::string& op, const AnchorPointFixAttr& attr);
    Description(const std::string& op, const AnchorPointPadOpAttr& attr);

  public:
    static Description create_by_json(const std::string& anchor_point_json);

  public:
    AnchorPointProto proto_;
  };
  static constexpr char IDENTITY_OP[] = "identity";

public:
  /** @brief create an identity anchor point.
   *
   * create an identity anchor point, node_arg might be on an
   * intermediate graph, so the idendity anchor point does not mean it
   * is idenitical to the origin node on the ogirinal graph.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  identity(const IPass& pass, const NodeArg& node_arg);
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  identity(const IPass& pass, const std::string& node_arg);
  /** @brief create an alias anchor point.
   *
   * @experimental this function is not encourage to be used, because the
   * caller must make sure the new_name is unique.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  alias1(const IPass& pass, const Graph& graph, const std::string& origin_name,
         const std::string& new_name);
  /** @brief create an identity anchor point via description.
   *
   * create an anchor point from a node arg. node_arg might be on an
   * intermediate graph.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  create(const IPass& pass, const NodeArg& node_arg,
         const Description& desciption);
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  create(const IPass& pass, const std::string& node_arg,
         const Description& desciption);

  /** @brief create new anchor point on top of another.
   *
   * @experimental this function is not encourage to be used, because the
   * caller must make sure the new_name is unique.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  create(const IPass& pass, const AnchorPointProto& proto,
         const std::string& name, const Description& desciption);

  /** @brief create a new anchor point from a AnchorPointProto */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  create(const AnchorPointProto& proto);
  // clang-format off
  /** @brief create a new anchor point from a SISO path
   *
   * a path is a vector of nodes, start from output to input,
   * e.g. [345,342,339]
   *
   * so the actual graph, there is
   *
   * [167]@167 [339:(ty=1,shape=[1,256,72,120])] Relu [338:(ty=1,shape=[1,256,72,120])]
   * [168]@168 [342:(ty=3,shape=[1,256,72,120])] QuantizeLinear [339:(ty=1,shape=[1,256,72,120]),ortshared_7_1_0:(ty=1,shape=[]),341:(ty=3,shape=[])]
   * [169]@169 [345:(ty=1,shape=[1,256,72,120])] DequantizeLinear [342:(ty=3,shape=[1,256,72,120]),ortshared_7_1_0:(ty=1,shape=[]),344:(ty=3,shape=[])]
   *
   * // clang-format on
   *
   * @return an anchor point whose original node the last node,
   * e.g. 339, we don't ignore the op_type of the producer of 339.
   *
   * @note it is tricky that every op must be reversible, e.g. Relu is not, and DequantizeLinear is.
   *
   * */
  // clang-format on
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  create_from_siso_path(const IPass& pass, const Graph& graph,
                        const std::vector<const Node*>& path);
  // for debugging purpose and testing.
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPointProto> create_proto();

private: // TODO: expose this API?
  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  append(const std::string& node_arg, const std::string& origin_node_name,
         const Description& desciption);

public:
  /** @brief find an anchor point in the current context.
   *
   * @return nullptr if not found
   */

  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  find_anchor_point(const IPass& pass, const std::string& name);
  /** @brief find an anchor point in the current context.
   *
   * find an anchor point in the current context, if not found, create
   * a new identiy anchor point. It used by mapping from xir.xmodel to
   * model, because some origin node might not be found but it is just
   * the very original node on the origin graph, it is not changed.
   *
   * @return nullptr if not found in origin nodes, and cannot be found
   * on the original graph.
   */

  MORPHIZEN_DLL_SPEC static std::unique_ptr<AnchorPoint>
  find_anchor_point(IPass& pass, const Graph& graph, const std::string& name);

public:
  MORPHIZEN_DLL_SPEC AnchorPoint();
  virtual ~AnchorPoint();

public:
  /** @breif return the original node name on the very first original model.
   */
  virtual std::string origin_node_arg_name() const;
  /** @brief return an op_debug_string represented.
   */
  virtual std::string op_debug_string() const;

  /** @brief  is it an identity node or not, all anchor points must be
   * identity.
   */
  virtual bool is_identity(bool test_all) const;

  virtual const AnchorPointProto& get_proto() const = 0;
  // find the op in the chain;
  // to avoid DLL issue, make it virtual and final.
  virtual const AnchorPointProto* find_op(const std::string& op) const;

  /** @brief find an anchor point in the current context.
   *
   * insert the anchor point into the current context, later on, it can be
   * found by `::identity` or `::create`
   *
   */
  virtual void insert_into_context(IPass& pass) const;

  /** @brief append a new element in the anchor point list.
   *
   *  node0(`origin0`) --- Q ---> node1(`origin1`) on `onnx.onnx`
   *
   *  this is the anchor point, returned by `find_anchor_point(node_arg)`
   *
   *  node1(`origin1`) -> .... siso and passes... -> nodex(`node_arg`)
   *
   *  this function return a new anchor point, looks likes below
   *
   *  node0(`origin0`) -> Q -> .... siso and passe... -> nodex(`node_arg`)
   *
   *
   *  @param node_arg is the name on an intermediate graph, i.e. the
   *  input graph of current pass.
   *
   *  @param origin_node_name is the name on the original graph, i.e.
   * `onnx.onnx`.
   *
   *  */

  // NOTE: We need MORPHIZEN_DLL_SPEC because there virtual functions are no
  // longer virtual functions in the DLL, due to the `final` keyword.
  MORPHIZEN_DLL_SPEC std::unique_ptr<AnchorPoint>
  append(const IPass& pass, const std::string& origin_node_name,
         const Description& desciption) const;

  MORPHIZEN_DLL_SPEC std::unique_ptr<AnchorPoint>
  append(const IPass& pass, const AnchorPoint& rest) const;

  /** @brief anchor point optimization
   */
  MORPHIZEN_DLL_SPEC std::unique_ptr<AnchorPoint>
  optimize(const IPass& pass) const;

  /** @brief iterate over anchor point
   */
  MORPHIZEN_DLL_SPEC void
  for_each(const std::function<void(const AnchorPointProto&)>& func) const;
};

} // namespace morphizen
