/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/anchor_point.hpp"

namespace morphizen_imp {
using namespace morphizen;
class AnchorPointImp : public AnchorPoint {
public:
  AnchorPointImp(const NodeArg &node_arg, const Description &desciption);
  virtual ~AnchorPointImp();
  AnchorPointImp(const AnchorPointProto &proto);

private:
  virtual const AnchorPointProto &get_proto() const override final;

public:
  AnchorPointImp();

private:
  AnchorPointProto merge_proto(const AnchorPointImp *other) const;

private:
  const AnchorPointProto proto_;
};
} // namespace morphizen_imp
