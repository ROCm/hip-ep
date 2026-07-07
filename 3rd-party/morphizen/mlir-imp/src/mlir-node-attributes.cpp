/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-node-attributes.hpp"
#include "mlir-constants.hpp" // For attribute name constants
#include "mlir-context-manager.hpp"
#include "mlir-named-attribute.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <glog/logging.h>
#include <sstream>

namespace morphizen {
namespace mlir_impl {

mlir::Operation* MLIRNodeAttributes::Create() {
  // Get the global MLIR context
  auto& context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // Create a placeholder operation with no operands, no results
  mlir::OperationState state(loc, attr_names::MORPHIZEN_PLACEHOLDER);
  return mlir::Operation::create(state);
}

// Scalar attribute getters
bool MLIRNodeAttributes::has_attribute(const std::string& name) const {
  return (*this)->hasAttr(name);
}

std::vector<std::string> MLIRNodeAttributes::get_attribute_names() const {
  std::vector<std::string> names;

  auto dict_attr = (*this)->getAttrDictionary();

  for (auto named_attr : dict_attr) {
    llvm::StringRef attr_name = named_attr.getName().strref();

    // Skip internal morphizen / onnx-mlir attributes that are not user-
    // visible. Everything under the `morphizen.` namespace is internal;
    // the remaining named entries are ONNX bookkeeping plus onnx.Custom's
    // function_name / domain_name stash.
    if (attr_name.starts_with("morphizen.") ||
        attr_name == attr_names::NODE_OUTPUTS ||
        attr_name == attr_names::ONNX_NAME ||
        attr_name == attr_names::ONNX_NODE_NAME ||
        attr_name == attr_names::ONNX_GRAPH_NAME ||
        attr_name == attr_names::CUSTOM_OP_FUNCTION_NAME ||
        attr_name == attr_names::CUSTOM_OP_DOMAIN_NAME) {
      continue;
    }

    names.push_back(attr_name.str());
  }

  return names;
}

const MLIRNamedAttribute&
MLIRNodeAttributes::get_mlir_attribute(const std::string& name) const {
  auto attrs = (*this)->getAttrs();
  for (auto it = attrs.begin(); it != attrs.end(); it++) {
    if (it->getName().str() == name) {
      return static_cast<const MLIRNamedAttribute&>(*it);
    }
  }
  CHECK(0) << "Should check whether it contains name: " << name;
  return static_cast<const MLIRNamedAttribute&>(*attrs.begin());
}

mlir::DictionaryAttr MLIRNodeAttributes::get_mlir_dictionary() const {
  if (*this) {
    return (*this)->getAttrDictionary();
  }
  return mlir::DictionaryAttr{};
}

void MLIRNodeAttributes::add(const mlir::NamedAttribute& named_attr) {
  (*this)->setAttr(named_attr.getName(), named_attr.getValue());
}

std::vector<std::string>
MLIRNodeAttributes::get_attribute_as_strings(const std::string& name) const {
  auto& named_attr = (const MLIRNamedAttribute&)get_mlir_attribute(name);
  return named_attr.get_strings();
}

} // namespace mlir_impl
} // namespace morphizen
