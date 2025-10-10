/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
namespace mlir_impl {

class MLIRGraph;

// MLIR-based Model implementation
class MLIRModel {
public:
  // PrivateTag pattern to allow factory methods while keeping constructor
  // private
  struct PrivateTag {};

  // Constructor that requires PrivateTag - only accessible to factory methods
  explicit MLIRModel(PrivateTag, mlir::OwningOpRef<mlir::ModuleOp> module);

  // Static factory method to create MLIRModel instances
  static std::unique_ptr<MLIRModel>
  create(mlir::OwningOpRef<mlir::ModuleOp> module);

  // Static factory method to create an empty MLIRModel with optional path and
  // opset
  static std::unique_ptr<MLIRModel>
  create_empty(const std::filesystem::path& path = {},
               const std::vector<std::pair<std::string, int64_t>>& opset = {});

  static std::unique_ptr<MLIRModel> load(const std::string& filename);

  std::unique_ptr<MLIRModel> clone(int64_t external_data_threshold) const;

  MLIRGraph& main_graph() const;

  void set_metadata_prop(const std::string& key, const std::string& value);

  std::string get_metadata_prop(const std::string& key) const;

  bool has_metadata_prop(const std::string& key) const;

  mlir::ModuleOp getModule() const;

  std::string serialize_as_string() const;

private:
  // Private static helper function to create main graph
  static std::unique_ptr<MLIRGraph>
  create_main_graph(MLIRModel& model, mlir::OpBuilder& builder,
                    mlir::ModuleOp& module, const std::string& graph_name);

  // Private member function to dump MLIR for debugging
  void maybe_dump_mlir(const std::filesystem::path& path) const;

  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mutable std::map<std::string, std::string> metadata_;
  mutable std::unique_ptr<MLIRGraph> main_graph_;
};

} // namespace mlir_impl
} // namespace morphizen
