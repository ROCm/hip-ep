/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "./mlir-node-arg-index.hpp"
#include "./mlir-node-arg.hpp"
#include "./mlir-node.hpp"
#include "./symbol-table.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "llvm/ADT/SmallVector.h"
#include <filesystem>
#include <functional>
#include <gsl/span>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace morphizen {
namespace mlir_impl {

// Forward declaration and using declaration
using ::mlir_impl::MLIRSymbolTable;

class MLIRModel;
class MLIRNodeAttributes;

// MLIR-based Graph implementation
class MLIRGraph {
public:
  explicit MLIRGraph(MLIRModel& model, mlir::func::FuncOp func,
                     uint32_t proposed_graph_id = 0);
  ~MLIRGraph();

  MLIRGraph* parent_graph() const;

  // mlir-imp backend of graph_new_subgraph. The orphan block is cleaned
  // up by ~MLIRGraph if add_node never consumes the matching attribute.
  MLIRGraph& new_subgraph();

  // Delete copy operations since MLIRSymbolTable is non-copyable
  MLIRGraph(const MLIRGraph&) = delete;
  MLIRGraph& operator=(const MLIRGraph&) = delete;

  // Enable move operations
  MLIRGraph(MLIRGraph&&) = default;
  MLIRGraph& operator=(MLIRGraph&&) = delete;

  const std::string& get_name() const;

  void set_name(const char* name);

  const MLIRModel& get_model() const;

  /**
   * @brief Get the symbol name of the function
   * @return The symbol name of the underlying MLIR function
   */
  std::string get_symbol_name() const;

  std::vector<mlir::Operation*> nodes_unsafe() const;

  llvm::SmallVector<MLIRNodeArgIndex> get_inputs() const;

  llvm::SmallVector<MLIRNodeArgIndex> get_outputs() const;

  void set_outputs(const llvm::SmallVector<MLIRNodeArgIndex>& outputs);

  void set_inputs(const llvm::SmallVector<MLIRNodeArgIndex>& inputs);

  const mlir::Operation* get_node(size_t index) const;

  mlir::Operation* producer_node(const std::string& node_arg_name) const;

  MLIRNodeArgIndex get_node_arg_index(const std::string& name) const;

  MLIRNodeArgIndex node_arg_new(const std::string& name,
                                const llvm::SmallVector<int64_t>* shape,
                                int element_type);

  const mlir::Operation*
  add_node(const std::string& name, const std::string& op_type,
           const std::string& description,
           const std::vector<MLIRNodeArgIndex>& input_args,
           const std::vector<MLIRNodeArgIndex>& output_args,
           const class MLIRNodeAttributes& attributes,
           const std::string& domain);

  void add_constant_initialized_tensor(const mlir_impl::MLIRNodeArg* tensor);

  const std::unordered_map<std::string, const void*>&
  get_all_initialized_tensors() const;

  void save(const std::string& filename, const std::string& dat_filename,
            size_t external_data_threshold) const;

  std::string save_string() const;

  // Graph resolution - ensures graph is in consistent state
  int resolve(bool force);

  const MLIRNodeArg* get_node_arg(MLIRNodeArgIndex node_arg_index) const;

  void remove_node(mlir::Operation* op);

  void remove_initialized_tensor(const std::string& name);

  mlir::Operation*
  fuse(const std::string& name, const std::string& op_type,
       const std::vector<const mlir::Operation*>& nodes,
       const std::vector<MLIRNodeArgIndex>& inputs,
       const std::vector<MLIRNodeArgIndex>& outputs,
       const std::vector<MLIRNodeArgIndex>& constant_initializers);

  std::vector<const mlir::Operation*>
  get_consumer_nodes(const std::string& node_arg_name) const;
  const MLIRGraph* add_subgraph(std::unique_ptr<MLIRGraph> graph);

  // Graph traversal functions
  void reverse_dfs_from_preemp(
      gsl::span<const mlir::Operation* const> from,
      const std::function<bool(const mlir::Operation*)>& enter,
      const std::function<bool(const mlir::Operation*)>& leave,
      const std::function<bool(const mlir::Operation*, const mlir::Operation*)>&
          comp,
      const std::function<bool(const mlir::Operation* /*from*/,
                               const mlir::Operation* /*to*/)>& stop) const;

private:
  explicit MLIRGraph(MLIRModel& model, mlir::Block& entry_block,
                     uint32_t proposed_graph_id = 0);

  void set_parent_graph(MLIRGraph* p);
  mlir::Block* take_orphan_block();
  void register_captured_alias(const std::string& name,
                               mlir::Value outer_value);

  void initialize();
  void initialize_graph_inputs();
  void initialize_graph_outputs();
  void initialize_constant_initializers();
  void initialize_node_args_map();
  void populate_node_arg_indexes();

  // helper functions
  std::string extract_value_name(const mlir::Value value);
  void canonicalize_optional_outputs();

  // Fusion helper methods
  std::pair<mlir::func::FuncOp, std::stack<mlir::Operation*>>
  create_func_func(const std::vector<MLIRNodeArgIndex>& inputs,
                   const std::vector<MLIRNodeArgIndex>& outputs,
                   const std::vector<const mlir::Operation*>& nodes,
                   const std::vector<MLIRNodeArgIndex>& constant_initializers);

  mlir::Operation*
  create_func_call(const std::string& name,
                   const std::vector<MLIRNodeArgIndex>& inputs,
                   const std::vector<MLIRNodeArgIndex>& outputs,
                   mlir::func::FuncOp fused_func,
                   const std::stack<mlir::Operation*>& cloned_ops_cache);
  void remove_func_ops(std::stack<mlir::Operation*>& cloned_ops_cache);

  mlir::func::FuncOp func() const;
  bool is_subgraph() const;

  MLIRModel& model_;
  // entry_block_ is what an MLIRGraph IS. Always valid.
  // Top-level:  &func.getBody().front() where the parent op is func::FuncOp.
  // Subgraph:   &parent_op.getRegion(i).front() once transplanted, or an
  //             orphan Block (getParent() == nullptr) during the
  //             lazy-valid IR window between new_subgraph and add_node.
  mlir::Block* entry_block_;

  uint32_t graph_id_;
  mlir::Operation* terminator_;
  mlir::Operation* none_;
  MLIRSymbolTable value_map_;

  // Sub MLIRGraphs owned by this graph. Entries come from:
  //   - new_subgraph()     — ONNX If/Loop/Scan body regions
  //   - getFunctionBody()  — func.call lookups cached for fused custom ops
  // unique_ptr keeps each MLIRGraph's address stable across vector growth,
  // so raw pointers encoded by attr_proto_new_graph remain valid.
  std::vector<std::unique_ptr<MLIRGraph>> subgraphs_cache_;

  // Map from node argument names to their indices
  // This is used to quickly look up node arguments by name
  // In ONNX, NodeArg names are crucial for indexing, DFS traversal, and model
  // storage. When using the ONNX DFS graph, node_arg names are heavily used. so
  // we need to maintain a mapping from names to their indices.
  std::unordered_map<std::string, MLIRNodeArgIndex> node_args_map_;
  // mlir::Value cannot be directly created by users.
  // It is a lightweight handle that refers to operation results or block
  // arguments. A Value is obtained through creating operations (e.g.,
  // Operation::getResult(i)) or block arguments, rather than being constructed
  // manually.
  std::vector<std::unique_ptr<MLIRNodeArg>> all_node_args_;

  llvm::SmallVector<MLIRNodeArgIndex> graph_inputs_;
  llvm::SmallVector<MLIRNodeArgIndex> graph_outputs_;
  // Output names captured from the onnx.Return terminator during the first
  // initialize_graph_outputs() call. Used as a stable reference filter in
  // set_outputs to reject spurious intermediate outputs from the ORT bridge.
  // Never repopulated on subsequent resolve() calls so that user-driven
  // set_outputs changes are not blocked.
  std::unordered_set<std::string> model_output_names_;
  bool model_output_names_frozen_ = false;
  std::vector<MLIRNodeArgIndex> constant_initializers_;
  // it is only used for cache, to return a reference
  std::unordered_map<std::string, const void*> initialized_tensors_cache_;

  // clang-format off
  // Acceleration cache for DFS traversal operations
  // - Provides O(1) lookup to quickly skip nodes during depth-first search
  // - Performance improvement for fuse_transpose: 1501.67ms -> 76.046ms (~20x speedup)
  // - Newly added nodes are excluded from DFS traversal to prevent infinite loops
  // - Edge case protection: certain graph modifications can create cycles during traversal
  // - Best practice: Use NodeBuilder to avoid these traversal issues and other problems
  // clang-format on
  std::unordered_set<const mlir::Operation*> staging_nodes_;

  MLIRGraph* parent_graph_ = nullptr;
};

} // namespace mlir_impl
} // namespace morphizen
