/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "matmul_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

namespace {

/**
 * Level-2 Pass: MatMul Pattern Matching (hipBLASLt)
 *
 * Matches MatMul operations and replaces them with ROCm custom ops.
 * Unlike Gemm, MatMul:
 * - Has exactly 2 inputs (A, B), no optional bias
 * - Has no alpha/beta scaling parameters
 * - Supports multi-dimensional batch matrix multiplication
 *
 * For GQA (Grouped Query Attention), MatMul is used for:
 * - Query-Key attention: Q @ K^T
 * - Attention-Value: softmax(QK) @ V
 */
struct Level2RocmMatmul {
  static constexpr const char* LOG_PREFIX = "[ROCm MatMul L2]";

  Level2RocmMatmul(IPass& self) : self_{self} {}

  // Calculate batch size from shape (product of all dims except last 2)
  int64_t calc_batch_size(const std::vector<int64_t>& shape) {
    if (shape.size() <= 2)
      return 1;
    int64_t batch = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
      batch *= shape[i];
    }
    return batch;
  }

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char*)matmul_json));

    return Rule::create_rule(
        pattern, [=](Graph* graph, binder_t& binder) -> bool {
          auto input_A = binder["input_A"];
          auto input_B = binder["input_B"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found MatMul pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("matmul");

          auto* matmul_params = rocm_param.mutable_matmul_params();
          matmul_params->set_algorithm_index(-1);  // Auto-select

          // Get input tensor shapes
          auto a_shape_opt = node_arg_get_shape_i64(*input_A.node_arg);
          auto b_shape_opt = node_arg_get_shape_i64(*input_B.node_arg);

          std::vector<int64_t> a_shape, b_shape;
          if (a_shape_opt)
            a_shape = *a_shape_opt;
          if (b_shape_opt)
            b_shape = *b_shape_opt;

          // MatMul requires at least 2D tensors
          if (a_shape.size() < 2 || b_shape.size() < 2) {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Skipping: requires at least 2D tensors";
            return false;
          }

          // Extract dimensions
          // A: [..., M, K], B: [..., K, N] -> Y: [..., M, N]
          int64_t m = a_shape[a_shape.size() - 2];
          int64_t k = a_shape[a_shape.size() - 1];
          int64_t n = b_shape[b_shape.size() - 1];
          int64_t k_b = b_shape[b_shape.size() - 2];

          // Verify K dimensions match
          if (k != k_b) {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Skipping: K dimension mismatch (A.K=" << k
                        << ", B.K=" << k_b << ")";
            return false;
          }

          // Calculate batch size
          int64_t batch_a = calc_batch_size(a_shape);
          int64_t batch_b = calc_batch_size(b_shape);
          int64_t batch_count = std::max(batch_a, batch_b);

          matmul_params->set_m(m);
          matmul_params->set_k(k);
          matmul_params->set_n(n);
          matmul_params->set_batch_count(batch_count);

          // Set leading dimensions (row-major)
          matmul_params->set_lda(k);  // A is [..., M, K]
          matmul_params->set_ldb(n);  // B is [..., K, N]
          matmul_params->set_ldd(n);  // D/Y is [..., M, N]

          // Set strides for batched operations
          matmul_params->set_stride_a(m * k);
          matmul_params->set_stride_b(k * n);
          matmul_params->set_stride_d(m * n);

          // Store full shapes
          for (auto dim : a_shape)
            matmul_params->add_shape_a(dim);
          for (auto dim : b_shape)
            matmul_params->add_shape_b(dim);

          // Calculate output shape
          std::vector<int64_t> y_shape;
          size_t max_batch_dims = std::max(a_shape.size(), b_shape.size()) - 2;
          for (size_t i = 0; i < max_batch_dims; ++i) {
            int64_t dim_a = (i < a_shape.size() - 2) ? a_shape[i] : 1;
            int64_t dim_b = (i < b_shape.size() - 2) ? b_shape[i] : 1;
            y_shape.push_back(std::max(dim_a, dim_b));
          }
          y_shape.push_back(m);
          y_shape.push_back(n);
          for (auto dim : y_shape)
            matmul_params->add_shape_y(dim);

          ROCM_LOG(2) << LOG_PREFIX << " A shape: [";
          for (size_t i = 0; i < a_shape.size(); ++i) {
            ROCM_LOG(2) << (i > 0 ? ", " : "") << a_shape[i];
          }
          ROCM_LOG(2) << "]";
          ROCM_LOG(2) << LOG_PREFIX << " B shape: [";
          for (size_t i = 0; i < b_shape.size(); ++i) {
            ROCM_LOG(2) << (i > 0 ? ", " : "") << b_shape[i];
          }
          ROCM_LOG(2) << "]";
          ROCM_LOG(2) << LOG_PREFIX << " M=" << m << ", K=" << k << ", N=" << n
                      << ", batch=" << batch_count;

          // Check if B is a constant (weight matrix)
          auto pass_context = self->get_context();
          auto weight_ref =
              NodeArgConstRef::from_node_arg(*graph, *input_B.node_arg);
          auto weight_name = node_arg_get_name(*input_B.node_arg);

          bool b_is_constant = weight_ref.is_constant();
          matmul_params->set_b_is_constant(b_is_constant);

          if (b_is_constant) {
            auto weight_data =
                node_arg_get_const_data_as_floats(*graph, *input_B.node_arg);
            std::string weight_filename =
                rocm_pass::generate_weight_filename("rocm_matmul", weight_name);

            if (rocm_pass::save_weight_to_cache(pass_context, weight_data,
                                                weight_filename, LOG_PREFIX)) {
              matmul_params->set_weight_file_path(weight_filename);
              matmul_params->set_weight_file_size(
                  static_cast<int64_t>(weight_data.size() * sizeof(float)));
            }
            ROCM_LOG(1) << LOG_PREFIX
                        << " B is constant weight: " << weight_name;
          } else {
            ROCM_LOG(1) << LOG_PREFIX << " B is runtime input: " << weight_name;
          }

          // Store input/output names
          matmul_params->add_input_names(node_arg_get_name(*input_A.node_arg));
          matmul_params->add_input_names(weight_name);
          matmul_params->add_output_names(node_arg_get_name(*output.node_arg));

          // Build input/output lists for fuse
          std::vector<std::string> input_names;
          input_names.push_back(node_arg_get_name(*input_A.node_arg));

          // If B is not constant, it's a runtime input
          if (!b_is_constant) {
            input_names.push_back(weight_name);
          }

          std::vector<std::string> output_names;
          output_names.push_back(node_arg_get_name(*output.node_arg));

          // Constant initializers (B if it's a constant)
          std::vector<std::string> constant_initializers;
          if (b_is_constant) {
            constant_initializers.push_back(weight_name);
          }

          // Get output name for param file and fused node naming
          std::string fused_output_name = node_arg_get_name(*output.node_arg);

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_matmul", input_names, output_names,
                             constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, fused_output_name,
                LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass& self, Graph& graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for MatMul patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};

}  // namespace

DEFINE_MORPHIZEN_PASS(Level2RocmMatmul, morphizen_pass_level2_rocm_matmul)
