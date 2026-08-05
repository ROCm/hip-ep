// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Positive coverage for -hipsr-materialize-init-tensors.
//
// This file grows with the pass instead of gaining a sibling per phase: each
// case keeps its input IR and only its CHECK lines are extended as later phases
// land. The transformation is end to end as of phase 4, and runs as:
//
//   phase 1, grouping -- every hipsr.placeholder in a domain moves to the front
//   of the domain block, keeping the relative order SSA form already gives it;
//   phase 2, shape computation -- each shape region body moves into an
//   scf.execute_region yielding !shape.shape, with the region's block arguments
//   replaced by the shapes of the placeholder inputs;
//   phase 3, allocation -- one tensor.empty per placeholder result, its dynamic
//   extents read out of the computed shape, all of them grouped after the last
//   scf.execute_region;
//   phase 4, cleanup -- every placeholder result use is rewired to its
//   tensor.empty and the placeholders are erased.
//
// So each domain ends up in the virtual 3-region form: shape computations, then
// allocations, then the data ops in their original order, with no
// hipsr.placeholder left behind.
//
// Error paths live in invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-materialize-init-tensors | FileCheck %s

// The canonical single-placeholder domain, and the case that carries the most
// CHECK detail as the later phases land. Its placeholder already leads the
// block, so grouping leaves the domain as it is. The two shape region arguments
// become shape.shape_of on the matmul inputs, and hipsr.shape_yield2 becomes
// scf.yield because it is bound to hipsr.placeholder by HasParent. Only dim 0
// of the result is dynamic, so the allocation reads a single extent back out of
// the shape and 512 stays in the tensor.empty type. The matmul ends up taking
// that tensor.empty as its outs operand, with the placeholder gone.
// CHECK-LABEL: func.func @matmul_domain(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>):
// CHECK:           %[[MATMUL_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             %[[A_SHAPE:.+]] = shape.shape_of %[[DA]] : tensor<?x256xf16> -> !shape.shape
// CHECK:             %[[B_SHAPE:.+]] = shape.shape_of %[[DB]] : tensor<256x512xf16> -> !shape.shape
// CHECK:             %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:             %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:             %[[RESULT_SHAPE:.+]] = shape.from_extents %[[M]], %[[N]] : !shape.size, !shape.size
// CHECK:             scf.yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK:           }
// CHECK:           %[[EXTENT:.+]] = shape.get_extent %[[MATMUL_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:           %[[DIM0:.+]] = shape.size_to_index %[[EXTENT]] : !shape.size
// CHECK:           %[[INIT:.+]] = tensor.empty(%[[DIM0]]) : tensor<?x512xf16>
// CHECK-NOT:       hipsr.placeholder
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[MATMUL]] : tensor<?x512xf16>
func.func @matmul_domain(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                         %b: tensor<256x512xf16>) -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %result_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %result_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// The same domain shape as @matmul_domain but with a fully static result. The
// shape computation is still built -- the pass does not try to prove it dead,
// it is left for canonicalization -- yet nothing is read back out of it,
// because the tensor.empty type already carries both extents. The CHECK-NEXT
// run is what asserts that: a stray shape.get_extent would break the chain.
// CHECK-LABEL: func.func @static_shape(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<128x256xf16>, %[[DB:.+]]: tensor<256x512xf16>):
// CHECK:           scf.execute_region -> !shape.shape {
// CHECK:             %[[A_SHAPE:.+]] = shape.shape_of %[[DA]] : tensor<128x256xf16> -> !shape.shape
// CHECK:             %[[B_SHAPE:.+]] = shape.shape_of %[[DB]] : tensor<256x512xf16> -> !shape.shape
// CHECK:             scf.yield %{{.+}} : !shape.shape
// CHECK:           }
// CHECK-NEXT:      %[[INIT:.+]] = tensor.empty() : tensor<128x512xf16>
// CHECK-NEXT:      %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<128x256xf16>, tensor<256x512xf16>) outs(%[[INIT]] : tensor<128x512xf16>) : tensor<128x512xf16>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[MATMUL]] : tensor<128x512xf16>
func.func @static_shape(%ctx: !hipsr.context, %a: tensor<128x256xf16>,
                        %b: tensor<256x512xf16>) -> tensor<128x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<128x256xf16>, tensor<256x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<128x256xf16>,
       %domain_b: tensor<256x512xf16>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<128x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<128x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %result_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %result_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<128x256xf16>, tensor<256x512xf16>)
        outs(%init : tensor<128x512xf16>) : tensor<128x512xf16>
    hipsr.pool_domain_yield %matmul : tensor<128x512xf16>
  } -> tensor<128x512xf16>
  return %0 : tensor<128x512xf16>
}

// -----

// A result with two dynamic dims and one static dim. The CHECK-NEXT run pins
// the whole tail down to the operation: dim 0 and dim 1 are read back out of
// the computed shape, dim 2 is not because 64 is already in the type, and
// nothing at all sits between the allocation and the data op that consumes it.
// CHECK-LABEL: func.func @multi_dynamic(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x?x64xf16>, %[[DB:.+]]: tensor<?x?x64xf16>):
// CHECK:           %[[SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             %[[A_SHAPE:.+]] = shape.shape_of %[[DA]] : tensor<?x?x64xf16> -> !shape.shape
// CHECK:             %[[B_SHAPE:.+]] = shape.shape_of %[[DB]] : tensor<?x?x64xf16> -> !shape.shape
// CHECK:             %[[BROADCAST:.+]] = shape.broadcast %[[A_SHAPE]], %[[B_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK:             scf.yield %[[BROADCAST]] : !shape.shape
// CHECK:           }
// CHECK:           %[[D0_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[D0_EXTENT:.+]] = shape.get_extent %[[SHAPE]], %[[D0_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D0:.+]] = shape.size_to_index %[[D0_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[D1_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT:      %[[D1_EXTENT:.+]] = shape.get_extent %[[SHAPE]], %[[D1_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D1:.+]] = shape.size_to_index %[[D1_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x64xf16>
// CHECK-NEXT:      %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x?x64xf16>, tensor<?x?x64xf16>) outs(%[[INIT]] : tensor<?x?x64xf16>) : tensor<?x?x64xf16>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[ADD]] : tensor<?x?x64xf16>
func.func @multi_dynamic(%ctx: !hipsr.context, %a: tensor<?x?x64xf16>,
                         %b: tensor<?x?x64xf16>) -> tensor<?x?x64xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x?x64xf16>, tensor<?x?x64xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x?x64xf16>,
       %domain_b: tensor<?x?x64xf16>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x?x64xf16> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %result_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield2 %result_shape : !shape.shape
    }
    %add = hipsr.add(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
        outs(%init : tensor<?x?x64xf16>) : tensor<?x?x64xf16>
    hipsr.pool_domain_yield %add : tensor<?x?x64xf16>
  } -> tensor<?x?x64xf16>
  return %0 : tensor<?x?x64xf16>
}

// -----

// The add placeholder is defined after the matmul in the input; grouping pulls
// it in front of both data ops without overtaking the matmul placeholder it
// depends on. That dependency then shows up in the shape graph: the add shape
// region took the matmul placeholder result as its first input, so its first
// argument becomes the matmul execute_region result rather than a shape.shape_of.
// Both allocations are grouped after the second execute_region, so once the
// placeholders are erased the domain reads as two shape computations, two
// allocations, then the two data ops.
// CHECK-LABEL: func.func @interleaved(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>, %[[DC:.+]]: tensor<?x512xf16>):
// CHECK:           %[[MATMUL_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             shape.shape_of %[[DA]] : tensor<?x256xf16> -> !shape.shape
// CHECK:             shape.shape_of %[[DB]] : tensor<256x512xf16> -> !shape.shape
// CHECK:             scf.yield %{{.+}} : !shape.shape
// CHECK:           }
// CHECK-NOT:       hipsr.placeholder
// CHECK:           %[[ADD_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             %[[C_SHAPE:.+]] = shape.shape_of %[[DC]] : tensor<?x512xf16> -> !shape.shape
// CHECK:             %[[BROADCAST:.+]] = shape.broadcast %[[MATMUL_SHAPE]], %[[C_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK:             scf.yield %[[BROADCAST]] : !shape.shape
// CHECK:           }
// CHECK:           %[[MATMUL_EXTENT:.+]] = shape.get_extent %[[MATMUL_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:           %[[MATMUL_DIM0:.+]] = shape.size_to_index %[[MATMUL_EXTENT]] : !shape.size
// CHECK:           %[[MATMUL_INIT:.+]] = tensor.empty(%[[MATMUL_DIM0]]) : tensor<?x512xf16>
// CHECK:           %[[ADD_EXTENT:.+]] = shape.get_extent %[[ADD_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:           %[[ADD_DIM0:.+]] = shape.size_to_index %[[ADD_EXTENT]] : !shape.size
// CHECK:           %[[ADD_INIT:.+]] = tensor.empty(%[[ADD_DIM0]]) : tensor<?x512xf16>
// CHECK-NOT:       hipsr.placeholder
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[MATMUL_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[MATMUL]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>) outs(%[[ADD_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[ADD]] : tensor<?x512xf16>
func.func @interleaved(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                       %b: tensor<256x512xf16>, %c: tensor<?x512xf16>)
    -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b, %c
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>,
        tensor<?x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>, %domain_c: tensor<?x512xf16>):
    %matmul_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %matmul_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %matmul_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%matmul_init : tensor<?x512xf16>) : tensor<?x512xf16>
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%matmul_init, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield2 %add_shape : !shape.shape
    }
    %add = hipsr.add(%domain_ctx)
        ins(%matmul, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        outs(%add_init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %add : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// Same domain as @interleaved but already grouped, so grouping is a no-op and
// the result is identical. Two leading placeholders in a row pin the skip that
// keeps a placeholder from being moved before itself, which would otherwise
// reverse their order.
// CHECK-LABEL: func.func @already_grouped(
// CHECK:         ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16>, %[[DB:.+]]: tensor<256x512xf16>, %[[DC:.+]]: tensor<?x512xf16>):
// CHECK:           %[[MATMUL_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             shape.shape_of %[[DA]] : tensor<?x256xf16> -> !shape.shape
// CHECK:             shape.shape_of %[[DB]] : tensor<256x512xf16> -> !shape.shape
// CHECK:             scf.yield %{{.+}} : !shape.shape
// CHECK:           }
// CHECK-NOT:       hipsr.placeholder
// CHECK:           %[[ADD_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK:             %[[C_SHAPE:.+]] = shape.shape_of %[[DC]] : tensor<?x512xf16> -> !shape.shape
// CHECK:             %[[BROADCAST:.+]] = shape.broadcast %[[MATMUL_SHAPE]], %[[C_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK:             scf.yield %[[BROADCAST]] : !shape.shape
// CHECK:           }
// CHECK:           %[[MATMUL_EXTENT:.+]] = shape.get_extent %[[MATMUL_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:           %[[MATMUL_DIM0:.+]] = shape.size_to_index %[[MATMUL_EXTENT]] : !shape.size
// CHECK:           %[[MATMUL_INIT:.+]] = tensor.empty(%[[MATMUL_DIM0]]) : tensor<?x512xf16>
// CHECK:           %[[ADD_EXTENT:.+]] = shape.get_extent %[[ADD_SHAPE]], %{{.+}} : !shape.shape, !shape.size -> !shape.size
// CHECK:           %[[ADD_DIM0:.+]] = shape.size_to_index %[[ADD_EXTENT]] : !shape.size
// CHECK:           %[[ADD_INIT:.+]] = tensor.empty(%[[ADD_DIM0]]) : tensor<?x512xf16>
// CHECK-NOT:       hipsr.placeholder
// CHECK:           %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16>, tensor<256x512xf16>) outs(%[[MATMUL_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[MATMUL]], %[[DC]] : tensor<?x512xf16>, tensor<?x512xf16>) outs(%[[ADD_INIT]] : tensor<?x512xf16>) : tensor<?x512xf16>
// CHECK:           hipsr.pool_domain_yield %[[ADD]] : tensor<?x512xf16>
func.func @already_grouped(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                           %b: tensor<256x512xf16>, %c: tensor<?x512xf16>)
    -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b, %c
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>,
        tensor<?x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>, %domain_c: tensor<?x512xf16>):
    %matmul_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %matmul_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %matmul_shape : !shape.shape
    }
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%matmul_init, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield2 %add_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%matmul_init : tensor<?x512xf16>) : tensor<?x512xf16>
    %add = hipsr.add(%domain_ctx)
        ins(%matmul, %domain_c : tensor<?x512xf16>, tensor<?x512xf16>)
        outs(%add_init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %add : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}

// -----

// A domain without placeholders is left alone.
// CHECK-LABEL: func.func @no_placeholder(
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[DIM:.+]] = tensor.dim %{{.+}}, %[[C1]] : tensor<3x4xf32>
// CHECK:         %[[BUFFER:.+]] = tensor.empty(%[[DIM]]) : tensor<2x?xi64>
// CHECK:         hipsr.pool_domain_yield %[[BUFFER]] : tensor<2x?xi64>
func.func @no_placeholder(%in: tensor<3x4xf32>) -> tensor<2x?xi64> {
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    %c1 = arith.constant 1 : index
    %n = tensor.dim %domain_in, %c1 : tensor<3x4xf32>
    %buffer = tensor.empty(%n) : tensor<2x?xi64>
    hipsr.pool_domain_yield %buffer : tensor<2x?xi64>
  } -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}
