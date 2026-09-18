// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Positive coverage for -hipsr-materialize-init-tensors.
//
// Every case spells out its whole function with CHECK-NEXT, so the CHECK block
// reads as the expected output rather than as a set of spot checks. That also
// makes the absences load bearing without a single CHECK-NOT: a leftover
// placeholder, or an extent read for a static dimension, shows up as an extra
// line that breaks the chain.
//
// Error paths live in invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-materialize-init-tensors | FileCheck %s

// A barrier reads the data of its inputs rather than their shapes, so its
// region arguments become ctx and the input tensors themselves, and the pass
// builds no shape.shape_of at region entry. Both tensor arguments are used as
// tensors here, which is what pins that: %[[DIN]] feeds a shape.shape_of inside
// the region yielding tensor<2xindex>, %[[DEXTENTS]] feeds a tensor.extract,
// and a !shape.shape in either spot would not verify. Neither extent of the
// result is in the result type: the shape comes from data the region reads,
// which is why a barrier lands one domain past its inputs. Both
// inputs enter the domain as block arguments, so an earlier domain filled those
// buffers and the region reads them as they are. That is the only form the pass
// accepts; an input this domain allocates is the error case in invalid.mlir.
// CHECK-LABEL:   func.func @barrier_domain(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<?x1xf32, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2xi64, #hipsr.mem<host>>) -> tensor<?x?xf32, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]] : !hipsr.context, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[VAL_2:.*]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:          %[[SHAPE_OF_0:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:          %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[EXTRACT_0:.*]] = tensor.extract %[[SHAPE_OF_0]]{{\[}}%[[CONSTANT_0]]] : tensor<2xindex>
// CHECK-NEXT:          %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[EXTRACT_1:.*]] = tensor.extract %[[VAL_2]]{{\[}}%[[CONSTANT_1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_1]] : i64 to index
// CHECK-NEXT:          %[[FROM_EXTENTS_0:.*]] = shape.from_extents %[[EXTRACT_0]], %[[INDEX_CAST_0]] : index, index
// CHECK-NEXT:          scf.yield %[[FROM_EXTENTS_0]] : !shape.shape
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CONST_SIZE_0:.*]] = shape.const_size 0
// CHECK-NEXT:        %[[GET_EXTENT_0:.*]] = shape.get_extent %[[EXECUTE_REGION_0]], %[[CONST_SIZE_0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[SIZE_TO_INDEX_0:.*]] = shape.size_to_index %[[GET_EXTENT_0]] : !shape.size
// CHECK-NEXT:        %[[CONST_SIZE_1:.*]] = shape.const_size 1
// CHECK-NEXT:        %[[GET_EXTENT_1:.*]] = shape.get_extent %[[EXECUTE_REGION_0]], %[[CONST_SIZE_1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[SIZE_TO_INDEX_1:.*]] = shape.size_to_index %[[GET_EXTENT_1]] : !shape.size
// CHECK-NEXT:        %[[EMPTY_0:.*]] = tensor.empty(%[[SIZE_TO_INDEX_0]], %[[SIZE_TO_INDEX_1]]) : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXPAND_0:.*]] = hipsr.expand(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_2]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EMPTY_0]] : tensor<?x?xf32, #hipsr.mem<device>>) : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[EXECUTE_REGION_0]], %[[EXPAND_0]] : !shape.shape, tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[EXPAND_0]] : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<?x?xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_0]] : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @barrier_domain(%ctx: !hipsr.context, %in: tensor<?x1xf32, #hipsr.mem<device>>,
                          %extents: tensor<2xi64, #hipsr.mem<host>>) -> tensor<?x?xf32, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %in, %extents
      : !hipsr.context, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_in: tensor<?x1xf32, #hipsr.mem<device>>,
       %domain_extents: tensor<2xi64, #hipsr.mem<host>>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_in, %domain_extents : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>)
        {placeholder_type = #hipsr.placeholder_type<barrier>}
        : tensor<?x?xf32, #hipsr.mem<device>> shape_region {
    ^bb0(%region_ctx: !hipsr.context, %region_in: tensor<?x1xf32, #hipsr.mem<device>>,
         %region_extents: tensor<2xi64, #hipsr.mem<host>>):
      %in_shape = shape.shape_of %region_in : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
      %row_index = arith.constant 0 : index
      %rows = tensor.extract %in_shape[%row_index] : tensor<2xindex>
      %column_index = arith.constant 1 : index
      %column = tensor.extract %region_extents[%column_index] : tensor<2xi64, #hipsr.mem<host>>
      %columns = arith.index_cast %column : i64 to index
      %result_shape = shape.from_extents %rows, %columns : index, index
      hipsr.shape_yield %result_shape : !shape.shape
    }
    %expand = hipsr.expand(%domain_ctx)
        ins(%domain_in, %domain_extents : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>)
        outs(%init : tensor<?x?xf32, #hipsr.mem<device>>) : tensor<?x?xf32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %expand : tensor<?x?xf32, #hipsr.mem<device>>
  } -> tensor<?x?xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<?x?xf32, #hipsr.mem<device>>
}

// -----

// The canonical form: a shape region argument standing for a domain input
// becomes shape.shape_of on it, hipsr.shape_yield becomes scf.yield because
// HasParent binds it to hipsr.placeholder, and only dim 0 of tensor<?x512xf16>
// is read back out of a computed shape -- 512 stays in the tensor.empty type.
// Each data op takes the tensor.empty as its outs operand, with the
// placeholders gone.
//
// The add placeholder comes after the matmul in the input, so its shape
// computation has to move ahead of both data ops without overtaking the matmul
// computation it reads. That dependency shows up in the shape graph: the add
// shape region took the matmul placeholder result as its first input, so its
// first argument becomes %[[MATMUL_SHAPE]] and only its second one gets a
// shape.shape_of. Both allocations land after the second execute_region, so the
// domain reads as two shape computations, two allocations, the two data ops,
// then the two shape links.
//
// A link names the result of the op that filled the buffer rather than the
// tensor.empty it was handed. The two are one buffer after bufferization, which
// is what leaves the shape on the memref.alloc for -hip-use-output-allocator,
// and naming the tensor.empty instead would keep it alive as an allocation of
// its own for an op whose result is only a view of another buffer.
// CHECK-LABEL:   func.func @interleaved(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<?x256xf16, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<256x512xf16, #hipsr.mem<device>>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<?x512xf16, #hipsr.mem<device>>) -> tensor<?x512xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<?x256xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: tensor<256x512xf16, #hipsr.mem<device>>, %[[VAL_3:.*]]: tensor<?x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:          %[[SHAPE_OF_0:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x256xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:          %[[SHAPE_OF_1:.*]] = shape.shape_of %[[VAL_2]] : tensor<256x512xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:          %[[CONST_SIZE_0:.*]] = shape.const_size 0
// CHECK-NEXT:          %[[GET_EXTENT_0:.*]] = shape.get_extent %[[SHAPE_OF_0]], %[[CONST_SIZE_0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:          %[[CONST_SIZE_1:.*]] = shape.const_size 1
// CHECK-NEXT:          %[[GET_EXTENT_1:.*]] = shape.get_extent %[[SHAPE_OF_1]], %[[CONST_SIZE_1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:          %[[FROM_EXTENTS_0:.*]] = shape.from_extents %[[GET_EXTENT_0]], %[[GET_EXTENT_1]] : !shape.size, !shape.size
// CHECK-NEXT:          scf.yield %[[FROM_EXTENTS_0]] : !shape.shape
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[EXECUTE_REGION_1:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:          %[[SHAPE_OF_2:.*]] = shape.shape_of %[[VAL_3]] : tensor<?x512xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:          %[[BROADCAST_0:.*]] = shape.broadcast %[[EXECUTE_REGION_0]], %[[SHAPE_OF_2]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:          scf.yield %[[BROADCAST_0]] : !shape.shape
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CONST_SIZE_2:.*]] = shape.const_size 0
// CHECK-NEXT:        %[[GET_EXTENT_2:.*]] = shape.get_extent %[[EXECUTE_REGION_0]], %[[CONST_SIZE_2]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[SIZE_TO_INDEX_0:.*]] = shape.size_to_index %[[GET_EXTENT_2]] : !shape.size
// CHECK-NEXT:        %[[CONST_SIZE_3:.*]] = shape.const_size 0
// CHECK-NEXT:        %[[GET_EXTENT_3:.*]] = shape.get_extent %[[EXECUTE_REGION_1]], %[[CONST_SIZE_3]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[SIZE_TO_INDEX_1:.*]] = shape.size_to_index %[[GET_EXTENT_3]] : !shape.size
// CHECK-NEXT:        %[[EMPTY_0:.*]] = tensor.empty(%[[SIZE_TO_INDEX_0]]) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EMPTY_1:.*]] = tensor.empty(%[[SIZE_TO_INDEX_1]]) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[MATMUL_0:.*]] = hipsr.matmul(%[[VAL_0]]) ins(%[[VAL_1]], %[[VAL_2]] : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>) outs(%[[EMPTY_0]] : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_0:.*]] = hipsr.add(%[[VAL_0]]) ins(%[[MATMUL_0]], %[[VAL_3]] : tensor<?x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>) outs(%[[EMPTY_1]] : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[EXECUTE_REGION_0]], %[[MATMUL_0]] : !shape.shape, tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[EXECUTE_REGION_1]], %[[ADD_0]] : !shape.shape, tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_0]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<?x512xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_0]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @interleaved(%ctx: !hipsr.context, %a: tensor<?x256xf16, #hipsr.mem<device>>,
                       %b: tensor<256x512xf16, #hipsr.mem<device>>, %c: tensor<?x512xf16, #hipsr.mem<device>>)
    -> tensor<?x512xf16, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %a, %b, %c
      : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>,
        tensor<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16, #hipsr.mem<device>>,
       %domain_b: tensor<256x512xf16, #hipsr.mem<device>>, %domain_c: tensor<?x512xf16, #hipsr.mem<device>>):
    %matmul_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16, #hipsr.mem<device>> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %matmul_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield %matmul_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        outs(%matmul_init : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%matmul_init, %domain_c : tensor<?x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16, #hipsr.mem<device>> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield %add_shape : !shape.shape
    }
    %add = hipsr.add(%domain_ctx)
        ins(%matmul, %domain_c : tensor<?x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>)
        outs(%add_init : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %add : tensor<?x512xf16, #hipsr.mem<device>>
  } -> tensor<?x512xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<?x512xf16, #hipsr.mem<device>>
}

// -----

// The weight the add placeholder reads is defined between the two placeholders,
// so it has to come over ahead of the shape computation that reads it. The
// constant landing at the front of the domain is the assertion -- left where it
// was, it would no longer dominate the read. Both results are fully static, so
// neither allocation reads an extent; the only reader of a computed shape here
// is the add region, which broadcasts the cast's.
// CHECK-LABEL:   func.func @constant_between_placeholders(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<2x2xf16, #hipsr.mem<device>>) -> tensor<2x2xf32, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<2x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<2x2xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = hipsr.constant {value = dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:          %[[SHAPE_OF_0:.*]] = shape.shape_of %[[VAL_1]] : tensor<2x2xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:          scf.yield %[[SHAPE_OF_0]] : !shape.shape
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[EXECUTE_REGION_1:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:          %[[SHAPE_OF_1:.*]] = shape.shape_of %[[CONSTANT_0]] : tensor<2x2xf32, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:          %[[BROADCAST_0:.*]] = shape.broadcast %[[EXECUTE_REGION_0]], %[[SHAPE_OF_1]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:          scf.yield %[[BROADCAST_0]] : !shape.shape
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[EMPTY_1:.*]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<2x2xf16, #hipsr.mem<device>>) outs(%[[EMPTY_0]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_0:.*]] = hipsr.add(%[[VAL_0]]) ins(%[[CAST_0]], %[[CONSTANT_0]] : tensor<2x2xf32, #hipsr.mem<device>>, tensor<2x2xf32, #hipsr.mem<device>>) outs(%[[EMPTY_1]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[EXECUTE_REGION_0]], %[[CAST_0]] : !shape.shape, tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[EXECUTE_REGION_1]], %[[ADD_0]] : !shape.shape, tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_0]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<2x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_0]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @constant_between_placeholders(%ctx: !hipsr.context,
                                         %a: tensor<2x2xf16, #hipsr.mem<device>>)
    -> tensor<2x2xf32, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %a
      : !hipsr.context, tensor<2x2xf16, #hipsr.mem<device>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<2x2xf16, #hipsr.mem<device>>):
    %cast_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a : tensor<2x2xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<2x2xf32, #hipsr.mem<device>> shape_region {
    ^bb0(%a_shape: !shape.shape):
      hipsr.shape_yield %a_shape : !shape.shape
    }
    %cast = hipsr.cast(%domain_ctx)
        ins(%domain_a : tensor<2x2xf16, #hipsr.mem<device>>)
        outs(%cast_init : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
    %weight = hipsr.constant {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>}
        : tensor<2x2xf32, #hipsr.mem<device>>
    %add_init = hipsr.placeholder(%domain_ctx)
        ins(%cast_init, %weight : tensor<2x2xf32, #hipsr.mem<device>>, tensor<2x2xf32, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<2x2xf32, #hipsr.mem<device>> shape_region {
    ^bb0(%lhs_shape: !shape.shape, %rhs_shape: !shape.shape):
      %add_shape = shape.broadcast %lhs_shape, %rhs_shape
          : !shape.shape, !shape.shape -> !shape.shape
      hipsr.shape_yield %add_shape : !shape.shape
    }
    %add = hipsr.add(%domain_ctx)
        ins(%cast, %weight : tensor<2x2xf32, #hipsr.mem<device>>, tensor<2x2xf32, #hipsr.mem<device>>)
        outs(%add_init : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %add : tensor<2x2xf32, #hipsr.mem<device>>
  } -> tensor<2x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<2x2xf32, #hipsr.mem<device>>
}

// -----

// A domain without placeholders is left alone. The tensor.empty here is the one
// the input already had, not one the pass built.
// CHECK-LABEL:   func.func @no_placeholder(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<3x4xf32>) -> tensor<2x?xi64> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]] : tensor<3x4xf32>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: tensor<3x4xf32>):
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM_0:.*]] = tensor.dim %[[VAL_0]], %[[CONSTANT_0]] : tensor<3x4xf32>
// CHECK-NEXT:        %[[EMPTY_0:.*]] = tensor.empty(%[[DIM_0]]) : tensor<2x?xi64>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[EMPTY_0]] : tensor<2x?xi64>
// CHECK-NEXT:      } -> tensor<2x?xi64> {domain_id = 0 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_0]] : tensor<2x?xi64>
// CHECK-NEXT:    }

func.func @no_placeholder(%in: tensor<3x4xf32>) -> tensor<2x?xi64> {
  %0 = hipsr.pool_domain(%in : tensor<3x4xf32>) {
  ^bb0(%domain_in: tensor<3x4xf32>):
    %c1 = arith.constant 1 : index
    %n = tensor.dim %domain_in, %c1 : tensor<3x4xf32>
    %buffer = tensor.empty(%n) : tensor<2x?xi64>
    hipsr.pool_domain_yield %buffer : tensor<2x?xi64>
  } -> tensor<2x?xi64> {domain_id = 0 : i64}
  return %0 : tensor<2x?xi64>
}
