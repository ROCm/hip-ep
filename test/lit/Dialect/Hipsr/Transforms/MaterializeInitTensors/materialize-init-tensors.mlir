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
// CHECK-LABEL: func.func @barrier_domain(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[EXTENTS:.+]]: tensor<2xi64, #hipsr.mem<host>>) -> tensor<?x?xf32, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[DOMAIN:.+]] = hipsr.pool_domain(%[[CTX]], %[[IN]], %[[EXTENTS]] : !hipsr.context, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DIN:.+]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[DEXTENTS:.+]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:        %[[IN_SHAPE:.+]] = shape.shape_of %[[DIN]] : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:        %[[ROW_INDEX:.+]] = arith.constant 0 : index
// CHECK-NEXT:        %[[ROWS:.+]] = tensor.extract %[[IN_SHAPE]]{{\[}}%[[ROW_INDEX]]] : tensor<2xindex>
// CHECK-NEXT:        %[[COLUMN_INDEX:.+]] = arith.constant 1 : index
// CHECK-NEXT:        %[[COLUMN:.+]] = tensor.extract %[[DEXTENTS]]{{\[}}%[[COLUMN_INDEX]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[COLUMNS:.+]] = arith.index_cast %[[COLUMN]] : i64 to index
// CHECK-NEXT:        %[[RESULT_SHAPE:.+]] = shape.from_extents %[[ROWS]], %[[COLUMNS]] : index, index
// CHECK-NEXT:        scf.yield %[[RESULT_SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[D0_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[D0_EXTENT:.+]] = shape.get_extent %[[SHAPE]], %[[D0_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D0:.+]] = shape.size_to_index %[[D0_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[D1_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT:      %[[D1_EXTENT:.+]] = shape.get_extent %[[SHAPE]], %[[D1_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[D1:.+]] = shape.size_to_index %[[D1_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXPAND:.+]] = hipsr.expand(%[[DCTX]]) ins(%[[DIN]], %[[DEXTENTS]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : tensor<?x?xf32, #hipsr.mem<device>>) : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[SHAPE]], %[[EXPAND]] : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[EXPAND]] : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<?x?xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:    return %[[DOMAIN]] : tensor<?x?xf32, #hipsr.mem<device>>
// CHECK-NEXT:  }
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
// CHECK-LABEL: func.func @interleaved(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<?x256xf16, #hipsr.mem<device>>, %[[B:.+]]: tensor<256x512xf16, #hipsr.mem<device>>, %[[C:.+]]: tensor<?x512xf16, #hipsr.mem<device>>) -> tensor<?x512xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[DOMAIN:.+]] = hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]], %[[C]] : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<?x256xf16, #hipsr.mem<device>>, %[[DB:.+]]: tensor<256x512xf16, #hipsr.mem<device>>, %[[DC:.+]]: tensor<?x512xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[MATMUL_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:        %[[A_SHAPE:.+]] = shape.shape_of %[[DA]] : tensor<?x256xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:        %[[B_SHAPE:.+]] = shape.shape_of %[[DB]] : tensor<256x512xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:        %[[M_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT:        %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[N_INDEX:.+]] = shape.const_size 1
// CHECK-NEXT:        %[[N:.+]] = shape.get_extent %[[B_SHAPE]], %[[N_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[FROM_EXTENTS:.+]] = shape.from_extents %[[M]], %[[N]] : !shape.size, !shape.size
// CHECK-NEXT:        scf.yield %[[FROM_EXTENTS]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[ADD_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:        %[[C_SHAPE:.+]] = shape.shape_of %[[DC]] : tensor<?x512xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:        %[[BROADCAST:.+]] = shape.broadcast %[[MATMUL_SHAPE]], %[[C_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        scf.yield %[[BROADCAST]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[MATMUL_D0_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[MATMUL_D0_EXTENT:.+]] = shape.get_extent %[[MATMUL_SHAPE]], %[[MATMUL_D0_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[MATMUL_D0:.+]] = shape.size_to_index %[[MATMUL_D0_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[MATMUL_INIT:.+]] = tensor.empty(%[[MATMUL_D0]]) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ADD_D0_INDEX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[ADD_D0_EXTENT:.+]] = shape.get_extent %[[ADD_SHAPE]], %[[ADD_D0_INDEX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[ADD_D0:.+]] = shape.size_to_index %[[ADD_D0_EXTENT]] : !shape.size
// CHECK-NEXT:      %[[ADD_INIT:.+]] = tensor.empty(%[[ADD_D0]]) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MATMUL:.+]] = hipsr.matmul(%[[DCTX]]) ins(%[[DA]], %[[DB]] : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>) outs(%[[MATMUL_INIT]] : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[MATMUL]], %[[DC]] : tensor<?x512xf16, #hipsr.mem<device>>, tensor<?x512xf16, #hipsr.mem<device>>) outs(%[[ADD_INIT]] : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MATMUL_SHAPE]], %[[MATMUL]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ADD_SHAPE]], %[[ADD]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[ADD]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<?x512xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:    return %[[DOMAIN]] : tensor<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
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
// CHECK-LABEL: func.func @constant_between_placeholders(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context, %[[A:.+]]: tensor<2x2xf16, #hipsr.mem<device>>) -> tensor<2x2xf32, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[DOMAIN:.+]] = hipsr.pool_domain(%[[CTX]], %[[A]] : !hipsr.context, tensor<2x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[DA:.+]]: tensor<2x2xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[WEIGHT:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<2x2xf32>} : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:        %[[A_SHAPE:.+]] = shape.shape_of %[[DA]] : tensor<2x2xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:        scf.yield %[[A_SHAPE]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[ADD_SHAPE:.+]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:        %[[WEIGHT_SHAPE:.+]] = shape.shape_of %[[WEIGHT]] : tensor<2x2xf32, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:        %[[BROADCAST:.+]] = shape.broadcast %[[CAST_SHAPE]], %[[WEIGHT_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        scf.yield %[[BROADCAST]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CAST_INIT:.+]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ADD_INIT:.+]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST:.+]] = hipsr.cast(%[[DCTX]]) ins(%[[DA]] : tensor<2x2xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ADD:.+]] = hipsr.add(%[[DCTX]]) ins(%[[CAST]], %[[WEIGHT]] : tensor<2x2xf32, #hipsr.mem<device>>, tensor<2x2xf32, #hipsr.mem<device>>) outs(%[[ADD_INIT]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_SHAPE]], %[[CAST]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ADD_SHAPE]], %[[ADD]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[ADD]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<2x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:    return %[[DOMAIN]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:  }
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
// CHECK-LABEL: func.func @no_placeholder(
// CHECK-SAME:      %[[IN:.+]]: tensor<3x4xf32>) -> tensor<2x?xi64> {
// CHECK-NEXT:    %[[DOMAIN:.+]] = hipsr.pool_domain(%[[IN]] : tensor<3x4xf32>) {
// CHECK-NEXT:    ^bb0(%[[DIN:.+]]: tensor<3x4xf32>):
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.dim %[[DIN]], %[[C1]] : tensor<3x4xf32>
// CHECK-NEXT:      %[[BUFFER:.+]] = tensor.empty(%[[DIM]]) : tensor<2x?xi64>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[BUFFER]] : tensor<2x?xi64>
// CHECK-NEXT:    } -> tensor<2x?xi64> {domain_id = 0 : i64}
// CHECK-NEXT:    return %[[DOMAIN]] : tensor<2x?xi64>
// CHECK-NEXT:  }
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
