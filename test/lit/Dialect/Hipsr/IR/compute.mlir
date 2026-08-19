// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks hipsr.compute syntax, boundary value forwarding, and invalid IR
// diagnostics.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// The context, the input, and the output all cross into the body as entry
// block arguments, and the yielded value becomes the result written into the
// output. Nothing follows the result type, so the operand segment sizes stay
// elided.
// CHECK-LABEL: func.func @roundtrip(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: tensor<6xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<6xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: tensor<6xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[FILLED:.+]] = tensor.insert_slice %[[FLAT]] into %[[DEST]][0] [6] [1] : tensor<6xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FILLED]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @roundtrip(%ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
                     %init: tensor<6xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
                             outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>, %dest: tensor<6xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    %filled = tensor.insert_slice %flat into %dest[0] [6] [1]
        : tensor<6xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    hipsr.compute_yield %filled : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return %out : tensor<6xf16, #hipsr.mem<device>>
}

// -----
// Several inputs and outputs, with one result written into each output.
// CHECK-LABEL: func.func @multi_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[SCALE:.+]]: tensor<f16, #hipsr.mem<device>>, %[[INIT0:.+]]: tensor<6xf16, #hipsr.mem<device>>, %[[INIT1:.+]]: tensor<3x2xf16, #hipsr.mem<device>>) -> (tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins(%[[DATA]], %[[SCALE]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>) outs(%[[INIT0]], %[[INIT1]] : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[BODY_SCALE:.+]]: tensor<f16, #hipsr.mem<device>>, %[[DEST0:.+]]: tensor<6xf16, #hipsr.mem<device>>, %[[DEST1:.+]]: tensor<3x2xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[SWAPPED:.+]] = tensor.expand_shape %[[FLAT]] {{\[\[}}0, 1]] output_shape [3, 2] : tensor<6xf16, #hipsr.mem<device>> into tensor<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]], %[[SWAPPED]] : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @multi_result(%ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
                        %scale: tensor<f16, #hipsr.mem<device>>, %init0: tensor<6xf16, #hipsr.mem<device>>,
                        %init1: tensor<3x2xf16, #hipsr.mem<device>>)
    -> (tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>) {
  %out:2 = hipsr.compute(%ctx) ins(%data, %scale : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>)
                               outs(%init0, %init1 : tensor<6xf16, #hipsr.mem<device>>,
                                                     tensor<3x2xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>,
       %body_scale: tensor<f16, #hipsr.mem<device>>, %dest0: tensor<6xf16, #hipsr.mem<device>>,
       %dest1: tensor<3x2xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    %swapped = tensor.expand_shape %flat [[0, 1]] output_shape [3, 2]
        : tensor<6xf16, #hipsr.mem<device>> into tensor<3x2xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat, %swapped : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
  return %out#0, %out#1 : tensor<6xf16, #hipsr.mem<device>>, tensor<3x2xf16, #hipsr.mem<device>>
}

// -----
// An output only supplies the destination buffer, so a body that reshapes its
// data yields a result whose type differs from that output. A DPS op could not
// express this.
// CHECK-LABEL: func.func @shape_changing_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: tensor<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @shape_changing_result(%ctx: !hipsr.context, %data: tensor<2x3xf16, #hipsr.mem<device>>,
                                 %init: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16, #hipsr.mem<device>>)
                             outs(%init : tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16, #hipsr.mem<device>>,
       %dest: tensor<2x3xf16, #hipsr.mem<device>>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return %out : tensor<6xf16, #hipsr.mem<device>>
}

// -----
// A post-bufferization form takes device memrefs as outputs. A compute that had
// no results still has none, so its yield stays empty and the printer omits it.
// CHECK-LABEL: func.func @memref_form(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: memref<6xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<6xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: memref<6xf16, #hipsr.mem<device>>):
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @memref_form(%ctx: !hipsr.context,
                       %data: memref<2x3xf16, #hipsr.mem<device>>,
                       %init: memref<6xf16, #hipsr.mem<device>>) {
  hipsr.compute(%ctx) ins(%data : memref<2x3xf16, #hipsr.mem<device>>)
                      outs(%init : memref<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: memref<2x3xf16, #hipsr.mem<device>>,
       %dest: memref<6xf16, #hipsr.mem<device>>):
    hipsr.compute_yield
  }
  return
}

// -----
// A compute whose result type differs from its output keeps that result through
// bufferization, as a device memref yielded out of the body. The output buffer
// cannot stand in for it: the two types disagree.
// CHECK-LABEL: func.func @memref_form_with_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[INIT:.+]]: memref<2x3xf16, #hipsr.mem<device>>) -> memref<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<2x3xf16, #hipsr.mem<device>>, %[[DEST:.+]]: memref<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[FLAT:.+]] = memref.collapse_shape %[[IN]] {{\[\[}}0, 1]] : memref<2x3xf16, #hipsr.mem<device>> into memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]] : memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : memref<6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT: return %[[RESULT]] : memref<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @memref_form_with_result(%ctx: !hipsr.context,
                                   %data: memref<2x3xf16, #hipsr.mem<device>>,
                                   %init: memref<2x3xf16, #hipsr.mem<device>>)
    -> memref<6xf16, #hipsr.mem<device>> {
  %out = hipsr.compute(%ctx) ins(%data : memref<2x3xf16, #hipsr.mem<device>>)
                             outs(%init : memref<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: memref<2x3xf16, #hipsr.mem<device>>,
       %dest: memref<2x3xf16, #hipsr.mem<device>>):
    %flat = memref.collapse_shape %in [[0, 1]]
        : memref<2x3xf16, #hipsr.mem<device>>
        into memref<6xf16, #hipsr.mem<device>>
    hipsr.compute_yield %flat : memref<6xf16, #hipsr.mem<device>>
  } : memref<6xf16, #hipsr.mem<device>>
  return %out : memref<6xf16, #hipsr.mem<device>>
}

// -----
// Device is not the only space a bufferized compute may use: extents are read
// on the host, so a compute that produces them holds its input, its output and
// its result in host memory.
// CHECK-LABEL: func.func @memref_form_host_space(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[EXTENTS:.+]]: memref<2xi64, #hipsr.mem<host>>, %[[INIT:.+]]: memref<2xi64, #hipsr.mem<host>>) -> memref<2xi64, #hipsr.mem<host>> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[EXTENTS]] : memref<2xi64, #hipsr.mem<host>>) outs(%[[INIT]] : memref<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: memref<2xi64, #hipsr.mem<host>>, %[[DEST:.+]]: memref<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT: hipsr.compute_yield %[[DEST]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT: } : memref<2xi64, #hipsr.mem<host>>{{$}}
// CHECK-NEXT: return %[[RESULT]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT: }
func.func @memref_form_host_space(%ctx: !hipsr.context,
                                  %extents: memref<2xi64, #hipsr.mem<host>>,
                                  %init: memref<2xi64, #hipsr.mem<host>>)
    -> memref<2xi64, #hipsr.mem<host>> {
  %out = hipsr.compute(%ctx) ins(%extents : memref<2xi64, #hipsr.mem<host>>)
                             outs(%init : memref<2xi64, #hipsr.mem<host>>) {
  ^bb0(%body_ctx: !hipsr.context, %in: memref<2xi64, #hipsr.mem<host>>,
       %dest: memref<2xi64, #hipsr.mem<host>>):
    hipsr.compute_yield %dest : memref<2xi64, #hipsr.mem<host>>
  } : memref<2xi64, #hipsr.mem<host>>
  return %out : memref<2xi64, #hipsr.mem<host>>
}

// -----
// Any space is accepted, but a buffer still has to name the one it lives in.
func.func @memref_without_space(%ctx: !hipsr.context, %init: memref<6xf16>) {
  // expected-error @+1 {{operand #1 must be variadic of ranked tensor or hipsr memref, but got 'memref<6xf16>'}}
  hipsr.compute(%ctx) ins() outs(%init : memref<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: memref<6xf16>):
    hipsr.compute_yield
  }
  return
}

// -----
// The generic form is invalid because the body has no block.
func.func @empty_body(%ctx: !hipsr.context) {
  // expected-error @+1 {{failed to verify constraint: region with 1 blocks}}
  "hipsr.compute"(%ctx) <{operandSegmentSizes = array<i32: 1, 0, 0>}> ({
  }) : (!hipsr.context) -> ()
  return
}

// -----
// The body has two blocks, but a compute body allows only one.
func.func @multi_block_body(%ctx: !hipsr.context) {
  // expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
  hipsr.compute(%ctx) ins() outs() {
  ^bb0(%body_ctx: !hipsr.context):
    hipsr.compute_yield
  ^bb1:
    hipsr.compute_yield
  }
  return
}

// -----
// The context has no entry block argument to be forwarded to.
func.func @missing_context_argument(%ctx: !hipsr.context) {
  // expected-error @+2 {{operands, but region successor needs 0}}
  // expected-note @+1 {{region branch point}}
  hipsr.compute(%ctx) ins() outs() {
    hipsr.compute_yield
  }
  return
}

// -----
// The f16 input does not match the f32 entry block argument it is forwarded to.
func.func @entry_argument_type_mismatch(%ctx: !hipsr.context,
                                        %data: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-error @+2 {{should match successor input type #1}}
  // expected-note @+1 {{region branch point}}
  hipsr.compute(%ctx) ins(%data : tensor<6xf16, #hipsr.mem<device>>) outs() {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<6xf32, #hipsr.mem<device>>):
    hipsr.compute_yield
  }
  return
}

// -----
// The op has one result, but the yield has no value.
func.func @missing_yield_value(%ctx: !hipsr.context, %init: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{operands, but region successor needs 1}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16, #hipsr.mem<device>>):
    // expected-note @+1 {{region branch point}}
    hipsr.compute_yield
  } : tensor<6xf16, #hipsr.mem<device>>
  return
}

// -----
// The yielded value does not match the result it becomes.
func.func @yield_type_mismatch(%ctx: !hipsr.context, %init: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{should match successor input type #0}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16, #hipsr.mem<device>>):
    %other = tensor.empty() : tensor<3x2xf16, #hipsr.mem<device>>
    // expected-note @+1 {{region branch point}}
    hipsr.compute_yield %other : tensor<3x2xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return
}

// -----
// A result without an output has no destination to be written into.
func.func @more_results_than_outputs(%ctx: !hipsr.context,
                                     %init: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{expects one result per output, but got 2 results and 1 outputs}}
  %out:2 = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16, #hipsr.mem<device>>):
    hipsr.compute_yield %dest, %dest : tensor<6xf16, #hipsr.mem<device>>, tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>, tensor<6xf16, #hipsr.mem<device>>
  return
}

// -----
// The isolated body uses the output directly instead of its entry block
// argument.
func.func @body_uses_parent_value(%ctx: !hipsr.context, %init: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-note @+1 {{required by region isolation constraints}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16, #hipsr.mem<device>>):
    // expected-error @+1 {{using value defined outside the region}}
    hipsr.compute_yield %init : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return
}

// -----
// Inputs carry tensor data, so a scalar cannot cross the boundary.
func.func @non_tensor_input(%ctx: !hipsr.context, %n: index) {
  // expected-error @+1 {{operand #1 must be variadic of ranked tensor or hipsr memref, but got 'index'}}
  hipsr.compute(%ctx) ins(%n : index) outs() {
  ^bb0(%body_ctx: !hipsr.context, %extent: index):
    hipsr.compute_yield
  }
  return
}


// -----
// The body must end with hipsr.compute_yield.
func.func @wrong_terminator(%ctx: !hipsr.context) {
  // expected-error @+2 {{expects regions to end with 'hipsr.compute_yield', found 'llvm.unreachable'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'hipsr.compute_yield'}}
  "hipsr.compute"(%ctx) <{operandSegmentSizes = array<i32: 1, 0, 0>}> ({
  ^bb0(%body_ctx: !hipsr.context):
    llvm.unreachable
  }) : (!hipsr.context) -> ()
  return
}

// -----
// The yield is invalid because it has no parent compute.
func.func @yield_without_parent(%init: tensor<6xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{expects parent op 'hipsr.compute'}}
  hipsr.compute_yield %init : tensor<6xf16, #hipsr.mem<device>>
}
