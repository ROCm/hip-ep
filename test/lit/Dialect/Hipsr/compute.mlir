// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks hipsr.compute syntax, boundary value forwarding, and invalid IR
// diagnostics.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// The context, the input, and the init all cross into the body as entry block
// arguments, and the yielded value becomes the result tied to the init. Nothing
// follows the result type, so the operand segment sizes stay elided.
// CHECK-LABEL: func.func @roundtrip(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16>, %[[INIT:.+]]: tensor<6xf16>) -> tensor<6xf16> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[DATA]] : tensor<2x3xf16>) outs(%[[INIT]] : tensor<6xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16>, %[[DEST:.+]]: tensor<6xf16>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16> into tensor<6xf16>
// CHECK-NEXT: %[[FILLED:.+]] = tensor.insert_slice %[[FLAT]] into %[[DEST]][0] [6] [1] : tensor<6xf16> into tensor<6xf16>
// CHECK-NEXT: hipsr.compute_yield %[[FILLED]] : tensor<6xf16>
// CHECK-NEXT: } : tensor<6xf16>{{$}}
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16>
// CHECK-NEXT: }
func.func @roundtrip(%ctx: !hipsr.context, %data: tensor<2x3xf16>,
                     %init: tensor<6xf16>) -> tensor<6xf16> {
  %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                             outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>, %dest: tensor<6xf16>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16> into tensor<6xf16>
    %filled = tensor.insert_slice %flat into %dest[0] [6] [1]
        : tensor<6xf16> into tensor<6xf16>
    hipsr.compute_yield %filled : tensor<6xf16>
  } : tensor<6xf16>
  return %out : tensor<6xf16>
}

// -----
// Several inputs and inits, including a non-tensor input, and one result tied to
// each init.
// CHECK-LABEL: func.func @multi_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[DATA:.+]]: tensor<2x3xf16>, %[[EXTENT:.+]]: index, %[[INIT0:.+]]: tensor<6xf16>, %[[INIT1:.+]]: tensor<3x2xf16>) -> (tensor<6xf16>, tensor<3x2xf16>) {
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins(%[[DATA]], %[[EXTENT]] : tensor<2x3xf16>, index) outs(%[[INIT0]], %[[INIT1]] : tensor<6xf16>, tensor<3x2xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.+]]: !hipsr.context, %[[IN:.+]]: tensor<2x3xf16>, %[[BODY_EXTENT:.+]]: index, %[[DEST0:.+]]: tensor<6xf16>, %[[DEST1:.+]]: tensor<3x2xf16>):
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1]] : tensor<2x3xf16> into tensor<6xf16>
// CHECK-NEXT: %[[SWAPPED:.+]] = tensor.expand_shape %[[FLAT]] {{\[\[}}0, 1]] output_shape [3, 2] : tensor<6xf16> into tensor<3x2xf16>
// CHECK-NEXT: hipsr.compute_yield %[[FLAT]], %[[SWAPPED]] : tensor<6xf16>, tensor<3x2xf16>
// CHECK-NEXT: } : tensor<6xf16>, tensor<3x2xf16>{{$}}
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<6xf16>, tensor<3x2xf16>
// CHECK-NEXT: }
func.func @multi_result(%ctx: !hipsr.context, %data: tensor<2x3xf16>, %n: index,
                        %init0: tensor<6xf16>, %init1: tensor<3x2xf16>)
    -> (tensor<6xf16>, tensor<3x2xf16>) {
  %out:2 = hipsr.compute(%ctx) ins(%data, %n : tensor<2x3xf16>, index)
                               outs(%init0, %init1 : tensor<6xf16>,
                                                     tensor<3x2xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>, %extent: index,
       %dest0: tensor<6xf16>, %dest1: tensor<3x2xf16>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16> into tensor<6xf16>
    %swapped = tensor.expand_shape %flat [[0, 1]] output_shape [3, 2]
        : tensor<6xf16> into tensor<3x2xf16>
    hipsr.compute_yield %flat, %swapped : tensor<6xf16>, tensor<3x2xf16>
  } : tensor<6xf16>, tensor<3x2xf16>
  return %out#0, %out#1 : tensor<6xf16>, tensor<3x2xf16>
}

// -----
// A post-bufferization form takes device memrefs as inits. Memref inits are not
// tensor results, so the op has none, and the printer omits the empty implicit
// yield.
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
// The body must have a block. The custom form always parses one, so this is what
// a caller that builds the op and skips the entry block runs into.
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
  // expected-error @+2 {{along control flow edge from parent to Region #0: region branch point has 1 operands, but region successor needs 0 inputs}}
  // expected-note @+1 {{region branch point}}
  hipsr.compute(%ctx) ins() outs() {
    hipsr.compute_yield
  }
  return
}

// -----
// The f16 input does not match the f32 entry block argument it is forwarded to.
func.func @entry_argument_type_mismatch(%ctx: !hipsr.context,
                                        %data: tensor<6xf16>) {
  // expected-error @+2 {{along control flow edge from parent to Region #0: successor operand type #1 'tensor<6xf16>' should match successor input type #1 'tensor<6xf32>'}}
  // expected-note @+1 {{region branch point}}
  hipsr.compute(%ctx) ins(%data : tensor<6xf16>) outs() {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<6xf32>):
    hipsr.compute_yield
  }
  return
}

// -----
// The op has one result, but the yield has no value.
func.func @missing_yield_value(%ctx: !hipsr.context, %init: tensor<6xf16>) {
  // expected-error @+1 {{along control flow edge from Operation hipsr.compute_yield to parent: region branch point has 0 operands, but region successor needs 1 inputs}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16>):
    // expected-note @+1 {{region branch point}}
    hipsr.compute_yield
  } : tensor<6xf16>
  return
}

// -----
// The yielded value does not match the result it becomes.
func.func @yield_type_mismatch(%ctx: !hipsr.context, %init: tensor<6xf16>) {
  // expected-error @+1 {{along control flow edge from Operation hipsr.compute_yield to parent: successor operand type #0 'tensor<3x2xf16>' should match successor input type #0 'tensor<6xf16>'}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16>):
    %other = tensor.empty() : tensor<3x2xf16>
    // expected-note @+1 {{region branch point}}
    hipsr.compute_yield %other : tensor<3x2xf16>
  } : tensor<6xf16>
  return
}

// -----
// Destination-passing style ties each init to the result at the same position,
// so their types must agree.
func.func @init_result_type_mismatch(%ctx: !hipsr.context,
                                     %init: tensor<3x2xf16>) {
  // expected-error @+1 {{expected type of operand #1 ('tensor<3x2xf16>') to match type of corresponding result ('tensor<6xf16>')}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<3x2xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<3x2xf16>):
    %flat = tensor.empty() : tensor<6xf16>
    hipsr.compute_yield %flat : tensor<6xf16>
  } : tensor<6xf16>
  return
}

// -----
// A result without an init has no destination to be written into.
func.func @more_results_than_inits(%ctx: !hipsr.context, %init: tensor<6xf16>) {
  // expected-error @+1 {{expected the number of tensor results (2) to be equal to the number of output tensors (1)}}
  %out:2 = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16>):
    hipsr.compute_yield %dest, %dest : tensor<6xf16>, tensor<6xf16>
  } : tensor<6xf16>, tensor<6xf16>
  return
}

// -----
// The isolated body uses the init directly instead of its entry block argument.
func.func @body_uses_parent_value(%ctx: !hipsr.context, %init: tensor<6xf16>) {
  // expected-note @+1 {{required by region isolation constraints}}
  %out = hipsr.compute(%ctx) ins() outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<6xf16>):
    // expected-error @+1 {{using value defined outside the region}}
    hipsr.compute_yield %init : tensor<6xf16>
  } : tensor<6xf16>
  return
}

// -----
// An init must live in device memory once bufferized.
func.func @host_memref_init(%ctx: !hipsr.context, %init: memref<6xf16>) {
  // expected-error @+1 {{operand #1 must be variadic of ranked tensor or device memref, but got 'memref<6xf16>'}}
  hipsr.compute(%ctx) ins() outs(%init : memref<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: memref<6xf16>):
    hipsr.compute_yield
  }
  return
}

// -----
// Only hipsr.compute_yield may terminate the body. The custom form always
// supplies it, so only the generic form can end with something else.
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
func.func @yield_without_parent(%init: tensor<6xf16>) {
  // expected-error @+1 {{expects parent op 'hipsr.compute'}}
  hipsr.compute_yield %init : tensor<6xf16>
}
