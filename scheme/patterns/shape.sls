#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Shape → hipsr.placeholder + hipsr.compute (host output)
;;
;; Extracts tensor dimension sizes [start, end) as i64 scalars.
;; Uses (current-block-builder) — bound via parameterize by :regions codegen —
;; so :scheme escapes in region bodies access the OpBuilder without gensyms.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns shape)
  (export populate-shape-patterns
          onnx-shape->hipsr)
  (import (except (rnrs (6)) =)
          (only (chezscheme) format)
          (mlir ffi)
          (mlir hipsr)
          (mlir pattern-macro))

  ;; Helper: emit shape.const_size with an index-typed value attr
  (define (emit-const-size val)
    (let* ([op (mlir-build-op-in-block (current-block-builder) #f
                  "shape.const_size" '()
                  (list (mlir-get-shape-size-type
                          (mlir-operation-get-context #f))))])
      op))

  (define-conversion-pattern (onnx-shape->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Shape (%input)
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [%ctx        (mlir-get-hipsr-context-arg op)]
         [!input-type (mlir-value-get-type %input)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-host   (mlir-tensor-type-in-host-space !out-type)]
         [input-rank  (mlir-type-get-rank !input-type)]
         [start       (mlir-operation-get-integer-attr op "start" 0)]
         [end-raw     (mlir-operation-get-integer-attr op "end" 0)]
         [end         (if (zero? end-raw) input-rank end-raw)]
         [num-dims    (- end start)]
         [!shape-type (mlir-get-shape-shape-type ctx)]
         [!size-type  (mlir-get-shape-size-type  ctx)]
         [!index-type (mlir-get-index-type       ctx)]
         [!i64-type   (mlir-get-i64-type         ctx)]
         [!ctx-type   (mlir-get-hipsr-context-type ctx)])
    :rewrite %output :with
        ;; Placeholder: shape region yields const shape [num-dims]
        (%placeholder = hipsr.placeholder (%ctx %input !out-host)
                        :attrs (operandSegmentSizes (list 1 1 1) :i32-array)
                        :regions ((^bb0 ((%s : !shape-type))
                                     ;; shape.const_size needs an index attr — use :scheme
                                     (%cN = (let* ([x (mlir-build-op-in-block
                                                         (current-block-builder) op
                                                         "shape.const_size" '() (list !size-type))])
                                               (mlir-operation-set-index-attr x "value" num-dims)
                                               (mlir-operation-get-result x 0)))
                                     (%r  = shape.from_extents (%cN) -> !shape-type)
                                     (%y  = hipsr.shape_yield (%r) -> ())))
                        -> !out-host)
        ;; Compute body: for each axis [start, end), emit tensor.dim + arith.index_cast
        (%result = hipsr.compute (%ctx %input %placeholder !out-host)
                   :attrs (operandSegmentSizes (list 1 1 1) :i32-array)
                   :regions ((^bb0 ((%c : !ctx-type) (%in : !input-type) (%dest : !out-host))
                                ;; Collect i64 dim values via loop — (current-block-builder) gives
                                ;; the fresh OpBuilder for this block, set by parameterize.
                                (%dim-vals = (let loop ([axis start] [acc '()])
                                              (if (>= axis end) (reverse acc)
                                                (let* ([ci (mlir-build-op-in-block
                                                              (current-block-builder) op
                                                              "arith.constant" '() (list !index-type))])
                                                  (mlir-operation-set-index-attr ci "value" axis)
                                                  (let* ([di (mlir-build-op-in-block
                                                                (current-block-builder) op "tensor.dim"
                                                                (list %in (mlir-operation-get-result ci 0))
                                                                (list !index-type))]
                                                         [ii (mlir-build-op-in-block
                                                                (current-block-builder) op "arith.index_cast"
                                                                (list (mlir-operation-get-result di 0))
                                                                (list !i64-type))])
                                                    (loop (+ axis 1)
                                                          (cons (mlir-operation-get-result ii 0)
                                                                acc)))))))
                                (%r  = (mlir-operation-get-result
                                         (mlir-build-op-in-block (current-block-builder) op
                                           "tensor.from_elements" %dim-vals (list !out-host)) 0))
                                (%y  = hipsr.compute_yield (%r) -> ())))
                   -> !out-host))

  (define (populate-shape-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Shape"
                                      onnx-shape->hipsr type-converter))

) ;; end library (patterns shape)
