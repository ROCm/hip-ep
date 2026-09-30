#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Min → chain of hipsr.min
;;
;; Binary case (N=2): DSL pattern with inline broadcast shape region.
;; N=1: identity (replace with single input).
;; N>2: chain binary hipsr.min ops (plain Scheme).
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr min)
  (export populate-min-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects shape)
          (mlir ddr rewrite)
          (mlir ddr))

  ;;===--------------------------------------------------------------------===;;
  ;; Binary case — DSL with inline broadcast shape region (identical to equal)
  ;;===--------------------------------------------------------------------===;;

  (define-conversion-pattern (onnx-min-2->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Min (%lhs %rhs)
    :then-let
        ([%ctx        (mlir-get-hipsr-context-arg op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (make-mlir-tensor-in-device-space !out-type)]
         [!shape-type (mlir-shape.shape-type (mlir-operation-get-context op))])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %lhs %rhs !out-device)
                        (^bb0 ((%ls : !shape-type) (%rs : !shape-type))
                              (%broadcast = shape.broadcast (%ls %rs) -> !shape-type)
                              (hipsr.shape_yield (%broadcast)))
                        -> !out-device)
        (%result = hipsr.min (%ctx %lhs %rhs %placeholder) -> !out-device))

  ;;===--------------------------------------------------------------------===;;
  ;; General case — N=1 identity; N>2 chain (binary DSL pattern handles N=2)
  ;;===--------------------------------------------------------------------===;;

  (define (make-binary-min! rewriter loc-op ctx lhs rhs out-type)
    (let ([!shape-type (mlir-shape.shape-type (mlir-operation-get-context loc-op))])
      (mlir-set-insertion-point-before rewriter loc-op)
      (with-rewrite-builder (rewriter loc-op)
        (with-mlir-ops
          (%ph = hipsr.placeholder (ctx lhs rhs)
                 (^bb0 ((%ls : !shape-type) (%rs : !shape-type))
                       (%bc = shape.broadcast (%ls %rs) -> !shape-type)
                       (hipsr.shape_yield (%bc)))
                 -> out-type)
          (%r = hipsr.min (ctx lhs rhs %ph) -> out-type)))))

  (define (onnx-min-general->hipsr op operands-ref rewriter type-converter)
    (let ([n (value-array-ref-size operands-ref)])
      (cond
        [(eqv? n 1)
         (mlir-replace-op rewriter op (value-array-ref-at operands-ref 0))
         #t]
        [(> n 2)
         (let* ([ctx      (mlir-get-hipsr-context-arg op)]
                [out-type (make-mlir-tensor-in-device-space
                            (mlir-value-get-type (mlir-operation-get-result op 0)))])
           (let loop ([i 2]
                      [acc (make-binary-min! rewriter op ctx
                             (value-array-ref-at operands-ref 0)
                             (value-array-ref-at operands-ref 1)
                             out-type)])
             (if (eqv? i n)
                 (begin (mlir-replace-op rewriter op acc) #t)
                 (loop (+ i 1)
                       (make-binary-min! rewriter op ctx acc
                         (value-array-ref-at operands-ref i) out-type)))))]
        [else #f])))

  (define (populate-min-patterns type-converter patterns ctx)
    ;; DSL pattern for the common binary case (N=2) — inline shape region
    (mlir-register-conversion-pattern patterns "onnx.Min" onnx-min-2->hipsr    type-converter)
    ;; Scheme fallback for N=1 (identity) and N>2 (chain)
    (mlir-register-conversion-pattern patterns "onnx.Min" onnx-min-general->hipsr type-converter))

) ;; end library (onnx-to-hipsr min)
