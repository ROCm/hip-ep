#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Transpose → hipsr.transpose
;;
;; DSL pattern: placeholder with inline shape region + hipsr.transpose.
;; The shape region permutes input extents according to the perm attribute
;; (absent perm = reverse permutation).
;; A :scheme helper builds the shape.const_size / shape.get_extent /
;; shape.from_extents chain inside the region using the fresh OpBuilder.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr transpose)
  (export populate-transpose-patterns)
  (import (except (rnrs (6)) =)
          (mlir ir)
          (mlir dialects conversion)
          (mlir hipsr)
          (mlir pattern-macro))

  ;; Build the permuted output shape inside a region block.
  ;; Uses mlir-build-operation — must be called inside with-current-block-builder.
  ;; Returns the output !shape.shape value.
  (define (build-permuted-shape! loc perm input-shape shape-type size-type)
    (let* ([extents
            (map (lambda (p)
                   (let* ([sz-op (mlir-build-operation "shape.const_size" '() (list size-type))])
                     (mlir-operation-set-attr sz-op "value" p)
                     (let* ([ext-op (mlir-build-operation "shape.get_extent"
                                      (list input-shape
                                            (mlir-operation-get-result sz-op 0))
                                      (list size-type))])
                       (mlir-operation-get-result ext-op 0))))
                 perm)]
           [out-op (mlir-build-operation "shape.from_extents" extents (list shape-type))])
      (mlir-operation-get-result out-op 0)))

  (define-conversion-pattern (onnx-transpose->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Transpose (%input)
    :then-let
        ([%ctx        (mlir-get-hipsr-context-arg op)]
         [!in-type    (mlir-value-get-type %input)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (mlir-tensor-type-in-device-space! !out-type)]
         [!shape-type (mlir-get-shape-shape-type (mlir-operation-get-context op))]
         [!size-type  (mlir-get-shape-size-type  (mlir-operation-get-context op))]
         [perm        (let ([raw (mlir-operation-get-integer-array-attr op "perm")])
                        (if (null? raw)
                            ;; absent perm → reverse permutation
                            (let ([rank (mlir-type-get-rank !in-type)])
                              (let loop ([i 0] [acc '()])
                                (if (eqv? i rank) acc (loop (+ i 1) (cons i acc)))))
                            raw))])
    :rewrite %output :with
        (%placeholder = "hipsr.placeholder" (%ctx %input !out-device)
                        (^bb0 ((%is : !shape-type))
                              (%out-shape = (build-permuted-shape!
                                              op perm %is !shape-type !size-type))
                              ("hipsr.shape_yield" (%out-shape)))
                        -> !out-device)
        ;; :scheme — create transpose op and set perm attribute via mlir-build-operation
        (%result = (let* ([new-op (mlir-build-operation "hipsr.transpose"
                                    (list %ctx %input %placeholder !out-device)
                                    (list !out-device))])
                     (mlir-operation-set-dense-i64-array new-op "perm" perm)
                     (mlir-operation-get-result new-op 0))))

  (define (populate-transpose-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Transpose"
                                      onnx-transpose->hipsr type-converter))

) ;; end library (onnx-to-hipsr transpose)
