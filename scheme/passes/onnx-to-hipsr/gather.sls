#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Gather → hipsr.gather  (device data path)
;;
;; DSL pattern: placeholder with inline shape region + hipsr.gather{axis}.
;; Shape region: split data_shape at axis, concat with indices_shape.
;; A :scheme helper builds the split/concat chain using the fresh OpBuilder.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr gather)
  (export populate-gather-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir ddr rewrite)
          (mlir ddr))

  ;; Build the gather output shape inside a region block using the DSL.
  ;; Shape logic:
  ;;   leading, _ = split_at(data_shape, axis)
  ;;   _, trailing = split_at(data_shape, axis+1)
  ;;   result = concat(concat(leading, indices_shape), trailing)
  ;; Build the gather output shape inside a region block.
  ;; Shape logic:
  ;;   leading, _ = split_at(data_shape, axis)
  ;;   _, trailing = split_at(data_shape, axis+1)
  ;;   result = concat(concat(leading, indices_shape), trailing)
  ;; Build the gather output shape inside a region block.
  ;; Shape logic:
  ;;   leading, _ = split_at(data_shape, axis)
  ;;   _, trailing = split_at(data_shape, axis+1)
  ;;   result = concat(concat(leading, indices_shape), trailing)
  (define (build-gather-shape! axis data-shape idx-shape shape-type size-type)
    (define (mk-sz n)
      (let ([op (mlir-build-operation "shape.const_size" '() (list size-type))])
        (mlir-operation-set-attr op "value" n :index)
        (mlir-operation-get-result op 0)))
    (let* ([sz1      (mk-sz axis)]
           [sp1      (mlir-build-operation "shape.split_at"
                       (list data-shape sz1) (list shape-type shape-type))]
           [leading  (mlir-operation-get-result sp1 0)]
           [sz2      (mk-sz (+ axis 1))]
           [sp2      (mlir-build-operation "shape.split_at"
                       (list data-shape sz2) (list shape-type shape-type))]
           [trailing (mlir-operation-get-result sp2 1)]
           [gathered-op (mlir-build-operation "shape.concat"
                          (list leading idx-shape) (list shape-type))]
           [gathered    (mlir-operation-get-result gathered-op 0)])
      (mlir-operation-get-result
        (mlir-build-operation "shape.concat"
          (list gathered trailing) (list shape-type))
        0)))

  (define-conversion-pattern (onnx-gather->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Gather (%data %indices)
    :then-let
        ([%ctx        (mlir-get-hipsr-context-arg op)]
         [!data-type  (mlir-value-get-type %data)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (mlir-tensor-type-in-device-space! !out-type)]
         [!shape-type (mlir-get-shape-shape-type (mlir-operation-get-context op))]
         [!size-type  (mlir-get-shape-size-type  (mlir-operation-get-context op))]
         [axis        (let ([a (mlir-operation-get-integer-attr op "axis" 0)])
                        (if (< a 0) (+ a (mlir-type-get-rank !data-type)) a))]
         ;; guard: only handle device data (eqv? avoids shadowed = keyword)
         [ok?         (eqv? 1 (mlir-type-is-device-tensor !data-type))])
    :rewrite %output :with
        ;; Guard: device data only. Emit a remark so diagnostics are visible,
        ;; then return #f so the conversion framework falls through to the
        ;; fallback pattern (mlir-populate-gather-conversion-patterns).
        (_ = (if (not ok?)
                 (begin (mlir-emit-remark! op "onnx-gather->hipsr: skipping host data")
                        #f)
                 #t))
        (%placeholder = "hipsr.placeholder" (%ctx %data %indices !out-device)
                        (^bb0 ((%ds : !shape-type) (%is : !shape-type))
                              (%result-shape = (build-gather-shape!
                                                 axis %ds %is !shape-type !size-type))
                              ("hipsr.shape_yield" (%result-shape)))
                        -> !out-device)
        (%result = hipsr.gather (%ctx %data %indices %placeholder)
                   (operandSegmentSizes = (list 1 1 1 1) :i32-array)
                   ("axis" = axis :i64)
                   -> !out-device))

  (define (populate-gather-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Gather"
                                      onnx-gather->hipsr type-converter))

) ;; end library (onnx-to-hipsr gather)
