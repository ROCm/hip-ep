#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Expand → hipsr.expand
;;
;; Creates a Barrier placeholder with ins=(input, shape) then hipsr.expand.
;; The placeholder must be Barrier because the shape is computed at runtime
;; from the shape operand (a host tensor). The shape region is filled by
;; hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr expand)
  (export populate-expand-patterns
          onnx-expand->hipsr)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir ddr))

  ;; The shape adaptor value may be wrapped in a builtin.unrealized_conversion_cast
  ;; by the dialect conversion framework when it maps tensor<Nxi64> → device space.
  ;; But hipsr.expand and hipsr.placeholder (barrier) both require host-space shape
  ;; operands. Unwrap the cast to obtain the actual hipsr.compute result (host space).
  (define (unwrap-cast v)
    (let ([def (mlir-value-get-defining-op v)])
      (if (and (not (zero? def))
               (string=? (mlir-operation-name def)
                          "builtin.unrealized_conversion_cast"))
          (mlir-operation-get-operand-value def 0)
          v)))

  (define-conversion-pattern (onnx-expand->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Expand (%input %shape-operand)
    :then-let
        ([%ctx        (mlir-get-hipsr-context-arg op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (mlir-tensor-type-in-device-space! !out-type)]
         ;; Unwrap any unrealized_conversion_cast to get the host-space shape value.
         ;; The type converter wraps the tensor<Nxi64> shape in a cast to device space,
         ;; but hipsr.expand and hipsr.placeholder (barrier) require the host-space value.
         [%shape-host (unwrap-cast %shape-operand)])
    :rewrite %output :with
        ;; Barrier placeholder: ins=(input, shape-host).
        (%placeholder = (let* ([ph-op (mlir-build-operation "hipsr.placeholder"
                                          (list %ctx %input %shape-host)
                                          (list !out-device))])
                          (mlir-placeholder-set-barrier-type ph-op)
                          (mlir-operation-get-result ph-op 0)))
        (%result = "hipsr.expand" (%ctx %input %shape-host %placeholder !out-device)
                   -> !out-device))

  (define (populate-expand-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.Expand pattern")
    (mlir-register-conversion-pattern patterns "onnx.Expand"
                                      onnx-expand->hipsr type-converter))

) ;; end library (onnx-to-hipsr expand)
