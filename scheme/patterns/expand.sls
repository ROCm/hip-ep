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

(library (patterns expand)
  (export populate-expand-patterns
          onnx-expand->hipsr)
  (import (except (rnrs (6)) =)
          (mlir ffi)
          (mlir hipsr)
          (mlir pattern-macro))

  (define-conversion-pattern (onnx-expand->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Expand (%input %shape-operand)
    :then-let
        ([%ctx        (mlir-get-hipsr-context-arg op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (mlir-tensor-type-in-device-space! !out-type)])
    :rewrite %output :with
        ;; Barrier placeholder: ins=(input, shape), type set after creation.
        (%placeholder = (let* ([ph-op ((current-mlir-build-fn) "hipsr.placeholder"
                                          (list %ctx %input %shape-operand !out-device)
                                          (list !out-device))])
                          (mlir-placeholder-set-barrier-type ph-op)
                          (mlir-operation-get-result ph-op 0)))
        (%result = "hipsr.expand" (%ctx %input %shape-operand %placeholder !out-device)
                   -> !out-device))

  (define (populate-expand-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.Expand pattern")
    (mlir-register-conversion-pattern patterns "onnx.Expand"
                                      onnx-expand->hipsr type-converter))

) ;; end library (patterns expand)
