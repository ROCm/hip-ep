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
;; from the shape operand (a host tensor).
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr expand)
  (export populate-expand-patterns
          onnx-expand->hipsr)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects tensor)
          (mlir ddr))

  ;; The shape operand may be wrapped in a builtin.unrealized_conversion_cast
  ;; by the type converter (tensor<Nxi64> → device space). Unwrap to get the
  ;; host-space value that hipsr.placeholder (barrier) requires.
  (define (unwrap-cast v)
    (let ([def (mlir-value-get-defining-op v)])
      (if (and (not (zero? def))
               (string=? (mlir-operation-name def)
                          "builtin.unrealized_conversion_cast"))
          (mlir-operation-get-operand-value def 0)
          v)))

  (define-conversion-pattern (onnx-expand->hipsr op operands-ref rewriter type-converter)
    :if-match
        %output = onnx.Expand (%input %shape-operand)
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [%ctx        (mlir-get-hipsr-context-arg op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-device (mlir-tensor-type-with-encoding !out-type
                        (make-hipsr-device-space-attr ctx))]
         [%shape-host (unwrap-cast %shape-operand)])
         ;; TODO: validate that %shape-host is host-space after unwrapping.
         ;; If unwrap-cast returns the original value unchanged and it is already
         ;; device-space, hipsr.placeholder (barrier) will receive a device tensor
         ;; which may fault at runtime. The C++ ExpandConversion.cpp validates rank,
         ;; element type, and static shape length — these checks are missing here.
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %input %shape-host)
                        ("placeholder_type" = (make-hipsr-barrier-type-attr ctx))
                        -> !out-device)
        (%result = hipsr.expand (%ctx %input %shape-host %placeholder)
                   -> !out-device))

  (define (populate-expand-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.Expand pattern")
    (mlir-register-conversion-pattern patterns "onnx.Expand"
                                      onnx-expand->hipsr type-converter))

) ;; end library (onnx-to-hipsr expand)
