#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Constant → hipsr.constant (or arith.constant for rank-0 scalars)
;;
;; Handled cases:
;;   - Rank > 0 dense inline value → hipsr.constant with copied `value` attr
;;   - Rank 0 scalar → arith.constant with copied `value` attr
;;
;; NOT handled here (returns #f, defers to the C++ fallback registered in the pass):
;;   - External data (absent `value`, present `location`/`offset`/`size`) —
;;     DenseResourceElementsAttr construction from memory-mapped or file data
;;     is not available in Scheme FFI. The pass registers
;;     mlir-populate-constant-conversion-patterns as a fallback for these cases.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr constant)
  (export populate-constant-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir ddr))

  (define-conversion-pattern (onnx-constant->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
    :then-let
        ([!out-type (mlir-value-get-type %output)]
         [!out-dev  (make-mlir-tensor-in-device-space !out-type)]
         [rank      (mlir-type-get-rank !out-type)])
    :rewrite %output :with
        ;; Mirror C++ ConstantOpLowering logic:
        ;;   value present   → inline path (handled below)
        ;;   location present → external data, defer to C++ fallback
        ;;   neither          → emit error (C++ would notifyMatchFailure)
        (_ = (cond
               [(not (zero? (mlir-operation-has-attr op "value")))  #t]
               [(not (zero? (mlir-operation-has-attr op "location"))) #f]
               [else
                (begin (mlir-emit-error! op "onnx.Constant has neither value nor location")
                       #f)]))
        ;; Rank 0 → arith.constant (host type); rank > 0 → hipsr.constant (device type)
        (%result = (let* ([!result-type (if (zero? rank) !out-type !out-dev)]
                          [c-op (mlir-build-operation
                                  (if (zero? rank) "arith.constant" "hipsr.constant")
                                  '() (list !result-type))])
                     (mlir-operation-copy-attr c-op "value" op "value")
                     (mlir-operation-get-result c-op 0))))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
