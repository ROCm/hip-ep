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
        ;; Guard: no inline value → emit remark and defer to C++ fallback
        (_ = (if (zero? (mlir-operation-has-attr op "value"))
                 (begin (mlir-emit-remark! op "onnx-constant: no inline value, using fallback")
                        #f)
                 #t))
        ;; Rank 0 → arith.constant (host type); rank > 0 → hipsr.constant (device type)
        (%result = (let* ([!result-type (if (= rank 0) !out-type !out-dev)]
                          [c-op (mlir-build-operation
                                  (if (= rank 0) "arith.constant" "hipsr.constant")
                                  '() (list !result-type))])
                     (mlir-operation-copy-attr c-op "value" op "value")
                     (mlir-operation-get-result c-op 0))))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
