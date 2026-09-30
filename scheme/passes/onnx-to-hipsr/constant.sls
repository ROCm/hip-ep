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
;; Mirrors C++ ConstantOpLowering exactly:
;;   - Inline value (ElementsAttr) → rank-0: arith.constant, rank>0: hipsr.constant
;;   - ORT in-memory ("*/_ORT_MEM_ADDR_/*"): zero-copy DenseResourceElementsAttr
;;   - File-backed: memory-mapped DenseResourceElementsAttr
;;   - Neither value nor location → emit error
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr constant)
  (export populate-constant-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir ddr))

  ;; Mirrors kOrtMemAddrTag in OnnxToHip.cpp.
  (define kOrtMemAddrTag "*/_ORT_MEM_ADDR_/*")

  (define-conversion-pattern (onnx-constant->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
    :then-let
        ([!out-type (mlir-value-get-type %output)]
         [!out-dev  (make-mlir-tensor-in-device-space !out-type)]
         [rank      (mlir-type-get-rank !out-type)])
    :rewrite %output :with
        (%result =
          (cond
            ;; Inline value: rank-0 → arith.constant, rank>0 → hipsr.constant
            [(mlir-operation-has-attr? op "value")
             (let* ([!result-type (if (zero? rank) !out-type !out-dev)]
                    [c-op (mlir-build-operation
                            (if (zero? rank) "arith.constant" "hipsr.constant")
                            '() (list !result-type))])
               (mlir-operation-copy-attr c-op "value" op "value")
               (mlir-operation-get-result c-op 0))]
            ;; External data: ORT in-memory address or file-backed resource
            [(mlir-operation-has-attr? op "location")
             (let* ([location (mlir-operation-get-attr op "location" :string)]
                    [offset   (mlir-operation-get-attr op "offset"   :i64 0)]
                    [size     (mlir-operation-get-attr op "size"     :i64 0)]
                    [result   (if (string=? location kOrtMemAddrTag)
                                  (mlir-build-hipsr-constant-from-ort-mem
                                    rewriter op !out-dev offset size)
                                  (mlir-build-hipsr-constant-from-file
                                    rewriter op !out-dev location offset size))])
               (if (zero? result)
                   (begin (mlir-emit-error! op "onnx.Constant: cannot build external resource")
                          #f)
                   result))]
            ;; Neither — mirrors C++ notifyMatchFailure
            [else
             (begin (mlir-emit-error! op "onnx.Constant has neither value nor location")
                    #f)])))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
