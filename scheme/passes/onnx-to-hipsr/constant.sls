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
;; Mirrors C++ ConstantOpLowering:
;;   - Inline value   → rank-0: arith.constant, rank>0: hipsr.constant
;;   - ORT in-memory  → hipsr.constant with DenseResourceElementsAttr (zero-copy)
;;   - File-backed    → hipsr.constant with DenseResourceElementsAttr (mmap)
;;   - Neither        → emit error, pattern failure
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr constant)
  (export populate-constant-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core attribute)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir ddr))

  (define kOrtMemAddrTag "*/_ORT_MEM_ADDR_/*")

  ;; Build a DenseResourceElementsAttr from external constant data.
  ;; location: kOrtMemAddrTag → ORT zero-copy; otherwise an absolute file path.
  ;; Returns attr uptr or 0 on failure.
  (define (build-dense-resource location offset size ctx !result-type)
    (if (string=? location kOrtMemAddrTag)
        (make-mlir-attribute ctx :dense-resource
          (list !result-type
                (string-append "mem|0x" (number->string offset 16))
                offset size))
        (let ([buf-addr (mlir-hipsr-load-file-map ctx location)])
          (if (zero? buf-addr)
              0
              (make-mlir-attribute ctx :dense-resource
                (list !result-type
                      (string-append "file|" location "|" (number->string offset))
                      (+ buf-addr offset) size))))))

  (define-conversion-pattern (onnx-constant->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-dev    (make-mlir-tensor-in-device-space !out-type)]
         [rank        (mlir-type-get-rank !out-type)]
         ;; Compute the value attr up-front; #f means error already emitted.
         [$value-attr (cond
                        [(mlir-operation-has-attr? op "value")
                         (mlir-operation-get-attribute op "value")]
                        [(mlir-operation-has-attr? op "location")
                         (let ([r (build-dense-resource
                                    (mlir-operation-get-attr op "location" :string)
                                    (mlir-operation-get-attr op "offset"   :i64 0)
                                    (mlir-operation-get-attr op "size"     :i64 0)
                                    ctx !out-dev)])
                           (or (and (not (zero? r)) r)
                               (begin (mlir-emit-error! op "onnx.Constant: cannot build resource")
                                      #f)))]
                        [else
                         (begin (mlir-emit-error! op "onnx.Constant has neither value nor location")
                                #f)])])
    :rewrite %output :with
        ;; Guard: fail fast if no value attr was built.
        (_ = $value-attr)
        ;; rank-0 → arith.constant (host type); rank>0 → hipsr.constant (device type)
        (%result = (let* ([!type (if (zero? rank) !out-type !out-dev)]
                          [name  (if (zero? rank) "arith.constant" "hipsr.constant")]
                          [c-op  (mlir-build-operation name '() (list !type))])
                     (mlir-operation-set-attribute! c-op "value" $value-attr)
                     (mlir-operation-get-result c-op 0))))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
