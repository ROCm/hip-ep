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
;; DenseResourceElementsAttr is constructed via make-mlir-attribute :dense-resource
;; with (list result-type key data-addr data-size).
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

  ;; Mirrors kOrtMemAddrTag in OnnxToHip.cpp.
  (define kOrtMemAddrTag "*/_ORT_MEM_ADDR_/*")

  ;; Build a DenseResourceElementsAttr using make-mlir-attribute :dense-resource.
  ;; Returns attr uptr or 0 on failure.
  (define (build-dense-resource-attr ctx !result-type location offset size)
    (if (string=? location kOrtMemAddrTag)
        ;; ORT in-memory: offset IS the raw address; build key from it.
        (let ([key (string-append "mem|0x" (number->string offset 16))])
          (make-mlir-attribute ctx :dense-resource
            (list !result-type key offset size)))
        ;; File-backed: memory-map the file, get the buffer start address.
        (let ([buf-addr (mlir-hipsr-load-file-map ctx location)])
          (if (zero? buf-addr)
              0
              (let ([key (string-append "file|" location "|"
                                        (number->string offset))])
                (make-mlir-attribute ctx :dense-resource
                  (list !result-type key (+ buf-addr offset) size)))))))

  (define-conversion-pattern (onnx-constant->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
    :then-let
        ([ctx       (mlir-operation-get-context op)]
         [!out-type (mlir-value-get-type %output)]
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
               (mlir-operation-set-attribute! c-op "value"
                 (mlir-operation-get-attribute op "value"))
               (mlir-operation-get-result c-op 0))]
            ;; External data: build DenseResourceElementsAttr, then hipsr.constant
            [(mlir-operation-has-attr? op "location")
             (let* ([location   (mlir-operation-get-attr op "location" :string)]
                    [offset     (mlir-operation-get-attr op "offset"   :i64 0)]
                    [size       (mlir-operation-get-attr op "size"     :i64 0)]
                    [value-attr (build-dense-resource-attr ctx !out-dev
                                   location offset size)])
               (if (zero? value-attr)
                   (begin (mlir-emit-error! op "onnx.Constant: cannot build external resource")
                          #f)
                   (let* ([c-op (mlir-build-operation "hipsr.constant"
                                   '() (list !out-dev))])
                     (mlir-operation-set-attribute! c-op "value" value-attr)
                     (mlir-operation-get-result c-op 0))))]
            ;; Neither — mirrors C++ notifyMatchFailure
            [else
             (begin (mlir-emit-error! op "onnx.Constant has neither value nor location")
                    #f)])))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
