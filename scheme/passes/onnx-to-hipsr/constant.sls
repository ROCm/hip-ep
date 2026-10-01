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
;; Two patterns, selected by rank via :where guard:
;;   onnx-constant-scalar→arith — rank-0 inline → arith.constant (host type)
;;   onnx-constant-tensor→hipsr — rank>0        → hipsr.constant (device type)
;;
;; External data (location/offset/size) is handled via DenseResourceElementsAttr
;; constructed through make-mlir-attribute :dense-resource.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr constant)
  (export populate-constant-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core attribute)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects tensor)
          (mlir ddr))

  (define kOrtMemAddrTag "*/_ORT_MEM_ADDR_/*")

  ;; Build a value attr (ElementsAttr) for the constant.
  ;; For inline: reads the "value" attr directly.
  ;; For external: constructs DenseResourceElementsAttr.
  ;; Emits an MLIR diagnostic and raises on error — never returns #f.
  (define (constant-value-attr op ctx !result-type)
    (define (fail msg)
      (mlir-emit-error! op msg)
      (error 'onnx-constant msg))
    (cond
      [(mlir-operation-has-attr? op "value")
       (mlir-operation-get-attribute op "value")]
      [(mlir-operation-has-attr? op "location")
       (let* ([location (mlir-operation-get-attr op "location" :string)]
              [offset   (mlir-operation-get-attr op "offset"   :i64 0)]
              [size     (mlir-operation-get-attr op "size"     :i64 0)]
              [r (if (string=? location kOrtMemAddrTag)
                     (make-mlir-attribute ctx :dense-resource
                       (list !result-type
                             (string-append "mem|0x" (number->string offset 16))
                             offset size))
                     (let ([buf (mlir-hipsr-load-file-map ctx location)])
                       (if (zero? buf)
                           (fail (string-append "cannot memory-map: " location))
                           (make-mlir-attribute ctx :dense-resource
                             (list !result-type
                                   (string-append "file|" location "|"
                                                  (number->string offset))
                                   (+ buf offset) size)))))])
         (if (zero? r) (fail "cannot build dense resource attr") r))]
      [else
       (fail "onnx.Constant has neither value nor location")]))

  ;; Pattern 1: rank-0 scalar → arith.constant (host result type)
  (define-conversion-pattern (onnx-constant-scalar->arith op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
            :where (zero? (mlir-type-get-rank (mlir-value-get-type %output)))
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [!out-type   (mlir-value-get-type %output)]
         [$value-attr (constant-value-attr op ctx !out-type)])
    :rewrite %output :with
        (%result = arith.constant () ("value" = $value-attr) -> !out-type))

  ;; Pattern 2: ranked tensor → hipsr.constant (device result type)
  (define-conversion-pattern (onnx-constant-tensor->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Constant ()
            :where (positive? (mlir-type-get-rank (mlir-value-get-type %output)))
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-dev    (mlir-tensor-type-with-encoding !out-type (make-hipsr-device-space-attr (mlir-type-get-context !out-type)))]
         [$value-attr (constant-value-attr op ctx !out-dev)])
    :rewrite %output :with
        (%result = hipsr.constant () ("value" = $value-attr) -> !out-dev))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant-scalar->arith type-converter)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant-tensor->hipsr type-converter))

) ;; end library (onnx-to-hipsr constant)
