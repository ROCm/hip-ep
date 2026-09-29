#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Shape → hipsr.placeholder + hipsr.compute (host output)
;;
;; Extracts tensor dimension sizes [start, end) as i64 scalars.
;; Uses mlir-build-operation inside with-op-location to target a specific op as location.
;; so :scheme escapes in region bodies access the OpBuilder without gensyms.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns shape)
  (export populate-shape-patterns
          onnx-shape->hipsr)
  (import (except (rnrs (6)) =)
          (mlir ffi)
          (mlir hipsr)
          (mlir ops)
          (rename (rime loop) (:with :rime-with))
          (mlir pattern-macro))

  (define-conversion-pattern (onnx-shape->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Shape (%input)
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [%ctx        (mlir-get-hipsr-context-arg op)]
         [!input-type (mlir-value-get-type %input)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-host   (mlir-tensor-type-in-host-space !out-type)]
         [input-rank  (mlir-type-get-rank !input-type)]
         [start       (mlir-operation-get-integer-attr op "start" 0)]
         [end-raw     (mlir-operation-get-integer-attr op "end" 0)]
         [end         (if (zero? end-raw) input-rank end-raw)]
         [num-dims    (- end start)]
         [!shape-type (mlir-get-shape-shape-type ctx)]
         [!size-type  (mlir-get-shape-size-type  ctx)]
         [!index-type (mlir-get-index-type       ctx)]
         [!i64-type   (mlir-get-i64-type         ctx)]
         [!ctx-type   (mlir-get-hipsr-context-type ctx)])
    :rewrite %output :with
        ;; Placeholder: shape region yields const shape [num-dims]
        (%placeholder = hipsr.placeholder (%ctx %input !out-host)
                        (operandSegmentSizes = (list 1 1 1) :i32-array)
                        (^bb0 ((%s : !shape-type))
                              (%cN = shape.const_size () (value = num-dims :index) -> !size-type)
                              (%r  = shape.from_extents (%cN) -> !shape-type)
                              (hipsr.shape_yield (%r)))
                        -> !out-host)
        ;; Compute body: for each axis [start, end), emit dim + cast; then from_elements + yield
        (%result = hipsr.compute (%ctx %input %placeholder !out-host)
                   (operandSegmentSizes = (list 1 1 1) :i32-array)
                   (^bb0 ((%c : !ctx-type) (%in : !input-type) (%dest : !out-host))
                         (%dim-vals = (loop :for axis :from start :below end
                                       :collect (with-mlir-ops
                                                  (%ci = arith.constant ()
                                                       (value = axis :index)
                                                       -> !index-type)
                                                  (%d  = tensor.dim  (%in %ci) -> !index-type)
                                                  (%i  = arith.index_cast (%d) -> !i64-type))))
                         (%r = tensor.from_elements (,@%dim-vals) -> !out-host)
                         (hipsr.compute_yield (%r)))
                   -> !out-host))

  (define (populate-shape-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Shape"
                                      onnx-shape->hipsr type-converter))

) ;; end library (patterns shape)
