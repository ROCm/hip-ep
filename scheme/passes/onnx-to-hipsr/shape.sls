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
;; Mirrors ShapeConversion.cpp: uses arith.constant (index) in the shape
;; region and tensor.insert in the compute body, with static-dim detection.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr shape)
  (export populate-shape-patterns
          onnx-shape->hipsr)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects shape)
          (mlir ddr rewrite)
          (rename (rime loop) (:with :rime-with))
          (mlir ddr))

  ;; MLIR uses kDynamic = std::numeric_limits<int64_t>::min() for unknown dims.
  (define (dynamic-dim? d) (< d 0))

  ;; Normalize an ONNX axis bound: add rank for negative, then clamp to [0, rank].
  ;; use-default? controls whether zero means "absent" (-> default-val).
  (define (normalize-bound raw rank use-default? default-val)
    (let* ([v (if (and use-default? (zero? raw)) default-val raw)]
           [v (if (< v 0) (+ v rank) v)])
      (max 0 (min rank v))))

  ;; Build the compute body: for each axis in [start, end), insert the extent
  ;; (static dim as arith.constant i64; dynamic dim via tensor.dim + index_cast)
  ;; into the destination tensor via tensor.insert. Returns the final tensor Value*.
  ;; Mirrors ShapeConversion.cpp::populateComputeBody.
  (define (build-compute-body! in-val dest-val input-shape start end
                                index-type i64-type out-host-type)
    (let loop ([axis start] [slot 0] [acc dest-val])
      (if (>= axis end)
          acc
          (let* ([dim (list-ref input-shape axis)]
                 ;; extent: arith.constant (static) or tensor.dim + index_cast (dynamic)
                 [ext (if (dynamic-dim? dim)
                          (let* ([ci-op (mlir-build-operation "arith.constant"
                                          '() (list index-type))]
                                 [_     (mlir-operation-set-attr ci-op "value" axis :index)]
                                 [ci    (mlir-operation-get-result ci-op 0)]
                                 [d-op  (mlir-build-operation "tensor.dim"
                                          (list in-val ci) (list index-type))]
                                 [d     (mlir-operation-get-result d-op 0)]
                                 [e-op  (mlir-build-operation "arith.index_cast"
                                          (list d) (list i64-type))])
                            (mlir-operation-get-result e-op 0))
                          (let* ([e-op (mlir-build-operation "arith.constant"
                                         '() (list i64-type))]
                                 [_    (mlir-operation-set-attr e-op "value" dim :i64)])
                            (mlir-operation-get-result e-op 0)))]
                 ;; slot constant (= axis - start within the output tensor)
                 [slot-op (mlir-build-operation "arith.constant"
                             '() (list index-type))]
                 [_       (mlir-operation-set-attr slot-op "value" slot :index)]
                 [slot-c  (mlir-operation-get-result slot-op 0)]
                 ;; tensor.insert %ext into %acc[%slot-c]
                 [ins-op  (mlir-build-operation "tensor.insert"
                             (list ext acc slot-c) (list out-host-type))]
                 [ins     (mlir-operation-get-result ins-op 0)])
            (loop (+ axis 1) (+ slot 1) ins)))))

  (define-conversion-pattern (onnx-shape->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Shape (%input)
    :then-let
        ([ctx         (mlir-operation-get-context op)]
         [!input-type (mlir-value-get-type %input)]
         [!out-type   (mlir-value-get-type %output)]
         [!out-host   (mlir-tensor-type-in-host-space !out-type)]
         [input-rank  (mlir-type-get-rank !input-type)]
         [input-shape (mlir-type-get-shape !input-type)]
         [%ctx        (mlir-get-hipsr-context-arg op)]
         [start-raw   (mlir-operation-get-integer-attr op "start" 0)]
         [end-raw     (mlir-operation-get-integer-attr op "end" 0)]
         ;; ONNX normalizes negative bounds by adding rank, then clamps to [0, rank].
         ;; A zero end means "absent" and defaults to the rank.
         [start       (normalize-bound start-raw input-rank #f 0)]
         [end         (normalize-bound end-raw   input-rank #t  input-rank)]
         [num-dims    (- end start)]
         [!shape-type (mlir-get-shape-shape-type ctx)]
         [!index-type (mlir-get-index-type       ctx)]
         [!i64-type   (mlir-get-i64-type         ctx)]
         [!ctx-type   (mlir-get-hipsr-context-type ctx)])
    :rewrite %output :with
        ;; Placeholder: shape region yields const shape [num-dims].
        ;; Uses arith.constant (index) → shape.from_extents, matching
        ;; ShapeConversion.cpp::populateShapeRegion.
        (%placeholder = hipsr.placeholder (%ctx %input !out-host)
                        (^bb0 ((%s : !shape-type))
                              (%cN = arith.constant () (value = num-dims :index) -> !index-type)
                              (%r  = shape.from_extents (%cN) -> !shape-type)
                              (hipsr.shape_yield (%r)))
                        -> !out-host)
        ;; Compute body: inserts extents one by one via tensor.insert.
        ;; build-compute-body! uses mlir-build-operation directly to mix
        ;; Scheme control flow with MLIR op creation.
        (%result = hipsr.compute (%ctx %input %placeholder !out-host)
                   (operandSegmentSizes = (list 1 1 1) :i32-array)
                   (^bb0 ((%c : !ctx-type) (%in : !input-type) (%dest : !out-host))
                         (%final = (build-compute-body!
                                     %in %dest input-shape start end
                                     !index-type !i64-type !out-host))
                         (hipsr.compute_yield (%final)))
                   -> !out-host))

  (define (populate-shape-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Shape"
                                      onnx-shape->hipsr type-converter))

) ;; end library (onnx-to-hipsr shape)
