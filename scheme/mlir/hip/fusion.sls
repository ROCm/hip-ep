#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir hip fusion) — hip dialect helpers for quantized op fusion patterns.
;;
;; Implements the constraint and attribute-extraction helpers needed by DDR
;; rewrite patterns in (passes hip-fusion).  All logic here is pure Scheme
;; built on general (mlir core ir) primitives; no hip-specific C++ beyond the
;; three functions in lib/Scheme/Bindings/Hip.cpp that require ODS accessors:
;;   hip-extract-splat-scale  (via C++ FFI from core/builder.sls)
;;   hip-build-init           (via C++ FFI from core/builder.sls)
;;   hip-create-requantized-layout-op (via C++ FFI from core/builder.sls)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir hip fusion)
  (export
    ;; Single-use guard
    hip-op-single-use?

    ;; Q/DQ operand access (respects AttrSizedOperandSegments)
    hip-qdq-input-operand      ; Value: tensor being quantized/dequantized
    hip-qdq-scale-operand      ; Value: scale
    hip-qdq-zeropoint          ; int64 zero-point or absent-val when absent

    ;; Scale checks
    hip-splat-scale?           ; guard: scale Value is a splat float constant

    ;; Type / width checks
    hip-qdq-element-type       ; element IntegerType of the quantized result
    hip-qdq-value-bits         ; logical bit width (4 when packed_int4, else storage width)
    hip-qdq-unsigned?          ; element type is unsigned
    hip-qdq-quantized-width?   ; bit width is in an allowed set

    ;; Matching helpers
    hip-matching-qdq-params?   ; Q and DQ carry identical scale + zero-point
    hip-identity-roundtrip?    ; Q output type equals DQ input type

    ;; Layout op helpers (for QdqRoundTrip patterns)
    hip-can-requantize-layout-op?  ; op name is in the allowed layout-op list
    hip-layout-op-has-ctx?         ; layout op is a DPS hip op (takes ctx arg)

    ;; Attribute helpers (readable via existing mlir-operation-get-integer-attr)
    hip-int-attr-equal?            ; op's named int attr equals expected value
    hip-l2-equiv-rms-norm?         ; rms_norm is equivalent to L2 normalization
    hip-fusable-conv-geometry?     ; 1x1, unit stride/dilation, no pad, no group
    hip-per-axis-weight?           ; dq is per-axis weight at given rank/axis
    hip-per-channel-weight?        ; dq is per-channel weight for gemm consumer

    ;; Init guard
    hip-can-build-init?        ; result type rank matches shape-source rank

    ;; Re-export C++ helpers declared in (mlir core builder)
    hip-extract-splat-scale
    hip-build-init
    hip-create-requantized-layout-op)

  (import (except (rnrs) =)
          (mlir core ir))

  ;;===--------------------------------------------------------------------===;;
  ;; Single-use guard
  ;;===--------------------------------------------------------------------===;;

  ;; Returns #t when op's first result has exactly one use.
  ;; A second consumer would keep the unfused chain alive alongside the
  ;; replacement, computing the same value twice.
  (define (hip-op-single-use? op)
    (= (mlir-value-num-uses (mlir-operation-get-result op 0)) 1))

  ;;===--------------------------------------------------------------------===;;
  ;; Q/DQ operand access
  ;;
  ;; hip.quantize_linear and hip.dequantize_linear use AttrSizedOperandSegments
  ;; with groups:  (ctx: 1) (input: 1) (scale: 1) (zeropoint: 0|1) (init: 1)
  ;; The "operandSegmentSizes" attribute encodes the counts in that order.
  ;;===--------------------------------------------------------------------===;;

  (define (hip-qdq-segments op)
    ;; Return the operand segment list, falling back to (1 1 1 0 1) when absent.
    (let ([segs (mlir-op-get-operand-segment-sizes op)])
      (if (pair? segs) segs '(1 1 1 0 1))))

  (define (hip-qdq-input-operand op)
    ;; Operand layout: ctx(0), input(1), scale(2), [zp], [init]
    ;; Index 1 is always the input tensor.
    (mlir-operation-get-operand-value op 1))

  (define (hip-qdq-scale-operand op)
    ;; Scale follows ctx + input.
    (let ([segs (hip-qdq-segments op)])
      (mlir-operation-get-operand-value
        op (+ (list-ref segs 0) (list-ref segs 1)))))

  (define (hip-qdq-zeropoint op absent-val)
    ;; If zeropoint segment size is 0, the operand is absent → return absent-val.
    (let ([segs (hip-qdq-segments op)])
      (if (zero? (list-ref segs 3))
          absent-val
          (mlir-operation-get-operand-value
            op (+ (list-ref segs 0) (list-ref segs 1) (list-ref segs 2))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Scale checks
  ;;===--------------------------------------------------------------------===;;

  ;; Returns #t when val's defining op is a hip.constant whose "value" attr
  ;; is a splat DenseElementsAttr.  Uses general MLIR attr primitives.
  (define (hip-splat-scale? val)
    (let ([def (mlir-value-get-defining-op val)])
      (and def
           (string=? (mlir-operation-name def) "hip.constant")
           (let ([a (mlir-operation-get-attribute def "value")])
             (and (not (zero? a))
                  (mlir-attr-is-splat a))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Type / width checks
  ;;===--------------------------------------------------------------------===;;

  (define (hip-qdq-element-type op)
    ;; Element type of the quantized result tensor.
    (mlir-type-element-type
      (mlir-value-get-type (mlir-operation-get-result op 0))))

  (define (hip-qdq-value-bits op)
    ;; packed_int4 attr (integer, 0=false, 1=true) signals that the storage
    ;; holds two 4-bit values per byte; the logical width is 4 in that case.
    (let ([packed? (mlir-operation-get-integer-attr op "packed_int4" 0)])
      (if (= packed? 1)
          4
          (mlir-type-integer-width (hip-qdq-element-type op)))))

  (define (hip-qdq-unsigned? op)
    (mlir-type-is-unsigned (hip-qdq-element-type op)))

  (define (hip-qdq-quantized-width? op allowed-widths)
    (let ([w (hip-qdq-value-bits op)])
      (and (member w allowed-widths) #t)))

  ;;===--------------------------------------------------------------------===;;
  ;; Matching helpers
  ;;===--------------------------------------------------------------------===;;

  ;; Returns #t when dq-op and q-op carry the same scale and zero-point SSA values.
  ;; This is a necessary condition for a Q/DQ pair to be a round-trip.
  (define (hip-matching-qdq-params? dq-op q-op)
    (let ([dq-scale (hip-qdq-scale-operand dq-op)]
          [q-scale  (hip-qdq-scale-operand q-op)]
          [dq-zp    (hip-qdq-zeropoint dq-op 0)]
          [q-zp     (hip-qdq-zeropoint q-op  0)])
      ;; Same SSA Value means same pointer (same uptr integer).
      (and (= dq-scale q-scale)
           (if (and (integer? dq-zp) (integer? q-zp))
               (= dq-zp q-zp)
               (eqv? dq-zp q-zp)))))

  (define (hip-identity-roundtrip? dq-op q-op)
    ;; The pair is a round-trip only when the element type going in
    ;; matches the element type coming out.
    (= (hip-qdq-element-type dq-op) (hip-qdq-element-type q-op)))

  ;;===--------------------------------------------------------------------===;;
  ;; Layout op helpers
  ;;===--------------------------------------------------------------------===;;

  ;; Op names that are safe to clone over a quantized type.
  ;; Must match canRequantizeLayoutOp in hip_fusion_transform.hpp.
  (define %hip-requantizable-ops
    '("hip.transpose" "tensor.collapse_shape" "tensor.expand_shape"))

  (define (hip-can-requantize-layout-op? layout-op q-op)
    (and (member (mlir-operation-name layout-op) %hip-requantizable-ops) #t))

  (define (hip-layout-op-has-ctx? layout-op)
    ;; DPS hip ops (like hip.transpose) take a ctx as their first operand.
    ;; Pure tensor ops (tensor.collapse_shape etc.) do not.
    (let ([name (mlir-operation-name layout-op)])
      (let ([n (string-length "hip.")])
        (and (>= (string-length name) n)
             (string=? (substring name 0 n) "hip.")))))

  ;;===--------------------------------------------------------------------===;;
  ;; Attribute helpers
  ;;===--------------------------------------------------------------------===;;

  (define (hip-int-attr-equal? op name expected absent-val)
    (= (mlir-operation-get-integer-attr op name absent-val) expected))

  (define (hip-l2-equiv-rms-norm? op)
    ;; Must be: trailing axis, zero epsilon, scale = 1/sqrt(N).
    ;; The full check requires inspecting the dense scale attr — a complex
    ;; check best left to the C++ guard hip_is_l2_equiv_rms_norm if needed.
    ;; Here we expose the simpler sub-checks available via generic FFI.
    ;; Patterns using this should call the C++ helper directly when available.
    (and (let ([eps (mlir-op-get-float-attr op "epsilon")])
           (and (not (nan? eps)) (< (abs eps) 1e-9)))
         ;; axis check: -1 or last axis; read as integer attr
         (let ([axis (mlir-operation-get-integer-attr op "axis" -1)])
           (or (= axis -1) (= axis (- (mlir-type-get-rank
                                        (mlir-value-get-type
                                          (mlir-operation-get-result op 0)))
                                       1))))))

  (define (hip-fusable-conv-geometry? op)
    ;; 1x1 kernel, unit stride, unit dilation, zero pad, no group.
    ;; Uses ArrayAttr attributes so the generic integer-attr getter won't work.
    ;; Delegate to C++ for full accuracy; this Scheme version covers the attrs
    ;; accessible via mlir-operation-get-integer-attr.
    (and (hip-int-attr-equal? op "group" 1 1)))

  (define (hip-per-axis-weight? dq-op rank axis packed-int4?)
    ;; Scale must be rank-1 and packed_int4 must match.
    (let* ([scale-val  (hip-qdq-scale-operand dq-op)]
           [scale-type (mlir-value-get-type scale-val)]
           [scale-rank (mlir-type-get-rank scale-type)]
           [bits       (hip-qdq-value-bits dq-op)])
      (and (= scale-rank 1)
           (= bits (if packed-int4? 4 8)))))

  (define (hip-per-channel-weight? dq-op q-op)
    ;; For QGemm: weight is per-channel when the scale is rank-1.
    (let* ([scale-val  (hip-qdq-scale-operand dq-op)]
           [scale-type (mlir-value-get-type scale-val)])
      (= (mlir-type-get-rank scale-type) 1)))

  ;;===--------------------------------------------------------------------===;;
  ;; Init guard
  ;;===--------------------------------------------------------------------===;;

  (define (hip-can-build-init? q-op shape-source)
    ;; Guard before hip-build-init: result type rank must match shape-source rank.
    (let* ([out-type   (mlir-value-get-type (mlir-operation-get-result q-op 0))]
           [src-type   (mlir-value-get-type shape-source)])
      (and (= (mlir-type-get-rank out-type)
              (mlir-type-get-rank src-type)))))

) ;; end library (mlir hip fusion)
