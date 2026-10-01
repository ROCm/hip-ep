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

    ;; Value-level single-use check
    hip-value-single-use?

    ;; C++ wrappers for zero-point and value-bits (require ODS accessors)
    hip-extractable-qdq-zeropoint?
    hip-extract-qdq-zeropoint-i64
    hip-qdq-value-bits-c

    ;; Re-export C++ helpers declared in (mlir core builder)
    hip-extract-splat-scale
    hip-build-init
    hip-create-requantized-layout-op)

  (import (rnrs)
          (only (chezscheme) foreign-procedure nan?)
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

  ;; hip.quantize_linear and hip.dequantize_linear operand layout:
  ;;   4 operands: ctx(0), input(1), scale(2), init(3)          — no zero_point
  ;;   5 operands: ctx(0), input(1), scale(2), zp(3), init(4)   — has zero_point
  ;; Do NOT rely on operandSegmentSizes — it is not present as a named attribute
  ;; on these ops; use operand count instead.

  (define (hip-qdq-has-zeropoint? op)
    (= (mlir-operation-num-operands op) 5))

  (define (hip-qdq-input-operand op)
    ;; input is always at index 1
    (mlir-operation-get-operand-value op 1))

  (define (hip-qdq-scale-operand op)
    ;; scale is always at index 2
    (mlir-operation-get-operand-value op 2))

  (define (hip-qdq-zeropoint op absent-val)
    ;; zp is at index 3 when present (5-operand form); absent → return absent-val
    (if (hip-qdq-has-zeropoint? op)
        (mlir-operation-get-operand-value op 3)
        absent-val))

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
    ;; Returns the INTEGER element type of the quantized tensor:
    ;; - For hip.quantize_linear: result 0 IS the integer tensor.
    ;; - For hip.dequantize_linear: operand 1 (the input) IS the integer tensor;
    ;;   result 0 is float.
    (if (string=? (mlir-operation-name op) "hip.quantize_linear")
        (mlir-type-element-type
          (mlir-value-get-type (mlir-operation-get-result op 0)))
        ;; DQ: the quantized integer tensor is the INPUT operand.
        (mlir-type-element-type
          (mlir-value-get-type (hip-qdq-input-operand op)))))

  (define (hip-qdq-value-bits op)
    ;; packed_int4 is a UnitAttr (presence means true) — use has-attr? to test.
    ;; Do NOT use get-integer-attr; UnitAttr has no integer value, returns 0.
    (if (mlir-operation-has-attr? op "packed_int4")
        4
        (mlir-type-integer-width (hip-qdq-element-type op))))

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

  ;; C++ FFI for the full L2-equivalence check (epsilon=0, trailing axis,
  ;; scale = 1/sqrt(N) rounded to element type precision).
  ;; Pure Scheme cannot replicate the APFloat bit-exact scale comparison.
  (define %hip-is-l2-equiv-rms-norm-c
    (foreign-procedure "hip_is_l2_equiv_rms_norm" (uptr) int))

  (define (hip-l2-equiv-rms-norm? op)
    (not (zero? (%hip-is-l2-equiv-rms-norm-c op))))

  (define %hip-is-fusable-conv-geometry-c
    (foreign-procedure "hip_is_fusable_conv_geometry" (uptr) int))

  (define (hip-fusable-conv-geometry? op)
    ;; 1x1 kernel, unit stride/dilation, zero pads, group=1.
    ;; Delegates to C++ which reads I64ArrayAttr attrs via MLIR APIs.
    (not (zero? (%hip-is-fusable-conv-geometry-c op))))

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

  ;;===--------------------------------------------------------------------===;;
  ;; Value-level single-use helper
  ;;===--------------------------------------------------------------------===;;

  ;; Returns #t when val (a result Value) has exactly one use.
  ;; Complement to hip-op-single-use? — use when you have the Value not the Op.
  (define (hip-value-single-use? val)
    (= (mlir-value-num-uses val) 1))

  ;;===--------------------------------------------------------------------===;;
  ;; C++ wrappers for ODS-accessor-based zero-point extraction
  ;;===--------------------------------------------------------------------===;;

  ;; Returns #t when the zero-point operand of op is absent or a splat constant.
  ;; Uses C++ ODS accessor getZeroPoint() — cannot be expressed with generic FFI.
  (define hip-extractable-qdq-zeropoint?
    (let ([f (foreign-procedure "hip_extractable_qdq_zeropoint" (uptr) int)])
      (lambda (op) (not (zero? (f op))))))

  ;; Extracts zero-point of op as i64. Returns absent-val when absent.
  ;; Returns (most-negative-fixnum) on failure (non-constant zero-point).
  (define hip-extract-qdq-zeropoint-i64
    (foreign-procedure "hip_extract_qdq_zeropoint_i64" (uptr integer-64) integer-64))

  ;; Logical quantized bit-width: 4 if packed_int4, else storage integer width.
  (define hip-qdq-value-bits-c
    (foreign-procedure "hip_qdq_value_bits" (uptr) integer-64))

) ;; end library (mlir hip fusion)
