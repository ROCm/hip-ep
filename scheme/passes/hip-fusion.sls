#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (passes hip-fusion) — hip quantized-op fusion patterns in Scheme DDR
;;
;; Ports the PDLL patterns from lib/Dialect/Transforms/fusion_pattern/ to
;; Scheme define-rewrite-pattern.  All constraints are expressed as Scheme
;; guards using (mlir hip fusion) helpers; no new C++ required beyond the
;; three ODS-accessor wrappers committed in Hip.cpp.
;;
;; Entry point: (run-pass module-op)
;;   Registers all patterns and applies them greedily.
;;
;; Pattern inventory (19 total):
;;   Q-fusion (12):
;;     hip-qadd-fusion, hip-qmul-fusion
;;     hip-qmatmul-fusion (per-tensor), hip-qmatmul-per-col-w4, hip-qmatmul-per-col-w8
;;     hip-qgemm-fusion, hip-qgemm-no-bias
;;     hip-qgemm-per-channel, hip-qgemm-no-bias-per-channel
;;     hip-qconv-fusion, hip-qsigmoid-fusion, hip-qlpnorm-fusion
;;   QdqRoundTrip (3):
;;     hip-qdq-roundtrip-dps, hip-qdq-roundtrip-tensor, hip-qdq-roundtrip-pair
;;
;;===----------------------------------------------------------------------===;;

(library (passes hip-fusion)
  (export run-pass)
  (import (except (rnrs) =)
          (only (chezscheme) nan? foreign-procedure)
          (rename (only (rnrs) =) (= num=))
          (mlir core ir)
          (mlir core conversion)
          (mlir hip fusion)
          (mlir ddr))

  ;;===--------------------------------------------------------------------===;;
  ;; Shared rewrite helper
  ;;===--------------------------------------------------------------------===;;
  ;;
  ;; Set all quantization scalar attributes on a freshly created fused op.
  ;; All attribute names follow the hip.q* ODS convention.

  (define (set-qdq-scale-zp-attrs! new-op
                                   lhs-scale lhs-zp
                                   rhs-scale rhs-zp
                                   out-scale out-zp)
    (mlir-operation-set-f32-attr! new-op "lhs_scale"    lhs-scale)
    (mlir-operation-set-f32-attr! new-op "rhs_scale"    rhs-scale)
    (mlir-operation-set-f32-attr! new-op "output_scale" out-scale)
    (mlir-operation-set-i64-attr! new-op "lhs_zp"       lhs-zp)
    (mlir-operation-set-i64-attr! new-op "rhs_zp"       rhs-zp)
    (mlir-operation-set-i64-attr! new-op "output_zp"    out-zp))

  (define (set-qdq-in-out-attrs! new-op in-scale in-zp out-scale out-zp)
    (mlir-operation-set-f32-attr! new-op "input_scale"  in-scale)
    (mlir-operation-set-i64-attr! new-op "input_zp"     in-zp)
    (mlir-operation-set-f32-attr! new-op "output_scale" out-scale)
    (mlir-operation-set-i64-attr! new-op "output_zp"    out-zp))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 1: QAdd  (benefit 10)
  ;; hip.dequantize_linear x2 → hip.add → hip.quantize_linear → hip.qadd
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qadd-fusion op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %add %out_scale)
        %add    = hip.add               (%ctx %dq_lhs %dq_rhs %add_init)
        %dq_lhs = hip.dequantize_linear (%ctx %lhs %lhs_scale)
        %dq_rhs = hip.dequantize_linear (%ctx %rhs %rhs_scale)
              :where (and (hip-value-single-use? %add)
                (hip-splat-scale? %lhs_scale)
                (hip-splat-scale? %rhs_scale)
                (hip-splat-scale? %out_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_lhs))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_rhs))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %add_init))
    :then-let
        ([!out-type  (mlir-value-get-type %q)]
         [%dq-lhs-op (mlir-value-get-defining-op %dq_lhs)]
         [%dq-rhs-op (mlir-value-get-defining-op %dq_rhs)]
         [lhs-scale  (hip-extract-splat-scale %lhs_scale)]
         [rhs-scale  (hip-extract-splat-scale %rhs_scale)]
         [out-scale  (hip-extract-splat-scale %out_scale)]
         [lhs-zp     (hip-extract-qdq-zeropoint-i64 %dq-lhs-op 0)]
         [rhs-zp     (hip-extract-qdq-zeropoint-i64 %dq-rhs-op 0)]
         [out-zp     (hip-extract-qdq-zeropoint-i64 op 0)]
         [%init      (hip-build-init rewriter !out-type %add_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qadd"
                                   (list %ctx %lhs %rhs %init)
                                   (list !out-type))])
                     (set-qdq-scale-zp-attrs! new-op lhs-scale lhs-zp
                                                     rhs-scale rhs-zp
                                                     out-scale out-zp)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 2: QMul  (benefit 10)
  ;; hip.dequantize_linear x2 → hip.mul → hip.quantize_linear → hip.qmul
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmul-fusion op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %mul %out_scale)
        %mul    = hip.mul               (%ctx %dq_lhs %dq_rhs %mul_init)
        %dq_lhs = hip.dequantize_linear (%ctx %lhs %lhs_scale)
        %dq_rhs = hip.dequantize_linear (%ctx %rhs %rhs_scale)
              :where (and (hip-value-single-use? %mul)
                (hip-splat-scale? %lhs_scale)
                (hip-splat-scale? %rhs_scale)
                (hip-splat-scale? %out_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_lhs))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_rhs))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %mul_init))
    :then-let
        ([!out-type  (mlir-value-get-type %q)]
         [%dq-lhs-op (mlir-value-get-defining-op %dq_lhs)]
         [%dq-rhs-op (mlir-value-get-defining-op %dq_rhs)]
         [lhs-scale  (hip-extract-splat-scale %lhs_scale)]
         [rhs-scale  (hip-extract-splat-scale %rhs_scale)]
         [out-scale  (hip-extract-splat-scale %out_scale)]
         [lhs-zp     (hip-extract-qdq-zeropoint-i64 %dq-lhs-op 0)]
         [rhs-zp     (hip-extract-qdq-zeropoint-i64 %dq-rhs-op 0)]
         [out-zp     (hip-extract-qdq-zeropoint-i64 op 0)]
         [%init      (hip-build-init rewriter !out-type %mul_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qmul"
                                   (list %ctx %lhs %rhs %init)
                                   (list !out-type))])
                     (set-qdq-scale-zp-attrs! new-op lhs-scale lhs-zp
                                                     rhs-scale rhs-zp
                                                     out-scale out-zp)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 3: QMatMul per-tensor  (benefit 10)
  ;; hip.dequantize_linear x2 → hip.matmul → hip.quantize_linear → hip.qmatmul
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmatmul-fusion op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %matmul %y_scale)
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scale)
              :where (and (hip-value-single-use? %matmul)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_b) '(8))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %b_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_b))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %matmul_init))
    :then-let
        ([!y-type    (mlir-value-get-type %q)]
         [%dq-a-op   (mlir-value-get-defining-op %dq_a)]
         [%dq-b-op   (mlir-value-get-defining-op %dq_b)]
         [%mm-op     (mlir-value-get-defining-op %matmul)]
         [a-scale    (hip-extract-splat-scale %a_scale)]
         [b-scale    (hip-extract-splat-scale %b_scale)]
         [y-scale    (hip-extract-splat-scale %y_scale)]
         [a-zp       (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [b-zp       (hip-extract-qdq-zeropoint-i64 %dq-b-op 0)]
         [y-zp       (hip-extract-qdq-zeropoint-i64 op 0)]
         [trans-a    (mlir-operation-get-integer-attr %mm-op "transA" 0)]
         [trans-b    (mlir-operation-get-integer-attr %mm-op "transB" 0)]
         [%init      (hip-build-init rewriter !y-type %matmul_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qmatmul"
                                   (list %ctx %a %b %init)
                                   (list !y-type))])
                     (mlir-operation-set-f32-attr! new-op "A_scale"       a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point"  a-zp)
                     (mlir-operation-set-f32-attr! new-op "B_scale"       b-scale)
                     (mlir-operation-set-i64-attr! new-op "B_zero_point"  b-zp)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"       y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point"  y-zp)
                     (mlir-operation-set-i64-attr! new-op "transA"        trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"        trans-b)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 4: QMatMul per-column W4  (benefit 9)
  ;; Per-column weight with packed 4-bit storage
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmatmul-per-col-w4 op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %matmul %y_scale)
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
              :where (and (hip-value-single-use? %matmul)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-per-axis-weight? (mlir-value-get-defining-op %dq_b) 2 1 #t)
                (num= (mlir-operation-get-integer-attr
                     (mlir-value-get-defining-op %matmul) "transB" 0) 0)
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %matmul_init))
    :then-let
        ([!y-type    (mlir-value-get-type %q)]
         [%dq-a-op   (mlir-value-get-defining-op %dq_a)]
         [%mm-op     (mlir-value-get-defining-op %matmul)]
         [a-scale    (hip-extract-splat-scale %a_scale)]
         [y-scale    (hip-extract-splat-scale %y_scale)]
         [a-zp       (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [y-zp       (hip-extract-qdq-zeropoint-i64 op 0)]
         [trans-a    (mlir-operation-get-integer-attr %mm-op "transA" 0)]
         [%init      (hip-build-init rewriter !y-type %matmul_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qmatmul"
                                   (list %ctx %a %b %b_scales %b_zps %init)
                                   (list !y-type))])
                     (mlir-operation-set-f32-attr! new-op "A_scale"       a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point"  a-zp)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"       y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point"  y-zp)
                     (mlir-operation-set-i64-attr! new-op "transA"        trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"        0)
                     (mlir-operation-set-i64-attr! new-op "B_quant_axis"  1)
                     (mlir-operation-set-unit-attr! new-op "packed_int4")
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 5: QMatMul per-column W8  (benefit 9)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmatmul-per-col-w8 op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %matmul %y_scale)
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
              :where (and (hip-value-single-use? %matmul)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-per-axis-weight? (mlir-value-get-defining-op %dq_b) 2 1 #f)
                (num= (mlir-operation-get-integer-attr
                     (mlir-value-get-defining-op %matmul) "transB" 0) 0)
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %matmul_init))
    :then-let
        ([!y-type    (mlir-value-get-type %q)]
         [%dq-a-op   (mlir-value-get-defining-op %dq_a)]
         [%mm-op     (mlir-value-get-defining-op %matmul)]
         [a-scale    (hip-extract-splat-scale %a_scale)]
         [y-scale    (hip-extract-splat-scale %y_scale)]
         [a-zp       (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [y-zp       (hip-extract-qdq-zeropoint-i64 op 0)]
         [trans-a    (mlir-operation-get-integer-attr %mm-op "transA" 0)]
         [%init      (hip-build-init rewriter !y-type %matmul_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qmatmul"
                                   (list %ctx %a %b %b_scales %b_zps %init)
                                   (list !y-type))])
                     (mlir-operation-set-f32-attr! new-op "A_scale"       a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point"  a-zp)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"       y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point"  y-zp)
                     (mlir-operation-set-i64-attr! new-op "transA"        trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"        0)
                     (mlir-operation-set-i64-attr! new-op "B_quant_axis"  1)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 6: QGemm per-tensor with bias  (benefit 10)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qgemm-fusion op rewriter)
    :if-match
        %q     = hip.quantize_linear   (%ctx %gemm %y_scale)
        %gemm  = hip.gemm              (%ctx %dq_a %dq_b %dq_c %gemm_init)
        %dq_a  = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b  = hip.dequantize_linear (%ctx %b %b_scale)
        %dq_c  = hip.dequantize_linear (%ctx %c %c_scale)
              :where (and (hip-value-single-use? %gemm)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_b) '(8))
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_c) '(8 16 32))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %b_scale)
                (hip-splat-scale? %c_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_b))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_c))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %gemm_init))
    :then-let
        ([!y-type   (mlir-value-get-type %q)]
         [%dq-a-op  (mlir-value-get-defining-op %dq_a)]
         [%dq-b-op  (mlir-value-get-defining-op %dq_b)]
         [%dq-c-op  (mlir-value-get-defining-op %dq_c)]
         [%gemm-op  (mlir-value-get-defining-op %gemm)]
         [a-scale   (hip-extract-splat-scale %a_scale)]
         [b-scale   (hip-extract-splat-scale %b_scale)]
         [c-scale   (hip-extract-splat-scale %c_scale)]
         [y-scale   (hip-extract-splat-scale %y_scale)]
         [a-zp      (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [b-zp      (hip-extract-qdq-zeropoint-i64 %dq-b-op 0)]
         [c-zp      (hip-extract-qdq-zeropoint-i64 %dq-c-op 0)]
         [y-zp      (hip-extract-qdq-zeropoint-i64 op 0)]
         [b-bits    (hip-qdq-value-bits-c %dq-b-op)]
         [alpha     (mlir-op-get-float-attr %gemm-op "alpha")]
         [beta      (mlir-op-get-float-attr %gemm-op "beta")]
         [trans-a   (mlir-operation-get-integer-attr %gemm-op "transA" 0)]
         [trans-b   (mlir-operation-get-integer-attr %gemm-op "transB" 0)]
         [%init     (hip-build-init rewriter !y-type %gemm_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qgemm"
                                   (list %ctx %a %b %c %init)
                                   (list !y-type))])
                     ;; operandSegmentSizes: ctx=1 A=1 B=1 B_scales=0 B_zps=0 C=1 Y=1
                     (mlir-operation-set-dense-i32-array! new-op "operandSegmentSizes"
                                                          '(1 1 1 0 0 1 1))
                     (mlir-operation-set-f32-attr! new-op "A_scale"      a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point" a-zp)
                     (mlir-operation-set-f32-attr! new-op "B_scale"      b-scale)
                     (mlir-operation-set-i64-attr! new-op "B_zero_point" b-zp)
                     (mlir-operation-set-i64-attr! new-op "B_bits"       b-bits)
                     (mlir-operation-set-f32-attr! new-op "C_scale"      c-scale)
                     (mlir-operation-set-i64-attr! new-op "C_zero_point" c-zp)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"      y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point" y-zp)
                     (unless (nan? alpha) (mlir-operation-set-f32-attr! new-op "alpha" alpha))
                     (unless (nan? beta)  (mlir-operation-set-f32-attr! new-op "beta"  beta))
                     (mlir-operation-set-i64-attr! new-op "transA"       trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"       trans-b)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 7: QGemm per-tensor no bias  (benefit 10)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qgemm-no-bias op rewriter)
    :if-match
        %q    = hip.quantize_linear   (%ctx %gemm %y_scale)
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %gemm_init)
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b = hip.dequantize_linear (%ctx %b %b_scale)
              :where (and (hip-value-single-use? %gemm)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_b) '(8))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %b_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_b))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %gemm_init))
    :then-let
        ([!y-type  (mlir-value-get-type %q)]
         [%dq-a-op (mlir-value-get-defining-op %dq_a)]
         [%dq-b-op (mlir-value-get-defining-op %dq_b)]
         [%gemm-op (mlir-value-get-defining-op %gemm)]
         [a-scale  (hip-extract-splat-scale %a_scale)]
         [b-scale  (hip-extract-splat-scale %b_scale)]
         [y-scale  (hip-extract-splat-scale %y_scale)]
         [a-zp     (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [b-zp     (hip-extract-qdq-zeropoint-i64 %dq-b-op 0)]
         [y-zp     (hip-extract-qdq-zeropoint-i64 op 0)]
         [b-bits   (hip-qdq-value-bits-c %dq-b-op)]
         [alpha    (mlir-op-get-float-attr %gemm-op "alpha")]
         [trans-a  (mlir-operation-get-integer-attr %gemm-op "transA" 0)]
         [trans-b  (mlir-operation-get-integer-attr %gemm-op "transB" 0)]
         [%init    (hip-build-init rewriter !y-type %gemm_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qgemm"
                                   (list %ctx %a %b %init)
                                   (list !y-type))])
                     ;; operandSegmentSizes: ctx=1 A=1 B=1 B_scales=0 B_zps=0 C=0 Y=1
                     (mlir-operation-set-dense-i32-array! new-op "operandSegmentSizes"
                                                          '(1 1 1 0 0 0 1))
                     (mlir-operation-set-f32-attr! new-op "A_scale"      a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point" a-zp)
                     (mlir-operation-set-f32-attr! new-op "B_scale"      b-scale)
                     (mlir-operation-set-i64-attr! new-op "B_zero_point" b-zp)
                     (mlir-operation-set-i64-attr! new-op "B_bits"       b-bits)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"      y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point" y-zp)
                     (unless (nan? alpha) (mlir-operation-set-f32-attr! new-op "alpha" alpha))
                     (mlir-operation-set-i64-attr! new-op "transA"       trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"       trans-b)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 8: QGemm per-channel weight with bias  (benefit 9)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qgemm-per-channel op rewriter)
    :if-match
        %q    = hip.quantize_linear   (%ctx %gemm %y_scale)
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %dq_c %gemm_init)
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
        %dq_c = hip.dequantize_linear (%ctx %c %c_scale)
              :where (and (hip-value-single-use? %gemm)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_c) '(8 16 32))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-per-channel-weight? (mlir-value-get-defining-op %dq_b)
                                         (mlir-value-get-defining-op %gemm))
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %c_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_c))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %gemm_init))
    :then-let
        ([!y-type  (mlir-value-get-type %q)]
         [%dq-a-op (mlir-value-get-defining-op %dq_a)]
         [%dq-b-op (mlir-value-get-defining-op %dq_b)]
         [%dq-c-op (mlir-value-get-defining-op %dq_c)]
         [%gemm-op (mlir-value-get-defining-op %gemm)]
         [a-scale  (hip-extract-splat-scale %a_scale)]
         [c-scale  (hip-extract-splat-scale %c_scale)]
         [y-scale  (hip-extract-splat-scale %y_scale)]
         [a-zp     (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [c-zp     (hip-extract-qdq-zeropoint-i64 %dq-c-op 0)]
         [y-zp     (hip-extract-qdq-zeropoint-i64 op 0)]
         [b-bits   (hip-qdq-value-bits-c %dq-b-op)]
         [alpha    (mlir-op-get-float-attr %gemm-op "alpha")]
         [beta     (mlir-op-get-float-attr %gemm-op "beta")]
         [trans-a  (mlir-operation-get-integer-attr %gemm-op "transA" 0)]
         [trans-b  (mlir-operation-get-integer-attr %gemm-op "transB" 0)]
         [%init    (hip-build-init rewriter !y-type %gemm_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qgemm"
                                   (list %ctx %a %b %b_scales %b_zps %c %init)
                                   (list !y-type))])
                     ;; operandSegmentSizes: ctx=1 A=1 B=1 B_scales=1 B_zps=1 C=1 Y=1
                     (mlir-operation-set-dense-i32-array! new-op "operandSegmentSizes"
                                                          '(1 1 1 1 1 1 1))
                     (mlir-operation-set-f32-attr! new-op "A_scale"      a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point" a-zp)
                     (mlir-operation-set-i64-attr! new-op "B_bits"       b-bits)
                     (mlir-operation-set-f32-attr! new-op "C_scale"      c-scale)
                     (mlir-operation-set-i64-attr! new-op "C_zero_point" c-zp)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"      y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point" y-zp)
                     (unless (nan? alpha) (mlir-operation-set-f32-attr! new-op "alpha" alpha))
                     (unless (nan? beta)  (mlir-operation-set-f32-attr! new-op "beta"  beta))
                     (mlir-operation-set-i64-attr! new-op "transA"       trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"       trans-b)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 9: QGemm per-channel no bias  (benefit 9)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qgemm-no-bias-per-channel op rewriter)
    :if-match
        %q    = hip.quantize_linear   (%ctx %gemm %y_scale)
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %gemm_init)
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
        %dq_b = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
              :where (and (hip-value-single-use? %gemm)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_a) '(8 16))
                (hip-qdq-quantized-width? op '(8 16))
                (hip-per-channel-weight? (mlir-value-get-defining-op %dq_b)
                                         (mlir-value-get-defining-op %gemm))
                (hip-splat-scale? %a_scale)
                (hip-splat-scale? %y_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_a))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %gemm_init))
    :then-let
        ([!y-type  (mlir-value-get-type %q)]
         [%dq-a-op (mlir-value-get-defining-op %dq_a)]
         [%dq-b-op (mlir-value-get-defining-op %dq_b)]
         [%gemm-op (mlir-value-get-defining-op %gemm)]
         [a-scale  (hip-extract-splat-scale %a_scale)]
         [y-scale  (hip-extract-splat-scale %y_scale)]
         [a-zp     (hip-extract-qdq-zeropoint-i64 %dq-a-op 0)]
         [y-zp     (hip-extract-qdq-zeropoint-i64 op 0)]
         [b-bits   (hip-qdq-value-bits-c %dq-b-op)]
         [alpha    (mlir-op-get-float-attr %gemm-op "alpha")]
         [trans-a  (mlir-operation-get-integer-attr %gemm-op "transA" 0)]
         [trans-b  (mlir-operation-get-integer-attr %gemm-op "transB" 0)]
         [%init    (hip-build-init rewriter !y-type %gemm_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qgemm"
                                   (list %ctx %a %b %b_scales %b_zps %init)
                                   (list !y-type))])
                     ;; operandSegmentSizes: ctx=1 A=1 B=1 B_scales=1 B_zps=1 C=0 Y=1
                     (mlir-operation-set-dense-i32-array! new-op "operandSegmentSizes"
                                                          '(1 1 1 1 1 0 1))
                     (mlir-operation-set-f32-attr! new-op "A_scale"      a-scale)
                     (mlir-operation-set-i64-attr! new-op "A_zero_point" a-zp)
                     (mlir-operation-set-i64-attr! new-op "B_bits"       b-bits)
                     (mlir-operation-set-f32-attr! new-op "Y_scale"      y-scale)
                     (mlir-operation-set-i64-attr! new-op "Y_zero_point" y-zp)
                     (unless (nan? alpha) (mlir-operation-set-f32-attr! new-op "alpha" alpha))
                     (mlir-operation-set-i64-attr! new-op "transA"       trans-a)
                     (mlir-operation-set-i64-attr! new-op "transB"       trans-b)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 10: QConv  (benefit 10)
  ;; dq input + per-channel packed-int4 weights → hip.conv → q → hip.qconv
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qconv-fusion op rewriter)
    :if-match
        %q          = hip.quantize_linear   (%ctx %conv %out_scale)
        %conv       = hip.conv              (%ctx %dq_in %dq_w %conv_init)
        %dq_in      = hip.dequantize_linear (%ctx %input %in_scale)
        %dq_w       = hip.dequantize_linear (%ctx %weights %w_scales %w_zps %w_init)
              :where (and (hip-value-single-use? %conv)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_in) '(16))
                (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                (hip-qdq-quantized-width? op '(16))
                (hip-qdq-unsigned? op)
                (hip-per-axis-weight? (mlir-value-get-defining-op %dq_w) 4 0 #t)
                (hip-fusable-conv-geometry? (mlir-value-get-defining-op %conv))
                (hip-splat-scale? %in_scale)
                (hip-splat-scale? %out_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_in))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %conv_init))
    :then-let
        ([!out-type (mlir-value-get-type %q)]
         [%dq-in-op (mlir-value-get-defining-op %dq_in)]
         [in-scale  (hip-extract-splat-scale %in_scale)]
         [out-scale (hip-extract-splat-scale %out_scale)]
         [in-zp     (hip-extract-qdq-zeropoint-i64 %dq-in-op 0)]
         [out-zp    (hip-extract-qdq-zeropoint-i64 op 0)]
         [%init     (hip-build-init rewriter !out-type %conv_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qconv"
                                   (list %ctx %input %weights %w_scales %w_zps %init)
                                   (list !out-type))])
                     (mlir-operation-set-f32-attr! new-op "input_scale"   in-scale)
                     (mlir-operation-set-i64-attr! new-op "input_zp"      in-zp)
                     (mlir-operation-set-f32-attr! new-op "output_scale"  out-scale)
                     (mlir-operation-set-i64-attr! new-op "output_zp"     out-zp)
                     (mlir-operation-set-i64-attr! new-op "weight_axis"   0)
                     (mlir-operation-set-dense-i64-array! new-op "kernel_shape" '(1 1))
                     (mlir-operation-set-dense-i64-array! new-op "strides"      '(1 1))
                     (mlir-operation-set-dense-i64-array! new-op "pads"         '(0 0 0 0))
                     (mlir-operation-set-dense-i64-array! new-op "dilations"    '(1 1))
                     (mlir-operation-set-i64-attr! new-op "group"          1)
                     (mlir-operation-set-unit-attr! new-op "packed_int4")
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 11: QSigmoid  (benefit 10)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qsigmoid-fusion op rewriter)
    :if-match
        %q       = hip.quantize_linear   (%ctx %sigmoid %out_scale)
        %sigmoid = hip.sigmoid           (%ctx %dq_in %sigmoid_init)
        %dq_in   = hip.dequantize_linear (%ctx %input %in_scale)
              :where (and (hip-value-single-use? %sigmoid)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_in) '(16))
                (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                (hip-qdq-quantized-width? op '(16))
                (hip-qdq-unsigned? op)
                (hip-splat-scale? %in_scale)
                (hip-splat-scale? %out_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_in))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %sigmoid_init))
    :then-let
        ([!out-type (mlir-value-get-type %q)]
         [%dq-in-op (mlir-value-get-defining-op %dq_in)]
         [in-scale  (hip-extract-splat-scale %in_scale)]
         [out-scale (hip-extract-splat-scale %out_scale)]
         [in-zp     (hip-extract-qdq-zeropoint-i64 %dq-in-op 0)]
         [out-zp    (hip-extract-qdq-zeropoint-i64 op 0)]
         [%init     (hip-build-init rewriter !out-type %sigmoid_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qsigmoid"
                                   (list %ctx %input %init)
                                   (list !out-type))])
                     (set-qdq-in-out-attrs! new-op in-scale in-zp out-scale out-zp)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 12: QLpNormalization  (benefit 10)
  ;; dq → hip.rms_norm → q → hip.qlpnormalization (L2/RMS equivalence)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qlpnorm-fusion op rewriter)
    :if-match
        %q    = hip.quantize_linear   (%ctx %rms %out_scale)
        %rms  = hip.rms_norm          (%ctx %dq_in %rms_scale %rms_init)
        %dq_in = hip.dequantize_linear (%ctx %input %in_scale)
              :where (and (hip-value-single-use? %rms)
                (hip-qdq-quantized-width? (mlir-value-get-defining-op %dq_in) '(16))
                (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                (hip-qdq-quantized-width? op '(16))
                (hip-qdq-unsigned? op)
                (hip-l2-equiv-rms-norm? (mlir-value-get-defining-op %rms))
                (hip-splat-scale? %in_scale)
                (hip-splat-scale? %out_scale)
                (hip-extractable-qdq-zeropoint? (mlir-value-get-defining-op %dq_in))
                (hip-extractable-qdq-zeropoint? op)
                (hip-can-build-init? op %rms_init))
    :then-let
        ([!out-type (mlir-value-get-type %q)]
         [%dq-in-op (mlir-value-get-defining-op %dq_in)]
         [in-scale  (hip-extract-splat-scale %in_scale)]
         [out-scale (hip-extract-splat-scale %out_scale)]
         [in-zp     (hip-extract-qdq-zeropoint-i64 %dq-in-op 0)]
         [out-zp    (hip-extract-qdq-zeropoint-i64 op 0)]
         [%init     (hip-build-init rewriter !out-type %rms_init)])
    :rewrite %q :with
        (%result = (let ([new-op (mlir-build-operation "hip.qlpnormalization"
                                   (list %ctx %input %init)
                                   (list !out-type))])
                     (set-qdq-in-out-attrs! new-op in-scale in-zp out-scale out-zp)
                     (mlir-operation-set-i64-attr! new-op "axis" -1)
                     (mlir-operation-set-i64-attr! new-op "p"    2)
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 13: QdqRoundTrip — hip DPS layout op  (benefit 10)
  ;; Q(LAYOUT_DPS(DQ(x))) → LAYOUT_DPS(x) when params match
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-dps op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %layout %out_scale)
        %layout = :any                  (%ctx %dq)
        %dq     = hip.dequantize_linear (%ctx %input %in_scale)
              :where (and (hip-value-single-use? %layout)
                (hip-matching-qdq-params? (mlir-value-get-defining-op %dq) op)
                (hip-can-requantize-layout-op? (mlir-value-get-defining-op %layout) op))
    :then-let
        ([%dq-op     (mlir-value-get-defining-op %dq)]
         [%layout-op (mlir-value-get-defining-op %layout)])
    :rewrite %q :with
        (%result = (hip-create-requantized-layout-op rewriter %dq-op %layout-op op)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 14: QdqRoundTrip — tensor layout op  (benefit 10)
  ;; Q(LAYOUT_TENSOR(DQ(x))) → LAYOUT_TENSOR(x)  (no ctx on layout)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-tensor op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %layout %out_scale)
        %layout = :any                  (%dq)
        %dq     = hip.dequantize_linear (%ctx %input %in_scale)
              :where (and (hip-value-single-use? %layout)
                (not (hip-layout-op-has-ctx? (mlir-value-get-defining-op %layout)))
                (hip-matching-qdq-params? (mlir-value-get-defining-op %dq) op)
                (hip-can-requantize-layout-op? (mlir-value-get-defining-op %layout) op))
    :then-let
        ([%dq-op     (mlir-value-get-defining-op %dq)]
         [%layout-op (mlir-value-get-defining-op %layout)])
    :rewrite %q :with
        (%result = (hip-create-requantized-layout-op rewriter %dq-op %layout-op op)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 15: QdqRoundTrip — adjacent pair  (benefit 10)
  ;; Q(DQ(x)) → x  when params match and types are identical
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-pair op rewriter)
    :if-match
        %q  = hip.quantize_linear   (%ctx %dq %out_scale)
        %dq = hip.dequantize_linear (%ctx %input %in_scale)
              :where (and (hip-matching-qdq-params? (mlir-value-get-defining-op %dq) op)
                (hip-identity-roundtrip? (mlir-value-get-defining-op %dq) op))
    :rewrite %q :with
        ;; Replace q with its dq's input directly
        (%result = (begin %input)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pass entry point
  ;;===--------------------------------------------------------------------===;;

  ;; Helper: need mlir-operation-set-dense-i32-array! and mlir-operation-set-dense-i64-array!
  ;; imported from (mlir core ir)
  (define mlir-operation-set-dense-i32-array!
    (foreign-procedure "mlir_operation_set_dense_i32_array" (uptr string scheme-object) void))
  (define mlir-operation-set-dense-i64-array!
    (foreign-procedure "mlir_operation_set_dense_i64_array" (uptr string scheme-object) void))

  (define (run-pass module-op)
    (let ([ctx (mlir-operation-get-context module-op)])
      (with-rewrite-pattern-set (patterns ctx)
        ;; Q-fusion patterns (benefit 10, then 9)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qadd-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qmul-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qmatmul-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qmatmul-per-col-w4 9)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qmatmul-per-col-w8 9)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qgemm-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qgemm-no-bias 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qgemm-per-channel 9)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qgemm-no-bias-per-channel 9)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qconv-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qsigmoid-fusion 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qlpnorm-fusion 10)
        ;; QdqRoundTrip patterns
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-dps 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-tensor 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-pair 10)
        ;; Apply greedily
        (mlir-apply-patterns-greedy module-op patterns))))

) ;; end library (passes hip-fusion)
