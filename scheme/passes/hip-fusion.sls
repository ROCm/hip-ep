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
;; guards using (mlir hip fusion) helpers.
;;
;; :where scoping rule (critical):
;;   Each :where fires after that op's own operands are bound but BEFORE
;;   parent ops' remaining operands are processed.  In a Q→compute→DQ chain:
;;     DQ :where  — sees DQ's own operands; %*_init and %out_scale NOT bound
;;     compute :where — sees compute's operands incl. %*_init; %out_scale NOT bound
;;     Q :where   — fires last, sees everything incl. %out_scale
;;
;; Entry point: (run-pass module-op)
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
  ;; Shared rewrite helpers
  ;;===--------------------------------------------------------------------===;;

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

  (define mlir-operation-set-dense-i32-array!
    (foreign-procedure "mlir_operation_set_dense_i32_array" (uptr string scheme-object) void))
  (define mlir-operation-set-dense-i64-array!
    (foreign-procedure "mlir_operation_set_dense_i64_array" (uptr string scheme-object) void))
  ;; For ODS I64ArrayAttr (ArrayAttr of IntegerAttr) — different from DenseI64ArrayAttr
  (define mlir-operation-set-i64-array-attr!
    (foreign-procedure "mlir_operation_set_i64_array_attr" (uptr string scheme-object) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 1: QAdd  (benefit 10)
  ;; dq x2 → hip.add → hip.quantize_linear → hip.qadd
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qadd-fusion op rewriter)
    :if-match
        %q   = hip.quantize_linear   (%ctx %sum %out_scale)
                 :where (and (hip-splat-scale? %out_scale)
                             (hip-extractable-qdq-zeropoint? op))
        %sum = hip.add               (%ctx %dq_lhs %dq_rhs %sum_init)
                 :where (and (hip-value-single-use? %sum)
                             (hip-can-build-init? op %sum_init))
        %dq_lhs = hip.dequantize_linear (%ctx %lhs %lhs_scale)
                 :where (and (hip-splat-scale? %lhs_scale)
                             (hip-extractable-qdq-zeropoint?
                               (mlir-value-get-defining-op %dq_lhs)))
        %dq_rhs = hip.dequantize_linear (%ctx %rhs %rhs_scale)
                 :where (and (hip-splat-scale? %rhs_scale)
                             (hip-extractable-qdq-zeropoint?
                               (mlir-value-get-defining-op %dq_rhs)))
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
         [%init      (hip-build-init rewriter !out-type %sum_init)])
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
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmul-fusion op rewriter)
    :if-match
        %q   = hip.quantize_linear   (%ctx %prod %out_scale)
                 :where (and (hip-splat-scale? %out_scale)
                             (hip-extractable-qdq-zeropoint? op))
        %prod = hip.mul              (%ctx %dq_lhs %dq_rhs %prod_init)
                 :where (and (hip-value-single-use? %prod)
                             (hip-can-build-init? op %prod_init))
        %dq_lhs = hip.dequantize_linear (%ctx %lhs %lhs_scale)
                 :where (and (hip-splat-scale? %lhs_scale)
                             (hip-extractable-qdq-zeropoint?
                               (mlir-value-get-defining-op %dq_lhs)))
        %dq_rhs = hip.dequantize_linear (%ctx %rhs %rhs_scale)
                 :where (and (hip-splat-scale? %rhs_scale)
                             (hip-extractable-qdq-zeropoint?
                               (mlir-value-get-defining-op %dq_rhs)))
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
         [%init      (hip-build-init rewriter !out-type %prod_init)])
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
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmatmul-fusion op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %matmul %y_scale)
                    :where (and (hip-splat-scale? %y_scale)
                                (hip-qdq-quantized-width? op '(8 16))
                                (hip-extractable-qdq-zeropoint? op))
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
                    :where (and (hip-value-single-use? %matmul)
                                (hip-can-build-init? op %matmul_init))
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
                    :where (and (hip-qdq-quantized-width?
                                  (mlir-value-get-defining-op %dq_a) '(8 16))
                                (hip-splat-scale? %a_scale)
                                (hip-extractable-qdq-zeropoint?
                                  (mlir-value-get-defining-op %dq_a)))
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scale)
                    :where (and (hip-qdq-quantized-width?
                                  (mlir-value-get-defining-op %dq_b) '(8))
                                (hip-splat-scale? %b_scale)
                                (hip-extractable-qdq-zeropoint?
                                  (mlir-value-get-defining-op %dq_b)))
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
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qmatmul-per-col-w4 op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %matmul %y_scale)
                    :where (and (hip-splat-scale? %y_scale)
                                (hip-qdq-quantized-width? op '(8 16))
                                (hip-extractable-qdq-zeropoint? op))
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
                    :where (and (hip-value-single-use? %matmul)
                                (hip-can-build-init? op %matmul_init)
                                (num= (mlir-operation-get-integer-attr
                                        (mlir-value-get-defining-op %matmul) "transB" 0)
                                      0))
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
                    :where (and (hip-qdq-quantized-width?
                                  (mlir-value-get-defining-op %dq_a) '(8 16))
                                (hip-splat-scale? %a_scale)
                                (hip-extractable-qdq-zeropoint?
                                  (mlir-value-get-defining-op %dq_a)))
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
                    :where (hip-per-axis-weight?
                              (mlir-value-get-defining-op %dq_b) 2 1 #t)
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
                    :where (and (hip-splat-scale? %y_scale)
                                (hip-qdq-quantized-width? op '(8 16))
                                (hip-extractable-qdq-zeropoint? op))
        %matmul = hip.matmul            (%ctx %dq_a %dq_b %matmul_init)
                    :where (and (hip-value-single-use? %matmul)
                                (hip-can-build-init? op %matmul_init)
                                (num= (mlir-operation-get-integer-attr
                                        (mlir-value-get-defining-op %matmul) "transB" 0)
                                      0))
        %dq_a   = hip.dequantize_linear (%ctx %a %a_scale)
                    :where (and (hip-qdq-quantized-width?
                                  (mlir-value-get-defining-op %dq_a) '(8 16))
                                (hip-splat-scale? %a_scale)
                                (hip-extractable-qdq-zeropoint?
                                  (mlir-value-get-defining-op %dq_a)))
        %dq_b   = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
                    :where (hip-per-axis-weight?
                              (mlir-value-get-defining-op %dq_b) 2 1 #f)
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
        %q    = hip.quantize_linear   (%ctx %gemm %y_scale)
                  :where (and (hip-splat-scale? %y_scale)
                              (hip-qdq-quantized-width? op '(8 16))
                              (hip-extractable-qdq-zeropoint? op))
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %dq_c %gemm_init)
                  :where (and (hip-value-single-use? %gemm)
                              (hip-can-build-init? op %gemm_init))
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_a) '(8 16))
                              (hip-splat-scale? %a_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_a)))
        %dq_b = hip.dequantize_linear (%ctx %b %b_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_b) '(8))
                              (hip-splat-scale? %b_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_b)))
        %dq_c = hip.dequantize_linear (%ctx %c %c_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_c) '(8 16 32))
                              (hip-splat-scale? %c_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_c)))
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
                  :where (and (hip-splat-scale? %y_scale)
                              (hip-qdq-quantized-width? op '(8 16))
                              (hip-extractable-qdq-zeropoint? op))
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %gemm_init)
                  :where (and (hip-value-single-use? %gemm)
                              (hip-can-build-init? op %gemm_init))
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_a) '(8 16))
                              (hip-splat-scale? %a_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_a)))
        %dq_b = hip.dequantize_linear (%ctx %b %b_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_b) '(8))
                              (hip-splat-scale? %b_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_b)))
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
                  :where (and (hip-splat-scale? %y_scale)
                              (hip-qdq-quantized-width? op '(8 16))
                              (hip-extractable-qdq-zeropoint? op))
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %dq_c %gemm_init)
                  :where (and (hip-value-single-use? %gemm)
                              (hip-can-build-init? op %gemm_init))
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_a) '(8 16))
                              (hip-splat-scale? %a_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_a)))
        %dq_b = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
                  :where (hip-per-channel-weight?
                            (mlir-value-get-defining-op %dq_b)
                            (mlir-value-get-defining-op %gemm))
        %dq_c = hip.dequantize_linear (%ctx %c %c_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_c) '(8 16 32))
                              (hip-splat-scale? %c_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_c)))
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
                  :where (and (hip-splat-scale? %y_scale)
                              (hip-qdq-quantized-width? op '(8 16))
                              (hip-extractable-qdq-zeropoint? op))
        %gemm = hip.gemm              (%ctx %dq_a %dq_b %gemm_init)
                  :where (and (hip-value-single-use? %gemm)
                              (hip-can-build-init? op %gemm_init))
        %dq_a = hip.dequantize_linear (%ctx %a %a_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_a) '(8 16))
                              (hip-splat-scale? %a_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_a)))
        %dq_b = hip.dequantize_linear (%ctx %b %b_scales %b_zps %b_init)
                  :where (hip-per-channel-weight?
                            (mlir-value-get-defining-op %dq_b)
                            (mlir-value-get-defining-op %gemm))
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
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qconv-fusion op rewriter)
    :if-match
        %q    = hip.quantize_linear   (%ctx %conv %out_scale)
                  :where (and (hip-splat-scale? %out_scale)
                              (hip-qdq-quantized-width? op '(16))
                              (hip-qdq-unsigned? op)
                              (hip-extractable-qdq-zeropoint? op))
        %conv = hip.conv              (%ctx %dq_in %dq_w %conv_init)
                  :where (and (hip-value-single-use? %conv)
                              (hip-fusable-conv-geometry?
                                (mlir-value-get-defining-op %conv))
                              (hip-can-build-init? op %conv_init))
        %dq_in = hip.dequantize_linear (%ctx %input %in_scale)
                  :where (and (hip-qdq-quantized-width?
                                (mlir-value-get-defining-op %dq_in) '(16))
                              (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                              (hip-splat-scale? %in_scale)
                              (hip-extractable-qdq-zeropoint?
                                (mlir-value-get-defining-op %dq_in)))
        %dq_w  = hip.dequantize_linear (%ctx %weights %w_scales %w_zps %w_init)
                  :where (hip-per-axis-weight?
                            (mlir-value-get-defining-op %dq_w) 4 0 #t)
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
                     (mlir-operation-set-i64-array-attr! new-op "kernel_shape" '(1 1))
                     (mlir-operation-set-i64-array-attr! new-op "strides"      '(1 1))
                     (mlir-operation-set-i64-array-attr! new-op "pads"         '(0 0 0 0))
                     (mlir-operation-set-i64-array-attr! new-op "dilations"    '(1 1))
                     (mlir-operation-set-i64-attr! new-op "group"          1)
                     (mlir-operation-set-unit-attr! new-op "packed_int4")
                     (mlir-operation-get-result new-op 0))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 11: QSigmoid  (benefit 10)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qsigmoid-fusion op rewriter)
    :if-match
        %q       = hip.quantize_linear   (%ctx %sigmoid %out_scale)
                     :where (and (hip-splat-scale? %out_scale)
                                 (hip-qdq-quantized-width? op '(16))
                                 (hip-qdq-unsigned? op)
                                 (hip-extractable-qdq-zeropoint? op))
        %sigmoid = hip.sigmoid           (%ctx %dq_in %sigmoid_init)
                     :where (and (hip-value-single-use? %sigmoid)
                                 (hip-can-build-init? op %sigmoid_init))
        %dq_in   = hip.dequantize_linear (%ctx %input %in_scale)
                     :where (and (hip-qdq-quantized-width?
                                   (mlir-value-get-defining-op %dq_in) '(16))
                                 (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                                 (hip-splat-scale? %in_scale)
                                 (hip-extractable-qdq-zeropoint?
                                   (mlir-value-get-defining-op %dq_in)))
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
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qlpnorm-fusion op rewriter)
    :if-match
        %q     = hip.quantize_linear   (%ctx %rms %out_scale)
                   :where (and (hip-splat-scale? %out_scale)
                               (hip-qdq-quantized-width? op '(16))
                               (hip-qdq-unsigned? op)
                               (hip-extractable-qdq-zeropoint? op))
        %rms   = hip.rms_norm          (%ctx %dq_in %rms_scale %rms_init)
                   :where (and (hip-value-single-use? %rms)
                               (hip-l2-equiv-rms-norm? (mlir-value-get-defining-op %rms))
                               (hip-can-build-init? op %rms_init))
        %dq_in = hip.dequantize_linear (%ctx %input %in_scale)
                   :where (and (hip-qdq-quantized-width?
                                 (mlir-value-get-defining-op %dq_in) '(16))
                               (hip-qdq-unsigned? (mlir-value-get-defining-op %dq_in))
                               (hip-splat-scale? %in_scale)
                               (hip-extractable-qdq-zeropoint?
                                 (mlir-value-get-defining-op %dq_in)))
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
  ;; Q(LAYOUT_DPS(DQ(x))) → clone layout with quantized type
  ;; The :where is on %dq; at that point %layout IS bound (Q's operand 1)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-dps op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %layout %out_scale)
        %layout = :any                  (%ctx %dq)
        %dq     = hip.dequantize_linear (%ctx %input %in_scale)
                    :where (and (hip-value-single-use? %layout)
                                (hip-matching-qdq-params?
                                  (mlir-value-get-defining-op %dq) op)
                                (hip-can-requantize-layout-op?
                                  (mlir-value-get-defining-op %layout) op))
    :then-let
        ([%dq-op     (mlir-value-get-defining-op %dq)]
         [%layout-op (mlir-value-get-defining-op %layout)])
    :rewrite %q :with
        (%result = (hip-create-requantized-layout-op rewriter %dq-op %layout-op op)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 14: QdqRoundTrip — tensor layout op (no ctx)  (benefit 10)
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-tensor op rewriter)
    :if-match
        %q      = hip.quantize_linear   (%ctx %layout %out_scale)
        %layout = :any                  (%dq)
        %dq     = hip.dequantize_linear (%ctx %input %in_scale)
                    :where (and (hip-value-single-use? %layout)
                                (not (hip-layout-op-has-ctx?
                                       (mlir-value-get-defining-op %layout)))
                                (hip-matching-qdq-params?
                                  (mlir-value-get-defining-op %dq) op)
                                (hip-can-requantize-layout-op?
                                  (mlir-value-get-defining-op %layout) op))
    :then-let
        ([%dq-op     (mlir-value-get-defining-op %dq)]
         [%layout-op (mlir-value-get-defining-op %layout)])
    :rewrite %q :with
        (%result = (hip-create-requantized-layout-op rewriter %dq-op %layout-op op)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern 15: QdqRoundTrip — adjacent pair  (benefit 10)
  ;; Q(DQ(x)) → x when params match and types are identical
  ;;===--------------------------------------------------------------------===;;

  (define-rewrite-pattern (hip-qdq-roundtrip-pair op rewriter)
    :if-match
        %q  = hip.quantize_linear   (%ctx %dq %out_scale)
        %dq = hip.dequantize_linear (%ctx %input %in_scale)
                :where (and (hip-matching-qdq-params?
                              (mlir-value-get-defining-op %dq) op)
                            (hip-identity-roundtrip?
                              (mlir-value-get-defining-op %dq) op))
    :rewrite %q :with
        (%result = (begin %input)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pass entry point
  ;;===--------------------------------------------------------------------===;;

  (define (run-pass module-op)
    (let ([ctx (mlir-operation-get-context module-op)])
      (with-rewrite-pattern-set (patterns ctx)
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
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-dps 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-tensor 10)
        (mlir-register-rewrite-pattern patterns "hip.quantize_linear"
                                       hip-qdq-roundtrip-pair 10)
        (mlir-apply-patterns-greedy module-op patterns))))

) ;; end library (passes hip-fusion)
