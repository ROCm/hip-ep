#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.MatMul → hipsr.matmul
;;
;; Inline shape region (mirrors HipsrMatMulOp.cpp::populateMatMulShapeRegion).
;; Only handles rank≥2 inputs (the common NN case). 1-D dot-product
;; falls through to PopulateShapeRegionPass via the C++ fallback.
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr matmul)
  (export populate-matmul-patterns
          onnx-matmul->hipsr)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects shape)
          (mlir ddr))

  (define-conversion-pattern (onnx-matmul->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.MatMul (%a %b)
    :then-let
        ([%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (make-mlir-tensor-in-device-space !output-type)]
         [!shape-type    (mlir-shape.shape-type   (mlir-operation-get-context op))]
         [!size-type     (mlir-shape.size-type    (mlir-operation-get-context op))]
         [!witness-type  (mlir-shape.witness-type (mlir-operation-get-context op))]
         ;; Rank info from operand types (runtime)
         [a-rank         (mlir-type-get-rank (mlir-value-get-type %a))]
         [b-rank         (mlir-type-get-rank (mlir-value-get-type %b))]
         ;; K indices: A's last dim; B's second-to-last (or last if 1-D)
         [k-a-idx        (- a-rank 1)]
         [k-b-idx        (if (eqv? b-rank 1) (- b-rank 1) (- b-rank 2))]
         ;; Batch rank = max(rank-2, 0)
         [a-batch-rank   (max 0 (- a-rank 2))]
         [b-batch-rank   (max 0 (- b-rank 2))]
         ;; M/N dim indices (valid only when rank≥2)
         [m-idx          (- a-rank 2)]
         [n-idx          (- b-rank 1)])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %a %b !output-device)
                        (^bb0 ((%a-shape : !shape-type) (%b-shape : !shape-type))
                              ;; K equality: compare single-element shape of K dims
                              (%cka   = shape.const_size () (value = k-a-idx :index) -> !size-type)
                              (%ckb   = shape.const_size () (value = k-b-idx :index) -> !size-type)
                              (%eka   = shape.get_extent (%a-shape %cka) -> !size-type)
                              (%ekb   = shape.get_extent (%b-shape %ckb) -> !size-type)
                              (%aksh  = shape.from_extents (%eka) -> !shape-type)
                              (%bksh  = shape.from_extents (%ekb) -> !shape-type)
                              (%wk    = shape.cstr_eq (%aksh %bksh) -> !witness-type)
                              ;; Batch shapes via split_at (head = batch dims)
                              (%cab   = shape.const_size () (value = a-batch-rank :index) -> !size-type)
                              (%cbb   = shape.const_size () (value = b-batch-rank :index) -> !size-type)
                              ((%ab %at) = shape.split_at (%a-shape %cab) -> (!shape-type !shape-type))
                              ((%bb %bt) = shape.split_at (%b-shape %cbb) -> (!shape-type !shape-type))
                              (%wbc   = shape.cstr_broadcastable (%ab %bb) -> !witness-type)
                              (%wall  = shape.assuming_all (%wk %wbc) -> !witness-type)
                              ;; M and N extents (used inside assuming — not IsolatedFromAbove)
                              (%cm    = shape.const_size () (value = m-idx :index) -> !size-type)
                              (%cn    = shape.const_size () (value = n-idx :index) -> !size-type)
                              (%em    = shape.get_extent (%a-shape %cm) -> !size-type)
                              (%en    = shape.get_extent (%b-shape %cn) -> !size-type)
                              ;; Result shape inside shape.assuming (non-isolated, sees %ab %bb %em %en)
                              (%res   = shape.assuming (%wall)
                                        (^bb0 ()
                                              (%batch  = shape.broadcast    (%ab %bb)    -> !shape-type)
                                              (%matrix = shape.from_extents (%em %en)   -> !shape-type)
                                              (%cat   = shape.concat        (%batch %matrix) -> !shape-type)
                                              (shape.assuming_yield (%cat)))
                                        -> !shape-type)
                              (hipsr.shape_yield (%res)))
                        -> !output-device)
        (%result = hipsr.matmul (%ctx %a %b %placeholder !output-device)
                   -> !output-device))

  (define (populate-matmul-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.MatMul pattern (inline shape region)")
    (mlir-register-conversion-pattern patterns "onnx.MatMul"
                                      onnx-matmul->hipsr type-converter))

) ;; end library (onnx-to-hipsr matmul)
