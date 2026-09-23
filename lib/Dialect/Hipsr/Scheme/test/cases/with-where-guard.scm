;; Test case: with-where-guard
;; Pattern with per-operation :where guard

(:pattern
  (:match %a = onnx.Conv (%x %w)
     :where (let ([$ks (mlir-operation-get-attribute %a "kernel_shape")])
              (and $ks (is-1x1-kernel? $ks)))
   :rewrite %a :with (hipsr.matmul (%x %w) -> !t)))
