;; Test case: where-and-then-let
;; Pattern with both :where guard and :then-let bindings

(:pattern
  (:match %a = onnx.Conv (%x %w)
     :where (mlir-operation-get-attribute %a "kernel_shape")
   :then-let ((%ctx (mlir-get-context %a))
              (%device (get-device-type %x)))
   :rewrite %a :with (hipsr.conv (%ctx %x %w) -> !t)))
