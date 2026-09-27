(:pattern
 (:match
  %output = onnx.Cast (%input)
  :then-let
  ([%ctx (mlir-get-hipsr-context-arg op)]
   [!t1 (mlir-value-get-type %output)])
  :rewrite %output :with
  (%result = new.op (%input) -> !t1))
 :expect-codegen ())
