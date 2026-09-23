;; Test case: root-operand
;; Root operation with operand - should use operands-ref not mlir-operation-get-operand-value
;; This verifies the fix at pattern-codegen.sls:228-236

(:pattern
  (:match %out = test.op (%in)
   :rewrite %out :with (new.op (%in) -> !t))

 ;; The analyze phase should generate a :bind-operand action for %in
 ;; When codegen generates code, it should check if op-idx == root-op-idx
 ;; and use (vector-ref operands-ref operand-idx) for root operation
 ;; instead of (mlir-operation-get-operand-value ...)

 ;; This test documents the expected behavior but cannot verify codegen output
 ;; without :debug-codegen support showing the generated lambda code
)