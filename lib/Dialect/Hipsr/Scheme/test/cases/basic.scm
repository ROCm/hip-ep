;; Test case: basic
;; Basic pattern with multiple operations in rewrite clause
;; Demonstrates: match one op, rewrite to sequence of ops

(:pattern
  (:match
      %out = test.op (%a %b)
   :then-let
      ([!t1 (mlir-value-get-type %out)])
   :rewrite %out :with
     (%x = temp.op (%a) -> !t1)
     (%y = new.op (%x %b) -> !t1))

 ;; Phase 1: Parse only - validation and analysis skipped
 ;; - Multiple rewrite operations demonstrate chaining: temp.op -> new.op
 ;; - :then-let binding captured in where field
 :expect-parse
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . #f)
  (root-op-index . #f)
  (root-result-idx . #f)
  (match
    ((result-var . %out)
     (op-name . test.op)
     (operands
       ((kind . required) (var . %a))
       ((kind . required) (var . %b)))
     (where-expr . #f)))
  (match-bindings . #f)
  (match-actions . #f)
  (rewrite
    ((result-var . %x)
     (op-name . temp.op)
     (operands %a)
     (regions)
     (attributes)
     (result-types . !t1))
    ((result-var . %y)
     (op-name . new.op)
     (operands %x %b)
     (regions)
     (attributes)
     (result-types . !t1)))
  (where ((var . !t1) (expr mlir-value-get-type %out)))
  (debug-parse? . #t)
  (debug-validate? . #f)
  (debug-analyze? . #f)
  (debug-codegen? . #f)
  (debug-matching? . #f))

 ;; Phase 2: Parse + Validate only, analysis skipped
 ;; - root-op-name converted to string, root-op-index/root-result-idx set
 ;; - Two rewrite operations with their operands
 :expect-validate
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . "test.op")
  (root-op-index . 0)
  (root-result-idx . 0)
  (match
    ((result-var %out)
     (op-name . "test.op")
     (operands
       ((kind . required) (var . %a))
       ((kind . required) (var . %b)))
     (where-expr . #f)))
  (match-bindings . #f)
  (match-actions . #f)
  (rewrite
    ((result-var . %x)
     (op-name . "temp.op")
     (operands %a)
     (regions)
     (attributes)
     (result-types . !t1))
    ((result-var . %y)
     (op-name . "new.op")
     (operands %x %b)
     (regions)
     (attributes)
     (result-types . !t1)))
  (where ((var . !t1) (expr mlir-value-get-type %out)))
  (debug-parse? . #f)
  (debug-validate? . #t)
  (debug-analyze? . #f)
  (debug-codegen? . #f)
  (debug-matching? . #f))

 ;; Phase 3: Parse + Validate + Analyze - all phases complete
 ;; - match-bindings: includes %a, %b (operands) and %out (result)
 ;; - match-actions: uses :bind-argument-operand for both %a and %b (root operands in conversion pattern)
 :expect-analyze
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . "test.op")
  (root-op-index . 0)
  (root-result-idx . 0)
  (match
    ((result-var %out)
     (op-name . "test.op")
     (operands
       ((kind . required) (var . %a))
       ((kind . required) (var . %b)))
     (where-expr . #f)))
  (match-bindings
    (bindings-table
      (%a (id . %a) (is-result? . #f) (result-op-idx . #f)
          (result-idx . #f) (operand-op-idx . 0) (operand-idx . 0)
          (bound? . #t))
      (%b (id . %b) (is-result? . #f) (result-op-idx . #f)
          (result-idx . #f) (operand-op-idx . 0) (operand-idx . 1)
          (bound? . #t))
      (%out (id . %out) (is-result? . #t) (result-op-idx . 0)
        (result-idx . 0) (operand-op-idx . #f) (operand-idx . #f)
        (bound? . #t))))
  (match-actions
    (:set-current-op (op-idx . 0) (var . %out))
    (:check-op (op-idx . 0))
    (:bind-argument-operand (operand-idx . 0) (var . %a))
    (:bind-argument-operand (operand-idx . 1) (var . %b)))
  (rewrite
    ((result-var . %x)
     (op-name . "temp.op")
     (operands %a)
     (regions)
     (attributes)
     (result-types . !t1))
    ((result-var . %y)
     (op-name . "new.op")
     (operands %x %b)
     (regions)
     (attributes)
     (result-types . !t1)))
  (where ((var . !t1) (expr mlir-value-get-type %out)))
  (debug-parse? . #f)
  (debug-validate? . #f)
  (debug-analyze? . #t)
  (debug-codegen? . #f)
  (debug-matching? . #f)))

