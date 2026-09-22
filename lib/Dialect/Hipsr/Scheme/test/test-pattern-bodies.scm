;; Test Pattern Bodies - Single Source of Truth
;; Each pattern defined ONCE, used by all test phases
;;
;; Format: (pattern-name pattern-body...)
;; The pattern body is everything that comes after define-conversion-pattern
;; and the test name (i.e., the :match/:rewrite clauses)

(
  ;; Pattern 1: Basic single operation
  (basic
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  ;; Pattern 2: Multiple required operands
  (required
    :match
      %out = "test.add" (%x %y)
    :rewrite %out :with
      (%sum = "new.add" (%x %y)))

  ;; Pattern 3: Optional operands
  (optional
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  ;; Pattern 4: Variadic operands
  (variadic
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  ;; Pattern 5: Mixed operands (required + optional + variadic)
  (mixed
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  ;; Pattern 6: :where guard
  (where
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  ;; Pattern 7: :then-let bindings
  (then-let
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%device (get-device)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %device)))

  ;; Pattern 8: Two operations (DAG)
  (two-ops
    :match
      %a = "test.op1" (%x)
      %b = "test.op2" (%a %y)
    :rewrite %b :with
      (%out = "new.op" (%x %y)))

  ;; Pattern 9: Variable reuse (same operand used twice)
  (reuse
    :match
      %a = "test.add" (%x %x)
    :rewrite %a :with
      (%out = "new.double" (%x)))

  ;; Pattern 10: Combined features (where + then-let + two ops)
  (combined
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
      %b = "onnx.Add" (%a %bias)
    :then-let ((%ctx (get-context)))
    :rewrite %b :with
      (%conv = "hipsr.conv" (%ctx %x %w))
      (%add = "hipsr.add" (%conv %bias)))

  ;; Pattern 11: Regions with blocks
  (region
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1] -> i32))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0] -> i32)))
          -> i32))
)
