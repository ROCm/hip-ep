#!r6rs
;;===----------------------------------------------------------------------===;;
;; Shared Test Patterns - Single Source of Truth
;;===----------------------------------------------------------------------===;;
;;
;; This library defines canonical test patterns used across all test phases.
;; Each pattern is defined multiple times with different debug flags:
;;   - *-parse:    :debug-parse flag (returns AST after parse phase)
;;   - *-validate: no debug flags, parsed and validated
;;   - *-analyze:  :debug-analyze flag (returns AST with actions after analyze)
;;   - *-codegen:  :debug-codegen flag (returns quoted code after codegen)
;;   - *-lambda:   no debug flags (normal lambda generation)
;;
;; Benefits:
;;   1. DRY: Define each test pattern ONCE with different flags
;;   2. Easy to see what each pattern tests
;;   3. Phase test files just import and assert on these patterns
;;   4. Adding new test: define pattern once, add assertions in phase tests
;;
;;===----------------------------------------------------------------------===;;

(library (test test-patterns)
  (export
    ;; Pattern 1: Basic single operation (no operands, simple rewrite)
    pattern-basic-parse
    pattern-basic-validate
    pattern-basic-analyze
    pattern-basic-codegen
    pattern-basic-lambda

    ;; Pattern 2: Required operands (multiple operands, all required)
    pattern-required-parse
    pattern-required-validate
    pattern-required-analyze
    pattern-required-codegen
    pattern-required-lambda

    ;; Pattern 3: Optional operands (operand group with &optional)
    pattern-optional-parse
    pattern-optional-validate
    pattern-optional-analyze
    pattern-optional-codegen
    pattern-optional-lambda

    ;; Pattern 4: Variadic operands (operand group with &variadic)
    pattern-variadic-parse
    pattern-variadic-validate
    pattern-variadic-analyze
    pattern-variadic-codegen
    pattern-variadic-lambda

    ;; Pattern 5: Mixed operands (required + optional + variadic)
    pattern-mixed-parse
    pattern-mixed-validate
    pattern-mixed-analyze
    pattern-mixed-codegen
    pattern-mixed-lambda

    ;; Pattern 6: :where guard (per-operation guard)
    pattern-where-parse
    pattern-where-validate
    pattern-where-analyze
    pattern-where-codegen
    pattern-where-lambda

    ;; Pattern 7: :then-let bindings (global bindings)
    pattern-then-let-parse
    pattern-then-let-validate
    pattern-then-let-analyze
    pattern-then-let-codegen
    pattern-then-let-lambda

    ;; Pattern 8: Two operations DAG (DAG traversal, operand dependency)
    pattern-two-ops-parse
    pattern-two-ops-validate
    pattern-two-ops-analyze
    pattern-two-ops-codegen
    pattern-two-ops-lambda

    ;; Pattern 9: Variable reuse (same operand twice, check-eq action)
    pattern-reuse-parse
    pattern-reuse-validate
    pattern-reuse-analyze
    pattern-reuse-codegen
    pattern-reuse-lambda

    ;; Pattern 10: Combined features (where + then-let + multiple ops)
    pattern-combined-parse
    pattern-combined-validate
    pattern-combined-analyze
    pattern-combined-codegen
    pattern-combined-lambda

    ;; Pattern 11: Regions with blocks
    pattern-region-parse
    pattern-region-validate
    pattern-region-analyze
    pattern-region-codegen
    pattern-region-lambda

    ;; Pattern 12: Multiple rewrite operations
    pattern-multi-rewrite-parse
    pattern-multi-rewrite-validate
    pattern-multi-rewrite-analyze
    pattern-multi-rewrite-codegen
    pattern-multi-rewrite-lambda)

  (import (except (chezscheme) =)
          (mlir pattern-macro))

  ;;=======================================================================
  ;; Pattern 1: Basic Single Operation
  ;; Tests: Basic parsing, simple match/rewrite structure
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-basic-parse
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  (define-conversion-pattern pattern-basic-validate
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  (define-conversion-pattern :debug-analyze pattern-basic-analyze
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  (define-conversion-pattern :debug-codegen pattern-basic-codegen
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  (define-conversion-pattern pattern-basic-lambda
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%new = "new.op" (%in)))

  ;;=======================================================================
  ;; Pattern 2: Required Operands
  ;; Tests: Multiple operands, all required (no groups)
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-required-parse
    :match
      %a = "test.op" (%x %y %z)
    :rewrite %a :with
      (%out = "new.op" (%x %y %z)))

  (define-conversion-pattern pattern-required-validate
    :match
      %a = "test.op" (%x %y %z)
    :rewrite %a :with
      (%out = "new.op" (%x %y %z)))

  (define-conversion-pattern :debug-analyze pattern-required-analyze
    :match
      %a = "test.op" (%x %y %z)
    :rewrite %a :with
      (%out = "new.op" (%x %y %z)))

  (define-conversion-pattern :debug-codegen pattern-required-codegen
    :match
      %a = "test.op" (%x %y %z)
    :rewrite %a :with
      (%out = "new.op" (%x %y %z)))

  (define-conversion-pattern pattern-required-lambda
    :match
      %a = "test.op" (%x %y %z)
    :rewrite %a :with
      (%out = "new.op" (%x %y %z)))

  ;;=======================================================================
  ;; Pattern 3: Optional Operands
  ;; Tests: Operand flattening, segment handling, optional group
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-optional-parse
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  (define-conversion-pattern pattern-optional-validate
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  (define-conversion-pattern :debug-analyze pattern-optional-analyze
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  (define-conversion-pattern :debug-codegen pattern-optional-codegen
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  (define-conversion-pattern pattern-optional-lambda
    :match
      %a = "test.op" (%x (&optional %y %z))
    :rewrite %a :with
      (%out = "new.op" (%x)))

  ;;=======================================================================
  ;; Pattern 4: Variadic Operands
  ;; Tests: Variadic operand handling, segment metadata
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-variadic-parse
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  (define-conversion-pattern pattern-variadic-validate
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  (define-conversion-pattern :debug-analyze pattern-variadic-analyze
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  (define-conversion-pattern :debug-codegen pattern-variadic-codegen
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  (define-conversion-pattern pattern-variadic-lambda
    :match
      %a = "test.concat" (%x (&variadic %rest))
    :rewrite %a :with
      (%out = "new.concat" (%x)))

  ;;=======================================================================
  ;; Pattern 5: Mixed Operands
  ;; Tests: Complex operand structure with all group types
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-mixed-parse
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  (define-conversion-pattern pattern-mixed-validate
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  (define-conversion-pattern :debug-analyze pattern-mixed-analyze
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  (define-conversion-pattern :debug-codegen pattern-mixed-codegen
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  (define-conversion-pattern pattern-mixed-lambda
    :match
      %a = "test.op" (%x (&optional %y) %z (&variadic %rest))
    :rewrite %a :with
      (%out = "new.op" (%x %z)))

  ;;=======================================================================
  ;; Pattern 6: :where Guard
  ;; Tests: Per-operation guard parsing, validation, action generation
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-where-parse
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  (define-conversion-pattern pattern-where-validate
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  (define-conversion-pattern :debug-analyze pattern-where-analyze
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  (define-conversion-pattern :debug-codegen pattern-where-codegen
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  (define-conversion-pattern pattern-where-lambda
    :match
      %a = "onnx.Conv" (%x %w)
         :where (mlir-operation-get-attribute %a "kernel_shape")
    :rewrite %a :with
      (%out = "hipsr.matmul" (%x %w)))

  ;;=======================================================================
  ;; Pattern 7: :then-let Bindings
  ;; Tests: Global binding parsing, validation, code generation
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-then-let-parse
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %val)))

  (define-conversion-pattern pattern-then-let-validate
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %val)))

  (define-conversion-pattern :debug-analyze pattern-then-let-analyze
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %val)))

  (define-conversion-pattern :debug-codegen pattern-then-let-codegen
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %val)))

  (define-conversion-pattern pattern-then-let-lambda
    :match
      %out = "test.op" (%in)
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
      (%new = "new.op" (%ctx %in %val)))

  ;;=======================================================================
  ;; Pattern 8: Two Operations DAG
  ;; Tests: DAG traversal, operation dependencies, action ordering
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-two-ops-parse
    :match
      %a = "op1" (%x)
      %b = "op2" (%a)
    :rewrite %b :with
      (%out = "new.op" (%a)))

  (define-conversion-pattern pattern-two-ops-validate
    :match
      %a = "op1" (%x)
      %b = "op2" (%a)
    :rewrite %b :with
      (%out = "new.op" (%a)))

  (define-conversion-pattern :debug-analyze pattern-two-ops-analyze
    :match
      %a = "op1" (%x)
      %b = "op2" (%a)
    :rewrite %b :with
      (%out = "new.op" (%a)))

  (define-conversion-pattern :debug-codegen pattern-two-ops-codegen
    :match
      %a = "op1" (%x)
      %b = "op2" (%a)
    :rewrite %b :with
      (%out = "new.op" (%a)))

  (define-conversion-pattern pattern-two-ops-lambda
    :match
      %a = "op1" (%x)
      %b = "op2" (%a)
    :rewrite %b :with
      (%out = "new.op" (%a)))

  ;;=======================================================================
  ;; Pattern 9: Variable Reuse
  ;; Tests: Same operand used twice, check-eq action generation
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-reuse-parse
    :match
      %mul = "onnx.Mul" (%x %x)
    :rewrite %mul :with
      (%sq = "onnx.Square" (%x)))

  (define-conversion-pattern pattern-reuse-validate
    :match
      %mul = "onnx.Mul" (%x %x)
    :rewrite %mul :with
      (%sq = "onnx.Square" (%x)))

  (define-conversion-pattern :debug-analyze pattern-reuse-analyze
    :match
      %mul = "onnx.Mul" (%x %x)
    :rewrite %mul :with
      (%sq = "onnx.Square" (%x)))

  (define-conversion-pattern :debug-codegen pattern-reuse-codegen
    :match
      %mul = "onnx.Mul" (%x %x)
    :rewrite %mul :with
      (%sq = "onnx.Square" (%x)))

  (define-conversion-pattern pattern-reuse-lambda
    :match
      %mul = "onnx.Mul" (%x %x)
    :rewrite %mul :with
      (%sq = "onnx.Square" (%x)))

  ;;=======================================================================
  ;; Pattern 10: Combined Features
  ;; Tests: Complex pattern with :where, :then-let, multiple operations
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-combined-parse
    :match
      %dq = "onnx.DequantizeLinear" (%input %scale %zp)
         :where (mlir-operation-has-one-use %dq)
      %conv = "onnx.Conv" (%dq %weight)
         :where (is-1x1-conv? %conv)
    :then-let ((%ctx (mlir-get-context %conv))
               (%device (get-device-type %input)))
    :rewrite %conv :with
      (%matmul = "hipsr.matmul" (%ctx %input %weight %device)))

  (define-conversion-pattern pattern-combined-validate
    :match
      %dq = "onnx.DequantizeLinear" (%input %scale %zp)
         :where (mlir-operation-has-one-use %dq)
      %conv = "onnx.Conv" (%dq %weight)
         :where (is-1x1-conv? %conv)
    :then-let ((%ctx (mlir-get-context %conv))
               (%device (get-device-type %input)))
    :rewrite %conv :with
      (%matmul = "hipsr.matmul" (%ctx %input %weight %device)))

  (define-conversion-pattern :debug-analyze pattern-combined-analyze
    :match
      %dq = "onnx.DequantizeLinear" (%input %scale %zp)
         :where (mlir-operation-has-one-use %dq)
      %conv = "onnx.Conv" (%dq %weight)
         :where (is-1x1-conv? %conv)
    :then-let ((%ctx (mlir-get-context %conv))
               (%device (get-device-type %input)))
    :rewrite %conv :with
      (%matmul = "hipsr.matmul" (%ctx %input %weight %device)))

  (define-conversion-pattern :debug-codegen pattern-combined-codegen
    :match
      %dq = "onnx.DequantizeLinear" (%input %scale %zp)
         :where (mlir-operation-has-one-use %dq)
      %conv = "onnx.Conv" (%dq %weight)
         :where (is-1x1-conv? %conv)
    :then-let ((%ctx (mlir-get-context %conv))
               (%device (get-device-type %input)))
    :rewrite %conv :with
      (%matmul = "hipsr.matmul" (%ctx %input %weight %device)))

  (define-conversion-pattern pattern-combined-lambda
    :match
      %dq = "onnx.DequantizeLinear" (%input %scale %zp)
         :where (mlir-operation-has-one-use %dq)
      %conv = "onnx.Conv" (%dq %weight)
         :where (is-1x1-conv? %conv)
    :then-let ((%ctx (mlir-get-context %conv))
               (%device (get-device-type %input)))
    :rewrite %conv :with
      (%matmul = "hipsr.matmul" (%ctx %input %weight %device)))

  ;;=======================================================================
  ;; Pattern 11: Regions with Blocks
  ;; Tests: Region parsing, block arguments, nested operations
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-region-parse
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1]))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0])))))

  (define-conversion-pattern pattern-region-validate
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1]))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0])))))

  (define-conversion-pattern :debug-analyze pattern-region-analyze
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1]))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0])))))

  (define-conversion-pattern :debug-codegen pattern-region-codegen
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1]))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0])))))

  (define-conversion-pattern pattern-region-lambda
    :match
      %out = "scf.if" (%cond)
    :rewrite %out :with
      (%r = "scf.if" (%cond)
          :regions
            ((^then ()
               (%t = "arith.constant" () :attrs [value 1]))
             (^else ()
               (%f = "arith.constant" () :attrs [value 0])))))

  ;;=======================================================================
  ;; Pattern 12: Multiple Rewrite Operations
  ;; Tests: Sequential rewrite operations, operand chaining
  ;;=======================================================================

  (define-conversion-pattern :debug-parse pattern-multi-rewrite-parse
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%temp = "temp.op" (%in))
      (%new = "new.op" (%temp))
      (%result = "final.op" (%new)))

  (define-conversion-pattern pattern-multi-rewrite-validate
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%temp = "temp.op" (%in))
      (%new = "new.op" (%temp))
      (%result = "final.op" (%new)))

  (define-conversion-pattern :debug-analyze pattern-multi-rewrite-analyze
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%temp = "temp.op" (%in))
      (%new = "new.op" (%temp))
      (%result = "final.op" (%new)))

  (define-conversion-pattern :debug-codegen pattern-multi-rewrite-codegen
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%temp = "temp.op" (%in))
      (%new = "new.op" (%temp))
      (%result = "final.op" (%new)))

  (define-conversion-pattern pattern-multi-rewrite-lambda
    :match
      %out = "test.op" (%in)
    :rewrite %out :with
      (%temp = "temp.op" (%in))
      (%new = "new.op" (%temp))
      (%result = "final.op" (%new)))

) ;; end library
