#!r6rs
;;===----------------------------------------------------------------------===;;
;; Phase 4: Codegen Test - Test code generation with :debug-codegen
;;===----------------------------------------------------------------------===;;
;;
;; Focus: Test ONLY what the codegen phase does
;; - Generate lambda with correct signature (op operands-ref rewriter type-converter)
;; - Generate variable initialization code (make-unbound-value)
;; - Generate root operation initialization (set! var (mlir-operation-get-result ...))
;; - Generate match check code from actions
;; - Generate :where guard code
;; - Generate :then-let binding code
;; - Generate rewrite code structure
;;
;; :debug-codegen returns quoted code for inspection
;;
;; All test patterns imported from (test test-patterns) - single source of truth
;;
;;===----------------------------------------------------------------------===;;

(library (test phase-4-codegen-test)
  (export run-tests)
  (import (except (chezscheme) =)
          (test test-framework)
          (test test-patterns))

  ;;=======================================================================
  ;; Helper Functions
  ;;=======================================================================

  (define (is-lambda? code)
    "Check if code is a lambda expression"
    (and (list? code)
         (>= (length code) 2)
         (eq? (car code) 'lambda)))

  (define (lambda-params code)
    "Extract lambda parameters"
    (if (is-lambda? code)
        (cadr code)
        #f))

  (define (lambda-body code)
    "Extract lambda body"
    (if (is-lambda? code)
        (cddr code)
        #f))

  (define (find-in-code pred code)
    "Search for an expression matching predicate in code structure"
    (cond
      [(pred code) code]
      [(pair? code)
       (or (find-in-code pred (car code))
           (find-in-code pred (cdr code)))]
      [else #f]))

  (define (contains-symbol? sym code)
    "Check if code contains a symbol"
    (find-in-code (lambda (x) (and (symbol? x) (eq? x sym))) code))

  ;;=======================================================================
  ;; Test Suite
  ;;=======================================================================

  (define (run-tests)
    (test-begin "phase-4-codegen")
    (display "\n=== Phase 4: Codegen Tests (Using Shared Patterns) ===\n\n")

    ;; Test 1: Basic lambda structure
    (test-equal "basic: returns quoted code" #t (list? pattern-basic-codegen))
    (test-equal "basic: is lambda" #t (is-lambda? pattern-basic-codegen))
    ;; TODO: Verify structure is (lambda (op operands-ref rewriter type-converter) ...)
    ;; TODO: Check lambda-params: (op operands-ref rewriter type-converter)

    ;; Test 2: Required operands - variable initialization
    (test-equal "required: returns code" #t (list? pattern-required-codegen))
    (test-equal "required: is lambda" #t (is-lambda? pattern-required-codegen))
    ;; TODO: Verify let bindings contain (make-unbound-value) for each variable
    ;; TODO: Check for (%a (make-unbound-value)) (%x ...) (%y ...) (%z ...)

    ;; Test 3: Optional operands - conditional access
    (test-equal "optional: returns code" #t (list? pattern-optional-codegen))
    (test-equal "optional: is lambda" #t (is-lambda? pattern-optional-codegen))
    ;; TODO: Verify optional operand access uses:
    ;;   - Operand count check
    ;;   - Conditional binding based on operand-segments attribute

    ;; Test 4: Variadic operands - collection
    (test-equal "variadic: returns code" #t (list? pattern-variadic-codegen))
    (test-equal "variadic: is lambda" #t (is-lambda? pattern-variadic-codegen))
    ;; TODO: Verify variadic operand access:
    ;;   - Collect remaining operands into list/vector
    ;;   - Use operand-segments for indexing

    ;; Test 5: Mixed operands - complex access pattern
    (test-equal "mixed: returns code" #t (list? pattern-mixed-codegen))
    (test-equal "mixed: is lambda" #t (is-lambda? pattern-mixed-codegen))
    ;; TODO: Verify correct handling of all operand types

    ;; Test 6: :where guard codegen
    (test-equal "where: returns code" #t (list? pattern-where-codegen))
    (test-equal "where: is lambda" #t (is-lambda? pattern-where-codegen))
    ;; TODO: Verify guard check appears in generated code
    ;; TODO: Check for (mlir-operation-get-attribute %a "kernel_shape")
    ;; TODO: Verify guard wrapped in early-return logic (and ...)

    ;; Test 7: :then-let codegen
    (test-equal "then-let: returns code" #t (list? pattern-then-let-codegen))
    (test-equal "then-let: is lambda" #t (is-lambda? pattern-then-let-codegen))
    ;; TODO: Verify :then-let bindings appear in code:
    ;;   (let* ([%ctx (get-context)]
    ;;          [%val (compute-value)])
    ;;     <rewrite-code>)

    ;; Test 8: Two operations - DAG match code
    (test-equal "two-ops: returns code" #t (list? pattern-two-ops-codegen))
    (test-equal "two-ops: is lambda" #t (is-lambda? pattern-two-ops-codegen))
    ;; TODO: Verify match check code:
    ;;   - Operation name checks: (string=? (mlir-operation-name ...) "op1")
    ;;   - Operand binding code
    ;;   - DAG traversal structure

    ;; Test 9: Variable reuse - equality check
    (test-equal "reuse: returns code" #t (list? pattern-reuse-codegen))
    (test-equal "reuse: is lambda" #t (is-lambda? pattern-reuse-codegen))
    ;; TODO: Verify check-eq action generates equality test

    ;; Test 10: Combined features - full structure
    (test-equal "combined: returns code" #t (list? pattern-combined-codegen))
    (test-equal "combined: is lambda" #t (is-lambda? pattern-combined-codegen))
    ;; TODO: Verify complete structure:
    ;;   - Lambda signature
    ;;   - Variable init
    ;;   - Root init
    ;;   - Match checks (with :where guards)
    ;;   - :then-let bindings
    ;;   - Rewrite code

    ;; Test 11: Regions - nested structure
    (test-equal "regions: returns code" #t (list? pattern-region-codegen))
    (test-equal "regions: is lambda" #t (is-lambda? pattern-region-codegen))
    ;; TODO: Verify region structure in rewrite
    ;; TODO: Test block arguments and operations

    ;; Test 12: Multiple rewrite operations - sequential code
    (test-equal "multi-rewrite: returns code" #t (list? pattern-multi-rewrite-codegen))
    (test-equal "multi-rewrite: is lambda" #t (is-lambda? pattern-multi-rewrite-codegen))
    ;; TODO: Verify rewrite operations generate:
    ;;   - Operation creation calls (mlir-create-operation)
    ;;   - Sequential binding (%temp = ..., then %new = ..., then %result = ...)
    ;;   - Final replacement (mlir-replace-op)

    (display "\n:debug-codegen returns quoted code for inspection.\n")
    (display "Detailed code structure tests require analyzing quoted S-expressions.\n")

    (test-end)))
