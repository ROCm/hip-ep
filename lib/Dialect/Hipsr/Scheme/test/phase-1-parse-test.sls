#!r6rs
;;===----------------------------------------------------------------------===;;
;; Phase 1: Parse Test - Test pattern parsing with :debug-parse
;;===----------------------------------------------------------------------===;;
;;
;; Focus: Test ONLY what the parse phase does
;; - Parse operands and flatten operand groups
;; - Parse :where guards (per-operation)
;; - Parse :then-let bindings (global)
;; - Parse region/block structure
;; - Parse debug flags
;;
;; Does NOT test:
;; - Validation (% prefix, duplicates) - that's phase 2
;; - Action generation - that's phase 3
;; - Code generation - that's phase 4
;;
;; All test patterns imported from (test test-patterns) - single source of truth
;;
;;===----------------------------------------------------------------------===;;

(library (test phase-1-parse-test)
  (export run-tests)
  (import (except (chezscheme) =)
          (test test-framework)
          (test test-patterns))

  ;;=======================================================================
  ;; Helper Functions
  ;;=======================================================================

  (define (ast-get ast key)
    "Extract field from AST list by key"
    (let loop ([rest ast])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  ;;=======================================================================
  ;; Test Suite
  ;;=======================================================================

  (define (run-tests)
    (test-begin "phase-1-parse")
    (display "\n=== Phase 1: Parse Tests (Using Shared Patterns) ===\n\n")

    ;; Test 1: Basic single operation
    (test-equal "basic: returns AST list" #t (list? pattern-basic-parse))
    (test-equal "basic: has function-name" 'pattern-basic-parse
                (ast-get pattern-basic-parse 'function-name))
    (test-equal "basic: has root-op-name" "test.op"
                (ast-get pattern-basic-parse 'root-op-name))
    (test-equal "basic: debug-parse flag set" #t
                (ast-get pattern-basic-parse 'debug-parse?))

    ;; Test 2: Required operands
    (test-equal "required: parses as AST" #t (list? pattern-required-parse))
    (test-equal "required: has match ops" #t
                (pair? (ast-get pattern-required-parse 'match)))
    ;; TODO: Verify operands list contains 3 items: %x %y %z

    ;; Test 3: Optional operands
    (test-equal "optional: parses as AST" #t (list? pattern-optional-parse))
    (test-equal "optional: has match ops" #t
                (pair? (ast-get pattern-optional-parse 'match)))
    ;; TODO: Verify operands field structure includes (&optional %y %z) group

    ;; Test 4: Variadic operands
    (test-equal "variadic: parses as AST" #t (list? pattern-variadic-parse))
    (test-equal "variadic: has match ops" #t
                (pair? (ast-get pattern-variadic-parse 'match)))
    ;; TODO: Verify operands field includes (&variadic %rest) group

    ;; Test 5: Mixed operands
    (test-equal "mixed: parses as AST" #t (list? pattern-mixed-parse))
    (test-equal "mixed: has match ops" #t
                (pair? (ast-get pattern-mixed-parse 'match)))
    ;; TODO: Verify operands structure: %x (&optional %y) %z (&variadic %rest)

    ;; Test 6: :where guard
    (test-equal "where: parses as AST" #t (list? pattern-where-parse))
    (test-equal "where: has match ops" #t
                (pair? (ast-get pattern-where-parse 'match)))
    ;; TODO: Verify first match operation has where-expr field

    ;; Test 7: :then-let
    (test-equal "then-let: parses as AST" #t (list? pattern-then-let-parse))
    (test-equal "then-let: has where bindings" #t
                (pair? (ast-get pattern-then-let-parse 'where)))
    (test-equal "then-let: has 2 bindings" 2
                (length (ast-get pattern-then-let-parse 'where)))
    ;; TODO: Verify binding structure: ((%ctx expr) (%val expr))

    ;; Test 8: Two operations DAG
    (test-equal "two-ops: parses as AST" #t (list? pattern-two-ops-parse))
    (test-equal "two-ops: has match ops" #t
                (pair? (ast-get pattern-two-ops-parse 'match)))
    (test-equal "two-ops: has 2 operations" 2
                (length (ast-get pattern-two-ops-parse 'match)))

    ;; Test 9: Variable reuse
    (test-equal "reuse: parses as AST" #t (list? pattern-reuse-parse))
    (test-equal "reuse: has match ops" #t
                (pair? (ast-get pattern-reuse-parse 'match)))
    ;; TODO: Verify operand list shows %x appears twice

    ;; Test 10: Combined features
    (test-equal "combined: parses as AST" #t (list? pattern-combined-parse))
    (test-equal "combined: has match ops" #t
                (pair? (ast-get pattern-combined-parse 'match)))
    (test-equal "combined: has where bindings" #t
                (pair? (ast-get pattern-combined-parse 'where)))
    (test-equal "combined: has 2 bindings" 2
                (length (ast-get pattern-combined-parse 'where)))

    ;; Test 11: Regions
    (test-equal "regions: parses as AST" #t (list? pattern-region-parse))
    (test-equal "regions: has rewrite ops" #t
                (pair? (ast-get pattern-region-parse 'rewrite)))
    ;; TODO: Verify rewrite operation has regions field

    ;; Test 12: Multiple rewrite operations
    (test-equal "multi-rewrite: parses as AST" #t (list? pattern-multi-rewrite-parse))
    (test-equal "multi-rewrite: has rewrite ops" #t
                (pair? (ast-get pattern-multi-rewrite-parse 'rewrite)))
    ;; TODO: Verify rewrite list contains 3 operations

    (display "\nAll parse tests use shared patterns from test-patterns.sls\n")

    (test-end)))
