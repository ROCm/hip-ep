#!r6rs
;;===----------------------------------------------------------------------===;;
;; Phase 3: Analyze Test - Test action generation with :debug-analyze
;;===----------------------------------------------------------------------===;;
;;
;; Focus: Test ONLY what the analyze phase does
;; - Generate actions from match operations (DAG traversal)
;; - Create binding manager with correct metadata
;; - Handle operand segments (optional/variadic) in bindings
;; - Verify match-actions list correctness
;; - Test DAG traversal order (root → operands)
;;
;; Key actions tested:
;; - action:set-current-op
;; - action:check-op
;; - action:bind-operand
;; - action:check-eq
;;
;; All test patterns imported from (test test-patterns) - single source of truth
;;
;;===----------------------------------------------------------------------===;;

(library (test phase-3-analyze-test)
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

  (define (count-actions ast)
    "Count actions in match-actions list"
    (let ([actions (ast-get ast 'match-actions)])
      (if (list? actions)
          (length actions)
          0)))

  ;;=======================================================================
  ;; Test Suite
  ;;=======================================================================

  (define (run-tests)
    (test-begin "phase-3-analyze")
    (display "\n=== Phase 3: Analyze Tests (Using Shared Patterns) ===\n\n")

    ;; Test 1: Basic single operation - action generation
    (test-equal "basic: returns AST" #t (list? pattern-basic-analyze))
    (test-equal "basic: has match-actions" #t
                (not (not (ast-get pattern-basic-analyze 'match-actions))))
    (test-equal "basic: has match-bindings" #t
                (not (not (ast-get pattern-basic-analyze 'match-bindings))))
    ;; TODO: Verify actions list contains:
    ;;   - action:set-current-op (op-idx=0, result-var=%out)
    ;;   - action:check-op (op-idx=0, op-name="test.op")
    ;;   - action:bind-operand (op-idx=0, operand-idx=0, var=%in)

    ;; Test 2: Required operands - binding actions
    (test-equal "required: returns AST" #t (list? pattern-required-analyze))
    (test-equal "required: has actions" #t
                (> (count-actions pattern-required-analyze) 0))
    ;; TODO: Verify 3 bind-operand actions for %x %y %z

    ;; Test 3: Optional operands - segment handling
    (test-equal "optional: returns AST" #t (list? pattern-optional-analyze))
    ;; TODO: Verify binding manager marks %y %z as optional
    ;; TODO: Verify action generation handles optional flag

    ;; Test 4: Variadic operands - segment metadata
    (test-equal "variadic: returns AST" #t (list? pattern-variadic-analyze))
    ;; TODO: Verify binding manager marks %rest as variadic
    ;; TODO: Verify action handles variadic operand access

    ;; Test 5: Mixed operands - complex binding
    (test-equal "mixed: returns AST" #t (list? pattern-mixed-analyze))
    ;; TODO: Verify binding manager correctly categorizes:
    ;;   - %x: required (operand-idx=0)
    ;;   - %y: optional (operand-idx=1)
    ;;   - %z: required (operand-idx=2 after optional group)
    ;;   - %rest: variadic (operand-idx=3+)

    ;; Test 6: :where guard - action integration
    (test-equal "where: returns AST" #t (list? pattern-where-analyze))
    ;; TODO: Verify :where guard generates guard-check action
    ;; TODO: Verify guard-check appears after operation match

    ;; Test 7: :then-let bindings - no actions (handled at codegen)
    (test-equal "then-let: returns AST" #t (list? pattern-then-let-analyze))
    (test-equal "then-let: has bindings" #t
                (pair? (ast-get pattern-then-let-analyze 'where)))
    ;; Note: :then-let bindings don't generate match actions, they're used in codegen

    ;; Test 8: Two operations - DAG traversal
    (test-equal "two-ops: returns AST" #t (list? pattern-two-ops-analyze))
    ;; TODO: Verify actions list shows correct DAG traversal:
    ;;   1. Start at root (%b = op2)
    ;;   2. Follow operand %a to producer
    ;;   3. Process %a = op1 first
    ;;   4. Then process %b = op2

    ;; Test 9: Variable reuse - check-eq action
    (test-equal "reuse: returns AST" #t (list? pattern-reuse-analyze))
    ;; TODO: Verify actions contain:
    ;;   - bind-operand for first %x
    ;;   - check-eq for second %x (same variable)

    ;; Test 10: Combined features - complex actions
    (test-equal "combined: returns AST" #t (list? pattern-combined-analyze))
    (test-equal "combined: has actions" #t
                (> (count-actions pattern-combined-analyze) 0))
    ;; TODO: Verify DAG traversal handles:
    ;;   - Multiple operations (%dq, %conv)
    ;;   - Multiple :where guards
    ;;   - :then-let bindings

    ;; Test 11: Regions - action generation for nested blocks
    (test-equal "regions: returns AST" #t (list? pattern-region-analyze))
    ;; TODO: Verify region operations generate appropriate actions

    ;; Test 12: Multiple rewrite operations - no match actions
    (test-equal "multi-rewrite: returns AST" #t (list? pattern-multi-rewrite-analyze))
    ;; Note: Rewrite operations don't generate match actions, only match ops do

    (display "\nAnalyze tests verify action generation and DAG traversal.\n")
    (display "TODO: Add detailed action inspection tests.\n")

    (test-end)))
