#!/usr/bin/env scheme-script
;; Comprehensive test cases for action generation
;; Verify DAG traversal and action ordering before codegen

(import (except (chezscheme) =)
        (only (chezscheme) pretty-print format)
        (mlir pattern-macro))

(define (get-actions pattern-data)
  (cadr (memq 'match-actions pattern-data)))

(define (show-actions name pattern-data)
  (format #t "\n=== ~a ===\n" name)
  (let ([actions-list (get-actions pattern-data)])
    (format #t "Actions as s-expr:\n  ~s\n\n" actions-list)
    (format #t "Actions formatted:\n")
    (for-each (lambda (action)
                (format #t "  ~a\n" action))
              actions-list)))

(define (verify-actions name pattern-data expected-actions)
  (let ([actual (get-actions pattern-data)])
    (if (equal? actual expected-actions)
        (format #t "  ✓ PASS\n")
        (begin
          (format #t "  ✗ FAIL\n")
          (format #t "    Expected: ~a\n" expected-actions)
          (format #t "    Actual:   ~a\n" actual)
          (error 'verify-actions "Test failed" name)))))

;; Test 1: Simple chain - A uses result from B
(define-conversion-pattern :debug-analyze test-chain-2ops
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%a) -> !t3)))

(show-actions "Chain 2 ops: %b uses %a" test-chain-2ops)
(verify-actions "Chain 2 ops"
                test-chain-2ops
                '((:bind-root %b 1 0)
                  (:set-current-op 1 %b 0)
                  (:check-op 1)
                  (:bind-operand 0 %x 0)
                  (:check-op 0)
                  (:set-current-op 0 %a 0)))

;; Test 2: Chain of 3 operations
(define-conversion-pattern :debug-analyze test-chain-3ops
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%b) () : (!t3) -> !t4))
  :rewrite %c :with
  ((%out = "op4" (%c) -> !t4)))

(show-actions "Chain 3 ops: %c uses %b uses %a" test-chain-3ops)
(verify-actions "Chain 3 ops"
                test-chain-3ops
                '((:bind-root %c 2 0)
                  (:set-current-op 2 %c 0)
                  (:check-op 2)
                  (:set-current-op 0 %a 0)
                  (:check-op 0)
                  (:bind-operand 0 %x 0)
                  (:check-op 1)
                  (:set-current-op 1 %b 0)))

;; Test 3: Diamond - two paths converge
(define-conversion-pattern :debug-analyze test-diamond
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%a) () : (!t2) -> !t4)
          (%d = "op4" (%b %c) () : (!t3 !t4) -> !t5))
  :rewrite %d :with
  ((%out = "op5" (%d) -> !t5)))

(show-actions "Diamond: %d uses %b and %c, both use %a" test-diamond)
(verify-actions "Diamond"
                test-diamond
                '((:bind-root %d 3 0)
                  (:set-current-op 3 %d 0)
                  (:check-op 3)
                  (:set-current-op 0 %a 0)
                  (:check-op 0)
                  (:bind-operand 0 %x 0)
                  (:check-op 1)
                  (:set-current-op 1 %b 0)
                  (:check-eq 2 0 %a)
                  (:check-op 2)
                  (:set-current-op 2 %c 0)))

;; Test 4: Multiple operands - same operation
(define-conversion-pattern :debug-analyze test-multi-operands
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a %a) () : (!t2 !t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%b) -> !t3)))

(show-actions "Multi operands: %b uses %a twice" test-multi-operands)
(verify-actions "Multi operands"
                test-multi-operands
                '((:bind-root %b 1 0)
                  (:set-current-op 1 %b 0)
                  (:check-op 1)
                  (:bind-operand 0 %x 0)
                  (:check-op 0)
                  (:set-current-op 0 %a 0)
                  (:check-eq 1 1 %a)))

;; Test 5: Tree - one source, multiple consumers
(define-conversion-pattern :debug-analyze test-tree
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%b) () : (!t3) -> !t4))
  :rewrite %c :with
  ((%out = "op4" (%a %b %c) -> !t5)))

(show-actions "Tree: %c references %a, %b, %c" test-tree)
(verify-actions "Tree"
                test-tree
                '((:bind-root %c 2 0)
                  (:set-current-op 2 %c 0)
                  (:check-op 2)
                  (:set-current-op 0 %a 0)
                  (:check-op 0)
                  (:bind-operand 0 %x 0)
                  (:check-op 1)
                  (:set-current-op 1 %b 0)))

;; Test 6: Multiple results - TODO: syntax not supported yet
;; (define-conversion-pattern :debug-analyze test-multi-results
;;   :match ((%a %b = "op1" (%x) () : (!t1) -> !t2 !t3)
;;           (%c = "op2" (%a %b) () : (!t2 !t3) -> !t4))
;;   :rewrite %c :with
;;   ((%out = "op3" (%c) -> !t4)))
;; (show-actions "Multi results: op1 produces %a and %b" test-multi-results)
(format #t "\n=== Multi results: SKIPPED (syntax not supported yet) ===\n")

;; Test 7: Unreachable operation (negative case)
(define-conversion-pattern :debug-analyze test-unreachable
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%y) () : (!t3) -> !t4))
  :rewrite %b :with
  ((%out = "op3" (%b) -> !t4)))

(show-actions "Unreachable: op1 not reachable from root %b" test-unreachable)
(verify-actions "Unreachable"
                test-unreachable
                '((:bind-root %b 1 0)
                  (:set-current-op 1 %b 0)
                  (:check-op 1)
                  (:bind-operand 1 %y 0)))

;; Test 8: Complex DAG
(define-conversion-pattern :debug-analyze test-complex-dag
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%a) () : (!t2) -> !t4)
          (%d = "op4" (%b %c) () : (!t3 !t4) -> !t5)
          (%e = "op5" (%d) () : (!t5) -> !t6))
  :rewrite %e :with
  ((%out = "op6" (%e) -> !t6)))

(show-actions "Complex DAG: 5 ops with diamond + chain" test-complex-dag)
(verify-actions "Complex DAG"
                test-complex-dag
                '((:bind-root %e 4 0)
                  (:set-current-op 4 %e 0)
                  (:check-op 4)
                  (:set-current-op 2 %c 0)
                  (:check-op 2)
                  (:check-eq 2 0 %a)
                  (:set-current-op 1 %b 0)
                  (:check-op 1)
                  (:bind-operand 0 %x 0)
                  (:check-op 0)
                  (:set-current-op 0 %a 0)
                  (:check-op 3)
                  (:set-current-op 3 %d 0)))

(format #t "\n=== All test cases complete ===\n")
(format #t "All action sequences verified automatically.\n")
