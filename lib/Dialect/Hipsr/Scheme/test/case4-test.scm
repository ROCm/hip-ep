#!/usr/bin/env scheme-script
;; Comprehensive test cases for action generation
;; Verify DAG traversal and action ordering before codegen

(import (except (chezscheme) =)
        (only (chezscheme) pretty-print format)
        (mlir pattern-macro))

(define (show-actions name pattern-data)
  (format #t "\n=== ~a ===\n" name)
  (let ([actions-list (cadr (memq 'match-actions pattern-data))])
    (for-each (lambda (action)
                (format #t "  ~a\n" action))
              actions-list)))

;; Test 1: Simple chain - A uses result from B
(define-conversion-pattern :debug-ast test-chain-2ops
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%a) -> !t3)))

(show-actions "Chain 2 ops: %b uses %a" test-chain-2ops)
(format #t "  Expected: bind-root %b → check op2 → recurse to op1 → check op1 → bind %x\n")

;; Test 2: Chain of 3 operations
(define-conversion-pattern :debug-ast test-chain-3ops
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%b) () : (!t3) -> !t4))
  :rewrite %c :with
  ((%out = "op4" (%c) -> !t4)))

(show-actions "Chain 3 ops: %c uses %b uses %a" test-chain-3ops)
(format #t "  Expected: bind-root %c → recurse to op2 → recurse to op1 → bind all\n")

;; Test 3: Diamond - two paths converge
(define-conversion-pattern :debug-ast test-diamond
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%a) () : (!t2) -> !t4)
          (%d = "op4" (%b %c) () : (!t3 !t4) -> !t5))
  :rewrite %d :with
  ((%out = "op5" (%d) -> !t5)))

(show-actions "Diamond: %d uses %b and %c, both use %a" test-diamond)
(format #t "  Expected: %a visited once, then %b and %c, then %d\n")

;; Test 4: Multiple operands - same operation
(define-conversion-pattern :debug-ast test-multi-operands
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a %a) () : (!t2 !t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%b) -> !t3)))

(show-actions "Multi operands: %b uses %a twice" test-multi-operands)
(format #t "  Expected: First %a binds, second %a checks equality\n")

;; Test 5: Tree - one source, multiple consumers
(define-conversion-pattern :debug-ast test-tree
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%b) () : (!t3) -> !t4))
  :rewrite %c :with
  ((%out = "op4" (%a %b %c) -> !t5)))

(show-actions "Tree: %c references %a, %b, %c" test-tree)

;; Test 6: Multiple results - TODO: syntax not supported yet
;; (define-conversion-pattern :debug-ast test-multi-results
;;   :match ((%a %b = "op1" (%x) () : (!t1) -> !t2 !t3)
;;           (%c = "op2" (%a %b) () : (!t2 !t3) -> !t4))
;;   :rewrite %c :with
;;   ((%out = "op3" (%c) -> !t4)))
;; (show-actions "Multi results: op1 produces %a and %b" test-multi-results)
(format #t "\n=== Multi results: SKIPPED (syntax not supported yet) ===\n")

;; Test 7: Unreachable operation (negative case)
(define-conversion-pattern :debug-ast test-unreachable
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%y) () : (!t3) -> !t4))
  :rewrite %b :with
  ((%out = "op3" (%b) -> !t4)))

(show-actions "Unreachable: op1 not reachable from root %b" test-unreachable)
(format #t "  Expected: WARNING about op1 being unreachable\n")

;; Test 8: Complex DAG
(define-conversion-pattern :debug-ast test-complex-dag
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%a) () : (!t2) -> !t4)
          (%d = "op4" (%b %c) () : (!t3 !t4) -> !t5)
          (%e = "op5" (%d) () : (!t5) -> !t6))
  :rewrite %e :with
  ((%out = "op6" (%e) -> !t6)))

(show-actions "Complex DAG: 5 ops with diamond + chain" test-complex-dag)
(format #t "  Expected: Proper DAG traversal visiting each op once\n")

(format #t "\n=== All test cases complete ===\n")
(format #t "Manually verify action sequences match expected DAG traversal order.\n")
