#!/usr/bin/env scheme-script
;; Test cases for code generation - review generated code
(import (except (chezscheme) =)
        (only (chezscheme) pretty-print format)
        (mlir pattern-macro))

;; Test 1: Simple chain (2 ops)
(define-conversion-pattern :debug-codegen test-chain-2
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%a) -> !t3)))

(format #t "\n=== Test 1: Chain 2 ops ===\n")
(pretty-print test-chain-2)

;; Test 2: Chain 3 ops
(define-conversion-pattern :debug-codegen test-chain-3
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%b) () : (!t3) -> !t4))
  :rewrite %c :with
  ((%out = "op4" (%c) -> !t4)))

(format #t "\n=== Test 2: Chain 3 ops ===\n")
(pretty-print test-chain-3)

;; Test 3: Diamond pattern
(define-conversion-pattern :debug-codegen test-diamond
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3)
          (%c = "op3" (%a) () : (!t2) -> !t4)
          (%d = "op4" (%b %c) () : (!t3 !t4) -> !t5))
  :rewrite %d :with
  ((%out = "op5" (%d) -> !t5)))

(format #t "\n=== Test 3: Diamond pattern ===\n")
(pretty-print test-diamond)

;; Test 4: Multiple operands (same variable used twice)
(define-conversion-pattern :debug-codegen test-multi-operands
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a %a) () : (!t2 !t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%b) -> !t3)))

(format #t "\n=== Test 4: Multi operands ===\n")
(pretty-print test-multi-operands)

;; Test 5: Single op with free variable
(define-conversion-pattern :debug-codegen test-single-op
  :match ((%a = "op1" (%x) () : (!t1) -> !t2))
  :rewrite %a :with
  ((%out = "op2" (%x) -> !t2)))

(format #t "\n=== Test 5: Single op with free variable ===\n")
(pretty-print test-single-op)

(format #t "\n=== All codegen tests complete ===\n")
