;; Test case: dag-unreachable
;; Unreachable operation (negative test case - op1 not reachable from root)

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%y)
   :rewrite %b :with (op3 (%b) -> !t)))
