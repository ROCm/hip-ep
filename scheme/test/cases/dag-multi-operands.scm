;; Test case: dag-multi-operands
;; Multiple operands - same variable used twice

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a %a)
   :rewrite %b :with (op3 (%b) -> !t)))
