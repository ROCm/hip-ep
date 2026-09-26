;; Test case: dag-diamond
;; Diamond - two paths converge (A→B,C→D)

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
          %c = op3 (%a)
          %d = op4 (%b %c)
   :rewrite %d :with (op5 (%d) -> !t)))
