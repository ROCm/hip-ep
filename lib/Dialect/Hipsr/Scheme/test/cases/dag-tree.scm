;; Test case: dag-tree
;; Tree - multiple variables referenced in rewrite

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
          %c = op3 (%b)
   :rewrite %c :with (op4 (%a %b %c) -> !t)))
