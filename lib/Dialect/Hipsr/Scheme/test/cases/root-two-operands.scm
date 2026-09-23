;; Test case: root-two-operands
;; Root operation has two operands: one from producer, one free variable

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a %y)
   :rewrite %b :with (op3 (%a %y) -> !t)))
