;; Test case: dag-chain-2ops
;; Simple chain - A uses result from B

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
   :rewrite %b :with (op3 (%a) -> !t)))
