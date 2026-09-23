;; Test case: dag-chain-3ops
;; Chain of 3 operations: A→B→C

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
          %c = op3 (%b)
   :rewrite %c :with (op4 (%c) -> !t)))
