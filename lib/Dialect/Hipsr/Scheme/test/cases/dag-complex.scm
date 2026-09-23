;; Test case: dag-complex
;; Complex DAG - 5 ops with diamond + chain

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
          %c = op3 (%a)
          %d = op4 (%b %c)
          %e = op5 (%d)
   :rewrite %e :with (op6 (%e) -> !t)))
