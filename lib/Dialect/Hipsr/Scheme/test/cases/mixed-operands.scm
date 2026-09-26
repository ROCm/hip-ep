;; Test case: mixed-operands
;; Pattern with mixed &optional and &variadic operands

(:pattern
  (:match %a = test.op (%x (&optional %y) %z (&variadic %rest))
   :rewrite %a :with (new.op (%x %z) -> !t)))
