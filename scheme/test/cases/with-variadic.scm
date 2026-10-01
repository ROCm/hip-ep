;; Test case: with-variadic
;; Pattern with &variadic operands

(:pattern
  (:if-match %a = test.concat (%x (&variadic %rest))
   :rewrite %a :with (new.concat (%x) -> !t)))
