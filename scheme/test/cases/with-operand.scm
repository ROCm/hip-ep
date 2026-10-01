;; Test case: with-operand
;; Pattern with one operand (input and output types)

(:pattern
  (:if-match %out = test.op (%in)
   :rewrite %out :with (new.op (%in) -> !t)))
