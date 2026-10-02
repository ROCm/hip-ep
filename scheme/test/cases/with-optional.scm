;; Test case: with-optional
;; Pattern with &optional operands

(:pattern
  (:if-match %a = test.op (%x (&optional %y %z))
   :rewrite %a :with (new.op (%x) -> !t)))
