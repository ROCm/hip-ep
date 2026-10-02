;; Test case: with-then-let
;; Pattern with :then-let clause for global bindings

(:pattern
  (:if-match %out = test.op (%in)
   :then-let ((%ctx (get-context))
              (%val (compute-value)))
   :rewrite %out :with (new.op (%ctx %in %val) -> !t)))
