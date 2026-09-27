;; Test case: codegen-test
;; Test :debug-codegen flag - should return generated code as datum

(:pattern
  (:match %out = test.op ()
   :debug-codegen
   :rewrite %out :with (new.op () -> !t)))
