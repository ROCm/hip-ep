;; Test Pattern Bodies - Single Source of Truth

(
  ;; Simplest test: operation with no operands
  (basic
   :pattern (
     :match %out = "test.op" ()
     :rewrite %out :with ("new.op" () -> !t)
   )
   :expect-parse (
     (has-function-name . #t)
     (root-op-name . "test.op")
     (match-count . 1)
   )
  )
)
