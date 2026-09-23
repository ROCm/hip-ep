#!r6rs
;;===----------------------------------------------------------------------===;;
;; Show Pattern Output - For Troubleshooting Individual Tests
;;===----------------------------------------------------------------------===;;

(library-directories '("." "libraries" "../../../../third_party/rime"))

(import (except (chezscheme) =)
        (mlir pattern-macro)
        (test test-helpers))

;; Import pattern-macro into interaction-environment
(eval '(import (mlir pattern-macro)) (interaction-environment))

(define (show-test test-name phase)
  (let* ([test-bodies (load-test-bodies)]
         [test-case (loop :for tc :in test-bodies
                          :if (eq? (car tc) test-name)
                          :break tc
                          :finally #f)])
    (if test-case
        (let ([pattern (get-field ':pattern test-case)])
          (display (format "Test: ~a, Phase: ~a\n\n" test-name phase))
          (show-pattern-output test-name
                               (case phase
                                 [(parse) ':debug-parse]
                                 [(validate) ':debug-validate]
                                 [(analyze) ':debug-analyze]
                                 [(codegen) #f]
                                 [else (error 'show-test "Unknown phase" phase)])
                               pattern))
        (display (format "Test ~a not found\n" test-name)))))

;; Usage: scheme --script test/show-test.scm
;; Then at REPL: (show-test 'basic 'parse)
(display "Usage: (show-test 'test-name 'phase)\n")
(display "Phases: parse, validate, analyze, codegen\n")
(display "Example: (show-test 'basic 'parse)\n\n")
