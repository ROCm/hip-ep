#!r6rs
(import (rnrs (6))
        (rnrs eval (6))
        (rime loop)
        (rime logging))
(define (show-deps scm_file)
  (call-with-input-file scm_file
    (lambda (port)
      (loop :for import-specs :in (read port)
            :if (and (list? import-specs)
                     (eq? (car import-specs)  'import))
            (:loop :for x-import-spec :in (cdr import-specs)
                   :with import-spec :=
                   (cond
                    [(and (list? x-import-spec)
                          (eq? (car x-import-spec)  'rime))
                     x-import-spec]
                    [(and (list? x-import-spec)
                          (eq? (car x-import-spec)  'for))
                     (cadr x-import-spec)]
                    [else '()])
                   :unless (null? import-spec)
                   :do (logger :info scm_file " "
                               (loop :initially :return-value := ""
                                     sep := ""
                                     :for s :in import-spec
                                     :with :return-value := (string-append :return-value sep (symbol->string s))
                                     :with sep := "/"
                                     :finally (string-append :return-value ".scm")
                                     )) :if #f
                   :do (begin
                         (display scm_file)
                         (display " => ")
                         (display import-spec)
                         (newline)
                         )
                                        )))))

(define (main args)
  (loop :for arg :in args
        :do (show-deps arg)))

(main (command-line))
