#!r6rs
(library (rime loop)
  (export loop
          :for :as :and :from :downfrom :upfrom :to :downto :upto :below :above :by  :collect :count :into :append :do :with :=
          :in :in-string :on :across :in-vector :offset
          :being :each :the :in-hashtable :hash-key :hash-keys :hash-value :hash-values :of :using
          :finally
          :break
          :repeat
          :if :when :unless
          :initially
          :loop
          :name
          :trace-parser
          :trace-codegen
          :join-string
          :seperator
          :expr
          :list
          :hash-table
          :make-hash-table
          :group
          :recur
          :then
          :in-directory
          :reverse
          )
  (import (rnrs (6))
          (rime loop keywords)
          (for (rime loop core) expand))

  (define-syntax loop
    (lambda (original-e)
      (let loop ([e original-e]
                 [trace-parser #f]
                 [trace-codegen #f])
        (syntax-case e (:loop :trace-parser :trace-codegen)
          [((k props ...) (:loop clauses ...))
           (let-values ([(clauses s-k)
                         (parse-loop-clauses
                          #'((k props ...)
                             (:loop clauses ...)))])
             (loop-codegen s-k clauses original-e))]
          [(k :trace-parser clauses ...)
           (identifier? #'k)
           (loop #'(k clauses ...) #t trace-codegen)
           ]
          [(k :trace-codegen clauses ...)
           (identifier? #'k)
           (loop #'(k clauses ...) trace-parser #t)
           ]
          [(k clauses ...)
           (identifier? #'k)
           (with-syntax ([k (loop-init-props #'k trace-parser trace-codegen)])
             (loop #'(k (:loop clauses ...)) trace-parser trace-codegen))
           ]
          )))))
