#!r6rs
(library (rime loop keywords)
  (export
   :for :as :and :from :downfrom :upfrom :to :downto :upto :below :above :by :collect :count :into :append :do :with :=
   :in :on :across :in-vector
   :in-string :offset
   :in-hashtable :being :each :the :hash-key :hash-keys :hash-value :hash-values :of :using
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
   assq-id
   new-sym
   keyword?
   keyword=?
   one-of
   object-to-string ;; for debugging purpose
   loop-level
   loop-level++
   loop-trace-parser
   loop-trace-codegen
   loop-init-props
   loop-return-value
   logger :info :debug :trace
   )
  (import (rnrs (6))
          (rime logging))

  (define-syntax define-keyword
    (syntax-rules ()
      [(_ keyword) (define-syntax keyword
                     (lambda (x)
                       (syntax-violation
                        'loop-keyword "misplaced aux keyword"
                        x)))]))
  (define-keyword :for)
  (define-keyword :as)
  (define-keyword :from)
  (define-keyword :downfrom)
  (define-keyword :upfrom)
  (define-keyword :to)
  (define-keyword :downto)
  (define-keyword :up)
  (define-keyword :upto)
  (define-keyword :below)
  (define-keyword :above)
  (define-keyword :by)
  (define-keyword :and)
  (define-keyword :collect)
  (define-keyword :count)
  (define-keyword :into)
  (define-keyword :append)
  (define-keyword :do)
  (define-keyword :with)
  (define-keyword :=)
  (define-keyword :in)
  (define-keyword :on)
  (define-keyword :in-string)
  (define-keyword :offset)
  (define-keyword :across)
  (define-keyword :in-vector)
  (define-keyword :being)
  (define-keyword :in-hashtable)
  (define-keyword :each)
  (define-keyword :the)
  (define-keyword :hash-key)
  (define-keyword :hash-keys)
  (define-keyword :hash-value)
  (define-keyword :hash-values)
  (define-keyword :using)
  (define-keyword :finally)
  (define-keyword :break)
  (define-keyword :repeat)
  (define-keyword :if)
  (define-keyword :when)
  (define-keyword :unless)
  (define-keyword :of)
  (define-keyword :initially)
  (define-keyword :loop)
  (define-keyword :name)
  (define-keyword :loop-level)
  (define-keyword :trace-parser)
  (define-keyword :trace-codegen)
  (define-keyword :join-string)
  (define-keyword :seperator)
  (define-keyword :expr)
  (define-keyword :list)
  (define-keyword :hash-table)
  (define-keyword :make-hash-table)
  (define-keyword :group)
  (define-keyword :recur)
  (define-keyword :then)
  (define-keyword :in-directory)
  (define-keyword :reverse)

  (define (keyword? e)
    (exists (lambda (keyword)
              (and (identifier? e)
                   (free-identifier=? e keyword)))
            (list
             (syntax :for)
             (syntax :as)
             (syntax :from)
             (syntax :downfrom)
             (syntax :upfrom)
             (syntax :to)
             (syntax :downto)
             (syntax :up)
             (syntax :upto)
             (syntax :below)
             (syntax :above)
             (syntax :by)
             (syntax :and)
             (syntax :collect)
             (syntax :count)
             (syntax :into)
             (syntax :append)
             (syntax :do)
             (syntax :with)
             (syntax :=)
             (syntax :in)
             (syntax :on)
             (syntax :in-string)
             (syntax :offset)
             (syntax :across)
             (syntax :in-vector)
             (syntax :being)
             (syntax :in-hashtable)
             (syntax :each)
             (syntax :the)
             (syntax :hash-key)
             (syntax :hash-keys)
             (syntax :hash-value)
             (syntax :hash-values)
             (syntax :using)
             (syntax :finally)
             (syntax :break)
             (syntax :repeat)
             (syntax :if)
             (syntax :when)
             (syntax :unless)
             (syntax :of)
             (syntax :initially)
             (syntax :loop)
             (syntax :name)
             (syntax :loop-level)
             (syntax :trace-parser)
             (syntax :trace-codegen)
             (syntax :join-string)
             (syntax :seperator)
             (syntax :expr)
             (syntax :list)
             (syntax :hash-table)
             (syntax :make-hash-table)
             (syntax :group)
             (syntax :recur)
             (syntax :then)
             (syntax :in-directory)
             (syntax :reverse)
             )))

  (define (keyword=? k1 k2)
    (and (keyword? k1)
         (keyword? k2)
         (free-identifier=? k1 k2)))

  (define (one-of e ids)
    (exists (lambda (id) (free-identifier=? id e)) ids))

  (define (loop-init-props k trace-parser trace-codegen)
    (with-syntax ([k k]
                  [trace-parser trace-parser]
                  [trace-codegen trace-codegen]
                  )
      #'(k (:loop-level . 0)
           (:trace-parser . trace-parser)
           (:trace-codegen . trace-codegen)
           )))
  (define (loop-level k)
    (loop-get-prop k #':loop-level))

  (define (loop-level++ k)
    (loop-add-prop k
                   #':loop-level
                   (fx+ 1 (loop-level k))))


  (define loop-trace-parser
    (case-lambda
      [(k) (loop-get-prop k #':trace-parser)]
      [(k v) (loop-add-prop k #':trace-parser v)]))

  (define loop-trace-codegen
    (case-lambda
      [(k) (loop-get-prop k #':trace-codegen)]
      [(k v) (loop-add-prop k #':trace-codegen v)]))


  (define (loop-get-prop origin-k prop)
    (let repeat ([k origin-k])
      (syntax-case k ()
        [(loop (key . value) rest-props ...)
         (and (identifier? #'loop)
              (free-identifier=? #'key prop))
         (syntax->datum #'value)
         ]
        [(loop p1 rest-props ...)
         (identifier? #'loop)
         (repeat #'(_k rest-props ...))
         ]
        [(loop-and-props rest ...)
         (not (identifier? #'loop-and-props))
         (repeat #'loop-and-props)
         ]
        [(_k)
         #f])))

  (define (loop-add-prop origin-k prop value)
    (let ([k (loop-get-k origin-k)])
      (with-syntax ([k k]
                    [(props ...) (loop-get-props origin-k)]
                    [prop  prop]
                    [value  (datum->syntax k value)])
        #'(k (prop . value ) props ...))))

  (define (loop-get-k e)
    (syntax-case e ()
      [(k props ...) #'k]))

  (define (loop-get-props e)
    (syntax-case e ()
      [(k props ...) #'(props ...)]))

  (define (new-sym e hint)
    (cond
     [(and (identifier? e) (eq? hint #f))
      (car (generate-temporaries (list e)))]

     [(and (identifier? e) (string? hint))
      (datum->syntax e (string->symbol
                        (string-append
                         (symbol->string (syntax->datum e))
                         "-" hint)))]
     [(string? hint)
      (car (generate-temporaries (list (loop-get-k e))))]
     [(symbol? hint)
      (datum->syntax (loop-get-k e) hint)]
     [else (raise "ERROR")])
    )

  (define (loop-return-value e)
    (new-sym e ':return-value))

  (define assq-id
    (case-lambda
      [(id prop-list)
       (cond
        [(assp (lambda (k)
                 (free-identifier=? (datum->syntax #'assp-id id) k)) prop-list) => cdr]
        [else (raise
               (condition
                (make-who-condition 'assp-id)
                (make-message-condition "cannot find required value")
                (make-irritants-condition
                 (list (cons 'id  id) (cons 'prop-list prop-list))))
               )])]
      [(id prop-list default)
       (cond
        [(assp (lambda (k)
                 (free-identifier=? (datum->syntax #'assp-id id) k)) prop-list) => cdr]
        [else default])])
    ))
