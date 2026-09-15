#!r6rs
(library (rime rime-0 __feature-registry)
  (export check-library check-feature)
  (import (rnrs (6))
          (rnrs eval (6)))

  (define (import-spec-exists? import-spec)
    (guard
        (exn
         [else #f])
      (eval #t (environment import-spec))))

  (define (define-feature-if-import-set feature-id import-spec)
    (cons feature-id (import-spec-exists? import-spec)))

  (define (check-window-is-available)
    (cond
     [(import-spec-exists? '(chezscheme))
      (let ([machine-type ((eval 'machine-type (environment '(chezscheme))))])
        (case machine-type
          [(i3nt ti3nt a6nt ta6nt arm64nt tarm64nt) #t]
          [else #f]))]
     [else #f]))

  (define (check-darwin-is-available)
    (cond
     [(import-spec-exists? '(chezscheme))
      (let ([machine-type ((eval 'machine-type (environment '(chezscheme))))])
        (case machine-type
          [(i3osx ti3osx a6osx  ta6osx ppc32osx tppc32osx arm64osx tarm64osx)
           #t]
          [else #f]))]

     [(import-spec-exists? '(guile))
      (string=? (eval '(utsname:sysname (uname)) (environment '(guile))) "Darwin")]
     [else #f]))

  (define (check-linux-is-available)
    (cond
     [(import-spec-exists? '(chezscheme))
      (let ([machine-type ((eval 'machine-type (environment '(chezscheme))))])
        (case machine-type
          [(i3le ti3le a6le ta6le ppc32le tppc32le arm32le tarm32le arm64le tarm64le rv64le trv64le la64le tla64le)
           #t]
          [else #f]))]
     [(import-spec-exists? '(guile))
      (string=? (eval '(utsname:sysname (uname)) (environment '(guile))) "Linux")]
     [else #f]))

  (define feature-registry
    (list
     (define-feature-if-import-set 'guile '(guile))
     (define-feature-if-import-set 'chezscheme '(chezscheme))
     (cons 'windows (check-window-is-available))
     (cons 'darwin (check-darwin-is-available))
     (cons 'linux (check-linux-is-available))
     ))

  (define (check-feature feature-id)
    (cond
     [(assq feature-id feature-registry) => cdr]
     [else #f]) )

  (define (check-library import-spec)
    (import-spec-exists? import-spec)))
