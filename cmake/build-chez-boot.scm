(import (chezscheme))

(define (main args)
  (when (< (length args) 3)
    (fprintf (current-error-port)
             "Usage: scheme --script build-chez-boot.scm petite.boot scheme.boot output.boot\n")
    (exit 1))

  (let ([petite-boot (list-ref args 0)]
        [scheme-boot (list-ref args 1)]
        [output-boot (list-ref args 2)])

    (fprintf (current-output-port) "Creating boot file...\n")
    (fprintf (current-output-port) "  Source: ~a\n" scheme-boot)
    (fprintf (current-output-port) "  Output: ~a\n" output-boot)

    (call-with-port (open-file-output-port output-boot (file-options replace))
      (lambda (out)
        (call-with-port (open-file-input-port scheme-boot)
          (lambda (in)
            (let loop ()
              (let ([byte (get-u8 in)])
                (unless (eof-object? byte)
                  (put-u8 out byte)
                  (loop))))))))

    (fprintf (current-output-port) "Boot file created successfully.\n")))

(main (cdr (command-line)))
