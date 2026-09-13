(library (test google protobuf test student)
  (export google-protobuf-test-student)
  (import
    (rnrs (6))
    (rime protobuf define-proto)
    (test google protobuf test school))
  (define google-protobuf-test-student-deps google-protobuf-test-school)
  (define-proto (library (google protobuf test student))
    (package (google protobuf test))
    (message
     (google protobuf test Student)
     ((optional int32 id 1 (option)) (optional string name 2 (option))
      (optional
       (message (google protobuf test Location))
       location
       3
       (option))
      (optional
       (message (google protobuf test School))
       school
       4
       (option))
      (optional sint32 age 5 (option))
      (repeated string friend 6 (option))
      (optional
       (message (google protobuf test Student Family))
       family
       7
       (option))
      (optional
       (enum (google protobuf test Student Sex))
       sex
       8
       (option))))
    (message
     (google protobuf test Student Family)
     ((optional string father 1 (option))
      (optional string mother 2 (option))))
    (message
     (google protobuf test Student Family Address)
     ((optional string country 1 (option))
      (optional string city 2 (option))))
    (message
     (google protobuf test Student Family TestNestedType)
     ((optional string f1 1 (option))
      (optional string f2 2 (option))))
    (message
     (google protobuf test Location)
     ((optional sint32 x 1 (option))
      (optional sint32 y 2 (option))))))
