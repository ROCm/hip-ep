//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_TfIdfVectorizerOp
//===----------------------------------------------------------------------===//

def Onnx_TfIdfVectorizerOp : Onnx_Op<"TfIdfVectorizer", [Pure]> {
  let summary = "ONNX TfIdfVectorizer, schema version 9";

  let description = [{
    This transform extracts n-grams from the input sequence and save them as a vector. Input can
    be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
    For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
    More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
    If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

    In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
    sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
    If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
    Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
    The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
    If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
    indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

    The output vector (denoted by Y) stores the count of each n-gram;
    Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
    between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
    ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
    respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
    Note that we may consider all skips up to S when generating the n-grams.

    The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
    the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
    this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

    Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
    If pool_strings is set, the input must be a string tensor.
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
    SI64Attr:$max_gram_length,
    SI64Attr:$max_skip_count,
    SI64Attr:$min_gram_length,
    StrAttr:$mode,
    I64ArrayAttr:$ngram_counts,
    I64ArrayAttr:$ngram_indexes,
    OptionalAttr<I64ArrayAttr>:$pool_int64s,
    OptionalAttr<StrArrayAttr>:$pool_strings,
    OptionalAttr<F32ArrayAttr>:$weights
  );

  let results = (outs
    TensorOf<[F32]>:$Y
  );
}
