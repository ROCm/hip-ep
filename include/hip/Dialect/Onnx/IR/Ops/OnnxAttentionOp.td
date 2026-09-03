//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_AttentionOp
//===----------------------------------------------------------------------===//

def Onnx_AttentionOp : Onnx_Op<"Attention", [Pure]> {
  let summary = "ONNX Attention, schema version 24";

  let description = [{
    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

    This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

    For self attention, `kv_sequence_length` equals to `q_sequence_length`.

    For cross attention, query and key might have different lengths.

    This operator also covers the 3 following variants based on the number of heads:
    1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
    2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
    3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

    Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
    1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
    2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.

    With respect to KV cache update, this operator allows the following two use cases:

    1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
    tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
    all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
    key and value, respectively. Note that this only works correctly for the special case where the past key and value
    do not contain padded tokens.
    2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
    case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
    present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
    provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
    Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
    as the maximum value of `nonpad_kv_seqlen`.

    Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
    The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

    ```
      The following pattern is applied by this operator:
          Q          K          V
          |          |          |
    Q*sqrt(scale) K*sqrt(scale) |
          |          |          |
          |       Transpose     |
          |          |          |
          ---MatMul---          |
                |               |
     at_mask---Add              |
                |               |
      softcap (if provided)     |
                |               |
             Softmax            |
                |               |
                -----MatMul------
                       |
                       Y
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Q,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$K,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$V,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, NoneType
    ]>:$attn_mask,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$past_key,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$past_value,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$nonpad_kv_seqlen,
    DefaultValuedAttr<SI64Attr, "0">:$is_causal,
    OptionalAttr<SI64Attr>:$kv_num_heads,
    OptionalAttr<SI64Attr>:$q_num_heads,
    DefaultValuedAttr<SI64Attr, "0">:$qk_matmul_output_mode,
    OptionalAttr<F32Attr>:$scale,
    DefaultValuedAttr<F32Attr, "0.0">:$softcap,
    OptionalAttr<SI64Attr>:$softmax_precision
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$present_key,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$present_value,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      NoneType
    ]>:$qk_matmul_output
  );
}
