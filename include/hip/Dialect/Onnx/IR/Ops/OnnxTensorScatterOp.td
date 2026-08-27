//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_TensorScatterOp
//===----------------------------------------------------------------------===//

def Onnx_TensorScatterOp : Onnx_Op<"TensorScatter", [Pure]> {
  let summary = "ONNX TensorScatter, schema version 24";

  let description = [{
    TensorScatter is a generic tensor update operation, motivated by the requirements for KV cache updates for Attention
    ops commonly found in LLMs. It is a functional operation that models an in-place update to a KV cache buffer.

    The past and present cache tensors have the same shape (batch_size, D1, D2, ..., max_sequence_length, ..., Dn), with
    the sequence dimension (indicated by the `axis` attribute) being max_sequence_length, so the sizes of these tensors do
    not need to grow between iterations. The `update` tensor's shape only differs from the cache tensors in the sequence
    dimension: (batch_size, D1, D2, ..., sequence_length, ..., Dn), where sequence_length <= max_sequence_length.

    The optional `write_indices` input indicates the write index for each sample in the batch, assumed to be zero
    if not provided. When the `mode` attribute is set to "circular", the write index is modulo max_sequence_length.
    The operation can be described using the following pseudocode:

    ```
    for prefix_idx in np.ndindex(past_cache.shape[:axis]):
        batch_idx = prefix_idx[0]
        for sequence_idx in range(sequence_length):
            cache_idx = (*prefix_idx, write_indices[batch_idx] + sequence_idx)
            if mode == "circular":
                cache_idx = tuple(np.mod(np.asarray(cache_idx), max_sequence_length))
            update_idx = (*prefix_idx, sequence_idx)
            present_cache[cache_idx] = update[update_idx]
    ```

    During the prefill phase of attention, only the first two inputs are needed. During the decode phase, `write_indices`
    is also needed so that the incoming key or value update can be appended after the last valid token for each sample
    in the batch.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>
    ]>:$past_cache,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>
    ]>:$update,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$write_indices,
    DefaultValuedAttr<SI64Attr, "-2">:$axis,
    DefaultValuedStrAttr<StrAttr, "linear">:$mode
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>
    ]>:$present_cache
  );
}
