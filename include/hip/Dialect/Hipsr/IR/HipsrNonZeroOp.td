//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_NonZeroOp : Hipsr_DpsOp<"nonzero"> {
  let summary = "Locate the non-zero elements of a tensor, "
                "destination-passing style";
  let description = [{
    Writes the position of each non-zero element of `input` into `indices` as
    one row per input axis, and how many were found into `count`.

    How many positions there are depends on the values in `input`, so
    `indices` is sized for the worst case, where every element is non-zero:
    its trailing extent is a capacity, and only the first `count` columns hold
    positions. Narrowing it to the positions actually found is a separate
    operation, because that needs the host to read `count` first, which is
    also why `count` is host-resident.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$input,
    Hipsr_TensorOrDeviceMemRef:$indices_init,
    Hipsr_TensorOrHostMemRef:$count_init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $indices_init `,` $count_init `:` type($indices_init) `,`
        type($count_init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
