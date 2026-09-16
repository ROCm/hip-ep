//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_NonZeroOp : Hipsr_DpsOp<"nonzero"> {
  let summary = "Locate the non-zero elements of a tensor, "
                "destination-passing style";
  let description = [{
    Writes the position of each non-zero element of `input` into `indices`,
    one row per input axis, and how many were found into `count`.

    How many positions there are follows the values in `input`, so `indices`
    is sized for the worst case, where every element is non-zero: only its
    first `count` columns hold positions. Narrowing it needs the host to read
    `count`, which `hipsr.copy_d2h` brings over, so that is a separate op.

    The search runs on the device, so both destinations are device-resident.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$input,
    Hipsr_DeviceTensorOrMemRef:$indices_init,
    Hipsr_DeviceTensorOrMemRef:$count_init
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
