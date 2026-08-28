//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Hipsr_CopyD2HOp
//===----------------------------------------------------------------------===//

def Hipsr_CopyD2HOp : Hipsr_DpsOp<"copy_d2h",
    [AllElementTypesMatch<["src", "init"]>,
     AllShapesMatch<["src", "init"]>]> {
  let summary = "Copy device memory to host memory, destination-passing style";
  let description = [{
    Copies `src` from device memory into `init` in host memory.

    The copy runs on the context's stream, after the work that produced `src`.
    It does not wait for the copy to finish, so the host can read `init` only
    after that stream is synchronized.

    Not `Pure`: the op must keep its place in stream order, and it must not be
    deleted when nothing reads its result.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$src,
    Hipsr_HostTensorOrMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}
