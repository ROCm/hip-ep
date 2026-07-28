//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_GetPoolOp : Hipsr_Op<"get_pool",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Get a runtime-owned GPU memory pool";
  let description = [{
    Returns a runtime-managed GPU memory pool.

    - `pool_size` : minimum size in bytes the returned buffer must hold.
    - `domain_id` : selects an independent pool (default 0); pools with distinct
                    domain_ids grow independently.
  }];

  let arguments = (ins Hipsr_ContextType:$ctx, Index:$pool_size,
                       DefaultValuedAttr<I64Attr, "0">:$domain_id);
  let results = (outs Hipsr_DeviceMemRef:$pool);
  let assemblyFormat = "`(` $ctx `,` $pool_size `)` attr-dict `:` type($pool)";
}
