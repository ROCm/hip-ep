#ifndef HIP_TYPES
#define HIP_TYPES

include "HipDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Hip_Type<string name, string typeMnemonic>
    : TypeDef<Hip_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Hip_ContextType : Hip_Type<"Context", "context"> {
  let summary = "Opaque HIP execution context";
  let description = [{
    Represents runtime state including stream, library handles (MIOpen,
    hipBLASLT), and GPU memory. Lowered to !llvm.ptr during HIP→LLVM conversion.

    The context is an opaque pointer to a runtime-managed struct that contains:
    - HIP stream for asynchronous execution
    - Library handles (miopenHandle_t, hipblasLtHandle_t, etc.)
    - GPU pointers to uploaded weights
    - Cached descriptors and workspace buffers

    This type is intentionally opaque to separate high-level IR from low-level
    implementation details, enabling easy extension without IR changes.
  }];
}

#endif // HIP_TYPES
