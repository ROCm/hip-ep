/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DATATYPE_ABI_H
#define HIP_DATATYPE_ABI_H

//===----------------------------------------------------------------------===//
// Backend-Independent Data Type Identifiers
//===----------------------------------------------------------------------===//
//
// Single source of truth for the element-type identifiers the compiler writes
// into generated runtime calls and the runtime reads back out. They are our own
// values -- do NOT assume they match cuDNN or any other library's enum. Each
// backend provides an explicit mapping function to convert these to
// library-specific types.
//
// Both sides include this header, so a value cannot be changed on one side
// alone. They stay plain #defines because the runtime's C ABI header uses them.
//
// To add a new type:
//   1. Add the #define here
//   2. Update hipdnn_ep_datatype_size() and hipdnn_ep_datatype_name() in
//      lib/Runtime/hipdnn_ep_runtime.h
//   3. Update the compiler mappings from an MLIR element type:
//   getHipdnnDataType()
//      in lib/Conversion/HipToLLVM/HipToLLVMUtils.h and in
//      lib/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.cpp, and
//      getHipdnnInputDataType() in lib/Conversion/OnnxToHip/OnnxToHipUtils.h
//   4. Update each backend mapping function
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_DATATYPE_FLOAT 0    // f32, 4 bytes
#define HIPDNN_EP_DATATYPE_HALF 1     // f16, 2 bytes
#define HIPDNN_EP_DATATYPE_BFLOAT16 2 // bf16, 2 bytes
#define HIPDNN_EP_DATATYPE_INT32 3    // i32, 4 bytes
#define HIPDNN_EP_DATATYPE_INT64 4    // i64, 8 bytes
#define HIPDNN_EP_DATATYPE_INT8 5     // i8, 1 byte
#define HIPDNN_EP_DATATYPE_DOUBLE 6   // f64, 8 bytes
#define HIPDNN_EP_DATATYPE_UINT8 7    // ui8, 1 byte
#define HIPDNN_EP_DATATYPE_INT16 8    // i16, 2 byte

// Returned by the compiler-side mappings for an element type with no runtime
// identifier, so a caller fails conversion explicitly instead of passing a
// wrong type.
#define HIPDNN_EP_DATATYPE_UNSUPPORTED (-1)

#endif // HIP_DATATYPE_ABI_H
