/*
 * Minimal MIOpen shim for the standalone GQA dispatch benchmark.
 *
 * gqa.cpp pulls in runtime_types.h -> <miopen/miopen.h> only for the
 * miopenHandle_t member of RuntimeState. The GQA path never touches MIOpen,
 * so a forward-declared opaque handle is all that is needed to compile the
 * real dispatcher off-tree (the full EP build uses the real MIOpen SDK).
 */
#ifndef HIPDNN_EP_BENCH_MIOPEN_SHIM_H
#define HIPDNN_EP_BENCH_MIOPEN_SHIM_H

typedef void *miopenHandle_t;

#endif // HIPDNN_EP_BENCH_MIOPEN_SHIM_H
