/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*===- hipep_backend.h - Running ggml graphs' splits by compiled code -----===
 *
 * libhipep_backend.so: what a ggml backend (llama.cpp's ggml-hip, built with
 * GGML_HIP_HIPEP) calls to have the splits the scheduler hands it compiled
 * and run on an AMD GPU. The
 * caller describes a split in plain C - its tensors' metadata, never their
 * data - and gets code for it; it runs the code on the tensors' memory.
 * Nothing here is ggml's: the library depends on no ggml or llama.cpp, and the
 * caller on nothing of the library but this header.
 *
 * A backend checks hipep_backend_abi_version() against
 * HIPEP_BACKEND_ABI_VERSION and opens a context per device
 * (hipep_backend_open). It asks hipep_backend_supports about each node of a
 * split; its graph_compute describes the split, takes hipep_backend_find's code
 * or has it made (hipep_backend_compile), runs it (hipep_backend_run), and
 * otherwise runs the split another way - for good once find or compile says
 * it is refused or invalid (HIPEP_BACKEND_STATUS_REFUSED, _INVALID).
 * hipep_backend_close at the end.
 *
 * A description is the caller's, valid during the call only: what the
 * library keeps of it, it copies (a compile in the background works on its
 * copy). Every function is safe to call from any thread.
 *
 * Sizes. Code is compiled with dynamic shapes - for splits of one structure
 * at other sizes. Which sizes vary, and which are one, the library derives
 * from the split alone (the operations' contracts; a weight's and a cache's
 * sizes are fixed), and it checks every extent, stride and offset of a split
 * before code compiled for another runs it.
 *===----------------------------------------------------------------------===*/

#ifndef HIPEP_BACKEND_H
#define HIPEP_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIPEP_BACKEND_ABI_VERSION 6

#if defined(_WIN32) && defined(HIPEP_BACKEND_BUILD)
#define HIPEP_BACKEND_API __declspec(dllexport)
#elif defined(_WIN32)
#define HIPEP_BACKEND_API __declspec(dllimport)
#else
#define HIPEP_BACKEND_API __attribute__((visibility("default")))
#endif

typedef struct hipep_backend_ctx hipep_backend_ctx;
typedef struct hipep_backend_code hipep_backend_code;

/* What a tensor read from outside the split is. */
enum hipep_backend_role {
  HIPEP_BACKEND_ROLE_NODE = 0,   /* computed in the graph */
  HIPEP_BACKEND_ROLE_WEIGHT = 1, /* a model weight, of fixed sizes */
  HIPEP_BACKEND_ROLE_KV_CACHE =
      2,                        /* memory kept across steps (caches, states) */
  HIPEP_BACKEND_ROLE_INPUT = 3, /* set by the host before the step */
};

/* One tensor, as ggml has it. Indices are into the same description; -1 for
   none. */
typedef struct hipep_backend_tensor_desc {
  const char *op;   /* ggml's operation name: "MUL_MAT", "NONE" for a leaf */
  const char *name; /* the tensor's name ("inp_pos", "blk.0.attn_q.weight") */
  int32_t type;     /* enum ggml_type */
  int64_t ne[4];
  uint64_t nb[4];
  int32_t src[10];
  int32_t view_src;
  uint64_t view_offs;
  int32_t op_params[16]; /* as ggml lays them out (ggml_version says which) */
  int32_t flags;    /* ggml's tensor flags (INPUT 1, OUTPUT 2, COMPUTE 16) */
  uint8_t role;     /* enum hipep_backend_role */
  uint8_t in_split; /* 1: a node the split computes */
  uint8_t exported; /* a node of the split whose value is read after it */
  uint8_t reserved;
} hipep_backend_tensor_desc;

/* A split: for each node in the split's order, the tensors it reads not
   described yet (a view of a tensor outside the split with what it views,
   depth first), then the node. */
typedef struct hipep_backend_split_desc {
  uint32_t abi;          /* HIPEP_BACKEND_ABI_VERSION */
  uint32_t tensor_size;  /* sizeof(hipep_backend_tensor_desc) */
  uint32_t ggml_version; /* major * 10000 + minor * 100 + patch */
  uint32_t reserved;
  const hipep_backend_tensor_desc *tensors;
  size_t n_tensors;
} hipep_backend_split_desc;

/* Whether a split has code (hipep_backend_find, hipep_backend_compile), and
   when it has none, whether it will. */
enum hipep_backend_status {
  /* the code returned */
  HIPEP_BACKEND_STATUS_READY = 0,
  /* none yet: it is being compiled - find it again */
  HIPEP_BACKEND_STATUS_COMPILING = 1,
  /* none: it is neither compiled nor being compiled - compile it */
  HIPEP_BACKEND_STATUS_NOT_COMPILED = 2,
  /* none from this context: the compiler refused it or a split of its
     structure (in this process or an earlier one), its code could not be made
     or loaded in this process (the disk, the device's memory), or compiling
     is off */
  HIPEP_BACKEND_STATUS_REFUSED = 3,
  /* none: no context or split, or a description the library does not read
     (another ABI or ggml, a tensor it does not know, an index out of range,
     tensors not numbered sources first) */
  HIPEP_BACKEND_STATUS_INVALID = 4,
};

HIPEP_BACKEND_API int hipep_backend_abi_version(void);

/* The library for HIP device `device`, its code cached in `cache_dir` (NULL:
   HIPEP_BACKEND_CACHE, else ~/.cache/hipep). Null when there is no such
   device. */
HIPEP_BACKEND_API hipep_backend_ctx *hipep_backend_open(int device,
                                                        const char *cache_dir);

/* Whether the library compiles the description's last tensor - an operation,
   its sources and the tensor it views before it. */
HIPEP_BACKEND_API int
hipep_backend_supports(hipep_backend_ctx *ctx,
                       const hipep_backend_split_desc *node);

/* Code for the split, compiled already (in this process or an earlier one,
   for its sizes or with dynamic shapes that fit it): null when there is none.
   `*status` (unless `status` is NULL): enum hipep_backend_status. The code is
   the context's until it closes. */
HIPEP_BACKEND_API hipep_backend_code *
hipep_backend_find(hipep_backend_ctx *ctx,
                   const hipep_backend_split_desc *split, int *status);

/* Compiles the split. With `wait`, on this thread: its code, or null when it
   cannot be compiled (or another call is compiling it). Without, in the
   background: null now, the code from `find` once it is ready. A split the
   compiler refused is not tried again, in this process or a later one; one
   whose code could not be made or loaded otherwise, not in this process:
   find and compile say HIPEP_BACKEND_STATUS_REFUSED. `status` as find's. */
HIPEP_BACKEND_API hipep_backend_code *
hipep_backend_compile(hipep_backend_ctx *ctx,
                      const hipep_backend_split_desc *split, int wait,
                      int *status);

/* Runs the code on the split's tensors: tensor_data[i] is tensor i's memory,
   numbered as described. The code reads its inputs and writes its outputs and
   the caches it updates where they are. `stream`: reserved, NULL - the code
   runs on the library's own stream; the caller orders it against other work on
   the device by synchronizing the device. 0, or nonzero (last_error). */
HIPEP_BACKEND_API int hipep_backend_run(hipep_backend_code *code,
                                        void *const *tensor_data, void *stream);

/* Waits for a compile under way, and frees the context and its code. */
HIPEP_BACKEND_API void hipep_backend_close(hipep_backend_ctx *ctx);

/* What went wrong last on this thread ("" when nothing did). */
HIPEP_BACKEND_API const char *hipep_backend_last_error(hipep_backend_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* HIPEP_BACKEND_H */
