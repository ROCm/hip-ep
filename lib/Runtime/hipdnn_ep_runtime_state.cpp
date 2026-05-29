/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hip/timing.h"
#include "hip_cleanup.h"
#include "hipdnn_ep_runtime.h"
#include "op_profile.h"
#include "runtime_state_internal.h"

#include "mm/mm_api.h"

#include "model_metadata_generated.h"
#include "morphizen-foundation/file_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Win32 API declarations for the OGA pipeline shared-constants cache
// (named shared memory + atomic ref counting). Model DLLs link static
// CRT, so kernel32 calls go through declared imports rather than
// <windows.h> (which would pull a much larger surface).
#ifdef _WIN32
extern "C" {
void *__stdcall CreateFileMappingA(void *, void *, unsigned long, unsigned long,
                                   unsigned long, const char *);
void *__stdcall OpenFileMappingA(unsigned long, int, const char *);
void *__stdcall MapViewOfFile(void *, unsigned long, unsigned long,
                              unsigned long, size_t);
int __stdcall UnmapViewOfFile(const void *);
int __stdcall CloseHandle(void *);
unsigned long __stdcall GetCurrentProcessId(void);
}

// Metadata block stored in the named shared memory. The actual constants
// live in the publishing model's hipMalloc'd blob; this struct only
// carries the GPU pointer + size + atomic ref count.
//
// NOTE: layout intentionally diverges from main: PR #109 had a `bool
// is_host` here to support a hipHostMalloc fallback path on iGPU. We
// dropped that path (constants blob is always hipMalloc'd VRAM), so
// is_host is removed. Cross-DLL sharing relies on both publisher and
// consumer using this same struct, which holds for OGA pipeline (both
// model.dll built from the same hip-compiler).
struct SharedConstantsMeta {
  void *blob_ptr;
  size_t blob_size;
  volatile long ref_count;
};

// Atomic ref-count helpers. Model DLL bitcode is compiled by Clang/LLVM
// where InterlockedIncrement/Decrement are not linkable symbols (MSVC
// intrinsics, not kernel32 exports). Use compiler builtins that lower
// to LLVM atomicrmw.
#if defined(__clang__) || defined(__GNUC__)
static inline long shm_ref_inc(volatile long *p) {
  return __sync_add_and_fetch(p, 1);
}
static inline long shm_ref_dec(volatile long *p) {
  return __sync_sub_and_fetch(p, 1);
}
#else
static inline long shm_ref_inc(volatile long *p) { return ++(*p); }
static inline long shm_ref_dec(volatile long *p) { return --(*p); }
#endif

#define SHM_FILE_MAP_ALL_ACCESS 0x000F001Fu
#define SHM_PAGE_READWRITE 0x04u
#endif

// Forward decl of static helpers defined later in this file.
static int initialize_state_handles(RuntimeState **out_state);
static int prepare_constants_array(RuntimeState *state,
                                   const mlir::hip::HipModelMetaInfo *meta);
static size_t
compute_constants_total_size(const mlir::hip::HipModelMetaInfo *meta);
static int hipmalloc_and_fixup(RuntimeState *state,
                               const mlir::hip::HipModelMetaInfo *meta,
                               size_t total_size);
static bool try_attach_shared_constants(RuntimeState *state,
                                        const mlir::hip::HipModelMetaInfo *meta,
                                        size_t total_size);
static void publish_shared_constants(RuntimeState *state, size_t total_size);
static int bulk_load_constants(RuntimeState *state, morphizen::FileSystem *fs,
                               const char *constants_filename);
static int per_entry_load_constants(RuntimeState *state,
                                    const mlir::hip::HipModelMetaInfo *meta,
                                    morphizen::FileSystem *fs,
                                    const char *constants_filename);

static int ensure_mm_initialized() {
  if (mm::is_initialized())
    return 0;
  mm::Config cfg = mm::config_default();
  cfg.device_id = 0;
  cfg.enable_debug_log = hipdnn_ep_debug_enabled();
  auto status = mm::init(&cfg);
  if (status != mm::Status::Ok) {
    fprintf(stderr, "Memory manager init failed (status=%d)\n",
            static_cast<int>(status));
    return 1;
  }
  return 0;
}

int hipdnn_ep_state_init_with_fs(RuntimeState **out_state, void *fs,
                                 const void *metadata_blob, size_t blob_size) {
  auto t0 = timing_now();

  if (!out_state || !fs) {
    fprintf(stderr, "Invalid arguments to hipdnn_ep_state_init_with_fs\n");
    return 1;
  }

  if (ensure_mm_initialized() != 0) {
    return 1;
  }

  if (int rc = initialize_state_handles(out_state); rc != 0) {
    return rc;
  }

  if (!metadata_blob || blob_size == 0) {
    TIMING_LOG("[Session] hipdnn_ep_state_init_with_fs total: %.3fs (no "
               "constants)\n",
               elapsed_since(t0));
    return 0;
  }

  auto *meta = flatbuffers::GetRoot<mlir::hip::HipModelMetaInfo>(metadata_blob);
  auto *constants = meta->constants();
  int64_t count = constants ? (int64_t)constants->size() : 0;
  if (count <= 0) {
    TIMING_LOG("[Session] hipdnn_ep_state_init_with_fs total: %.3fs (no "
               "constants)\n",
               elapsed_since(t0));
    return 0;
  }

  // 1. Pre-allocate the gpu_constants[] pointer array (cheap, needed by
  //    both shared-cache hit and miss paths). num_constants is set here.
  if (int rc = prepare_constants_array(*out_state, meta); rc != 0) {
    hipdnn_ep_state_cleanup(*out_state);
    *out_state = nullptr;
    return rc;
  }

  // 2. Try to attach a process-wide shared constants blob (OGA pipeline
  //    optimization: prefill+decode share the same constants, so the
  //    second model can skip both hipMalloc and the data load entirely).
  size_t total_size = compute_constants_total_size(meta);
  if (try_attach_shared_constants(*out_state, meta, total_size)) {
    TIMING_LOG("[Session] hipdnn_ep_state_init_with_fs total: %.3fs "
               "(shared blob attached)\n",
               elapsed_since(t0));
    return 0;
  }

  // 3. Cache miss: allocate our own VRAM blob and fix up gpu_constants[i].
  if (int rc = hipmalloc_and_fixup(*out_state, meta, total_size); rc != 0) {
    hipdnn_ep_state_cleanup(*out_state);
    *out_state = nullptr;
    return rc;
  }

  auto *fileSystem = static_cast<morphizen::FileSystem *>(fs);

  // 4. Dispatch by metadata semantics: if any constant carries a per-entry
  //    source descriptor (Splat / FileRef), do per-tensor upload driven by
  //    the source union. Otherwise fall back to the bulk sidecar path (used
  //    by EPContext export / import and has_mem_addr downgrade).
  bool anyPerEntry = false;
  for (int64_t i = 0; i < count; ++i) {
    if (constants->Get(i)->source_type() != mlir::hip::ConstantSource::NONE) {
      anyPerEntry = true;
      break;
    }
  }
  const char *constants_filename = "constants.bin";
  if (meta->constants_filename())
    constants_filename = meta->constants_filename()->c_str();
  int rc;
  if (anyPerEntry) {
    // Hybrid path also goes through per-entry; SidecarSource entries open
    // constants_filename through fs to fetch their slice on demand. Pure
    // streaming (only Splat / FileRef) ignores fs/constants_filename.
    rc = per_entry_load_constants(*out_state, meta, fileSystem,
                                  constants_filename);
  } else {
    rc = bulk_load_constants(*out_state, fileSystem, constants_filename);
  }
  if (rc != 0) {
    hipdnn_ep_state_cleanup(*out_state);
    *out_state = nullptr;
    return rc;
  }

  // 5. Publish the freshly loaded blob to the shared cache (best-effort;
  //    failure means subsequent models won't be able to share).
  publish_shared_constants(*out_state, total_size);

  TIMING_LOG("[Session] hipdnn_ep_state_init_with_fs total: %.3fs\n",
             elapsed_since(t0));
  return 0;
}

// Shared initialization that brings up HIP device, stream, MIOpen and
// hipBLASLt handles in a fresh RuntimeState. On any failure the partially
// initialized state is released and a non-zero error code (matching the
// historical exit codes 1-9) is returned. On success *out_state holds the
// RuntimeState ready for constants_blob allocation; no constant memory has
// been touched yet.
static int initialize_state_handles(RuntimeState **out_state) {
  auto t_prev = timing_now();

  RuntimeState *state = (RuntimeState *)malloc(sizeof(RuntimeState));
  if (!state) {
    fprintf(stderr, "Failed to allocate runtime state\n");
    return 1;
  }

  state->stream = nullptr;
  state->miopen_handle = nullptr;
  state->hipblas_handle = nullptr;
  state->gpu_constants_blob = nullptr;
  state->gpu_constants = nullptr;
  state->num_constants = 0;
  state->gpu_constants_handle = mm::kInvalidHandle;
  state->constants_is_shared = false;
  state->shared_constants_mapping = nullptr;
  state->shared_constants_view = nullptr;
  state->pool_base = nullptr;
  state->pool_size = 0;
  state->buffer_offsets = nullptr;
  state->num_buffers = 0;
  state->workspace = nullptr;
  state->workspace_size = 0;
  state->workspace_handle = mm::kInvalidHandle;
  state->qmoe_scratch = nullptr;
  state->qmoe_scratch_size = 0;
  state->qmoe_scratch_handle = mm::kInvalidHandle;
  state->qmoe_host_scratch = nullptr;
  state->qmoe_host_scratch_size = 0;
  state->gqa_gemm_cache = nullptr;
  state->mha_gemm_cache = nullptr;
  state->causal_conv_cache = nullptr;
  state->zp_unpack_cache = nullptr;
  state->op_profile = hipdnn_ep_perf_enabled() ? op_profile_create() : nullptr;
  state->device_error_flag = nullptr;
  state->hipdnn_handle = nullptr;
  state->hipdnn_graph_registry = nullptr;
  state->seqlens_k_cached_valid = false;
  state->seqlens_k_cached_val = 0;
  state->seqlens_k_cached_ptr = nullptr;
  state->loop_iter_cpu_buf = nullptr;
  state->loop_iter_capacity = 0;
  state->loop_iter_dev = nullptr;
  state->loop_cond_host = nullptr;
  state->loop_cond_dev = nullptr;
  state->loop_event = nullptr;

  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    fprintf(stderr, "Failed to get HIP device count or no devices available\n");
    free(state);
    return 2;
  }

  if (hipSetDevice(0) != hipSuccess) {
    fprintf(stderr, "Failed to set HIP device 0\n");
    free(state);
    return 3;
  }

  TIMING_LOG("[Session] HIP device init: %.3fs\n", record_elapsed(t_prev));

  if (hipStreamCreate(&state->stream) != hipSuccess) {
    fprintf(stderr, "Failed to create HIP stream\n");
    free(state);
    return 6;
  }

  TIMING_LOG("[Session] hipStreamCreate: %.3fs\n", record_elapsed(t_prev));

  if (miopenCreate(&state->miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr, "Failed to create MIOpen handle\n");
    if (state->stream)
      HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 7;
  }

  if (miopenSetStream(state->miopen_handle, state->stream) !=
      miopenStatusSuccess) {
    fprintf(stderr, "Failed to set MIOpen stream\n");
    miopenDestroy(state->miopen_handle);
    if (state->stream)
      HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 8;
  }

  TIMING_LOG("[Session] MIOpen init: %.3fs\n", record_elapsed(t_prev));

  if (hipblasLtCreate(&state->hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to create hipBLASLt handle\n");
    miopenDestroy(state->miopen_handle);
    if (state->stream)
      HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 9;
  }

  TIMING_LOG("[Session] hipBLASLt init: %.3fs\n", record_elapsed(t_prev));

  // Allocate device-side error flag used by kernels for runtime error
  // propagation (e.g., Range delta==0).
  if (hipMalloc((void **)&state->device_error_flag, sizeof(int)) !=
      hipSuccess) {
    fprintf(stderr, "Failed to allocate device error flag\n");
    hipblasLtDestroy(state->hipblas_handle);
    miopenDestroy(state->miopen_handle);
    if (state->stream)
      HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 10;
  }
  if (hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream) !=
      hipSuccess) {
    fprintf(stderr, "Failed to initialize device error flag\n");
    HIP_CLEANUP(hipFree(state->device_error_flag));
    state->device_error_flag = nullptr;
    hipblasLtDestroy(state->hipblas_handle);
    miopenDestroy(state->miopen_handle);
    if (state->stream)
      HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 11;
  }

  *out_state = state;
  return 0;
}

// Allocate the gpu_constants[] pointer array (one slot per constant)
// and record num_constants. This is a few KB at most and is needed by
// both shared-cache hit and miss paths -- doing it up front lets
// try_attach_shared_constants fix up the slots without an extra malloc.
static int prepare_constants_array(RuntimeState *state,
                                   const mlir::hip::HipModelMetaInfo *meta) {
  auto *constants = meta ? meta->constants() : nullptr;
  int64_t count = constants ? (int64_t)constants->size() : 0;
  if (count <= 0) {
    return 0;
  }
  state->gpu_constants =
      (void **)calloc(static_cast<size_t>(count), sizeof(void *));
  if (!state->gpu_constants) {
    fprintf(stderr, "Failed to allocate gpu_constants array\n");
    return 1;
  }
  state->num_constants = static_cast<size_t>(count);
  return 0;
}

// Sum of max(end = offset + size) across all constants -- this is the
// size of the single VRAM blob that backs gpu_constants_blob.
static size_t
compute_constants_total_size(const mlir::hip::HipModelMetaInfo *meta) {
  auto *constants = meta ? meta->constants() : nullptr;
  if (!constants)
    return 0;
  size_t total = 0;
  for (int64_t i = 0, n = (int64_t)constants->size(); i < n; ++i) {
    size_t end = static_cast<size_t>(constants->Get(i)->offset()) +
                 static_cast<size_t>(constants->Get(i)->size());
    if (end > total)
      total = end;
  }
  return total;
}

// hipMalloc the VRAM blob and fix up gpu_constants[i] to point inside it.
// Caller must have already invoked prepare_constants_array.
static int hipmalloc_and_fixup(RuntimeState *state,
                               const mlir::hip::HipModelMetaInfo *meta,
                               size_t total_size) {
  auto t_prev = timing_now();
  mm::AllocHints hints;
  hints.mem_class = mm::MemoryClass::Weight;
  hints.lifetime = mm::Lifetime::Static;
  hints.alignment = 256;
  mm::handle_t handle = mm::alloc(total_size, &hints);
  if (handle == mm::kInvalidHandle) {
    fprintf(stderr, "mm::alloc failed for constants blob (%zu bytes)\n",
            total_size);
    return 1;
  }
  state->gpu_constants_handle = handle;
  state->gpu_constants_blob = mm::get_ptr(handle);
  if (!state->gpu_constants_blob) {
    fprintf(stderr, "mm::get_ptr returned null for constants blob\n");
    (void)mm::free(handle);
    state->gpu_constants_handle = mm::kInvalidHandle;
    return 1;
  }
  TIMING_LOG("[Session] MM alloc VRAM: %.3fs (%zu bytes)\n",
             record_elapsed(t_prev), total_size);
  auto *constants = meta->constants();
  for (int64_t i = 0, n = (int64_t)constants->size(); i < n; ++i) {
    size_t offset = static_cast<size_t>(constants->Get(i)->offset());
    state->gpu_constants[i] =
        static_cast<char *>(state->gpu_constants_blob) + offset;
  }
  return 0;
}

// Try to attach to a process-wide shared constants blob keyed by
// (pid, total_size). On hit: state->gpu_constants_blob is set to the
// shared GPU pointer, gpu_constants[i] are fixed up against it, and
// state->constants_is_shared is set to true. The shared blob's
// ref_count is incremented; cleanup decrements and only the last
// reference frees the GPU memory.
//
// Cache key matches the "OGA prefill+decode share identical
// model.constants.bin" invariant: same pid + same total_size implies
// same content. If two models with the same total_size diverge in
// content (rare), the consumer will silently get the wrong constants.
//
// Precondition: state->gpu_constants_blob == nullptr and
// gpu_constants[] has been prepared. On miss / non-Windows: returns
// false without touching state, caller proceeds with hipMalloc.
static bool try_attach_shared_constants(RuntimeState *state,
                                        const mlir::hip::HipModelMetaInfo *meta,
                                        size_t total_size) {
#ifndef _WIN32
  (void)state;
  (void)meta;
  (void)total_size;
  return false;
#else
  if (total_size == 0)
    return false;

  char shm_name[128];
  snprintf(shm_name, sizeof(shm_name), "Local\\hipdnn_const_%lu_%zu",
           (unsigned long)GetCurrentProcessId(), total_size);

  void *existing_map = OpenFileMappingA(SHM_FILE_MAP_ALL_ACCESS, 0, shm_name);
  if (!existing_map)
    return false;

  auto *smeta = (SharedConstantsMeta *)MapViewOfFile(
      existing_map, SHM_FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedConstantsMeta));
  if (!smeta || smeta->blob_size != total_size || !smeta->blob_ptr) {
    if (smeta)
      UnmapViewOfFile(smeta);
    CloseHandle(existing_map);
    return false;
  }

  long new_ref = shm_ref_inc(&smeta->ref_count);

  state->gpu_constants_blob = smeta->blob_ptr;
  state->constants_is_shared = true;
  state->shared_constants_mapping = existing_map;
  state->shared_constants_view = smeta;

  auto *constants = meta->constants();
  for (int64_t i = 0, n = (int64_t)constants->size(); i < n; ++i) {
    size_t offset = static_cast<size_t>(constants->Get(i)->offset());
    state->gpu_constants[i] =
        static_cast<char *>(state->gpu_constants_blob) + offset;
  }

  fprintf(stderr,
          "[SHARED_CONSTANTS] Reusing existing blob "
          "(%zu bytes, ref_count=%ld)\n",
          total_size, new_ref);
  return true;
#endif
}

// Publish the freshly loaded blob into the process-wide shared cache so
// later models with the same total_size can attach. Best-effort:
// failures are silently tolerated (the model still works, just no
// sharing). On non-Windows: no-op.
static void publish_shared_constants(RuntimeState *state, size_t total_size) {
#ifndef _WIN32
  (void)state;
  (void)total_size;
#else
  if (total_size == 0)
    return;

  char shm_name[128];
  snprintf(shm_name, sizeof(shm_name), "Local\\hipdnn_const_%lu_%zu",
           (unsigned long)GetCurrentProcessId(), total_size);

  void *new_map =
      CreateFileMappingA((void *)(intptr_t)-1, nullptr, SHM_PAGE_READWRITE, 0,
                         (unsigned long)sizeof(SharedConstantsMeta), shm_name);
  if (!new_map)
    return;

  auto *smeta = (SharedConstantsMeta *)MapViewOfFile(
      new_map, SHM_FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedConstantsMeta));
  if (!smeta) {
    CloseHandle(new_map);
    return;
  }

  smeta->blob_ptr = state->gpu_constants_blob;
  smeta->blob_size = total_size;
  smeta->ref_count = 1;
  state->shared_constants_mapping = new_map;
  state->shared_constants_view = smeta;
  fprintf(stderr, "[SHARED_CONSTANTS] Published new blob (%zu bytes)\n",
          total_size);
#endif
}

// Load the entire constants sidecar as one blob and hipMemcpy it into the
// pre-allocated gpu_constants_blob. Used when OnnxToHip wrote
// model.constants.bin (non-streaming path: hip-compiler CLI, EPContext
// import, mem-addr downgrade).
static int bulk_load_constants(RuntimeState *state, morphizen::FileSystem *fs,
                               const char *constants_filename) {
  auto t_prev = timing_now();

  auto reader = fs->create_reader_template(constants_filename);
  if (!reader) {
    fprintf(stderr, "Failed to open %s via FileSystem\n", constants_filename);
    return 1;
  }
  size_t total_size = reader->size();

  TIMING_LOG("[Session] open sidecar %s: %.3fs (%zu bytes)\n",
             constants_filename, record_elapsed(t_prev), total_size);

  const void *src = reader->mmap();
  void *cpu_buf = nullptr;

  TIMING_LOG("[Session] VRAM path: mmap %s\n",
             src ? "succeeded" : "failed, using fread fallback");

  if (!src) {
    cpu_buf = malloc(total_size);
    if (!cpu_buf) {
      fprintf(stderr, "Failed to allocate staging buffer (%zu bytes)\n",
              total_size);
      return 1;
    }
    size_t bytes_read = reader->fread(cpu_buf, total_size);
    if (bytes_read != total_size) {
      fprintf(stderr, "Short read: got %zu of %zu bytes\n", bytes_read,
              total_size);
      free(cpu_buf);
      return 1;
    }
    src = cpu_buf;
    TIMING_LOG("[Session] fread constants.bin: %.3fs (%zu bytes)\n",
               record_elapsed(t_prev), total_size);
  }

  if (hipMemcpy(state->gpu_constants_blob, src, total_size,
                hipMemcpyHostToDevice) != hipSuccess) {
    fprintf(stderr, "hipMemcpy failed for constants blob\n");
    free(cpu_buf);
    return 1;
  }

  TIMING_LOG("[Session] hipMemcpy H2D bulk: %.3fs (%zu bytes)\n",
             record_elapsed(t_prev), total_size);

  free(cpu_buf); // no-op when mmap was used
  return 0;
}

// Per-process single-slot cache for the external-data file referenced by
// streaming file-ref entries. LLM models typically have one giant
// weights.data shared by all file-ref tensors, so a size-1 cache keeps
// the FILE* open for the entire streaming loop. Different path closes
// old and opens new.
//
// Uses stdio FILE* rather than morphizen::FileSystem because file-ref
// sources (weights.data) are absolute OS paths authored by OnnxToHip
// from ORT's external data info. PassContext's FileSystem only resolves
// in-memory tar / cache entries, which weights.data is not.
struct FrefCache {
  char path[1024];
  std::FILE *fp;
};

static void fref_cache_init(FrefCache *c) {
  c->path[0] = '\0';
  c->fp = nullptr;
}

static void fref_cache_release(FrefCache *c) {
  if (c->fp) {
    std::fclose(c->fp);
    c->fp = nullptr;
  }
}

// Ensure c->fp points at `path` (null-terminated C string; flatbuffer
// String::c_str() is guaranteed null-terminated). Returns true on success.
static bool fref_cache_open(FrefCache *c, const char *path) {
  if (c->fp && std::strcmp(c->path, path) == 0) {
    return true; // same path, reuse
  }
  fref_cache_release(c);
  size_t path_len = std::strlen(path);
  if (path_len + 1 > sizeof(c->path)) {
    fprintf(stderr, "file-ref path too long (%zu bytes)\n", path_len);
    return false;
  }
  memcpy(c->path, path, path_len + 1);
  c->fp = std::fopen(c->path, "rb");
  if (!c->fp) {
    fprintf(stderr, "Failed to open file-ref source: %s\n", c->path);
    return false;
  }
  return true;
}

// Copy `size` bytes at `file_offset` from cached FILE* into `staging`.
static int fref_fetch(FrefCache *c, int64_t file_offset, size_t size,
                      void *staging) {
#ifdef _WIN32
  if (_fseeki64(c->fp, file_offset, SEEK_SET) != 0) {
    fprintf(stderr, "fref_fetch: fseek64 to %lld failed\n",
            (long long)file_offset);
    return 1;
  }
#else
  if (std::fseek(c->fp, (long)file_offset, SEEK_SET) != 0) {
    fprintf(stderr, "fref_fetch: fseek to %lld failed\n",
            (long long)file_offset);
    return 1;
  }
#endif
  size_t got = std::fread(staging, 1, size, c->fp);
  if (got != size) {
    fprintf(stderr, "fref_fetch: short read (got %zu of %zu)\n", got, size);
    return 1;
  }
  return 0;
}

// Lazy reader for the partial mem-addr sidecar (hybrid mode). Opened
// through the EP FileSystem on first SidecarSource entry; if the
// FileSystem can mmap the sidecar we keep the base pointer so the case
// becomes a memcpy + hipMemcpy (no extra fread). On non-mmap backends
// we fall back to fread, which still wins because the per-entry staging
// reuse keeps host peak bounded to the largest single tensor.
struct SidecarReaderCache {
  std::unique_ptr<morphizen::FileReader,
                  morphizen::FileSystem::Deleter<morphizen::FileReader>>
      reader; // default-init: null pointer, deleter unused
  const uint8_t *mmap_base = nullptr; // non-null when reader->mmap() succeeded
  size_t total_size = 0;
  bool tried_open = false;
};

// Returns true on success (or no-op if already open). On first call
// opens the sidecar through fs and tries mmap.
static bool sidecar_cache_open(SidecarReaderCache *c, morphizen::FileSystem *fs,
                               const char *constants_filename) {
  if (c->tried_open)
    return c->reader != nullptr;
  c->tried_open = true;
  if (!fs || !constants_filename) {
    fprintf(stderr,
            "per_entry: SidecarSource entry but fs / constants_filename "
            "unavailable\n");
    return false;
  }
  c->reader = fs->create_reader_template(constants_filename);
  if (!c->reader) {
    fprintf(stderr, "per_entry: failed to open partial sidecar %s\n",
            constants_filename);
    return false;
  }
  c->total_size = c->reader->size();
  c->mmap_base = static_cast<const uint8_t *>(c->reader->mmap());
  return true;
}

// Copy `size` bytes at `offset` from the partial sidecar into `staging`.
static int sidecar_fetch(SidecarReaderCache *c, int64_t offset, size_t size,
                         void *staging) {
  if ((uint64_t)offset + size > (uint64_t)c->total_size) {
    fprintf(stderr,
            "per_entry: sidecar fetch out of range (offset=%lld size=%zu "
            "total=%zu)\n",
            (long long)offset, size, c->total_size);
    return 1;
  }
  if (c->mmap_base) {
    std::memcpy(staging, c->mmap_base + offset, size);
    return 0;
  }
  // Non-mmap reader: rewind + skip + fread. morphizen::FileReader does
  // not expose a seek primitive, so we read-and-discard up to offset.
  // Sidecar is touched once per entry in entry order, so this still
  // streams linearly through the file.
  c->reader->rewind();
  static constexpr size_t kSkipChunk = 64 * 1024;
  uint8_t skip_buf[kSkipChunk];
  uint64_t remaining_skip = (uint64_t)offset;
  while (remaining_skip > 0) {
    size_t toRead = (size_t)std::min<uint64_t>(remaining_skip, kSkipChunk);
    size_t got = c->reader->fread(skip_buf, toRead);
    if (got != toRead) {
      fprintf(stderr, "per_entry: sidecar skip short read (got %zu of %zu)\n",
              got, toRead);
      return 1;
    }
    remaining_skip -= toRead;
  }
  size_t got = c->reader->fread(staging, size);
  if (got != size) {
    fprintf(stderr, "per_entry: sidecar payload short read (got %zu of %zu)\n",
            got, size);
    return 1;
  }
  return 0;
}

// Per-entry path: upload constants one tensor at a time driven by the
// ConstantSource union embedded in __metadata_blob. Called when the
// dispatch in hipdnn_ep_state_init_with_fs detected any constant with a
// non-NONE source. gpu_constants_blob is already allocated by
// allocate_constants_blob; this function just fills it tensor-by-tensor
// through a single reusable staging buffer, bounding host memory to the
// largest single tensor.
//
// `fs` and `constants_filename` are only consulted by SidecarSource
// entries (hybrid path). Pure streaming modules (only Splat / FileRef)
// never open the sidecar.
static int per_entry_load_constants(RuntimeState *state,
                                    const mlir::hip::HipModelMetaInfo *meta,
                                    morphizen::FileSystem *fs,
                                    const char *constants_filename) {
  auto t_prev = timing_now();

  auto *constants = meta->constants();
  int64_t count = constants ? (int64_t)constants->size() : 0;

  // Pre-scan for max tensor size -> one-shot staging alloc.
  size_t max_tensor_size = 0;
  for (int64_t i = 0; i < count; ++i) {
    size_t sz = (size_t)constants->Get(i)->size();
    if (sz > max_tensor_size)
      max_tensor_size = sz;
  }
  uint8_t *staging =
      max_tensor_size ? (uint8_t *)malloc(max_tensor_size) : nullptr;
  if (max_tensor_size && !staging) {
    fprintf(stderr, "per_entry: failed to alloc %zu bytes staging\n",
            max_tensor_size);
    return 1;
  }

  TIMING_LOG("[Session] per_entry pre-scan + staging alloc: %.3fs "
             "(max tensor = %zu bytes, count = %lld)\n",
             record_elapsed(t_prev), max_tensor_size, (long long)count);

  FrefCache fcache;
  fref_cache_init(&fcache);

  SidecarReaderCache scache; // RAII; default-initialized above

  for (int64_t i = 0; i < count; ++i) {
    auto *c = constants->Get(i);
    size_t sz = (size_t)c->size();
    void *dst = static_cast<char *>(state->gpu_constants_blob) + c->offset();

    switch (c->source_type()) {
    case mlir::hip::ConstantSource::SplatSource: {
      auto *splat = c->source_as_SplatSource();
      auto *eb = splat->elem_bytes();
      size_t elem_size = eb ? eb->size() : 0;
      if (elem_size == 0 || elem_size > 8) {
        fprintf(stderr, "per_entry: entry %lld invalid splat elem_size %zu\n",
                (long long)i, elem_size);
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      const uint8_t *elem_bytes = eb->data();
      for (size_t off = 0; off < sz; off += elem_size) {
        memcpy(staging + off, elem_bytes, elem_size);
      }
      if (hipMemcpy(dst, staging, sz, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "per_entry: hipMemcpy failed for splat entry %lld\n",
                (long long)i);
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      break;
    }
    case mlir::hip::ConstantSource::FileRefSource: {
      auto *fref = c->source_as_FileRefSource();
      const char *path = fref->path() ? fref->path()->c_str() : "";
      int64_t file_offset = fref->file_offset();
      if (!fref_cache_open(&fcache, path)) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (fref_fetch(&fcache, file_offset, sz, staging) != 0) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (hipMemcpy(dst, staging, sz, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "per_entry: hipMemcpy failed for file-ref entry %lld\n",
                (long long)i);
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      break;
    }
    case mlir::hip::ConstantSource::SidecarSource: {
      auto *side = c->source_as_SidecarSource();
      int64_t side_offset = side->sidecar_offset();
      if (!sidecar_cache_open(&scache, fs, constants_filename)) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (sidecar_fetch(&scache, side_offset, sz, staging) != 0) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (hipMemcpy(dst, staging, sz, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "per_entry: hipMemcpy failed for sidecar entry %lld\n",
                (long long)i);
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      break;
    }
    default:
      // NONE in a per-entry-mode module is invalid: the dispatch only
      // calls us when at least one entry has a source, but every entry
      // with source=NONE has no data anywhere for us to load.
      fprintf(stderr,
              "per_entry: entry %lld has source=NONE in per-entry mode\n",
              (long long)i);
      free(staging);
      fref_cache_release(&fcache);
      return 1;
    }
  }

  TIMING_LOG("[Session] per_entry upload: %.3fs (%lld constants)\n",
             record_elapsed(t_prev), (long long)count);

  free(staging);
  fref_cache_release(&fcache);
  return 0;
}

int hipdnn_ep_state_cleanup(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state in cleanup\n");
    return 0; // Best-effort - don't fail
  }

  // Best-effort cleanup - continue even if operations fail
  // Cleanup in reverse order of initialization (LIFO)

  // Synchronize stream to ensure all GPU operations complete
  if (state->stream) {
    HIP_CLEANUP(hipStreamSynchronize(state->stream));
  }

  // Free shared workspace (if allocated)
  if (state->workspace_handle != mm::kInvalidHandle) {
    (void)mm::free(state->workspace_handle);
    state->workspace_handle = mm::kInvalidHandle;
    state->workspace = nullptr;
    state->workspace_size = 0;
  }

  // Free qmoe device scratch + pinned host mirror (if allocated)
  if (state->qmoe_scratch_handle != mm::kInvalidHandle) {
    (void)mm::free(state->qmoe_scratch_handle);
    state->qmoe_scratch_handle = mm::kInvalidHandle;
    state->qmoe_scratch = nullptr;
    state->qmoe_scratch_size = 0;
  }
  if (state->qmoe_host_scratch) {
    HIP_CLEANUP(hipHostFree(state->qmoe_host_scratch));
    state->qmoe_host_scratch = nullptr;
    state->qmoe_host_scratch_size = 0;
  }

  // Free ONNX Loop driver host-mapped buffers + reusable sync event (if
  // allocated). The stream sync at the top of cleanup has already drained
  // any in-flight kernel that may have been holding loop_*_dev pointers,
  // so hipHostFree is safe here.
  if (state->loop_event) {
    HIP_CLEANUP(hipEventDestroy(static_cast<hipEvent_t>(state->loop_event)));
  }
  if (state->loop_iter_cpu_buf) {
    HIP_CLEANUP(hipHostFree(state->loop_iter_cpu_buf));
  }
  if (state->loop_iter_dev) {
    HIP_CLEANUP(hipFree(state->loop_iter_dev));
  }
  if (state->loop_cond_host) {
    HIP_CLEANUP(hipHostFree(state->loop_cond_host));
  }

  if (state->device_error_flag) {
    HIP_CLEANUP(hipFree(state->device_error_flag));
    state->device_error_flag = nullptr;
  }

  // Free memory pool (if allocated)
  if (state->pool_base) {
    HIP_CLEANUP(hipFree(state->pool_base));
  }
  if (state->buffer_offsets) {
    free(state->buffer_offsets);
  }

  // Free the single constants blob and the pointer array.
  // With shared constants, only the last reference frees the GPU memory.
  if (state->shared_constants_view) {
#ifdef _WIN32
    auto *smeta = (SharedConstantsMeta *)state->shared_constants_view;
    long remaining = shm_ref_dec(&smeta->ref_count);
    fprintf(stderr, "[SHARED_CONSTANTS] Cleanup: ref_count=%ld\n", remaining);
    if (remaining <= 0) {
      if (state->gpu_constants_handle != mm::kInvalidHandle) {
        (void)mm::free(state->gpu_constants_handle);
      } else if (state->gpu_constants_blob) {
        HIP_CLEANUP(hipFree(state->gpu_constants_blob));
      }
    }
    UnmapViewOfFile(state->shared_constants_view);
    if (state->shared_constants_mapping)
      CloseHandle(state->shared_constants_mapping);
    state->shared_constants_view = nullptr;
    state->shared_constants_mapping = nullptr;
#endif
    state->gpu_constants_handle = mm::kInvalidHandle;
    state->gpu_constants_blob = nullptr;
  } else if (state->gpu_constants_handle != mm::kInvalidHandle) {
    (void)mm::free(state->gpu_constants_handle);
    state->gpu_constants_handle = mm::kInvalidHandle;
    state->gpu_constants_blob = nullptr;
  } else if (state->gpu_constants_blob) {
    HIP_CLEANUP(hipFree(state->gpu_constants_blob));
    state->gpu_constants_blob = nullptr;
  }
  if (state->gpu_constants)
    free(state->gpu_constants);

  // Free GQA GEMM descriptor cache
  if (state->gqa_gemm_cache) {
    hipdnn_ep_gqa_gemm_cache_destroy(state->gqa_gemm_cache);
    state->gqa_gemm_cache = nullptr;
  }

  // Free MultiHeadAttention GEMM descriptor cache
  if (state->mha_gemm_cache) {
    hipdnn_ep_mha_gemm_cache_destroy(state->mha_gemm_cache);
    state->mha_gemm_cache = nullptr;
  }

  // Free CausalConvWithState descriptor/algo cache
  if (state->causal_conv_cache) {
    hipdnn_ep_causal_conv_cache_destroy(state->causal_conv_cache);
    state->causal_conv_cache = nullptr;
  }

  // Free MatMulNBits asym zero_points unpack cache
  if (state->zp_unpack_cache) {
    hipdnn_ep_zp_unpack_cache_destroy(state->zp_unpack_cache);
    state->zp_unpack_cache = nullptr;
  }

  // Free op profiling state
  if (state->op_profile) {
    op_profile_destroy(static_cast<OpProfileState *>(state->op_profile));
    state->op_profile = nullptr;
  }

  // Destroy hipBLASLt handle
  if (state->hipblas_handle) {
    hipblasLtDestroy(state->hipblas_handle);
  }

  // Destroy MIOpen handle
  if (state->miopen_handle) {
    miopenDestroy(state->miopen_handle);
  }

  // Destroy HIP stream
  if (state->stream) {
    HIP_CLEANUP(hipStreamDestroy(state->stream));
  }

  // Free the context struct itself
  free(state);

  return 0; // Best-effort cleanup always returns success
}

void *hipdnn_ep_constant_get(RuntimeState *state, int64_t index) {
  if (!state || index < 0 || (size_t)index >= state->num_constants) {
    fprintf(stderr, "hipdnn_ep_constant_get: invalid state or index %lld\n",
            (long long)index);
    return nullptr;
  }
  return state->gpu_constants[index];
}

void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state ? static_cast<void *>(state->stream) : nullptr;
}

void *hipdnn_ep_state_get_miopen_handle(RuntimeState *state) {
  return state ? static_cast<void *>(state->miopen_handle) : nullptr;
}

void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *state) {
  return state ? static_cast<void *>(state->hipblas_handle) : nullptr;
}

void *hipdnn_ep_state_get_op_profile(RuntimeState *state) {
  return state ? state->op_profile : nullptr;
}

// Per-Compute() cache invalidation hook. Today this only resets the GQA
// seqlens_k cache; future per-forward-pass caches should be cleared here
// as well so the EP-side hook stays a single call.
//
// __declspec(dllexport) matches the convention in real/test_hip_from_dll.cpp
// (belt-and-suspenders with the .def export list in CompilerDriver.cpp) and
// guarantees the symbol survives LLVM optimization in the compiled
// model.dll, which dlsym/GetProcAddress resolves it from.
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void hipdnn_ep_runtime_begin_compute(RuntimeState *state) {
  if (!state) {
    return;
  }
  state->seqlens_k_cached_valid = false;
  state->seqlens_k_cached_ptr = nullptr;
}

// Per-op profile flush hook. Moved out of hipdnn_ep_stream_sync (which is on
// the inference_compute hot path) so the resolve + map + fprintf cost lands
// AFTER the EP closes its wall_ms timing window. Same dllexport contract as
// begin_compute above so dlsym/GetProcAddress can find it; symbol is
// optional from the EP's perspective (older DLLs no-op).
//
// Body intentionally minimal: op_profile_resolve_and_print itself is a no-op
// when the pending queue is empty or the state pointer is null, so callers
// don't need to gate on HIPDNN_EP_PERF.
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void hipdnn_ep_runtime_flush_op_profile(RuntimeState *state) {
  if (!state) {
    return;
  }
  op_profile_resolve_and_print(
      static_cast<OpProfileState *>(state->op_profile));
}

//===----------------------------------------------------------------------===//
// Memory Pooling Support
//===----------------------------------------------------------------------===//

extern "C" {

int hipdnn_ep_pool_init(RuntimeState *state, size_t pool_size,
                        const size_t *buffer_offsets, size_t num_buffers) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_pool_init\n");
    return 1;
  }

  // Allocate the memory pool
  if (pool_size > 0) {
    if (hipMalloc(&state->pool_base, pool_size) != hipSuccess) {
      fprintf(stderr, "Failed to allocate memory pool of size %zu bytes\n",
              pool_size);
      return 2; // Pool allocation failed
    }
  } else {
    state->pool_base = nullptr;
  }

  // Store pool metadata
  state->pool_size = pool_size;
  state->num_buffers = num_buffers;

  // Copy buffer offsets array
  if (num_buffers > 0 && buffer_offsets) {
    state->buffer_offsets = (size_t *)malloc(sizeof(size_t) * num_buffers);
    if (!state->buffer_offsets) {
      fprintf(stderr, "Failed to allocate buffer offsets array\n");
      if (state->pool_base) {
        HIP_CLEANUP(hipFree(state->pool_base));
        state->pool_base = nullptr;
      }
      return 1; // Allocation failed
    }
    memcpy(state->buffer_offsets, buffer_offsets, sizeof(size_t) * num_buffers);
  } else {
    state->buffer_offsets = nullptr;
  }

  return 0; // Success
}

void *hipdnn_ep_get_buffer_from_pool(RuntimeState *state, size_t index) {
  if (!state || !state->pool_base) {
    fprintf(stderr, "Invalid state or pool not initialized\n");
    return nullptr;
  }

  if (index >= state->num_buffers) {
    fprintf(stderr, "Buffer index %zu out of range (num_buffers = %zu)\n",
            index, state->num_buffers);
    return nullptr;
  }

  // Return pointer at pool_base + offset
  char *pool_ptr = static_cast<char *>(state->pool_base);
  size_t offset = state->buffer_offsets[index];
  return pool_ptr + offset;
}

void *hipdnn_ep_get_pool_base(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_get_pool_base\n");
    return nullptr;
  }
  return state->pool_base;
}

//===----------------------------------------------------------------------===//
// Shared Workspace Support
//===----------------------------------------------------------------------===//

void *hipdnn_ep_state_get_workspace(RuntimeState *state) {
  return state ? state->workspace : nullptr;
}

size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state) {
  return state ? state->workspace_size : 0;
}

int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->workspace_size >= needed_size)
    return 0;

  // Amortize growth: when enlarging an existing workspace, round the new
  // size up to at least 1.5x the current buffer. Callers whose request
  // size grows monotonically by a small increment per inference (e.g. the
  // GQA decode path, which sizes S buffers to B*H*total_seq and adds B*H
  // elements per token) would otherwise trigger a hipStreamSynchronize +
  // hipFree + hipMalloc cycle on every decode step; with the 1.5x factor
  // that drops to O(log N) reallocations over the whole generation.
  // Cold-start (no existing workspace) keeps the exact requested size so
  // warmup doesn't silently double large initial allocations.
  size_t alloc_size = needed_size;
  if (state->workspace_size > 0) {
    size_t grown = state->workspace_size + state->workspace_size / 2; // 1.5x
    if (grown > alloc_size)
      alloc_size = grown;
  }

  // Grow: free old, allocate new.
  // Sync the stream first to ensure no in-flight kernel is still using the
  // old workspace buffer (prevents use-after-free on async GPU execution).
  if (state->workspace_handle != mm::kInvalidHandle) {
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    (void)mm::free(state->workspace_handle);
    state->workspace_handle = mm::kInvalidHandle;
    state->workspace = nullptr;
    state->workspace_size = 0;
  }

  mm::AllocHints hints;
  hints.mem_class = mm::MemoryClass::Activation;
  hints.lifetime = mm::Lifetime::Request;
  mm::handle_t handle = mm::alloc(alloc_size, &hints);
  if (handle == mm::kInvalidHandle) {
    fprintf(
        stderr,
        "hipdnn_ep_state_ensure_workspace: mm::alloc failed for %zu bytes\n",
        alloc_size);
    std::abort();
  }

  state->workspace_handle = handle;
  state->workspace = mm::get_ptr(handle);
  state->workspace_size = alloc_size;
  RUNTIME_DEBUG_LOG(
      "[workspace] Allocated shared workspace: %zu bytes (requested %zu)\n",
      alloc_size, needed_size);
  return 0;
}

//===----------------------------------------------------------------------===//
// QMoE scratch helpers (device + pinned-host)
//===----------------------------------------------------------------------===//
//
// Why this lives in the dynseqlen PR
// ----------------------------------
// Conceptually independent of dynamic sequence length, but the same
// grow-on-demand infrastructure (1.5x amortization, sync-before-free,
// monotonic-grow, never-shrink) introduced by dynseqlen for the runtime
// pool and the shared workspace already encodes the policy this code
// wants.  Bundling keeps the policy in one place and avoids repeating
// the same sync-and-free dance in three near-identical helpers.
//
// Sub-buffer offset validation
// ----------------------------
// `wrap_qmoe` (lib/Runtime/real/qmoe.cpp) is the sole writer of
// sub-buffer offsets, and it sums them into `total_scratch` before
// calling `hipdnn_ep_state_ensure_qmoe_scratch`.  No defensive bound
// check is performed at the use sites today; that is intentional in the
// hot path but means a future refactor that adds a sub-buffer without
// updating `total_scratch` would silently overrun the allocation.  A
// follow-up should add a `qmoe_scratch_size` parameter exchange so the
// caller can `assert(offset + size <= cap)` per sub-buffer.
//
// Cleanup-after-sync ordering
// ---------------------------
// Both grow paths sync the stream BEFORE freeing the old buffer
// (`HIP_CLEANUP(hipStreamSynchronize) → HIP_CLEANUP(hipFree)`) and BEFORE
// hipMalloc'ing the replacement.  Synchronizing first is required by HIP
// semantics: hipFree on a buffer with in-flight kernel reads is undefined
// behavior.  Both calls go through HIP_CLEANUP so a sync/free failure is
// logged but does not abort the runtime — losing this scratch is a
// recoverable error (the next call re-allocates).
//
// Mock-CI coverage
// ----------------
// The mock GPU (`lib/Runtime/mock/mock_gpu.cpp`) stubs `hipMalloc` /
// `hipHostMalloc` so the grow path runs under mock CI, but no mock-CI
// test exercises a shape progression that triggers an actual
// reallocation.  Real-CI dynamic-shape benchmarks DO exercise the grow
// path; document this gap rather than paper over it with mock fixtures.
//===----------------------------------------------------------------------===//

// qmoe device scratch (single contiguous buffer, sub-buffer offsets computed
// per-call by wrap_qmoe). Same grow-on-demand policy as workspace; never
// shrinks. Caller is responsible for ensuring no in-flight kernel still reads
// the old buffer when growing -- we sync the stream before hipFree+hipMalloc.
void *hipdnn_ep_state_get_qmoe_scratch(RuntimeState *state) {
  return state ? state->qmoe_scratch : nullptr;
}

int hipdnn_ep_state_ensure_qmoe_scratch(RuntimeState *state,
                                        size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->qmoe_scratch_size >= needed_size)
    return 0;

  // Same 1.5x growth amortization as the shared workspace -- prefill->decode
  // shape transitions and any future autotune retries grow monotonically.
  size_t alloc_size = needed_size;
  if (state->qmoe_scratch_size > 0) {
    size_t grown = state->qmoe_scratch_size + state->qmoe_scratch_size / 2;
    if (grown > alloc_size)
      alloc_size = grown;
  }

  if (state->qmoe_scratch_handle != mm::kInvalidHandle) {
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    (void)mm::free(state->qmoe_scratch_handle);
    state->qmoe_scratch_handle = mm::kInvalidHandle;
    state->qmoe_scratch = nullptr;
    state->qmoe_scratch_size = 0;
  }

  mm::AllocHints hints;
  hints.mem_class = mm::MemoryClass::Activation;
  hints.lifetime = mm::Lifetime::Request;
  mm::handle_t handle = mm::alloc(alloc_size, &hints);
  if (handle == mm::kInvalidHandle) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_qmoe_scratch: mm::alloc failed for %zu "
            "bytes\n",
            alloc_size);
    return -1;
  }
  state->qmoe_scratch_handle = handle;
  state->qmoe_scratch = mm::get_ptr(handle);
  state->qmoe_scratch_size = alloc_size;
  return 0;
}

// Pinned host mirror used for the small (k * sizeof(int32_t) + k * elem_size)
// readback of expert routing decisions per qmoe call. hipHostMalloc'd once,
// reused; no sync on grow because grow only fires when a larger num_tokens*k
// is seen and growth is rare relative to call frequency.
void *hipdnn_ep_state_get_qmoe_host_scratch(RuntimeState *state) {
  return state ? state->qmoe_host_scratch : nullptr;
}

int hipdnn_ep_state_ensure_qmoe_host_scratch(RuntimeState *state,
                                             size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->qmoe_host_scratch_size >= needed_size)
    return 0;

  size_t alloc_size = needed_size;
  if (state->qmoe_host_scratch_size > 0) {
    size_t grown =
        state->qmoe_host_scratch_size + state->qmoe_host_scratch_size / 2;
    if (grown > alloc_size)
      alloc_size = grown;
  }

  if (state->qmoe_host_scratch) {
    // Sync first: any in-flight hipMemcpyAsync(D2H) targeting this pinned
    // buffer must complete before we free it. Cheap relative to alloc.
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    HIP_CLEANUP(hipHostFree(state->qmoe_host_scratch));
    state->qmoe_host_scratch = nullptr;
    state->qmoe_host_scratch_size = 0;
  }

  if (hipHostMalloc(&state->qmoe_host_scratch, alloc_size,
                    hipHostMallocDefault) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_qmoe_host_scratch: hipHostMalloc failed "
            "for %zu bytes\n",
            alloc_size);
    return -1;
  }
  state->qmoe_host_scratch_size = alloc_size;
  return 0;
}

void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state) {
  return state ? static_cast<void *>(state->device_error_flag) : nullptr;
}

int hipdnn_ep_state_reset_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    fprintf(stderr, "hipdnn_ep_state_reset_error_flag: invalid state\n");
    return -1;
  }
  hipError_t err =
      hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  return (err == hipSuccess) ? 0 : -1;
}

int hipdnn_ep_state_read_and_clear_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    fprintf(stderr,
            "hipdnn_ep_state_read_and_clear_error_flag: invalid state\n");
    return -1;
  }

  int host_error = 0;
  hipError_t err =
      hipMemcpyAsync(&host_error, state->device_error_flag, sizeof(int),
                     hipMemcpyDeviceToHost, state->stream);
  if (err != hipSuccess)
    return -1;
  err = hipStreamSynchronize(state->stream);
  if (err != hipSuccess)
    return -1;
  if (host_error != 0)
    return host_error;

  err = hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  return (err == hipSuccess) ? 0 : -1;
}

} // extern "C"
