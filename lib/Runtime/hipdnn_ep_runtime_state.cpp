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

#include "model_metadata_generated.h"
#include "morphizen-foundation/file_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Forward decl of static helpers defined later in this file.
static int initialize_state_handles(RuntimeState **out_state);
static int prepare_constants_array(RuntimeState *state,
                                   const mlir::hip::HipModelMetaInfo *meta);
static size_t
compute_constants_total_size(const mlir::hip::HipModelMetaInfo *meta);
static int hipmalloc_and_fixup(RuntimeState *state,
                               const mlir::hip::HipModelMetaInfo *meta,
                               size_t total_size);
static int bulk_load_constants(RuntimeState *state, morphizen::FileSystem *fs,
                               const char *constants_filename);
static int per_entry_load_constants(RuntimeState *state,
                                    const mlir::hip::HipModelMetaInfo *meta,
                                    morphizen::FileSystem *fs,
                                    const char *constants_filename);

int hipdnn_ep_state_init_with_fs(RuntimeState **out_state, void *fs,
                                 const void *metadata_blob, size_t blob_size) {
  auto t0 = timing_now();

  if (!out_state || !fs) {
    fprintf(stderr, "Invalid arguments to hipdnn_ep_state_init_with_fs\n");
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

  // 1. Pre-allocate the gpu_constants[] pointer array (cheap).
  //    num_constants is set here.
  if (int rc = prepare_constants_array(*out_state, meta); rc != 0) {
    hipdnn_ep_state_cleanup(*out_state);
    *out_state = nullptr;
    return rc;
  }

  // 2. Allocate the VRAM blob and fix up gpu_constants[i].
  size_t total_size = compute_constants_total_size(meta);
  if (int rc = hipmalloc_and_fixup(*out_state, meta, total_size); rc != 0) {
    hipdnn_ep_state_cleanup(*out_state);
    *out_state = nullptr;
    return rc;
  }

  auto *fileSystem = static_cast<morphizen::FileSystem *>(fs);

  // 3. Dispatch by metadata semantics: if any constant carries a per-entry
  //    source descriptor (Splat / FileRef), do per-tensor upload driven by
  //    the source union. Otherwise fall back to the bulk constants-file path
  //    (used by EPContext export / import and has_mem_addr downgrade).
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
    // Hybrid path also goes through per-entry; MemSource entries open
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

  // Marker (gated on HIPDNN_EP_DEBUG=1) of how much VRAM the model's constants
  // occupy -- handy to confirm at a glance the constants blob size loaded for
  // this session.
  RUNTIME_DEBUG_LOG("[CONSTANTS] Loaded constants blob: %zu bytes\n",
                    total_size);

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
  // Per-domain pool arrays start empty; grown on demand (no compile-time cap).
  state->num_pool_domains = 0;
  state->pool_base = nullptr;
  state->pool_size = nullptr;
  state->buffer_offsets = nullptr;
  state->num_buffers = 0;
  state->workspace = nullptr;
  state->workspace_size = 0;
  state->host_scratch_base = nullptr;
  state->host_scratch_size = 0;
  // Start with no output allocator (null context + callback). The EP installs
  // one via hipdnn_ep_set_output_allocator before the first inference_compute,
  // and hipdnn_ep_alloc_output then forwards each graph-output request to it.
  state->output_allocator.self = nullptr;
  state->output_allocator.allocate = nullptr;
  state->qmoe_scratch = nullptr;
  state->qmoe_scratch_size = 0;
  state->qmoe_host_scratch = nullptr;
  state->qmoe_host_scratch_size = 0;
  state->conv_scratch = nullptr;
  state->conv_scratch_size = 0;
  state->matmul_dp4a_scratch = nullptr;
  state->matmul_dp4a_scratch_size = 0;
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
  state->op_states = nullptr;
  state->num_op_states = 0;

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

  // Skip vendor-handle creation when the vendor BLAS/DNN backends are disabled:
  // the stubbed miopenCreate/hipblasLtCreate would fail and abort session
  // creation even for a model that never dispatches a vendor op. Handles stay
  // null; cleanup is already null-guarded.
#ifndef HIPDNN_EP_DISABLE_VENDOR_BLAS
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
#endif // HIPDNN_EP_DISABLE_VENDOR_BLAS

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
// and record num_constants. This is a few KB at most; hipmalloc_and_fixup
// fixes up the slots to point inside the VRAM blob.
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
  if (hipMalloc(&state->gpu_constants_blob, total_size) != hipSuccess) {
    fprintf(stderr, "hipMalloc failed for constants blob (%zu bytes)\n",
            total_size);
    return 1;
  }
  TIMING_LOG("[Session] hipMalloc VRAM: %.3fs (%zu bytes)\n",
             record_elapsed(t_prev), total_size);
  auto *constants = meta->constants();
  for (int64_t i = 0, n = (int64_t)constants->size(); i < n; ++i) {
    size_t offset = static_cast<size_t>(constants->Get(i)->offset());
    state->gpu_constants[i] =
        static_cast<char *>(state->gpu_constants_blob) + offset;
  }
  return 0;
}

// Load the entire constants file as one blob and hipMemcpy it into the
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

  TIMING_LOG("[Session] open constants file %s: %.3fs (%zu bytes)\n",
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

// Lazy reader for the partial mem-addr constants file (hybrid mode). Opened
// through the EP FileSystem on first MemSource entry; if the
// FileSystem can mmap the constants file we keep the base pointer so the case
// becomes a memcpy + hipMemcpy (no extra fread). On non-mmap backends
// we fall back to fread, which still wins because the per-entry staging
// reuse keeps host peak bounded to the largest single tensor.
struct ConstantsFileReaderCache {
  std::unique_ptr<morphizen::FileReader,
                  morphizen::FileSystem::Deleter<morphizen::FileReader>>
      reader; // default-init: null pointer, deleter unused
  const uint8_t *mmap_base = nullptr; // non-null when reader->mmap() succeeded
  size_t total_size = 0;
  bool tried_open = false;
};

// Returns true on success (or no-op if already open). On first call
// opens the constants file through fs and tries mmap.
static bool constants_file_cache_open(ConstantsFileReaderCache *c,
                                      morphizen::FileSystem *fs,
                                      const char *constants_filename) {
  if (c->tried_open)
    return c->reader != nullptr;
  c->tried_open = true;
  if (!fs || !constants_filename) {
    fprintf(stderr, "per_entry: MemSource entry but fs / constants_filename "
                    "unavailable\n");
    return false;
  }
  c->reader = fs->create_reader_template(constants_filename);
  if (!c->reader) {
    fprintf(stderr, "per_entry: failed to open partial constants file %s\n",
            constants_filename);
    return false;
  }
  c->total_size = c->reader->size();
  c->mmap_base = static_cast<const uint8_t *>(c->reader->mmap());
  return true;
}

// Copy `size` bytes at `offset` from the partial constants file into
// `staging`.
static int constants_file_fetch(ConstantsFileReaderCache *c, int64_t offset,
                                size_t size, void *staging) {
  if ((uint64_t)offset + size > (uint64_t)c->total_size) {
    fprintf(
        stderr,
        "per_entry: constants-file fetch out of range (offset=%lld size=%zu "
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
  // The constants file is touched once per entry in entry order, so this
  // still streams linearly through the file.
  c->reader->rewind();
  static constexpr size_t kSkipChunk = 64 * 1024;
  uint8_t skip_buf[kSkipChunk];
  uint64_t remaining_skip = (uint64_t)offset;
  while (remaining_skip > 0) {
    size_t toRead = (size_t)std::min<uint64_t>(remaining_skip, kSkipChunk);
    size_t got = c->reader->fread(skip_buf, toRead);
    if (got != toRead) {
      fprintf(stderr,
              "per_entry: constants-file skip short read (got %zu of %zu)\n",
              got, toRead);
      return 1;
    }
    remaining_skip -= toRead;
  }
  size_t got = c->reader->fread(staging, size);
  if (got != size) {
    fprintf(stderr,
            "per_entry: constants-file payload short read (got %zu of %zu)\n",
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
// `fs` and `constants_filename` are only consulted by MemSource
// entries (hybrid path). Pure streaming modules (only Splat / FileRef)
// never open the constants file.
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

  ConstantsFileReaderCache cfcache; // RAII; default-initialized above

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
    case mlir::hip::ConstantSource::MemSource: {
      auto *mem = c->source_as_MemSource();
      int64_t mem_offset = mem->mem_offset();
      if (!constants_file_cache_open(&cfcache, fs, constants_filename)) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (constants_file_fetch(&cfcache, mem_offset, sz, staging) != 0) {
        free(staging);
        fref_cache_release(&fcache);
        return 1;
      }
      if (hipMemcpy(dst, staging, sz, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "per_entry: hipMemcpy failed for mem entry %lld\n",
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
  if (state->workspace) {
    HIP_CLEANUP(hipFree(state->workspace));
  }

  // Free qmoe device scratch + pinned host mirror (if allocated)
  if (state->qmoe_scratch) {
    HIP_CLEANUP(hipFree(state->qmoe_scratch));
  }
  if (state->qmoe_host_scratch) {
    HIP_CLEANUP(hipHostFree(state->qmoe_host_scratch));
  }

  // Free the MIOpen convolution workspace pool (if allocated). The stream
  // sync above has drained any in-flight conv that may still be reading it.
  if (state->conv_scratch) {
    HIP_CLEANUP(hipFree(state->conv_scratch));
  }

  // Free the W4A8 dp4a matmul_nbits scratch (if allocated).
  if (state->matmul_dp4a_scratch) {
    HIP_CLEANUP(hipFree(state->matmul_dp4a_scratch));
  }

  // Tear down per-op state slots. Each entry's deletor destroys its concrete
  // type; slots reference nothing in other slots, so order is irrelevant. The
  // stream sync at the top has drained any in-flight op that may read a slot.
  if (state->op_states) {
    for (int i = 0; i < state->num_op_states; ++i) {
      OpState *os = state->op_states[i];
      if (os && os->deletor)
        os->deletor(os);
    }
    free(state->op_states);
    state->op_states = nullptr;
    state->num_op_states = 0;
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
  }

  // Free host-mapped scratch buffer (if allocated)
  if (state->host_scratch_base) {
    HIP_CLEANUP(hipHostFree(state->host_scratch_base));
  }

  // Free memory pools (if allocated) — one hipFree per non-null domain — then
  // the per-domain arrays themselves.
  if (state->pool_base) {
    for (int i = 0; i < state->num_pool_domains; ++i) {
      if (state->pool_base[i]) {
        HIP_CLEANUP(hipFree(state->pool_base[i]));
      }
    }
    free(state->pool_base);
    state->pool_base = nullptr;
  }
  if (state->pool_size) {
    free(state->pool_size);
    state->pool_size = nullptr;
  }
  state->num_pool_domains = 0;
  if (state->buffer_offsets) {
    free(state->buffer_offsets);
  }

  // Free the single constants blob and the pointer array.
  if (state->gpu_constants_blob) {
    HIP_CLEANUP(hipFree(state->gpu_constants_blob));
  }
  if (state->gpu_constants)
    free(state->gpu_constants);

  // (The GQA GEMM descriptor cache is no longer freed here: it moved into each
  // gqa instance's GqaState op-state slot, freed by the slot's deletor in the
  // per-op-state teardown just below.)

  // (The MultiHeadAttention GEMM descriptor cache is no longer freed here: it
  // moved into each multi_head_attention instance's MhaState op-state slot,
  // freed by the slot's deletor in the per-op-state teardown just below.)

  // (The CausalConvWithState descriptor/algo cache is no longer freed here:
  // it moved into each causal_conv_with_state instance's CausalConvState
  // op-state slot, freed by the slot's deletor in the per-op-state teardown
  // just below.)

  // Free the asym zero_points unpack cache (qmoe-owned; matmul_nbits keeps a
  // per-instance cache in its op-state slot).
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
    // Stop ABI-fixed helpers (memrefCopy) from reading this stream after it is
    // destroyed. begin_compute re-publishes the stream each forward pass, so
    // this only matters for a stray call after teardown -- but it keeps the
    // per-thread slot from dangling on a freed hipStream_t.
    if (hipdnn_ep_get_current_stream() == static_cast<void *>(state->stream)) {
      hipdnn_ep_set_current_stream(nullptr);
    }
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
// and guarantees the symbol survives LLVM optimization in the runtime.bc
// merged with the per-model bitcode, where LlvmIrJit's lookup_raw resolves
// it from the JITDylib's symbol table.
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
  // Publish the session stream for ABI-fixed helpers (memrefCopy) that run
  // before/within main_graph and otherwise default to stream 0.
  hipdnn_ep_set_current_stream(static_cast<void *>(state->stream));
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

// Outer (whole-scope) CPU timing sink for the EP. Same dllexport/optional
// contract as the two hooks above. `cpu_ms` is a steady_clock measurement from
// the EP side (see the header contract); op_profile_add_cpu records it as a
// CPU-only row so the resolve/print step can surface the bubble against the
// per-op CPU rows. No-op when perf is disabled (state->op_profile is null).
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void hipdnn_ep_runtime_add_cpu_profile(RuntimeState *state,
                                               const char *name,
                                               double cpu_start_us,
                                               double cpu_ms) {
  if (!state || !name) {
    return;
  }
  op_profile_add_cpu_total(static_cast<OpProfileState *>(state->op_profile),
                           name, cpu_start_us, cpu_ms);
}

//===----------------------------------------------------------------------===//
// Memory Pooling Support
//===----------------------------------------------------------------------===//

// Grow the per-domain pool arrays so that index [needed_count - 1] is valid.
// New slots are zero-filled (null base, size 0) so they behave exactly like the
// fresh inline entries the fixed-size array used to provide. Mirrors the
// grow-on-demand contract of the individual pools: never shrinks, and at steady
// state (every domain already seen) this is a no-op. Returns false on OOM,
// leaving the existing arrays intact. realloc-move is safe — callers re-derive
// pool_base[domain_id] from state on every access, never caching element ptrs.
static bool ensure_pool_domains(RuntimeState *state, int needed_count) {
  if (needed_count <= state->num_pool_domains) {
    return true;
  }
  auto *new_base = static_cast<void **>(
      realloc(state->pool_base, sizeof(void *) * needed_count));
  if (!new_base) {
    fprintf(stderr, "Failed to grow pool_base array to %d domains\n",
            needed_count);
    return false;
  }
  state->pool_base = new_base;
  auto *new_size = static_cast<size_t *>(
      realloc(state->pool_size, sizeof(size_t) * needed_count));
  if (!new_size) {
    // pool_base already grew; leaving it larger than num_pool_domains is safe
    // because num_pool_domains (updated below) still bounds every loop.
    fprintf(stderr, "Failed to grow pool_size array to %d domains\n",
            needed_count);
    return false;
  }
  state->pool_size = new_size;
  for (int i = state->num_pool_domains; i < needed_count; ++i) {
    state->pool_base[i] = nullptr;
    state->pool_size[i] = 0;
  }
  state->num_pool_domains = needed_count;
  return true;
}

extern "C" {

int hipdnn_ep_pool_init(RuntimeState *state, size_t pool_size,
                        const size_t *buffer_offsets, size_t num_buffers) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_pool_init\n");
    return 1;
  }

  // Ensure domain 0's array slot exists before we write it. Lazy domains 1..N
  // grow their slots on first hipdnn_ep_get_pool_base call.
  if (!ensure_pool_domains(state, 1)) {
    return 1;
  }

  // Eagerly size domain 0's pool from the static metadata baked at compile
  // time. Multi-domain functions leave domains 1..N empty here; those are grown
  // on first hip.get_pool call (lazy init).
  if (pool_size > 0) {
    if (hipMalloc(&state->pool_base[0], pool_size) != hipSuccess) {
      fprintf(stderr, "Failed to allocate memory pool of size %zu bytes\n",
              pool_size);
      return 2; // Pool allocation failed
    }
  } else {
    state->pool_base[0] = nullptr;
  }

  // Store pool metadata
  state->pool_size[0] = pool_size;
  state->num_buffers = num_buffers;

  // Copy buffer offsets array
  if (num_buffers > 0 && buffer_offsets) {
    state->buffer_offsets = (size_t *)malloc(sizeof(size_t) * num_buffers);
    if (!state->buffer_offsets) {
      fprintf(stderr, "Failed to allocate buffer offsets array\n");
      if (state->pool_base[0]) {
        HIP_CLEANUP(hipFree(state->pool_base[0]));
        state->pool_base[0] = nullptr;
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
  // Static buffers always live in domain 0 (the legacy single-pool case).
  // Multi-domain functions only use the dynamic hip.get_pool path; this entry
  // is kept for the eager-static-offset path.
  // pool_base is a lazily-allocated heap array (null until pool_init or the
  // first ensure_pool_domains). Check the array pointer and that domain 0 has
  // a slot BEFORE indexing [0] — otherwise calling this before pool_init (or
  // on a pool_size==0 model whose arrays were never grown) is a null deref.
  if (!state || !state->pool_base || state->num_pool_domains < 1 ||
      !state->pool_base[0]) {
    fprintf(stderr, "Invalid state or pool not initialized\n");
    return nullptr;
  }

  if (index >= state->num_buffers) {
    fprintf(stderr, "Buffer index %zu out of range (num_buffers = %zu)\n",
            index, state->num_buffers);
    return nullptr;
  }

  char *pool_ptr = static_cast<char *>(state->pool_base[0]);
  size_t offset = state->buffer_offsets[index];
  return pool_ptr + offset;
}

void *hipdnn_ep_get_pool_base(RuntimeState *state, int domain_id,
                              size_t needed_size) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_get_pool_base\n");
    return nullptr;
  }
  if (domain_id < 0) {
    fprintf(stderr,
            "hipdnn_ep_get_pool_base: negative domain_id %d — this is a "
            "compiler bug (hip-pool-allocs assigns domain ids starting at 0)\n",
            domain_id);
    return nullptr;
  }
  // Grow the per-domain arrays the first time this domain_id is seen. There is
  // no compile-time cap: the array grows to whatever the compiled function's
  // hip.get_pool ids require, which is a cold-path event on the first
  // inference.
  if (!ensure_pool_domains(state, domain_id + 1)) {
    return nullptr;
  }
  // Zero-byte pool requests arise when a dynamic temp has no elements (e.g.
  // embedding NonZero count=0 → transpose/scatter scratch). hipMalloc(0) is
  // invalid and an unallocated domain slot is nullptr, which poisons every
  // memref.view built from that pool base. Reserve one byte so views always
  // get a non-null alignedPtr; no kernel reads it when num_elements==0.
  if (needed_size == 0)
    needed_size = 1;
  // Grow-on-demand, per domain: when dynamic shapes produce larger
  // intermediates than the current allocation for this domain, reallocate
  // its pool. Pools never shrink, and other domains are untouched —
  // growing domain N is independent of domain M.
  if (needed_size > state->pool_size[domain_id]) {
    // Sync the stream before freeing: the previous inference may have
    // dispatched async kernels that still hold pointers into THIS domain's
    // pool. hipFree on an in-flight buffer is undefined behavior. Other
    // domains' in-flight pointers are unaffected since their backing memory
    // is independent — but we still need a single stream sync because all
    // domains share the runtime's compute stream. Grow events are rare so
    // the cost is amortized.
    if (state->pool_base[domain_id]) {
      if (state->stream)
        HIP_CLEANUP(hipStreamSynchronize(state->stream));
      fprintf(stderr,
              "hipdnn_ep_get_pool_base: growing pool[%d] %zu -> %zu bytes "
              "(rare; first time this large input shape was seen)\n",
              domain_id, state->pool_size[domain_id], needed_size);
      fflush(stderr);
      HIP_CLEANUP(hipFree(state->pool_base[domain_id]));
    }
    void *new_base = nullptr;
    if (hipMalloc(&new_base, needed_size) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_get_pool_base: hipMalloc failed for pool[%d] grow "
              "(%zu -> %zu bytes)\n",
              domain_id, state->pool_size[domain_id], needed_size);
      state->pool_base[domain_id] = nullptr;
      state->pool_size[domain_id] = 0;
      return nullptr;
    }
    state->pool_base[domain_id] = new_base;
    state->pool_size[domain_id] = needed_size;
  }
  return state->pool_base[domain_id];
}

void *hipdnn_ep_get_host_scratch_base(RuntimeState *state, size_t needed_size) {
  if (!state) {
    fprintf(stderr,
            "Invalid state parameter to hipdnn_ep_get_host_scratch_base\n");
    return nullptr;
  }
  // Mirrors hipdnn_ep_get_pool_base growth semantics for the host-mapped
  // scratch buffer that backs hip.get_host_scratch. One allocation per
  // function for all tiny host-fed scalars routed away from the GPU pool by
  // hip-materialize-host-scalars; grown only when shape changes increase the
  // total demand; never shrinks. hipHostMalloc(hipHostMallocMapped) memory is
  // host-writable AND GPU-readable via the device pointer mapping, so the
  // same pointer can be stored into from host code and then read by
  // subsequent GPU kernels.
  if (needed_size > state->host_scratch_size) {
    if (state->host_scratch_base) {
      if (state->stream)
        HIP_CLEANUP(hipStreamSynchronize(state->stream));
      fprintf(stderr,
              "hipdnn_ep_get_host_scratch_base: growing host scratch "
              "%zu -> %zu bytes (rare; first time this large)\n",
              state->host_scratch_size, needed_size);
      fflush(stderr);
      HIP_CLEANUP(hipHostFree(state->host_scratch_base));
    }
    void *new_base = nullptr;
    if (hipHostMalloc(&new_base, needed_size, hipHostMallocMapped) !=
        hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_get_host_scratch_base: hipHostMalloc failed "
              "(%zu -> %zu bytes)\n",
              state->host_scratch_size, needed_size);
      state->host_scratch_base = nullptr;
      state->host_scratch_size = 0;
      return nullptr;
    }
    state->host_scratch_base = new_base;
    state->host_scratch_size = needed_size;
  }
  return state->host_scratch_base;
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
  if (state->workspace) {
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    HIP_CLEANUP(hipFree(state->workspace));
    state->workspace = nullptr;
    state->workspace_size = 0;
  }

  if (hipMalloc(&state->workspace, alloc_size) != hipSuccess) {
    fprintf(
        stderr,
        "hipdnn_ep_state_ensure_workspace: hipMalloc failed for %zu bytes\n",
        alloc_size);
    std::abort();
  }

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
// Per-session grow-on-demand scratch shared by all qmoe instances. Single-
// buffer reuse is safe because the HIP stream is serialised: the next qmoe
// launches only after the previous one's kernels have consumed the buffer.
// Both grow paths sync the stream BEFORE freeing the old buffer (hipFree on a
// buffer with in-flight kernel reads is undefined behavior) and never shrink.
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

  if (state->qmoe_scratch) {
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    HIP_CLEANUP(hipFree(state->qmoe_scratch));
    state->qmoe_scratch = nullptr;
    state->qmoe_scratch_size = 0;
  }

  if (hipMalloc(&state->qmoe_scratch, alloc_size) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_qmoe_scratch: hipMalloc failed for %zu "
            "bytes\n",
            alloc_size);
    return -1;
  }
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

// MIOpen convolution workspace pool. Same grow-on-demand policy as qmoe_scratch
// above. Single-buffer reuse is safe because the stream is serialised: the
// next conv only launches after the previous miopenConvolutionForward (+ bias
// add) has consumed the workspace.
void *hipdnn_ep_state_get_conv_scratch(RuntimeState *state) {
  return state ? state->conv_scratch : nullptr;
}

int hipdnn_ep_state_ensure_conv_scratch(RuntimeState *state,
                                        size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->conv_scratch_size >= needed_size)
    return 0;

  // 1.5x growth amortisation mirrors workspace / qmoe_scratch.
  size_t alloc_size = needed_size;
  if (state->conv_scratch_size > 0) {
    size_t grown = state->conv_scratch_size + state->conv_scratch_size / 2;
    if (grown > alloc_size)
      alloc_size = grown;
  }

  if (state->conv_scratch) {
    // Drain any in-flight conv that may still be reading the old workspace
    // before we free it. Growth is rare (only on first call per new shape)
    // so the sync cost is amortised away.
    if (state->stream) {
      hipStreamSynchronize(state->stream);
    }
    HIP_CLEANUP(hipFree(state->conv_scratch));
    state->conv_scratch = nullptr;
    state->conv_scratch_size = 0;
  }

  if (hipMalloc(&state->conv_scratch, alloc_size) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_conv_scratch: hipMalloc failed for %zu "
            "bytes\n",
            alloc_size);
    return -1;
  }
  state->conv_scratch_size = alloc_size;
  return 0;
}

void *hipdnn_ep_state_get_matmul_dp4a_scratch(RuntimeState *state) {
  return state ? state->matmul_dp4a_scratch : nullptr;
}

int hipdnn_ep_state_ensure_matmul_dp4a_scratch(RuntimeState *state,
                                               size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->matmul_dp4a_scratch_size >= needed_size)
    return 0;

  // 1.5x growth amortisation mirrors conv_scratch / qmoe_scratch.
  size_t alloc_size = needed_size;
  if (state->matmul_dp4a_scratch_size > 0) {
    size_t grown =
        state->matmul_dp4a_scratch_size + state->matmul_dp4a_scratch_size / 2;
    if (grown > alloc_size)
      alloc_size = grown;
  }

  if (state->matmul_dp4a_scratch) {
    // Drain any in-flight dp4a gemv that may still be reading the old buffer
    // before we free it. Growth is rare (only when a larger K is first seen).
    if (state->stream) {
      hipStreamSynchronize(state->stream);
    }
    HIP_CLEANUP(hipFree(state->matmul_dp4a_scratch));
    state->matmul_dp4a_scratch = nullptr;
    state->matmul_dp4a_scratch_size = 0;
  }

  if (hipMalloc(&state->matmul_dp4a_scratch, alloc_size) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_matmul_dp4a_scratch: hipMalloc failed for "
            "%zu bytes\n",
            alloc_size);
    return -1;
  }
  state->matmul_dp4a_scratch_size = alloc_size;
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

int hipdnn_ep_state_set_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    fprintf(stderr, "hipdnn_ep_state_set_error_flag: invalid state\n");
    return -1;
  }
  // Byte-wise 0xff is the two's-complement representation of the runtime's
  // generic error code (-1). Queue it on the inference stream so the interface
  // boundary observes it after the existing stream synchronization.
  hipError_t err = hipMemsetAsync(state->device_error_flag, 0xff, sizeof(int),
                                  state->stream);
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
