/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft NGramHashMapping (Engram n-gram token hashing).
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxGrid = 4096;

template <typename T> __device__ T wrapMul(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}

template <typename T> __device__ T wrapAdd(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}

template <typename T> __device__ T positiveMod(T value, T mod) {
  T r = static_cast<T>(value % mod);
  if (r < 0) {
    r = static_cast<T>(r + mod);
  }
  return r;
}

template <typename T>
__global__ void
ngram_hash_kernel(const T *__restrict__ ids, const T *__restrict__ mult,
                  const T *__restrict__ vocab, const T *past,
                  const T *__restrict__ offsets, const T *__restrict__ eos,
                  const int32_t *__restrict__ seg, T *__restrict__ out,
                  int64_t batch, int64_t seq, int64_t ngram,
                  int64_t heads_per_ngram, T pad, bool reset_on_eos) {
  const int64_t hist = ngram - 1;
  const int64_t num_heads = hist * heads_per_ngram;
  const T fill = eos ? *eos : pad;
  const bool do_reset = reset_on_eos && eos != nullptr;
  const int64_t total = batch * seq;

  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = linear / seq;
    const int64_t t = linear % seq;
    const int64_t base = b * seq;
    auto token = [&](int64_t j) -> T {
      if (j < hist) {
        return past ? past[b * hist + j] : fill;
      }
      return ids[base + j - hist];
    };

    const int64_t idx = hist + t;
    int64_t last_reset = -(hist + 2);
    for (int64_t j = idx - 1; j >= idx - hist; --j) {
      bool boundary = do_reset && token(j) == fill;
      if (!boundary && seg && j >= hist) {
        const int64_t tj = j - hist;
        boundary = seg[base + tj] != seg[base + tj + 1];
      }
      if (boundary) {
        last_reset = j;
        break;
      }
    }

    T *o = out + linear * num_heads;
    T mix = 0;
    for (int64_t k = 0; k < ngram; ++k) {
      const int64_t source = idx - k;
      const T tok = last_reset >= source ? fill : token(source);
      const T product = wrapMul(tok, mult[k]);
      mix = k == 0 ? product : static_cast<T>(mix ^ product);
      if (k == 0) {
        continue;
      }
      const int64_t head_base = (k - 1) * heads_per_ngram;
      for (int64_t h = 0; h < heads_per_ngram; ++h) {
        const int64_t col = head_base + h;
        const T mod = vocab[col];
        T r = 0;
        if (mod > 0) {
          r = positiveMod(mix, mod);
          if (offsets) {
            r = wrapAdd(r, offsets[col]);
          }
        }
        o[col] = r;
      }
    }
  }
}

// One block per batch row; present may alias past, so the whole row is read
// before any of it is written.
template <typename T>
__global__ void ngram_present_kernel(const T *ids, const T *past, const T *eos,
                                     T *present, int64_t seq, int64_t hist,
                                     T pad) {
  extern __shared__ __align__(sizeof(int64_t)) unsigned char smem[];
  T *row = reinterpret_cast<T *>(smem);
  const int64_t b = blockIdx.x;
  const T fill = eos ? *eos : pad;
  for (int64_t j = threadIdx.x; j < hist; j += blockDim.x) {
    const int64_t source_t = seq - hist + j;
    if (source_t >= 0) {
      row[j] = ids[b * seq + source_t];
    } else {
      row[j] = past ? past[b * hist + hist + source_t] : fill;
    }
  }
  __syncthreads();
  for (int64_t j = threadIdx.x; j < hist; j += blockDim.x) {
    present[b * hist + j] = row[j];
  }
}

template <typename T>
int launch(hipStream_t stream, const void *ids, const void *mult,
           const void *vocab, const void *past, const void *offsets,
           const void *eos, const int32_t *seg, void *hash, void *present,
           int64_t batch, int64_t seq, int64_t ngram, int64_t heads_per_ngram,
           int64_t pad_id, bool reset_on_eos) {
  const T pad = static_cast<T>(pad_id);
  const int64_t hist = ngram - 1;
  const int64_t total = batch * seq;
  if (total > 0) {
    const int64_t blocks = (total + kBlockSize - 1) / kBlockSize;
    const int grid = static_cast<int>(blocks < kMaxGrid ? blocks : kMaxGrid);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(ngram_hash_kernel<T>), dim3(grid),
                       dim3(kBlockSize), 0, stream, static_cast<const T *>(ids),
                       static_cast<const T *>(mult),
                       static_cast<const T *>(vocab),
                       static_cast<const T *>(past),
                       static_cast<const T *>(offsets),
                       static_cast<const T *>(eos), seg, static_cast<T *>(hash),
                       batch, seq, ngram, heads_per_ngram, pad, reset_on_eos);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "[custom_kernels] ngram_hash_kernel launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }
  }
  if (present && batch > 0) {
    const int threads = hist < 64 ? 64 : kBlockSize;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(ngram_present_kernel<T>),
                       dim3(static_cast<unsigned>(batch)), dim3(threads),
                       static_cast<size_t>(hist) * sizeof(T), stream,
                       static_cast<const T *>(ids),
                       static_cast<const T *>(past),
                       static_cast<const T *>(eos), static_cast<T *>(present),
                       seq, hist, pad);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] ngram_present_kernel launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }
  }
  return 0;
}

} // namespace

extern "C" int hip_ngram_hash_mapping(
    void *stream, const void *input_ids, const void *multipliers,
    const void *vocab_sizes, const void *past_ids, const void *head_offsets,
    const void *eos_token_id, const int32_t *segment_ids, void *hash_ids,
    void *present_ids, int64_t batch, int64_t seq_len, int64_t max_ngram_size,
    int64_t n_head_per_ngram, int64_t pad_id, int64_t reset_on_eos,
    int element_size_bytes) {
  if (!input_ids || !multipliers || !vocab_sizes || !hash_ids ||
      max_ngram_size < 2 || n_head_per_ngram < 1 || batch < 0 || seq_len < 0) {
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_ngram_hash_mapping: B=%lld S=%lld N=%lld "
      "heads=%lld elem=%d past=%d offsets=%d eos=%d seg=%d present=%d\n",
      (long long)batch, (long long)seq_len, (long long)max_ngram_size,
      (long long)n_head_per_ngram, element_size_bytes, past_ids != nullptr,
      head_offsets != nullptr, eos_token_id != nullptr, segment_ids != nullptr,
      present_ids != nullptr);

  hipStream_t s = static_cast<hipStream_t>(stream);
  const bool reset = reset_on_eos != 0;
  switch (element_size_bytes) {
  case 4:
    return launch<int32_t>(s, input_ids, multipliers, vocab_sizes, past_ids,
                           head_offsets, eos_token_id, segment_ids, hash_ids,
                           present_ids, batch, seq_len, max_ngram_size,
                           n_head_per_ngram, pad_id, reset);
  case 8:
    return launch<int64_t>(s, input_ids, multipliers, vocab_sizes, past_ids,
                           head_offsets, eos_token_id, segment_ids, hash_ids,
                           present_ids, batch, seq_len, max_ngram_size,
                           n_head_per_ngram, pad_id, reset);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_ngram_hash_mapping: unsupported element "
            "size %d\n",
            element_size_bytes);
    return -1;
  }
}
