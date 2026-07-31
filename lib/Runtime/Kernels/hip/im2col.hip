// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "hip_custom_kernels.h"

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void Im2d2Col_f16(const __half *__restrict__ im, const int h,
                             const int w, const int wei_h, const int wei_w,
                             const int out_h, const int out_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, __half *__restrict__ col,
                             const int num_im_blks_x, const int num_im_blks,
                             const int tile_sz_x, const int tile_sz_y,
                             const int local_mem_size) {
  extern __shared__ __half local_im_fp16[];

  int lid = threadIdx.x;
  int64_t gid = blockIdx.x;

  int wg_ch = gid / num_im_blks;
  int im_x = ((gid % num_im_blks) % num_im_blks_x) * tile_sz_x;
  int im_y = ((gid % num_im_blks) / num_im_blks_x) * tile_sz_y;

  int out_cols_wg = (im_x + tile_sz_x) <= out_w ? tile_sz_x : (out_w - im_x);
  int out_rows_wg = (im_y + tile_sz_y) <= out_h ? tile_sz_y : (out_h - im_y);

  int im_cols_wg = (tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1;

  int inner_lid = lid;
  while (inner_lid < local_mem_size) {
    int row_to_use = inner_lid / im_cols_wg;
    int col_to_use = inner_lid % im_cols_wg;
    int lm_offset = row_to_use * im_cols_wg + col_to_use;

    int im_y_off = im_y * stride_h + row_to_use;
    int im_x_off = im_x * stride_w + col_to_use;

    if (im_y_off >= pad_h && im_y_off < h + pad_h && im_x_off >= pad_w &&
        im_x_off < w + pad_w) {
      int im_off_h = im_y_off - pad_h;
      int im_off_w = im_x_off - pad_w;
      int64_t im_off_id =
          static_cast<int64_t>(wg_ch) * h * w + im_off_h * w + im_off_w;
      local_im_fp16[lm_offset] = im[im_off_id];
    } else {
      local_im_fp16[lm_offset] = __float2half(0.0f);
    }

    inner_lid += blockDim.x;
  }
  __syncthreads();

  inner_lid = lid;
  while (inner_lid < out_cols_wg * out_rows_wg) {
    int out_x = inner_lid % out_cols_wg;
    int out_y = inner_lid / out_cols_wg;

    int64_t col_x = static_cast<int64_t>(im_y + out_y) * out_w + im_x + out_x;
    int64_t col_y = static_cast<int64_t>(wg_ch) * out_h * out_w * wei_h * wei_w;

    for (int y = 0; y < wei_h; y++) {
      for (int x = 0; x < wei_w; x++) {
        int im_off_h = out_y * stride_h + y * dilation_h;
        int im_off_w = out_x * stride_w + x * dilation_w;
        int64_t col_off =
            col_y + col_x + static_cast<int64_t>(y * wei_w + x) * out_h * out_w;
        col[col_off] = local_im_fp16[im_off_h * im_cols_wg + im_off_w];
      }
    }

    inner_lid += blockDim.x;
  }
}

__global__ void Im2d2Col_f32(const float *__restrict__ im, const int h,
                             const int w, const int wei_h, const int wei_w,
                             const int out_h, const int out_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, float *__restrict__ col,
                             const int num_im_blks_x, const int num_im_blks,
                             const int tile_sz_x, const int tile_sz_y,
                             const int local_mem_size) {
  extern __shared__ float local_im[];

  int lid = threadIdx.x;
  int64_t gid = blockIdx.x;

  int wg_ch = gid / num_im_blks;
  int im_x = ((gid % num_im_blks) % num_im_blks_x) * tile_sz_x;
  int im_y = ((gid % num_im_blks) / num_im_blks_x) * tile_sz_y;

  int out_cols_wg = (im_x + tile_sz_x) <= out_w ? tile_sz_x : (out_w - im_x);
  int out_rows_wg = (im_y + tile_sz_y) <= out_h ? tile_sz_y : (out_h - im_y);

  int im_cols_wg = (tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1;

  // Phase 1: Cooperatively load input patch into LDS
  int inner_lid = lid;
  while (inner_lid < local_mem_size) {
    int row_to_use = inner_lid / im_cols_wg;
    int col_to_use = inner_lid % im_cols_wg;
    int lm_offset = row_to_use * im_cols_wg + col_to_use;

    int im_y_off = im_y * stride_h + row_to_use;
    int im_x_off = im_x * stride_w + col_to_use;

    if (im_y_off >= pad_h && im_y_off < h + pad_h && im_x_off >= pad_w &&
        im_x_off < w + pad_w) {
      int im_off_h = im_y_off - pad_h;
      int im_off_w = im_x_off - pad_w;
      int64_t im_off_id =
          static_cast<int64_t>(wg_ch) * h * w + im_off_h * w + im_off_w;
      local_im[lm_offset] = im[im_off_id];
    } else {
      local_im[lm_offset] = 0.0f;
    }

    inner_lid += blockDim.x;
  }
  __syncthreads();

  // Phase 2: Scatter from LDS into im2col output
  inner_lid = lid;
  while (inner_lid < out_cols_wg * out_rows_wg) {
    int out_x = inner_lid % out_cols_wg;
    int out_y = inner_lid / out_cols_wg;

    int64_t col_x = static_cast<int64_t>(im_y + out_y) * out_w + im_x + out_x;
    int64_t col_y = static_cast<int64_t>(wg_ch) * out_h * out_w * wei_h * wei_w;

    for (int y = 0; y < wei_h; y++) {
      for (int x = 0; x < wei_w; x++) {
        int im_off_h = out_y * stride_h + y * dilation_h;
        int im_off_w = out_x * stride_w + x * dilation_w;
        int64_t col_off =
            col_y + col_x + static_cast<int64_t>(y * wei_w + x) * out_h * out_w;
        col[col_off] = local_im[im_off_h * im_cols_wg + im_off_w];
      }
    }

    inner_lid += blockDim.x;
  }
}

extern "C" int hip_im2d2col(void *stream, const void *input, int64_t data_type,
                            int64_t C, int64_t H, int64_t W, int64_t kh,
                            int64_t kw, int64_t pad_top, int64_t pad_bottom,
                            int64_t pad_left, int64_t pad_right,
                            int64_t stride_h, int64_t stride_w,
                            int64_t dilation_h, int64_t dilation_w,
                            void *output, int64_t out_h, int64_t out_w) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  if (data_type == HIP_DTYPE_FLOAT16) {
    int tile_sz_x = 32;
    int tile_sz_y = 8;
    int num_blks_x = (out_w + tile_sz_x - 1) / tile_sz_x;
    int num_blks = num_blks_x * ((out_h + tile_sz_y - 1) / tile_sz_y);
    int local_mem_sz =
        ((tile_sz_x - 1) * stride_w + (kw - 1) * dilation_w + 1) *
        ((tile_sz_y - 1) * stride_h + (kh - 1) * dilation_h + 1);
    // Shrink tiles until LDS fits in 64KB (type_size = 2 for fp16)
    while (local_mem_sz * 2 > 65536) {
      tile_sz_x =
          (tile_sz_x == 1) ? 1 : (tile_sz_y == 1 ? tile_sz_x / 2 : tile_sz_x);
      tile_sz_y = (tile_sz_y == 1) ? 1 : tile_sz_y / 2;
      num_blks_x = (out_w + tile_sz_x - 1) / tile_sz_x;
      num_blks = num_blks_x * ((out_h + tile_sz_y - 1) / tile_sz_y);
      local_mem_sz = ((tile_sz_x - 1) * stride_w + (kw - 1) * dilation_w + 1) *
                     ((tile_sz_y - 1) * stride_h + (kh - 1) * dilation_h + 1);
    }
    int block_size = 256;
    int group_cnt = C * num_blks;
    size_t shared_bytes = local_mem_sz * sizeof(__half);
    hipLaunchKernelGGL(
        Im2d2Col_f16, dim3(group_cnt), dim3(block_size), shared_bytes,
        hip_stream, static_cast<const __half *>(input), static_cast<int>(H),
        static_cast<int>(W), static_cast<int>(kh), static_cast<int>(kw),
        static_cast<int>(out_h), static_cast<int>(out_w),
        static_cast<int>(pad_top), static_cast<int>(pad_left),
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(dilation_h), static_cast<int>(dilation_w),
        static_cast<__half *>(output), num_blks_x, num_blks, tile_sz_x,
        tile_sz_y, local_mem_sz);
  } else {
    int tile_sz_x = 32;
    int tile_sz_y = 8;
    int num_blks_x = (out_w + tile_sz_x - 1) / tile_sz_x;
    int num_blks = num_blks_x * ((out_h + tile_sz_y - 1) / tile_sz_y);
    int local_mem_sz =
        ((tile_sz_x - 1) * stride_w + (kw - 1) * dilation_w + 1) *
        ((tile_sz_y - 1) * stride_h + (kh - 1) * dilation_h + 1);
    // Shrink tiles until LDS fits in 64KB (type_size = 4 for fp32)
    while (local_mem_sz * 4 > 65536) {
      tile_sz_x =
          (tile_sz_x == 1) ? 1 : (tile_sz_y == 1 ? tile_sz_x / 2 : tile_sz_x);
      tile_sz_y = (tile_sz_y == 1) ? 1 : tile_sz_y / 2;
      num_blks_x = (out_w + tile_sz_x - 1) / tile_sz_x;
      num_blks = num_blks_x * ((out_h + tile_sz_y - 1) / tile_sz_y);
      local_mem_sz = ((tile_sz_x - 1) * stride_w + (kw - 1) * dilation_w + 1) *
                     ((tile_sz_y - 1) * stride_h + (kh - 1) * dilation_h + 1);
    }
    int block_size = 256;
    int group_cnt = C * num_blks;
    int global_threads = block_size * group_cnt;
    size_t shared_bytes = local_mem_sz * sizeof(float);
    hipLaunchKernelGGL(
        Im2d2Col_f32, dim3(group_cnt), dim3(block_size), shared_bytes,
        hip_stream, static_cast<const float *>(input), static_cast<int>(H),
        static_cast<int>(W), static_cast<int>(kh), static_cast<int>(kw),
        static_cast<int>(out_h), static_cast<int>(out_w),
        static_cast<int>(pad_top), static_cast<int>(pad_left),
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(dilation_h), static_cast<int>(dilation_w),
        static_cast<float *>(output), num_blks_x, num_blks, tile_sz_x,
        tile_sz_y, local_mem_sz);
  }
  return hipGetLastError();
}
