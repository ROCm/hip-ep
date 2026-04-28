/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../runtime_state_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char *npy_descr(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return "<f4";
  case HIPDNN_EP_DATATYPE_HALF:
    return "<f2";
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return "<V2"; // bfloat16 has no native NumPy descr; use raw 2-byte void
  case HIPDNN_EP_DATATYPE_INT32:
    return "<i4";
  case HIPDNN_EP_DATATYPE_INT64:
    return "<i8";
  default:
    return "<f4";
  }
}

// Write a NumPy .npy v1.0 file.
// Format: 6-byte magic + 2-byte version + 2-byte header_len + header + data.
bool write_npy(const char *path, const void *data, const int64_t *shape,
               int64_t rank, int64_t data_type) {
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  // Build the header dict string.
  std::string header = "{'descr': '";
  header += npy_descr(data_type);
  header += "', 'fortran_order': False, 'shape': (";
  for (int64_t i = 0; i < rank; ++i) {
    if (i > 0)
      header += ", ";
    header += std::to_string(shape[i]);
  }
  if (rank == 1)
    header += ",";
  header += "), }";

  // Pad header to 64-byte alignment (including the 10-byte preamble).
  size_t preamble = 10; // magic(6) + version(2) + header_len(2)
  size_t total = preamble + header.size() + 1; // +1 for trailing newline
  size_t pad = (64 - (total % 64)) % 64;
  header.append(pad, ' ');
  header += '\n';

  uint16_t header_len = static_cast<uint16_t>(header.size());

  FILE *fp = fopen(path, "wb");
  if (!fp)
    return false;

  // Magic number + version 1.0.
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, fp);
  fwrite(&header_len, 1, 2, fp); // little-endian on x86/ARM
  fwrite(header.data(), 1, header.size(), fp);

  // Raw data.
  int64_t num_elements = 1;
  for (int64_t i = 0; i < rank; ++i)
    num_elements *= shape[i];
  fwrite(data, static_cast<size_t>(elem_size),
         static_cast<size_t>(num_elements), fp);
  fclose(fp);
  return true;
}

} // namespace

extern "C" void hipdnn_ep_dump_tensor(RuntimeState *state, void *gpu_ptr,
                                      const int64_t *shape, int64_t rank,
                                      int64_t data_type, const char *name,
                                      const char *dump_tensors_dir) {
  if (!gpu_ptr || !shape || !name || !dump_tensors_dir)
    return;

  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  if (elem_size <= 0)
    return;

  int64_t num_elements = 1;
  for (int64_t i = 0; i < rank; ++i)
    num_elements *= shape[i];
  size_t size_bytes =
      static_cast<size_t>(num_elements) * static_cast<size_t>(elem_size);

  if (size_bytes == 0)
    return;

  std::vector<uint8_t> host_buf(size_bytes);

  hipError_t err =
      hipMemcpy(host_buf.data(), gpu_ptr, size_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    RUNTIME_DEBUG_LOG("[DUMP] hipMemcpy D2H failed for '%s': %s\n", name,
                      hipGetErrorString(err));
    return;
  }

  if (state && state->stream) {
    hipStreamSynchronize(state->stream);
  }

  std::string path = std::string(dump_tensors_dir) + "/" + name + ".npy";

  if (!write_npy(path.c_str(), host_buf.data(), shape, rank, data_type)) {
    RUNTIME_DEBUG_LOG("[DUMP] Failed to write '%s'\n", path.c_str());
    return;
  }

  RUNTIME_DEBUG_LOG("[DUMP] Saved tensor '%s' -> %s (%lld elements, dtype=%s)\n",
                    name, path.c_str(), (long long)num_elements,
                    hipdnn_ep_datatype_name(data_type));
}
