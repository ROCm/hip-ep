/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include <cstdio>

extern "C" void hipdnn_ep_dump_tensor(RuntimeState * /*state*/,
                                      void * /*gpu_ptr*/,
                                      const int64_t * /*shape*/,
                                      int64_t /*rank*/, int64_t /*data_type*/,
                                      const char *name,
                                      const char * /*dump_tensors_dir*/) {
  fprintf(stderr, "[DUMP MOCK] hipdnn_ep_dump_tensor called for '%s'\n",
          name ? name : "(null)");
}
