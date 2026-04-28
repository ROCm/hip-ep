#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

// NAN_TRACE is a heavyweight debug tool that syncs the GPU and copies every
// tensor back to the host after each op to scan for NaN/Inf.  It is OFF by
// default; set HIPDNN_EP_NAN_TRACE=1 to enable.
inline bool nan_trace_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *v = ::getenv("HIPDNN_EP_NAN_TRACE");
    enabled = (v && v[0] == '1') ? 1 : 0;
  }
  return enabled != 0;
}

// Lightweight sync-only mode: just hipDeviceSynchronize() after each op
// to prevent async-related crashes without the D2H copy overhead.
// Set HIPDNN_EP_SYNC_OPS=1 to enable.
inline bool sync_ops_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *v = ::getenv("HIPDNN_EP_SYNC_OPS");
    enabled = (v && v[0] == '1') ? 1 : 0;
  }
  return enabled != 0;
}

static int g_nan_trace_counter = 0;
static bool g_nan_first_found = false;
static int g_nan_target_op = 0;

// Saved buffer to re-check between ops
static const void *g_watched_ptr = nullptr;
static int64_t g_watched_n = 0;
static int g_watched_last_op = 0;

inline void nan_watch_check(const char *caller, int op_id) {
  if (!nan_trace_enabled())
    return;
  if (!g_watched_ptr || g_watched_n <= 0)
    return;
  hipDeviceSynchronize();
  float probe[4] = {0};
  int to_read = g_watched_n > 4 ? 4 : (int)g_watched_n;
  hipError_t e =
      hipMemcpy(probe, g_watched_ptr, to_read * sizeof(float),
                hipMemcpyDeviceToHost);
  bool has_nan = false;
  for (int i = 0; i < to_read; i++) {
    if (std::isnan(probe[i]) || std::isinf(probe[i])) {
      has_nan = true;
      break;
    }
  }
  if (has_nan) {
    fprintf(stderr,
            "[NAN_WATCH] at op#%d %s: watched ptr=%p corrupted! "
            "val=[%.4g,%.4g,%.4g,%.4g] last_write_op=%d\n",
            op_id, caller, g_watched_ptr, probe[0], probe[1], probe[2],
            probe[3], g_watched_last_op);
    fflush(stderr);
  }
}

inline void nan_trace_check(const char *op_name, const void *gpu_buf,
                            int64_t num_elements, int elem_bytes = 4) {
  if (!nan_trace_enabled()) {
    if (sync_ops_enabled())
      hipDeviceSynchronize();
    return;
  }
  int op_id = ++g_nan_trace_counter;

  // Re-check watched buffer
  nan_watch_check(op_name, op_id);

  if (g_nan_first_found || !gpu_buf || num_elements <= 0 || elem_bytes != 4)
    return;

  hipDeviceSynchronize();
  hipError_t sync_err = hipGetLastError();

  float *host_buf = (float *)malloc(num_elements * sizeof(float));
  if (!host_buf)
    return;
  hipError_t cpy_err =
      hipMemcpy(host_buf, gpu_buf, num_elements * sizeof(float),
                hipMemcpyDeviceToHost);

  int64_t nan_pos = -1;
  int64_t extreme_pos = -1;
  float extreme_val = 0.0f;
  float max_abs = 0.0f;
  if (cpy_err == hipSuccess) {
    for (int64_t i = 0; i < num_elements; i++) {
      float v = host_buf[i];
      float av = std::abs(v);
      if (av > max_abs) max_abs = av;
      if (nan_pos < 0 && (std::isnan(v) || std::isinf(v))) {
        nan_pos = i;
      }
      if (extreme_pos < 0 && av > 1e10f && !std::isnan(v) && !std::isinf(v)) {
        extreme_pos = i;
        extreme_val = v;
      }
    }
  }

  fprintf(stderr, "[NAN_TRACE] op#%d %s: ptr=%p n=%lld nan_pos=%lld",
          op_id, op_name, gpu_buf, (long long)num_elements,
          (long long)nan_pos);
  if (cpy_err == hipSuccess && num_elements > 0) {
    int nf = num_elements < 4 ? (int)num_elements : 4;
    fprintf(stderr, " first=[");
    for (int fi = 0; fi < nf; fi++)
      fprintf(stderr, "%s%.4g", fi ? "," : "", host_buf[fi]);
    fprintf(stderr, "]");
  }
  if (extreme_pos >= 0)
    fprintf(stderr, " EXTREME@%lld=%.4g", (long long)extreme_pos,
            extreme_val);
  if (max_abs > 1e4f)
    fprintf(stderr, " max_abs=%.4g", max_abs);
  if (cpy_err != hipSuccess)
    fprintf(stderr, " COPY_ERR=%d", (int)cpy_err);
  if (sync_err != hipSuccess)
    fprintf(stderr, " SYNC_ERR=%d", (int)sync_err);
  fprintf(stderr, "\n");
  fflush(stderr);

  if (nan_pos >= 0) {
    g_nan_first_found = true;
    fprintf(stderr,
            "[NAN_TRACE] *** op#%d %s: FIRST NaN/Inf at pos %lld! "
            "max_abs=%.4g n=%lld "
            "first=[%.4g,%.4g,%.4g,%.4g]\n",
            op_id, op_name, (long long)nan_pos,
            max_abs, (long long)num_elements,
            host_buf[0], host_buf[1], host_buf[2], host_buf[3]);
    fflush(stderr);
  }

  // If this is a 512-element buffer and clean, watch it
  if (nan_pos < 0 && num_elements == 512) {
    g_watched_ptr = gpu_buf;
    g_watched_n = num_elements;
    g_watched_last_op = op_id;
  }

  free(host_buf);
}

inline void nan_trace_check_input(const char *op_name, int op_id,
                                  const char *input_name, const void *gpu_buf,
                                  int64_t num_elements) {
  if (!nan_trace_enabled())
    return;
  if (!gpu_buf || num_elements <= 0)
    return;
  hipDeviceSynchronize();

  float *host_buf = (float *)malloc(num_elements * sizeof(float));
  if (!host_buf)
    return;
  hipMemcpy(host_buf, gpu_buf, num_elements * sizeof(float),
            hipMemcpyDeviceToHost);

  int64_t nan_pos = -1;
  int64_t extreme_pos = -1;
  float extreme_val = 0.0f;
  float max_abs = 0.0f;
  for (int64_t i = 0; i < num_elements; i++) {
    float v = host_buf[i];
    float av = std::abs(v);
    if (av > max_abs) max_abs = av;
    if (nan_pos < 0 && (std::isnan(v) || std::isinf(v))) {
      nan_pos = i;
    }
    if (extreme_pos < 0 && av > 1e10f && !std::isnan(v) && !std::isinf(v)) {
      extreme_pos = i;
      extreme_val = v;
    }
  }

  fprintf(stderr,
          "[NAN_TRACE] op#%d %s input_%s: ptr=%p n=%lld nan=%s pos=%lld",
          op_id, op_name, input_name, gpu_buf, (long long)num_elements,
          nan_pos >= 0 ? "YES" : "no", (long long)nan_pos);
  if (extreme_pos >= 0)
    fprintf(stderr, " EXTREME@%lld=%.4g", (long long)extreme_pos,
            extreme_val);
  if (max_abs > 1e4f)
    fprintf(stderr, " max_abs=%.4g", max_abs);
  fprintf(stderr, " first=[%.4g,%.4g,%.4g,%.4g]\n",
          host_buf[0], host_buf[1],
          num_elements > 2 ? host_buf[2] : 0.0f,
          num_elements > 3 ? host_buf[3] : 0.0f);
  fflush(stderr);

  free(host_buf);
}
