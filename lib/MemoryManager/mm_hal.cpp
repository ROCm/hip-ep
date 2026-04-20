#include "mm_hal.h"
#include <cstring>

static const mm_hal_t *g_hal = nullptr;

int mm_hal_register(const mm_hal_t *hal) {
  if (!hal)
    return MM_ERROR_INVALID_ARG;

  /* Validate required function pointers */
  if (!hal->raw_alloc || !hal->raw_free || !hal->get_device_count ||
      !hal->get_total_memory || !hal->get_free_memory)
    return MM_ERROR_INVALID_ARG;

  g_hal = hal;
  return MM_OK;
}

const mm_hal_t *mm_hal_get(void) { return g_hal; }
