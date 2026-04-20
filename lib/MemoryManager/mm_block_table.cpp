#include "mm_block_table.h"
#include "mm_hal.h"

namespace mm {

void BlockTable::add_block(void *gpu_ptr) {
    ptrs_.push_back(gpu_ptr);
}

void **BlockTable::data() {
    return ptrs_.empty() ? nullptr : ptrs_.data();
}

const void *const *BlockTable::data() const {
    return ptrs_.empty() ? nullptr :
           reinterpret_cast<const void *const *>(ptrs_.data());
}

uint32_t BlockTable::size() const {
    return static_cast<uint32_t>(ptrs_.size());
}

void BlockTable::clear() {
    ptrs_.clear();
}

int BlockTable::materialize_contiguous(void *dst, size_t block_size_bytes,
                                       const mm_hal_t *hal,
                                       mm_stream_t stream) const {
    if (!dst || !hal || ptrs_.empty())
        return MM_ERROR_INVALID_ARG;

    char *out = static_cast<char *>(dst);
    for (size_t i = 0; i < ptrs_.size(); ++i) {
        if (!ptrs_[i])
            return MM_ERROR_INVALID_ARG;

        int err = hal->async_copy(out + i * block_size_bytes,
                                  ptrs_[i], block_size_bytes,
                                  MM_COPY_DEVICE_TO_DEVICE, stream, nullptr);
        if (err != MM_OK)
            return err;
    }
    return MM_OK;
}

} // namespace mm
