#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    size_t best = 0;
    T best_val = vals[0];
    for (size_t i = 1; i < size; i++) {
        if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(best_val)) {
            best = i;
            best_val = vals[i];
        }
    }
    max_idx[0] = (int64_t)best;
    max_val[0] = best_val;
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
