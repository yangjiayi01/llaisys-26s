#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t d, float eps) {
    for (size_t r = 0; r < rows; r++) {
        const T *xrow = in + r * d;
        T *orow = out + r * d;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float x = llaisys::utils::cast<float>(xrow[j]);
            sum_sq += x * x;
        }
        float inv = 1.0f / std::sqrt(sum_sq / (float)d + eps);
        for (size_t j = 0; j < d; j++) {
            float x = llaisys::utils::cast<float>(xrow[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            orow[j] = llaisys::utils::cast<T>(x * inv * w);
        }
    }
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t d, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
