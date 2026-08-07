#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead,
           size_t d, float theta) {
    const size_t half = d / 2;
    for (size_t s = 0; s < seqlen; s++) {
        float p = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            const T *xrow = in + (s * nhead + h) * d;
            T *orow = out + (s * nhead + h) * d;
            for (size_t j = 0; j < half; j++) {
                float phi = p / std::pow(theta, 2.0f * (float)j / (float)d);
                float cosv = std::cos(phi);
                float sinv = std::sin(phi);
                float a = llaisys::utils::cast<float>(xrow[j]);
                float b = llaisys::utils::cast<float>(xrow[j + half]);
                orow[j] = llaisys::utils::cast<T>(a * cosv - b * sinv);
                orow[j + half] = llaisys::utils::cast<T>(b * cosv + a * sinv);
            }
        }
    }
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
