#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        // SwiGLU: out = up * gate * sigmoid(gate)
        float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * g * sig);
    }
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), n);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), n);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
