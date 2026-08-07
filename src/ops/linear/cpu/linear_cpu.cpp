#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
namespace {

// Y = X @ W^T + b ; X: [m,k], W: [n,k], b: [n]
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    // Parallelize over the output columns j: decode has m==1 (only one row),
    // but n is large (1536/8960/151936), so this scales well on many cores.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long j = 0; j < (long long)n; j++) {
        const T *wrow = weight + j * k;
        for (size_t i = 0; i < m; i++) {
            float acc = 0.0f;
            const T *xrow = in + i * k;
            for (size_t p = 0; p < k; p++) {
                acc += llaisys::utils::cast<float>(xrow[p]) * llaisys::utils::cast<float>(wrow[p]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias), m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias), m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias), m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
