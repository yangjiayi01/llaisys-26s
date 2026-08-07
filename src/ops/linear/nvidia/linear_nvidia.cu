#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// Y = X @ W^T + b ; X: [m,k], W: [n,k], b: [n]
// One thread computes one output element. Grid: blocks over (m*n).
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias, size_t m,
                              size_t k, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = m * n;
    if (idx >= total)
        return;
    size_t i = idx / n;
    size_t j = idx % n;
    to_float f;
    float acc = 0.0f;
    const T *xrow = in + i * k;
    const T *wrow = weight + j * k;
    for (size_t p = 0; p < k; p++) {
        acc += f(xrow[p]) * f(wrow[p]);
    }
    if (bias != nullptr) {
        acc += f(bias[j]);
    }
    out[idx] = cast_back<T>(acc);
}

template <typename T>
void linear_launch(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k,
                   size_t n, cudaStream_t stream) {
    size_t total = m * n;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;
    linear_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), 0, stream>>>(out, in, weight, bias, m, k, n);
}

void linear(void *out, const void *in, const void *weight, const void *bias, llaisysDataType_t type,
            size_t m, size_t k, size_t n, void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_launch(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                             reinterpret_cast<const float *>(weight),
                             reinterpret_cast<const float *>(bias), m, k, n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return linear_launch(reinterpret_cast<llaisys::fp16_t *>(out),
                             reinterpret_cast<const llaisys::fp16_t *>(in),
                             reinterpret_cast<const llaisys::fp16_t *>(weight),
                             reinterpret_cast<const llaisys::fp16_t *>(bias), m, k, n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return linear_launch(reinterpret_cast<llaisys::bf16_t *>(out),
                             reinterpret_cast<const llaisys::bf16_t *>(in),
                             reinterpret_cast<const llaisys::bf16_t *>(weight),
                             reinterpret_cast<const llaisys::bf16_t *>(bias), m, k, n, reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
