#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        to_float f;
        c[i] = cast_back<T>(f(a[i]) + f(b[i]));
    }
}

template <typename T>
void add_launch(T *c, const T *a, const T *b, size_t n, cudaStream_t stream) {
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    add_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), 0, stream>>>(c, a, b, n);
}

void add(void *c, const void *a, const void *b, llaisysDataType_t type, size_t n,
         void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_launch(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                          reinterpret_cast<const float *>(b), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return add_launch(reinterpret_cast<llaisys::fp16_t *>(c),
                          reinterpret_cast<const llaisys::fp16_t *>(a),
                          reinterpret_cast<const llaisys::fp16_t *>(b), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return add_launch(reinterpret_cast<llaisys::bf16_t *>(c),
                          reinterpret_cast<const llaisys::bf16_t *>(a),
                          reinterpret_cast<const llaisys::bf16_t *>(b), n, reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
