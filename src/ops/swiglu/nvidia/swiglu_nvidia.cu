#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        to_float f;
        float gatev = f(gate[i]);
        float upv = f(up[i]);
        float sig = 1.0f / (1.0f + expf(-gatev));
        out[i] = cast_back<T>(upv * gatev * sig);
    }
}

template <typename T>
void swiglu_launch(T *out, const T *gate, const T *up, size_t n, cudaStream_t stream) {
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    swiglu_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), 0, stream>>>(out, gate, up, n);
}

void swiglu(void *out, const void *gate, const void *up, llaisysDataType_t type, size_t n,
            void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_launch(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                             reinterpret_cast<const float *>(up), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return swiglu_launch(reinterpret_cast<llaisys::fp16_t *>(out),
                             reinterpret_cast<const llaisys::fp16_t *>(gate),
                             reinterpret_cast<const llaisys::fp16_t *>(up), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return swiglu_launch(reinterpret_cast<llaisys::bf16_t *>(out),
                             reinterpret_cast<const llaisys::bf16_t *>(gate),
                             reinterpret_cast<const llaisys::bf16_t *>(up), n, reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
