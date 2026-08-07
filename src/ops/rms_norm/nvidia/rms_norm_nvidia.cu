#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t rows, size_t d,
                                float eps) {
    size_t r = blockIdx.x; // one block per row
    if (r >= rows)
        return;
    const T *xrow = in + r * d;
    T *orow = out + r * d;
    to_float f;

    // block reduction of sum of squares
    extern __shared__ float ssum[];
    size_t tid = threadIdx.x;
    float local = 0.0f;
    for (size_t j = tid; j < d; j += blockDim.x) {
        float x = f(xrow[j]);
        local += x * x;
    }
    ssum[tid] = local;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(ssum[0] / (float)d + eps);
    for (size_t j = tid; j < d; j += blockDim.x) {
        orow[j] = cast_back<T>(f(xrow[j]) * inv * f(weight[j]));
    }
}

template <typename T>
void rms_norm_launch(T *out, const T *in, const T *weight, size_t rows, size_t d, float eps,
                     cudaStream_t stream) {
    size_t threads = 256;
    rms_norm_kernel<T><<<(unsigned int)((unsigned int)(rows)), (unsigned int)(threads), (unsigned int)((unsigned int)(threads) * sizeof(float)), stream>>>(out, in, weight, rows,
                                                                           d, eps);
}

void rms_norm(void *out, const void *in, const void *weight, llaisysDataType_t type, size_t rows,
              size_t d, float eps, void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_launch(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                               reinterpret_cast<const float *>(weight), rows, d, eps, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return rms_norm_launch(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(in),
                               reinterpret_cast<const llaisys::fp16_t *>(weight), rows, d, eps,
                               reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return rms_norm_launch(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(in),
                               reinterpret_cast<const llaisys::bf16_t *>(weight), rows, d, eps,
                               reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
