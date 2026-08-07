#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// Simple 1D argmax: single block reduction over the whole array.
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    to_float f;
    // global reduction via one block
    size_t tid = threadIdx.x;
    // block-level reduction using shared memory
    extern __shared__ float sval[];
    __shared__ int64_t sidx[256];

    float local_max = -INFINITY;
    int64_t local_idx = 0;
    for (size_t i = tid; i < n; i += blockDim.x) {
        float v = f(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = (int64_t)i;
        }
    }
    sval[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_idx[0] = sidx[0];
        max_val[0] = cast_back<T>(sval[0]);
    }
}

template <typename T>
void argmax_launch(int64_t *max_idx, T *max_val, const T *vals, size_t n, cudaStream_t stream) {
    const size_t threads = 256;
    argmax_kernel<T><<<(unsigned int)(1), (unsigned int)(threads), (unsigned int)((unsigned int)(threads) * sizeof(float)), stream>>>(max_idx, max_val, vals, n);
}

void argmax(void *max_idx, void *max_val, const void *vals, llaisysDataType_t type, size_t n,
            void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_launch(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<float *>(max_val),
                             reinterpret_cast<const float *>(vals), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return argmax_launch(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<llaisys::fp16_t *>(max_val),
                             reinterpret_cast<const llaisys::fp16_t *>(vals), n, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return argmax_launch(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<llaisys::bf16_t *>(max_val),
                             reinterpret_cast<const llaisys::bf16_t *>(vals), n, reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
