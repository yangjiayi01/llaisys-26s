#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t nindex, size_t emb_dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // row index
    if (i < nindex) {
        int64_t idx = index[i];
        for (size_t j = 0; j < emb_dim; j++) {
            out[i * emb_dim + j] = weight[idx * emb_dim + j];
        }
    }
}

template <typename T>
void embedding_launch(T *out, const int64_t *index, const T *weight, size_t nindex,
                      size_t emb_dim, cudaStream_t stream) {
    size_t threads = 256;
    size_t blocks = (nindex + threads - 1) / threads;
    embedding_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), 0, stream>>>(out, index, weight, nindex, emb_dim);
}

void embedding(void *out, const void *index, const void *weight, llaisysDataType_t type,
               size_t nindex, size_t emb_dim, void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_launch(reinterpret_cast<float *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const float *>(weight), nindex, emb_dim, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return embedding_launch(reinterpret_cast<llaisys::fp16_t *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const llaisys::fp16_t *>(weight), nindex, emb_dim,
                                reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return embedding_launch(reinterpret_cast<llaisys::bf16_t *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const llaisys::bf16_t *>(weight), nindex, emb_dim,
                                reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
