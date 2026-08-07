#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// Self-attention with causal mask and GQA head expansion.
// q: [qlen, nh, d], k/v: [kvlen, nkvh, d], out: [qlen, nh, d]
// One thread block per (i, h) pair.
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                                      size_t d, float scale, size_t attend_offset) {
    size_t t = blockIdx.x;
    size_t i = t / nh;
    size_t h = t % nh;
    size_t kvh = h / (nh / nkvh);

    const T *qrow = q + (i * nh + h) * d;
    T *orow = attn_val + (i * nh + h) * d;
    const size_t attend_max = i + attend_offset; // kvlen - qlen

    extern __shared__ float sscores[]; // kvlen floats
    to_float f;

    // score = q . k_j * scale (causal masked)
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float acc = 0.0f;
        if (j <= attend_max) {
            const T *krow = k + j * (nkvh * d) + kvh * d;
            for (size_t p = 0; p < d; p++) {
                acc += f(qrow[p]) * f(krow[p]);
            }
            acc *= scale;
        } else {
            acc = -INFINITY;
        }
        sscores[j] = acc;
    }
    __syncthreads();

    // softmax — single-threaded to avoid shared-memory data races
    if (threadIdx.x == 0) {
        float maxv = -INFINITY;
        for (size_t j = 0; j < kvlen; j++) {
            if (sscores[j] > maxv)
                maxv = sscores[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            float e = expf(sscores[j] - maxv);
            sscores[j] = e;
            sum += e;
        }
        for (size_t j = 0; j < kvlen; j++) {
            sscores[j] /= sum;
        }
    }
    __syncthreads();

    // out = softmax @ v
    for (size_t p = threadIdx.x; p < d; p += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            acc += sscores[j] * f(v[j * (nkvh * d) + kvh * d + p]);
        }
        orow[p] = cast_back<T>(acc);
    }
}

template <typename T>
void self_attention_launch(T *attn_val, const T *q, const T *k, const T *v, size_t qlen,
                           size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale,
                           cudaStream_t stream) {
    size_t blocks = qlen * nh;
    size_t threads = 128;
    size_t smem = kvlen * sizeof(float);
    self_attention_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), smem, stream>>>(attn_val, q, k, v, qlen, kvlen,
                                                                nh, nkvh, d, scale, kvlen - qlen);
}

void self_attention(void *attn_val, const void *q, const void *k, const void *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t d, float scale, void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_launch(reinterpret_cast<float *>(attn_val),
                                     reinterpret_cast<const float *>(q),
                                     reinterpret_cast<const float *>(k),
                                     reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh, d,
                                     scale, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return self_attention_launch(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                     reinterpret_cast<const llaisys::fp16_t *>(q),
                                     reinterpret_cast<const llaisys::fp16_t *>(k),
                                     reinterpret_cast<const llaisys::fp16_t *>(v), qlen, kvlen, nh,
                                     nkvh, d, scale, reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return self_attention_launch(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                     reinterpret_cast<const llaisys::bf16_t *>(q),
                                     reinterpret_cast<const llaisys::bf16_t *>(k),
                                     reinterpret_cast<const llaisys::bf16_t *>(v), qlen, kvlen, nh,
                                     nkvh, d, scale, reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
