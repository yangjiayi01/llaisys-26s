#include "../../nvidia_common/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, size_t seqlen,
                            size_t nhead, size_t d, float theta) {
    // one thread per (seq, head, half-dim)
    size_t half = d / 2;
    size_t total = seqlen * nhead * half;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    size_t j = idx % half;
    size_t t2 = idx / half;
    size_t h = t2 % nhead;
    size_t s = t2 / nhead;

    float p = (float)pos_ids[s];
    float phi = p / powf(theta, 2.0f * (float)j / (float)d);
    float cosv = cosf(phi);
    float sinv = sinf(phi);

    to_float f;
    const T *xrow = in + (s * nhead + h) * d;
    T *orow = out + (s * nhead + h) * d;
    float a = f(xrow[j]);
    float b = f(xrow[j + half]);
    orow[j] = cast_back<T>(a * cosv - b * sinv);
    orow[j + half] = cast_back<T>(b * cosv + a * sinv);
}

template <typename T>
void rope_launch(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead,
                 size_t d, float theta, cudaStream_t stream) {
    size_t half = d / 2;
    size_t total = seqlen * nhead * half;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;
    rope_kernel<T><<<(unsigned int)((unsigned int)(blocks)), (unsigned int)(threads), 0, stream>>>(out, in, pos_ids, seqlen, nhead, d, theta);
}

void rope(void *out, const void *in, const void *pos_ids, llaisysDataType_t type, size_t seqlen,
          size_t nhead, size_t d, float theta, void *stream) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_launch(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta,
                           reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_F16:
        return rope_launch(reinterpret_cast<llaisys::fp16_t *>(out),
                           reinterpret_cast<const llaisys::fp16_t *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta,
                           reinterpret_cast<cudaStream_t>(stream));
    case LLAISYS_DTYPE_BF16:
        return rope_launch(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta,
                           reinterpret_cast<cudaStream_t>(stream));
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
