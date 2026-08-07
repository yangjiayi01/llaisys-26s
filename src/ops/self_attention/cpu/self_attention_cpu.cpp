#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {
namespace {

// Causal softmax attention with GQA head expansion.
// q: [qlen, nh, d], k/v: [kvlen, nkvh, d], out: [qlen, nh, d]
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t qlen,
                     size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale) {
    const size_t nrep = nh / nkvh;
    const long long total = (long long)qlen * (long long)nh;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long t = 0; t < total; t++) {
        {
            const size_t i = (size_t)(t / (long long)nh);
            const size_t h = (size_t)(t % (long long)nh);
            std::vector<float> scores(kvlen);
            std::vector<float> probs(kvlen);
            const size_t kvh = h / nrep;
            const T *qrow = q + (i * nh + h) * d;
            // k/v layout: [kvlen, nkvh, d] -> row j of head kvh is at
            // j * (nkvh * d) + kvh * d
            T *orow = attn_val + (i * nh + h) * d;

            // scores = q @ k^T * scale, with causal mask
            // causal: query i attends to key j iff j <= i + (kvlen - qlen)
            const size_t attend_max = i + (kvlen - qlen);
            for (size_t j = 0; j < kvlen; j++) {
                float acc = 0.0f;
                if (j <= attend_max) {
                    const T *krow = k + j * (nkvh * d) + kvh * d;
                    for (size_t p = 0; p < d; p++) {
                        acc += llaisys::utils::cast<float>(qrow[p]) * llaisys::utils::cast<float>(krow[p]);
                    }
                    acc *= scale;
                } else {
                    acc = -INFINITY;
                }
                scores[j] = acc;
            }

            // softmax
            float maxv = scores[0];
            for (size_t j = 1; j < kvlen; j++) {
                if (scores[j] > maxv) maxv = scores[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                float e = std::exp(scores[j] - maxv);
                probs[j] = e;
                sum += e;
            }
            for (size_t j = 0; j < kvlen; j++) {
                probs[j] /= sum;
            }

            // out = probs @ v
            for (size_t p = 0; p < d; p++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    acc += probs[j] * llaisys::utils::cast<float>(v[j * (nkvh * d) + kvh * d + p]);
                }
                orow[p] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh, d, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v), qlen, kvlen, nh, nkvh, d, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v), qlen, kvlen, nh, nkvh, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
