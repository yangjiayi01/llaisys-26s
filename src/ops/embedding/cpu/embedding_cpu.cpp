#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t nindex, size_t emb_dim) {
    for (size_t i = 0; i < nindex; i++) {
        int64_t idx = index[i];
        std::memcpy(out + i * emb_dim, weight + idx * emb_dim, emb_dim * sizeof(T));
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t nindex, size_t emb_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight), nindex, emb_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), nindex, emb_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), nindex, emb_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
