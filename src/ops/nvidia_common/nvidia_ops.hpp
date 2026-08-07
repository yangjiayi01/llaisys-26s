#pragma once
#include "llaisys.h"

// NOTE: this header intentionally avoids including CUDA headers so that the
// operator dispatch code (op.cpp) can be compiled for CPU-only builds.
// The CUDA kernels (compiled as .cu) pass their cudaStream_t as void*.

namespace llaisys::ops::nvidia {

void add(void *c, const void *a, const void *b, llaisysDataType_t type, size_t n, void *stream);

void argmax(void *max_idx, void *max_val, const void *vals, llaisysDataType_t type, size_t n,
            void *stream);

void embedding(void *out, const void *index, const void *weight, llaisysDataType_t type,
               size_t nindex, size_t emb_dim, void *stream);

void linear(void *out, const void *in, const void *weight, const void *bias, llaisysDataType_t type,
            size_t m, size_t k, size_t n, void *stream);

void rms_norm(void *out, const void *in, const void *weight, llaisysDataType_t type, size_t rows,
              size_t d, float eps, void *stream);

void rope(void *out, const void *in, const void *pos_ids, llaisysDataType_t type, size_t seqlen,
          size_t nhead, size_t d, float theta, void *stream);

void self_attention(void *attn_val, const void *q, const void *k, const void *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t d, float scale, void *stream);

void swiglu(void *out, const void *gate, const void *up, llaisysDataType_t type, size_t n,
            void *stream);

} // namespace llaisys::ops::nvidia
