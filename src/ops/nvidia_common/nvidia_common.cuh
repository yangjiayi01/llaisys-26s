#pragma once
#include "llaisys.h"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

// Convert fp16/bf16/float elements to float for computation.
struct to_float {
    __device__ float operator()(float v) const { return v; }
    __device__ float operator()(const llaisys::fp16_t &v) const {
        // manual fp16 -> fp32
        unsigned short h = v._v;
        unsigned sign = (h & 0x8000u) << 16;
        unsigned exp = (h >> 10) & 0x1Fu;
        unsigned mant = h & 0x3FFu;
        unsigned f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FFu;
                f = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            exp = exp - 15 + 127;
            f = sign | (exp << 23) | (mant << 13);
        }
        return __uint_as_float(f);
    }
    __device__ float operator()(const llaisys::bf16_t &v) const {
        unsigned short h = v._v;
        return __uint_as_float((unsigned)h << 16);
    }
};

// Convert float back to fp16/bf16/float. Use explicit convert<T>() template
// with specializations (cannot overload on return type alone).
struct from_float {
    __device__ float convert(float v) const { return v; }

    __device__ llaisys::fp16_t convert_f16(float v) const {
        __half_raw r = __float2half_rn(v);
        llaisys::fp16_t out;
        out._v = r.x;
        return out;
    }

    __device__ llaisys::bf16_t convert_bf16(float v) const {
        __nv_bfloat16_raw r = __float2bfloat16_rn(v);
        llaisys::bf16_t out;
        out._v = r.x;
        return out;
    }
};

// Helper: g(acc) writes back into T with correct rounding.
template <typename T>
__device__ T cast_back(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        from_float g;
        return g.convert_f16(v);
    } else {
        from_float g;
        return g.convert_bf16(v);
    }
}

} // namespace llaisys::ops::nvidia
