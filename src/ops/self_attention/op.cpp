#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
                   "SelfAttention: q/k/v/out must be 3-D.");
    CHECK_ARGUMENT(q->shape()[0] == attn_val->shape()[0], "SelfAttention: seqlen mismatch.");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2] && q->shape()[2] == attn_val->shape()[2],
                   "SelfAttention: head dim mismatch.");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0], "SelfAttention: kvlen mismatch.");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "SelfAttention: nkvh mismatch.");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention: nh must be multiple of nkvh.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t nkvh = k->shape()[1];
    size_t d = q->shape()[2];
    size_t kvlen = k->shape()[0];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, d, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, d, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                               attn_val->dtype(), qlen, kvlen, nh, nkvh, d, scale, stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
