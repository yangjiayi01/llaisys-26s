#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "ROPE: all tensors must be contiguous.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "ROPE: pos_ids must be Int64.");
    CHECK_ARGUMENT(in->ndim() == 3 && out->ndim() == 3, "ROPE: in/out must be 3-D.");
    CHECK_ARGUMENT(out->shape() == in->shape(), "ROPE: shape mismatch.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0],
                   "ROPE: pos_ids must be 1-D of length seqlen.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "ROPE: head dim must be even.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seqlen, nhead, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seqlen, nhead, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d,
                     theta, stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
