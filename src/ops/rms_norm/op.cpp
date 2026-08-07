#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");
    CHECK_ARGUMENT(in->ndim() == 2 && out->ndim() == 2, "RmsNorm: in/out must be 2-D.");
    CHECK_ARGUMENT(out->shape() == in->shape(), "RmsNorm: shape mismatch.");
    CHECK_ARGUMENT(weight->ndim() == 1 && weight->shape()[0] == in->shape()[1],
                   "RmsNorm: weight must be 1-D of size d.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t rows = in->shape()[0];
    size_t d = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, d, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps,
                         stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
