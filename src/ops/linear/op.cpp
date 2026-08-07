#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out/in/weight must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2,
                   "Linear: out/in/weight must be 2-D.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: input dim mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0],
                   "Linear: output shape mismatch.");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
                       "Linear: bias shape mismatch.");
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), m, k, n);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), m, k, n);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::linear(out->data(), in->data(), weight->data(),
                       bias ? bias->data() : nullptr, out->dtype(), m, k, n, stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
