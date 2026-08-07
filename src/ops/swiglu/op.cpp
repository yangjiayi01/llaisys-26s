#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");
    CHECK_ARGUMENT(gate->ndim() == 2 && up->ndim() == 2 && out->ndim() == 2,
                   "SwiGLU: out/gate/up must be 2-D.");
    CHECK_ARGUMENT(gate->shape() == up->shape() && out->shape() == up->shape(),
                   "SwiGLU: shape mismatch.");
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    size_t n = gate->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), n);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), n);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), n, stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
