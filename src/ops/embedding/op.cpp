#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include "../nvidia_common/nvidia_ops.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be Int64.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2-D.");
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1-D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2-D.");
    CHECK_ARGUMENT(index->shape()[0] == out->shape()[0], "Embedding: index length mismatch.");
    CHECK_ARGUMENT(weight->shape()[1] == out->shape()[1], "Embedding: embedding dim mismatch.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->shape()[0], weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto stream = llaisys::core::context().runtime().stream();
        nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(),
                          index->shape()[0], weight->shape()[1], stream);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
