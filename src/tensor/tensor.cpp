#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // A tensor is contiguous if it can be flattened into a single
    // continuous memory block without reordering. For row-major (C-order)
    // tensors this means strides[i] == product(shape[i+1:]).
    size_t ndim = _meta.shape.size();
    if (ndim == 0) {
        return true;
    }
    // stride of the last dim must be 1
    if (_meta.strides[ndim - 1] != 1) {
        return false;
    }
    size_t expected = 1;
    for (size_t i = ndim; i-- > 0;) {
        if (_meta.strides[i] != (ptrdiff_t)expected) {
            return false;
        }
        expected *= _meta.shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim = _meta.shape.size();
    CHECK_ARGUMENT(order.size() == ndim, "Permute: order size must equal tensor ndim.");
    // Validate that order is a valid permutation of [0, ndim)
    std::vector<bool> seen(ndim, false);
    for (size_t i = 0; i < ndim; i++) {
        CHECK_ARGUMENT(order[i] < ndim && !seen[order[i]], "Permute: order must be a valid permutation.");
        seen[order[i]] = true;
    }
    TensorMeta meta{_meta.dtype, {}, {}};
    meta.shape.reserve(ndim);
    meta.strides.reserve(ndim);
    for (size_t i = 0; i < ndim; i++) {
        meta.shape.push_back(_meta.shape[order[i]]);
        meta.strides.push_back(_meta.strides[order[i]]);
    }
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

static bool try_view_strides(const Tensor &self, const std::vector<size_t> &shape,
                               std::vector<ptrdiff_t> &out_strides) {
    // Try to derive new strides for `shape` from the current tensor without
    // copying data. Mirrors PyTorch's computeStride semantics: each new
    // dimension may merge a contiguous block of the old trailing dimensions.
    size_t ndim = shape.size();
    out_strides.assign(ndim, 0);
    const auto &old_shape = self.shape();
    const auto &old_strides = self.strides();
    size_t old_dim = old_shape.size();
    size_t old_size = 1;
    ptrdiff_t old_stride = 1;
    for (size_t new_dim = ndim; new_dim-- > 0;) {
        if (shape[new_dim] == 1) {
            // A dim of size 1 is stride-agnostic; keep the last old stride.
            out_strides[new_dim] = old_stride;
        } else {
            while (old_size < shape[new_dim]) {
                if (old_dim == 0) {
                    return false;
                }
                old_dim--;
                old_size *= old_shape[old_dim];
                old_stride = old_strides[old_dim];
            }
            if (old_size != shape[new_dim]) {
                return false;
            }
            out_strides[new_dim] = old_stride;
            old_size = 1;
            old_stride = 1;
        }
    }
    // Any remaining old dims must be of size 1 (stride-agnostic).
    for (; old_dim-- > 0;) {
        if (old_shape[old_dim] != 1) {
            return false;
        }
    }
    return true;
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // Total element count must match.
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    CHECK_ARGUMENT(new_numel == this->numel(), "View: new shape must have the same number of elements.");

    std::vector<ptrdiff_t> new_strides;
    if (this->isContiguous()) {
        // Row-major default strides.
        new_strides.resize(shape.size());
        size_t stride = 1;
        for (size_t i = 1; i <= shape.size(); i++) {
            new_strides[shape.size() - i] = (ptrdiff_t)stride;
            stride *= shape[shape.size() - i];
        }
    } else {
        bool ok = try_view_strides(*this, shape, new_strides);
        CHECK_ARGUMENT(ok, "View: new view is not compatible with the current tensor layout.");
    }

    TensorMeta meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim = _meta.shape.size();
    CHECK_ARGUMENT(dim < ndim, "Slice: dim out of range.");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "Slice: invalid range.");
    if (start == end) {
        CHECK_ARGUMENT(false, "Slice: empty slice is not supported (start == end).");
    }
    TensorMeta meta{_meta.dtype, _meta.shape, _meta.strides};
    meta.shape[dim] = end - start;
    size_t offset = _offset + start * _meta.strides[dim] * utils::dsize(_meta.dtype);
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    const auto *api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        api->memcpy_sync(this->data(), src_, bytes, LLAISYS_MEMCPY_H2H);
    } else {
        api->memcpy_sync(this->data(), src_, bytes, LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    // Allocate a fresh contiguous tensor and copy element-by-element
    // following the strided layout.
    tensor_t out = Tensor::create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    const auto &shape = _meta.shape;
    const auto &strides = _meta.strides;
    size_t dtype_size = this->elementSize();
    size_t ndim = shape.size();

    // Iterate over the flattened index space.
    std::vector<size_t> idx(ndim, 0);
    for (size_t linear = 0; linear < this->numel(); linear++) {
        size_t src_off = _offset;
        for (size_t d = 0; d < ndim; d++) {
            src_off += idx[d] * strides[d] * dtype_size;
        }
        std::memcpy(out->data() + linear * dtype_size, this->data() - _offset + src_off, dtype_size);
        // increment indices
        for (size_t d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // If a compatible view exists, return it without copying; otherwise
    // make a contiguous copy first (same semantics as PyTorch's reshape).
    std::vector<ptrdiff_t> dummy;
    if (try_view_strides(*this, shape, dummy) || this->isContiguous()) {
        return this->view(shape);
    }
    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device < 0) {
        device = this->deviceId();
    }
    if (this->deviceType() == device_type && this->deviceId() == device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    tensor_t out = Tensor::create(_meta.shape, _meta.dtype, device_type, device);
    core::context().setDevice(device_type, device);
    const auto *api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        api->memcpy_sync(out->data(), this->data(), bytes, LLAISYS_MEMCPY_H2H);
    } else if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        api->memcpy_sync(out->data(), this->data(), bytes, LLAISYS_MEMCPY_H2D);
    } else if (device_type == LLAISYS_DEVICE_CPU) {
        api->memcpy_sync(out->data(), this->data(), bytes, LLAISYS_MEMCPY_D2H);
    } else {
        api->memcpy_sync(out->data(), this->data(), bytes, LLAISYS_MEMCPY_D2D);
    }
    return out;
}

} // namespace llaisys
