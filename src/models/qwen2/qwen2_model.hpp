#pragma once

#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    // Weight tensors (populated by the Python frontend through the C API).
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w, attn_q_b;
    std::vector<tensor_t> attn_k_w, attn_k_b;
    std::vector<tensor_t> attn_v_w, attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w, mlp_up_w, mlp_down_w;

    // KV cache: [nlayer] x [maxseq, nkvh, dh]
    std::vector<tensor_t> k_cache, v_cache;

    const LlaisysQwen2Meta &meta() const { return _meta; }
    llaisysDeviceType_t deviceType() const { return _device; }
    int deviceId() const { return _device_id; }
    size_t seqLen() const { return _seq_len; }
    void setSeqLen(size_t n) { _seq_len = n; }

    // Run the transformer over `token_ids` (prefill or single-token decode)
    // and return the argmax next token id. Updates the KV cache.
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _seq_len = 0;
};

} // namespace llaisys::models
