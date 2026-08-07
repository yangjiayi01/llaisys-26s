#include "qwen2_model.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    const size_t nlayer = _meta.nlayer;
    const size_t hs = _meta.hs;
    const size_t nh = _meta.nh;
    const size_t nkvh = _meta.nkvh;
    const size_t dh = _meta.dh;
    const size_t di = _meta.di;
    const size_t voc = _meta.voc;
    const size_t maxseq = _meta.maxseq;
    const auto dtype = _meta.dtype;

    core::context().setDevice(_device, _device_id);

    in_embed = Tensor::create({voc, hs}, dtype, _device, _device_id);
    out_embed = Tensor::create({voc, hs}, dtype, _device, _device_id);
    out_norm_w = Tensor::create({hs}, dtype, _device, _device_id);

    attn_norm_w.resize(nlayer);
    attn_q_w.resize(nlayer);  attn_q_b.resize(nlayer);
    attn_k_w.resize(nlayer);  attn_k_b.resize(nlayer);
    attn_v_w.resize(nlayer);  attn_v_b.resize(nlayer);
    attn_o_w.resize(nlayer);
    mlp_norm_w.resize(nlayer);
    mlp_gate_w.resize(nlayer); mlp_up_w.resize(nlayer); mlp_down_w.resize(nlayer);

    for (size_t l = 0; l < nlayer; l++) {
        attn_norm_w[l] = Tensor::create({hs}, dtype, _device, _device_id);
        attn_q_w[l] = Tensor::create({nh * dh, hs}, dtype, _device, _device_id);
        attn_q_b[l] = Tensor::create({nh * dh}, dtype, _device, _device_id);
        attn_k_w[l] = Tensor::create({nkvh * dh, hs}, dtype, _device, _device_id);
        attn_k_b[l] = Tensor::create({nkvh * dh}, dtype, _device, _device_id);
        attn_v_w[l] = Tensor::create({nkvh * dh, hs}, dtype, _device, _device_id);
        attn_v_b[l] = Tensor::create({nkvh * dh}, dtype, _device, _device_id);
        attn_o_w[l] = Tensor::create({hs, nh * dh}, dtype, _device, _device_id);
        mlp_norm_w[l] = Tensor::create({hs}, dtype, _device, _device_id);
        mlp_gate_w[l] = Tensor::create({di, hs}, dtype, _device, _device_id);
        mlp_up_w[l] = Tensor::create({di, hs}, dtype, _device, _device_id);
        mlp_down_w[l] = Tensor::create({hs, di}, dtype, _device, _device_id);

        // KV cache
        k_cache.push_back(Tensor::create({maxseq, nkvh, dh}, dtype, _device, _device_id));
        v_cache.push_back(Tensor::create({maxseq, nkvh, dh}, dtype, _device, _device_id));
    }
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    const size_t nlayer = _meta.nlayer;
    const size_t hs = _meta.hs;
    const size_t nh = _meta.nh;
    const size_t nkvh = _meta.nkvh;
    const size_t dh = _meta.dh;
    const size_t di = _meta.di;
    const size_t voc = _meta.voc;
    const float eps = _meta.epsilon;
    const float theta = _meta.theta;
    const auto dtype = _meta.dtype;

    core::context().setDevice(_device, _device_id);
    const auto *api = core::context().runtime().api();

    // ---- token id tensor ----
    auto tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    tokens->load(token_ids);

    // ---- position ids: [seq_len, seq_len + ntoken) ----
    std::vector<int64_t> pos_vec(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_vec[i] = (int64_t)(_seq_len + i);
    }
    auto pos = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    pos->load(pos_vec.data());

    // ---- embedding ----
    auto hidden = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
    ops::embedding(hidden, tokens, in_embed);

    // ---- layers ----
    for (size_t l = 0; l < nlayer; l++) {
        // input layernorm
        auto h_norm = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
        ops::rms_norm(h_norm, hidden, attn_norm_w[l], eps);

        // qkv projections
        auto q = Tensor::create({ntoken, nh * dh}, dtype, _device, _device_id);
        auto k = Tensor::create({ntoken, nkvh * dh}, dtype, _device, _device_id);
        auto v = Tensor::create({ntoken, nkvh * dh}, dtype, _device, _device_id);
        ops::linear(q, h_norm, attn_q_w[l], attn_q_b[l]);
        ops::linear(k, h_norm, attn_k_w[l], attn_k_b[l]);
        ops::linear(v, h_norm, attn_v_w[l], attn_v_b[l]);

        auto q3 = q->view({ntoken, nh, dh});
        auto k3 = k->view({ntoken, nkvh, dh});
        auto v3 = v->view({ntoken, nkvh, dh});

        // RoPE
        auto qr = Tensor::create({ntoken, nh, dh}, dtype, _device, _device_id);
        auto kr = Tensor::create({ntoken, nkvh, dh}, dtype, _device, _device_id);
        ops::rope(qr, q3, pos, theta);
        ops::rope(kr, k3, pos, theta);

        // write new k/v into the KV cache
        {
            auto kc_slice = k_cache[l]->slice(0, _seq_len, _seq_len + ntoken);
            auto vc_slice = v_cache[l]->slice(0, _seq_len, _seq_len + ntoken);
            size_t bytes = ntoken * nkvh * dh * 2;  // bf16/fp16 = 2 bytes
            if (dtype == LLAISYS_DTYPE_F32) {
                bytes = ntoken * nkvh * dh * 4;
            }
            // device -> device (or host -> host on CPU); same-device copy
            const llaisysMemcpyKind_t kv_kind =
                (_device == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
            api->memcpy_sync(kc_slice->data(), kr->data(), bytes, kv_kind);
            api->memcpy_sync(vc_slice->data(), v3->data(), bytes, kv_kind);
        }

        // attention over the whole cached k/v (first _seq_len+ntoken rows)
        auto kc_full = k_cache[l]->slice(0, 0, _seq_len + ntoken);
        auto vc_full = v_cache[l]->slice(0, 0, _seq_len + ntoken);
        auto attn = Tensor::create({ntoken, nh, dh}, dtype, _device, _device_id);
        ops::self_attention(attn, qr, kc_full, vc_full, 1.0f / std::sqrt((float)dh));

        // output projection + residual
        auto attn2 = attn->view({ntoken, nh * dh});
        auto o = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
        ops::linear(o, attn2, attn_o_w[l], nullptr);
        ops::add(hidden, hidden, o);

        // MLP: rms_norm -> gate/up -> swiglu -> down -> residual
        auto h_norm2 = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
        ops::rms_norm(h_norm2, hidden, mlp_norm_w[l], eps);

        auto gate = Tensor::create({ntoken, di}, dtype, _device, _device_id);
        auto up = Tensor::create({ntoken, di}, dtype, _device, _device_id);
        ops::linear(gate, h_norm2, mlp_gate_w[l], nullptr);
        ops::linear(up, h_norm2, mlp_up_w[l], nullptr);

        auto glu = Tensor::create({ntoken, di}, dtype, _device, _device_id);
        ops::swiglu(glu, gate, up);

        auto down = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
        ops::linear(down, glu, mlp_down_w[l], nullptr);
        ops::add(hidden, hidden, down);
    }

    // ---- final norm + lm head ----
    auto h_final = Tensor::create({ntoken, hs}, dtype, _device, _device_id);
    ops::rms_norm(h_final, hidden, out_norm_w, eps);

    auto logits = Tensor::create({ntoken, voc}, dtype, _device, _device_id);
    ops::linear(logits, h_final, out_embed, nullptr);

    // slice -> [1, voc], then flatten to 1-D for argmax
    auto last = logits->slice(0, ntoken - 1, ntoken)->view({voc});
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = Tensor::create({1}, dtype, _device, _device_id);
    ops::argmax(max_idx, max_val, last);

    // read back the argmax index
    int64_t result = 0;
    if (_device == LLAISYS_DEVICE_CPU) {
        std::memcpy(&result, max_idx->data(), sizeof(int64_t));
    } else {
        // device -> host copy, then read
        api->device_synchronize();
        api->memcpy_sync(&result, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        api->device_synchronize();
    }

    _seq_len += ntoken;
    return result;
}

} // namespace llaisys::models
