#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2_model.hpp"

#include <vector>
#include <iostream>
#include <exception>

namespace {

struct Qwen2ModelHandle {
    llaisys::models::Qwen2Model *model;
    // C handle wrappers for weight tensors (owned here).
    std::vector<LlaisysTensor *> wrappers;
    LlaisysQwen2Weights weights;

    explicit Qwen2ModelHandle(llaisys::models::Qwen2Model *m) : model(m) {
        weights.in_embed = nullptr;
        weights.out_embed = nullptr;
        weights.out_norm_w = nullptr;
        weights.attn_norm_w = nullptr;
        weights.attn_q_w = nullptr;
        weights.attn_q_b = nullptr;
        weights.attn_k_w = nullptr;
        weights.attn_k_b = nullptr;
        weights.attn_v_w = nullptr;
        weights.attn_v_b = nullptr;
        weights.attn_o_w = nullptr;
        weights.mlp_norm_w = nullptr;
        weights.mlp_gate_w = nullptr;
        weights.mlp_up_w = nullptr;
        weights.mlp_down_w = nullptr;
    }

    ~Qwen2ModelHandle() {
        for (auto *w : wrappers) {
            delete w;
        }
        delete model;
    }
};

LlaisysTensor *wrap(llaisys::tensor_t t, std::vector<LlaisysTensor *> &wrappers) {
    auto *w = new LlaisysTensor{t};
    wrappers.push_back(w);
    return w;
}

} // namespace

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                      llaisysDeviceType_t device,
                                                      int *device_ids, int ndevice) {
        int device_id = (device_ids != nullptr && ndevice > 0) ? device_ids[0] : 0;
        auto *cpp_model = new llaisys::models::Qwen2Model(*meta, device, device_id);
        auto *handle = new Qwen2ModelHandle(cpp_model);
        auto *m = cpp_model;

        // Populate the weight handle struct once so the Python frontend can
        // fill in the weights afterwards.
        handle->weights.in_embed = wrap(m->in_embed, handle->wrappers);
        handle->weights.out_embed = wrap(m->out_embed, handle->wrappers);
        handle->weights.out_norm_w = wrap(m->out_norm_w, handle->wrappers);

        size_t nlayer = m->meta().nlayer;
        handle->weights.attn_norm_w = new LlaisysTensor *[nlayer];
        handle->weights.attn_q_w = new LlaisysTensor *[nlayer];
        handle->weights.attn_q_b = new LlaisysTensor *[nlayer];
        handle->weights.attn_k_w = new LlaisysTensor *[nlayer];
        handle->weights.attn_k_b = new LlaisysTensor *[nlayer];
        handle->weights.attn_v_w = new LlaisysTensor *[nlayer];
        handle->weights.attn_v_b = new LlaisysTensor *[nlayer];
        handle->weights.attn_o_w = new LlaisysTensor *[nlayer];
        handle->weights.mlp_norm_w = new LlaisysTensor *[nlayer];
        handle->weights.mlp_gate_w = new LlaisysTensor *[nlayer];
        handle->weights.mlp_up_w = new LlaisysTensor *[nlayer];
        handle->weights.mlp_down_w = new LlaisysTensor *[nlayer];

        for (size_t l = 0; l < nlayer; l++) {
            handle->weights.attn_norm_w[l] = wrap(m->attn_norm_w[l], handle->wrappers);
            handle->weights.attn_q_w[l] = wrap(m->attn_q_w[l], handle->wrappers);
            handle->weights.attn_q_b[l] = wrap(m->attn_q_b[l], handle->wrappers);
            handle->weights.attn_k_w[l] = wrap(m->attn_k_w[l], handle->wrappers);
            handle->weights.attn_k_b[l] = wrap(m->attn_k_b[l], handle->wrappers);
            handle->weights.attn_v_w[l] = wrap(m->attn_v_w[l], handle->wrappers);
            handle->weights.attn_v_b[l] = wrap(m->attn_v_b[l], handle->wrappers);
            handle->weights.attn_o_w[l] = wrap(m->attn_o_w[l], handle->wrappers);
            handle->weights.mlp_norm_w[l] = wrap(m->mlp_norm_w[l], handle->wrappers);
            handle->weights.mlp_gate_w[l] = wrap(m->mlp_gate_w[l], handle->wrappers);
            handle->weights.mlp_up_w[l] = wrap(m->mlp_up_w[l], handle->wrappers);
            handle->weights.mlp_down_w[l] = wrap(m->mlp_down_w[l], handle->wrappers);
        }
        return reinterpret_cast<struct LlaisysQwen2Model *>(handle);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete reinterpret_cast<Qwen2ModelHandle *>(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &reinterpret_cast<Qwen2ModelHandle *>(model)->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                   int64_t *token_ids, size_t ntoken) {
        try {
            return reinterpret_cast<Qwen2ModelHandle *>(model)->model->infer(token_ids, ntoken);
        } catch (const std::exception &e) {
            std::cerr << "[Qwen2 infer exception] " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cerr << "[Qwen2 infer exception] unknown" << std::endl;
            return -1;
        }
    }
}
