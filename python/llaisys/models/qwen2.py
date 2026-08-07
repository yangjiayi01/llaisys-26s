from typing import Sequence
import json
import ctypes

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # ---- read config ----
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hidden_size = cfg["hidden_size"]
        num_hidden_layers = cfg["num_hidden_layers"]
        num_attention_heads = cfg["num_attention_heads"]
        num_key_value_heads = cfg["num_key_value_heads"]
        head_dim = cfg.get("head_dim", hidden_size // num_attention_heads)
        intermediate_size = cfg["intermediate_size"]
        vocab_size = cfg["vocab_size"]
        max_position_embeddings = cfg.get("max_position_embeddings", 32768)
        rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        rope_theta = cfg.get("rope_theta", 10000.0)
        end_token = cfg.get("eos_token_id", 151645)

        meta = LlaisysQwen2Meta()
        meta.dtype = int(DataType.BF16)
        meta.nlayer = num_hidden_layers
        meta.hs = hidden_size
        meta.nh = num_attention_heads
        meta.nkvh = num_key_value_heads
        meta.dh = head_dim
        meta.di = intermediate_size
        meta.maxseq = max_position_embeddings
        meta.voc = vocab_size
        meta.epsilon = rms_norm_eps
        meta.theta = rope_theta
        meta.end_token = int(end_token)

        self._device = device
        self._meta = meta
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), None, 0
        )
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)

        self._end_token = int(end_token)

        # ---- load weights ----
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        w = self._weights.contents
        nlayer = self._meta.nlayer

        def get_tensor_array(ptr, n):
            return [ptr[i] for i in range(n)]

        attn_norm = get_tensor_array(w.attn_norm_w, nlayer)
        attn_q_w = get_tensor_array(w.attn_q_w, nlayer)
        attn_q_b = get_tensor_array(w.attn_q_b, nlayer)
        attn_k_w = get_tensor_array(w.attn_k_w, nlayer)
        attn_k_b = get_tensor_array(w.attn_k_b, nlayer)
        attn_v_w = get_tensor_array(w.attn_v_w, nlayer)
        attn_v_b = get_tensor_array(w.attn_v_b, nlayer)
        attn_o_w = get_tensor_array(w.attn_o_w, nlayer)
        mlp_norm = get_tensor_array(w.mlp_norm_w, nlayer)
        mlp_gate = get_tensor_array(w.mlp_gate_w, nlayer)
        mlp_up = get_tensor_array(w.mlp_up_w, nlayer)
        mlp_down = get_tensor_array(w.mlp_down_w, nlayer)

        def load_into(tensor_handle, arr):
            # arr: torch bf16 tensor (contiguous)
            if not arr.is_contiguous():
                arr = arr.contiguous()
            LIB_LLAISYS.tensorLoad(tensor_handle, arr.data_ptr())

        # iterate safetensors files
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    arr = data_.get_tensor(name_)
                    if name_ == "model.embed_tokens.weight":
                        load_into(w.in_embed, arr)
                    elif name_ == "lm_head.weight":
                        load_into(w.out_embed, arr)
                    elif name_ == "model.norm.weight":
                        load_into(w.out_norm_w, arr)
                    elif name_.startswith("model.layers."):
                        parts = name_.split(".")
                        # model.layers.{l}.{module}.{param}
                        l = int(parts[2])
                        module = ".".join(parts[3:])
                        if module == "input_layernorm.weight":
                            load_into(attn_norm[l], arr)
                        elif module == "self_attn.q_proj.weight":
                            load_into(attn_q_w[l], arr)
                        elif module == "self_attn.q_proj.bias":
                            load_into(attn_q_b[l], arr)
                        elif module == "self_attn.k_proj.weight":
                            load_into(attn_k_w[l], arr)
                        elif module == "self_attn.k_proj.bias":
                            load_into(attn_k_b[l], arr)
                        elif module == "self_attn.v_proj.weight":
                            load_into(attn_v_w[l], arr)
                        elif module == "self_attn.v_proj.bias":
                            load_into(attn_v_b[l], arr)
                        elif module == "self_attn.o_proj.weight":
                            load_into(attn_o_w[l], arr)
                        elif module == "post_attention_layernorm.weight":
                            load_into(mlp_norm[l], arr)
                        elif module == "mlp.gate_proj.weight":
                            load_into(mlp_gate[l], arr)
                        elif module == "mlp.up_proj.weight":
                            load_into(mlp_up[l], arr)
                        elif module == "mlp.down_proj.weight":
                            load_into(mlp_down[l], arr)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _infer(self, token_ids: Sequence[int]) -> int:
        n = len(token_ids)
        arr = (ctypes.c_int64 * n)(*token_ids)
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, n))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        outputs = list(inputs)
        current = list(inputs)
        for _ in range(max_new_tokens):
            tok = self._infer(current)
            outputs.append(tok)
            current = [tok]
            if tok == self._end_token:
                break
        return outputs
