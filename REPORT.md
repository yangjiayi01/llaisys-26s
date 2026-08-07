# LLAISYS 26 Summer Camp — Assignment Report

**Author**: winter (yangjiayi01)
**Direction**: 大模型推理服务系统 (LLM Inference and Serving System)
**Repository**: fork of [wooway777/llaisys-26s](https://github.com/wooway777/llaisys-26s)

---

## Reproduction Procedure

### Environment
| Item | Value |
|---|---|
| OS | Windows 11 (also passes on Ubuntu via GitHub Actions CI) |
| Compiler | MSVC 14.42 (VS 2022), GCC/Clang on Linux |
| Build tool | xmake v3.0.9 |
| Python | 3.12 (torch 2.13, transformers 5.x, safetensors, numpy) |
| Model | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (bf16) |

### Build & Test Steps
```bash
# build C++ backend and install shared library
xmake
xmake install

# install python package
pip install ./python/

# run all tests (same sequence as GitHub Actions build.yaml)
python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/ops/add.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
python test/test_infer.py --test
```

### CUDA build (optional, platform: NVIDIA)
```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
python test/test_runtime.py --device nvidia
python test/ops/linear.py --device nvidia   # etc. for all ops
python test/test_infer.py --test --device nvidia
```

---

## Results

### Assignment #0 — Getting Started ✅
- Environment installed (xmake, MSVC, Python, PyTorch).
- `test_runtime.py --device cpu` passes.

### Assignment #1 — Tensor ✅
Implemented in `src/tensor/tensor.cpp`:
- `load`: host→device copy via runtime memcpy.
- `isContiguous`: stride check against row-major layout.
- `view`: PyTorch-compatible stride derivation with compatibility validation
  (rejects non-mergeable layouts, e.g. shape (2,3,5) with strides (30,10,1)
  cannot be viewed as (2,15) without a copy).
- `permute`: dimension reorder with permutation validation, no data movement.
- `slice`: offset + shape adjustment along one dim.
- Bonus: `contiguous`, `reshape`, `to` (device transfer).

`python test/test_tensor.py` → **Test passed!** (load/view/permute/slice all
match PyTorch shapes, strides and values.)

### Assignment #2 — CPU Operators ✅
Implemented 7 operators in `src/ops/<name>/` with per-op `cpu/` kernels,
supporting **F32, F16 and BF16**:
- `argmax`, `embedding`, `linear`, `rms_norm`, `rope`, `self_attention`
  (causal mask + GQA head expansion), `swiglu`.

All 8 operator tests (incl. `add`) pass on every dtype, including the
512×4096 large-shape cases.

### Assignment #3 — LLM Inference ✅
Implemented a full Qwen2-style forward pass in C++:
- `src/models/qwen2/qwen2_model.{hpp,cpp}`: embedding → 28 transformer layers
  (rms_norm → qkv → RoPE → KV-cache attention → o_proj → SwiGLU MLP) →
  final norm → lm_head, **with KV cache** for efficient decode.
- `src/llaisys/qwen2.cc`: C API (create/destroy/weights/infer).
- `python/llaisys/models/qwen2.py`: safetensors weight loading + generate loop.

`python test/test_infer.py --model <model_dir> --test` →
**Test passed!** — all 94 generated tokens are **identical** to the PyTorch
reference output (including the `<think>` block and end token).

### Assignment #4 — CUDA (NVIDIA) ✅
- `src/device/nvidia/nvidia_runtime_api.cu`: full CUDA runtime API
  (device/stream/memory/memcpy).
- CUDA kernels for all 7 operators under `src/ops/<name>/nvidia/`.
- `xmake/nvidia.lua` build rules, gated by `--nv-gpu=y`.
- Model runs on NVIDIA with device-aware KV cache copies.

`test_runtime.py --device nvidia` and operator tests `--device nvidia` pass.

---

## Supported Platforms & Status

| Platform | Status | Notes |
|---|---|---|
| CPU (Windows MSVC) | ✅ All tests pass | OpenMP parallel linear/attention |
| CPU (Ubuntu GCC) | ✅ All tests pass | CI-verified |
| NVIDIA CUDA | ✅ runtime + ops + infer pass | RTX 4060, CUDA 13.1 |
| Iluvatar / Metax / Moore Threads | ⏳ not verified | no access granted |

## Notes
- Attention biases of DeepSeek-R1-Distill-Qwen-1.5B are supported (the model
  has q/k/v biases, unlike plain Qwen2).
- Performance: decode ≈1.3 s/token on CPU (24-core, OpenMP), much faster on
  GPU.
