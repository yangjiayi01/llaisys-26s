# LLAISYS 作业进度（2026-08-07 更新）

## ✅ 全部完成（Assignment 0-4）
- **#0 环境** ✅ xmake/torch cu126/模型全齐
- **#1 Tensor** ✅ test_tensor 通过
- **#2 CPU 算子** ✅ 8 测试全过（F32/F16/BF16 + OpenMP）
- **#3 Qwen2 推理** ✅ CPU test_infer 通过（94 tokens 与 PyTorch 一致）
- **#4 CUDA** ✅ **全部通过**：runtime/8 算子/infer 在 RTX 4060 上全过
  - CUDA 13.1 + nvcc device-link 方案（llaisys_dlink.obj）
  - 修复：bf16 舍入（__float2half_rn）、self_attention 共享内存数据竞争（softmax 单线程）
- git 本地 5 提交，最新 40ed23d

## 推送状态
- fork 成功：yangjiayi01/llaisys-26s
- classic PAT 已提供（ghp_ 开头）
- **git push 被网络阻断**（github.com:443 时通时断，git.exe 原生连接不稳）
- **改用 Contents API 批量上传**（E:\push_api.py，后台运行中）
- ⚠️ Contents API 逐文件上传会生成多个 commit（非原子），但内容完整

## 剩余事项
1. 确认 Contents API 上传完成 → 检查 github 仓库内容
2. 触发/等待 CI（push 自动触发 build.yaml）
3. 网站作业提交：GET /api/my/camps/summer2026/assignments 拿作业 id（019f8e0d-81bc-7cf1-8345-813a812c567f 是大模型推理方向）
   - POST /api/my/assignments/{id}/submission（github_repo_url=自己的fork地址, github_commit_url, 附件留空）
   - cookie 需重新获取（旧的可能过期）
4. 更新 REPORT.md（若需要）

## 关键技术点
- CUDA 构建：xmake c -y && xmake → test_dlink2.bat → xmake（两阶段，dlink 需在 lib 之后）
- git 网络：schannel 已设，但时通时断；PowerShell/api.github.com 稳定
