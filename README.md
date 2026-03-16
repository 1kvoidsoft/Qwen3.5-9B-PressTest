# Qwen3.5-9B Press Test

本地运行 Qwen3.5-9B 的 VRAM 感知型流式对话脚本，附自动化压测工具，适用于 NVIDIA RTX 40/50 系显卡。

- `qwen35_console_chatbot_streaming.py` — 交互式对话，支持流式输出与思考模式
- `qwen35_stress_test.py` — 批量任务压测，监控 VRAM 并输出 JSON 报告

## 特性

- **流式输出** — TextIteratorStreamer 实现逐 token 打印
- **思考模式** — 支持 `<think>` 块显示/隐藏，用 `--thinking` 开启
- **VRAM 精细管理** — 历史记录截断 + `empty_cache()`，全程 reserved ≤ 9.9 GB
- **多轮对话上下文控制** — `--max_rounds` / `--store_chars` 防止 prompt 爆炸

## 环境要求

- NVIDIA RTX 40/50 系（或任意 ≥ 12 GB VRAM 的 CUDA 显卡）
- Python 3.12 · Transformers 5.x · bitsandbytes 0.49+
- PyTorch 安装版本因卡而异：
  - RTX 40 系 → `torch 2.5.1+cu121`
  - RTX 50 系 → `torch 2.7+cu128`（需 CUDA 12.8+，尚未经过实机验证）
```bash
# RTX 40 系
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RTX 50 系
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---
## 部署环境
## 1 进入项目目录

```powershell
cd your-project-folder
```

---

## 2 创建虚拟环境

```powershell
py -m venv .venv
```

---

## 3 升级 pip

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

---

## 4 安装 PyTorch（CUDA 12.1，适配 4060 Ti）

```powershell
.\.venv\Scripts\pip.exe install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## 5 安装其余依赖库

```powershell
.\.venv\Scripts\pip.exe install "transformers>=5.0.0" accelerate bitsandbytes Pillow qwen-vl-utils
```

---

## 6 验证安装是否正确

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

正常输出应该是：
```
torch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

---

## 7 验证其他库

```powershell
.\.venv\Scripts\python.exe -c "import transformers, accelerate, bitsandbytes, PIL; print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('bitsandbytes:', bitsandbytes.__version__)"
```

## 8 下载模型

使用相对路径（以项目根目录为基准）：

```bash
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir models/qwen3.5-9b \
  --local-dir-use-symlinks False
```

Windows PowerShell 镜像版：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download Qwen/Qwen3.5-9B `
  --local-dir models/qwen3.5-9b `
  --local-dir-use-symlinks False
```

下载完成后的项目目录

```bash
    qwen35_console_chatbot_streaming.py
    qwen35_stress_test.py
	models
        qwen3.5-9b
            model.safetensors-00001-of-00004.safetensors
            model.safetensors-00002-of-00004.safetensors
            model.safetensors-00003-of-00004.safetensors
            model.safetensors-00004-of-00004.safetensors
            model.safetensors.index.json
            ...
```

## 9 试跑 chatbot

```powershell
.\.venv\Scripts\python.exe -u .\qwen35_console_chatbot_streaming.py --model .\models\qwen3.5-9b --load_in_4bit --max_new_tokens 2500
```

## 10 压力测试

```powershell
.\.venv\Scripts\python.exe -u .\qwen35_stress_test.py --model .\models\qwen3.5-9b --load_in_4bit --max_new_tokens 16000 --tasks .\press_test\tasks_landing.json
```
