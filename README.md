# Early Experience ALFWorld IL Checkpoint Reproduction

本文档记录如何直接使用官方发布的 `osunlp/early-experience-alfworld/il` checkpoint，在 ALFWorld 上复现实验结果。这里不从零训练 IL 模型，只做 checkpoint 下载和评测。

## 复现口径

- 论文/项目：Agent Learning via Early Experience, arXiv:2510.08558
- 评测环境：ALFWorld TextWorld
- 运行框架：`langfengQ/verl-agent`
- 模型：`Qwen/Qwen2.5-7B-Instruct` 的官方 IL checkpoint
- checkpoint：`osunlp/early-experience-alfworld` 仓库下的 `il/`
- 目标结果：官方开源 release README 中 ALFWorld / Qwen2.5 7B Instruct / IL 的 `58.6%`

注意：官方 release README 说明开源 release 使用统一训练设置，和论文 appendix 中 per-env 超参数设置略有差异。因此直接跑这个 checkpoint 时，应以 release 表格中的 `58.6%` 作为主要对齐目标，而不是论文 appendix 中从零训练设置得到的数值。

## 目录约定

本文档和脚本不绑定本机绝对路径。默认把外部仓库、checkpoint、临时数据和输出都放在当前项目目录下：

```text
.
├── external/verl-agent/
├── checkpoints/early-experience-alfworld/il/
├── checkpoints/early-experience-alfworld/iwm/
├── checkpoints/early-experience-alfworld/sr/
├── checkpoints/early-experience-scienceworld/il/
├── checkpoints/early-experience-webshop/il/
├── data/verl-agent/text/
├── outputs/alfworld_il_eval/
└── scripts/
```

可替换变量：

- `VERL_AGENT_DIR`：`verl-agent` 仓库位置，默认 `external/verl-agent`
- `CHECKPOINT_DIR`：checkpoint 下载位置，默认 `checkpoints/early-experience-alfworld`
- `DATA_DIR`：`verl-agent` 评测占位 parquet 数据位置，默认 `data/verl-agent`
- `OUTPUT_DIR`：评测输出和日志目录，默认 `outputs/alfworld_il_eval`
- `CONDA_ENV_NAME`：conda 环境名，默认 `early-exp-alfworld`
- `N_GPUS`：使用 GPU 数量，默认 `2`
- `TP_SIZE`：vLLM tensor parallel size，默认等于 `N_GPUS`
- `VAL_BATCH_SIZE`：ALFWorld 验证任务数，默认 `128`
- `ALFWORLD_EVAL_DATASET`：`verl-agent` 的 ALFWorld eval split，默认 `eval_in_distribution`
- `ADV_ESTIMATOR`：优势估计器，默认 `grpo`；纯评测不要使用会额外创建 critic 的默认 `gae`
- `MODEL_DTYPE`：actor/FSDP 加载 dtype，默认 `bf16`，避免评测时用 FP32 加载造成显存峰值
- `ACTOR_PARAM_OFFLOAD`：actor 参数 CPU offload，默认 `True`
- `ACTOR_MICRO_BATCH_SIZE_PER_GPU`、`ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`、`REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`、`CRITIC_MICRO_BATCH_SIZE_PER_GPU`：兼容当前 `verl-agent` 配置校验，默认均为 `1`

## 安装环境

先准备 conda，然后执行：

```bash
bash scripts/setup_verl_agent_alfworld.sh
```

如果要换目录或环境名：

```bash
VERL_AGENT_DIR=external/verl-agent \
CONDA_ENV_NAME=early-exp-alfworld \
bash scripts/setup_verl_agent_alfworld.sh
```

脚本会执行以下操作：

1. clone `https://github.com/langfengQ/verl-agent.git` 到 `VERL_AGENT_DIR`。
2. 创建并使用 Python 3.12 的 conda 环境。
3. 安装 `vllm==0.11.0`、`flash-attn==2.7.4.post1` 和 `verl-agent`。
4. 安装 ALFWorld 相关包：`gymnasium==0.29.1`、`stable-baselines3==2.6.0`、`alfworld`。
5. 执行 `alfworld-download -f` 下载 ALFWorld PDDL、game files 和 detector 资源。

## 下载 checkpoint

```bash
bash scripts/download_il_checkpoint.sh
```

若服务器不能直连 Hugging Face，可只为本次命令使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com CONDA_ENV_NAME=verl-agent \
  bash scripts/download_il_checkpoint.sh
```

下载 `iwm` 和 `sr` 时，建议同时放宽元数据/文件超时并降低并发数，避免镜像端的 HEAD 请求超时。命令支持断点续传：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ETAG_TIMEOUT=120 \
HF_HUB_DOWNLOAD_TIMEOUT=120 \
conda run -n verl-agent hf download osunlp/early-experience-alfworld \
  --include 'iwm/*' 'sr/*' \
  --local-dir checkpoints/early-experience-alfworld \
  --max-workers 2
```

自定义下载位置：

```bash
CHECKPOINT_DIR=checkpoints/early-experience-alfworld \
  bash scripts/download_il_checkpoint.sh
```

下载后三套模型路径应为：

```text
checkpoints/early-experience-alfworld/il/
checkpoints/early-experience-alfworld/iwm/
checkpoints/early-experience-alfworld/sr/
```

当前 `lab` 服务器状态（2026-09-27）：`il`、`iwm` 和 `sr` 已全部下载，每套约 15 GB。`iwm` 和 `sr` 均已通过离线 config/tokenizer 加载和 safetensors 头部校验，各包含 4 个权重分片、339 个张量，且没有残留 `.incomplete` 文件。

### ScienceWorld 和 WebShop IL checkpoint

下载这两个仓库的 `il/` 目录：

```bash
bash scripts/download_additional_il_checkpoints.sh
```

脚本默认使用 `https://hf-mirror.com`，通过 `curl` 逐文件下载，并对 `.part` 文件断点续传。这可规避 Hugging Face Python 客户端连接 Xet CDN 时偶发的 SSL EOF。可替换参数：

- `CHECKPOINT_ROOT`：下载根目录，默认 `checkpoints`
- `HF_ENDPOINT`：下载端点，默认 `https://hf-mirror.com`
- `MAX_FILE_ATTEMPTS`：每个文件最多尝试次数，默认 `50`
- `RETRY_DELAY_SECONDS`：外层重试间隔，默认 `10`

后台运行和查看日志：

```bash
setsid -f bash scripts/download_additional_il_checkpoints.sh \
  > outputs/download_additional_il_checkpoints.log 2>&1 < /dev/null
tail -f outputs/download_additional_il_checkpoints.log
```

下载路径：

```text
checkpoints/early-experience-scienceworld/il/
checkpoints/early-experience-webshop/il/
```

### Qwen 基座模型

下载 `Qwen/Qwen3.8-27B` 和 `Qwen/Qwen2.5-7B-Instruct`：

```bash
bash scripts/download_hf_models_curl.sh \
  Qwen/Qwen3.8-27B Qwen3.8-27B \
  Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct
```

该脚本先通过 Hugging Face API 获取每个仓库的完整文件清单，再使用 `curl` 逐文件下载。大文件写入 `.part` 后原子移动到正式路径，重跑同一命令即可续传。默认使用 `https://hf-mirror.com`，并对每个文件自动重试。

后台运行和查看日志：

```bash
setsid -f bash scripts/download_hf_models_curl.sh \
  Qwen/Qwen3.8-27B Qwen3.8-27B \
  Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct \
  > outputs/download_qwen_models.log 2>&1 < /dev/null
tail -f outputs/download_qwen_models.log
```

当前 `lab` 服务器状态：`Qwen3.8-27B` 已下载完整 18 个权重分片（约 52 GB），`Qwen2.5-7B-Instruct` 已下载完整 4 个权重分片（约 15 GB）。两者均已通过离线 config 解析和 safetensors 头部校验，且没有残留 `.part` 文件。

如果 Hugging Face 需要认证，先运行：

```bash
huggingface-cli login
```

## 运行评测

### 当前服务器上的环境与显存结论

已在 RTX 4090 24GB 上实测单卡冒烟：数据预处理、ALFWorld 初始化、配置校验和 BF16 FSDP actor 加载均可通过，但 `verl-agent` 的混合引擎还需要同时初始化一份未量化 vLLM 权重。单卡此时只剩约 8.25 GiB，低于 vLLM 约 14 GiB 的权重需求，因此减小 `VAL_BATCH_SIZE` 或 `GPU_MEMORY_UTILIZATION` 不能让单张 4090 完成评测。请使用两张空闲 4090，并设置 `N_GPUS=2`、`TP_SIZE=2`。

本机已有环境和源码时无需重装，使用：

```bash
CONDA_ENV_NAME=verl-agent
VERL_AGENT_DIR=/data1/shn/verl-agent
```

预处理脚本首次会读取 `hiyouga/geometry3k` 作为占位数据。不能直连 Hugging Face 时，首次使用 `HF_ENDPOINT=https://hf-mirror.com`；缓存完成后可设置 `HF_HUB_OFFLINE=1`。

确认 conda 环境已激活或让脚本自动通过 `conda run` 运行：

```bash
bash scripts/eval_alfworld_il_checkpoint.sh
```

常用可替换参数：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
HF_ENDPOINT=https://hf-mirror.com \
CONDA_ENV_NAME=verl-agent \
VERL_AGENT_DIR=/data1/shn/verl-agent \
MODEL_PATH=checkpoints/early-experience-alfworld/il \
DATA_DIR=data/verl-agent \
OUTPUT_DIR=outputs/alfworld_il_eval \
N_GPUS=2 \
TP_SIZE=2 \
VAL_BATCH_SIZE=128 \
ALFWORLD_EVAL_DATASET=eval_in_distribution \
bash scripts/eval_alfworld_il_checkpoint.sh
```

评测关键设置：

- `trainer.val_only=True`：只评测，不训练。
- `algorithm.adv_estimator=grpo`：避免纯评测创建不需要的 critic。
- `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`：避免 FP32 actor 加载峰值。
- `actor_rollout_ref.actor.fsdp_config.param_offload=True`：允许 actor 参数按框架流程卸载到 CPU。
- `actor_rollout_ref.model.path=$MODEL_PATH`：使用下载的 IL checkpoint。
- `data.max_prompt_length=4096`
- `data.max_response_length=1024`
- `actor_rollout_ref.rollout.val_kwargs.temperature=0.4`
- `env.env_name=alfworld/AlfredTWEnv`
- `env.max_steps=50`
- `trainer.logger=['console']`：默认只写控制台日志，不依赖 wandb。

日志会写到：

```text
outputs/alfworld_il_eval/eval.log
```

评测结束后，在日志中搜索 success rate、validation metrics 或 ALFWorld reward 相关字段，记录最终成功率。若要和官方 release 结果对齐，目标是接近 `58.6%`，但具体数值可能受 `verl-agent` 版本、ALFWorld 数据缓存、GPU/vLLM 版本和采样随机性影响。

## 常见问题

1. `flash-attn` 安装失败

   先确认 CUDA、PyTorch 和 GPU 驱动版本匹配。必要时按机器 CUDA 版本改用兼容的 `vllm`/`flash-attn` 组合，并在实验记录里写清楚版本。

2. `alfworld-download -f` 很慢或失败

   这是 ALFWorld 资源下载步骤。重跑安装脚本即可继续；如果机器不能联网，需要手动把 ALFWorld cache 准备好。

3. 显存不足

   先把 `N_GPUS` 和 `TP_SIZE` 设为机器实际 GPU 数。如果仍然 OOM，可降低 `VAL_BATCH_SIZE` 或 `actor_rollout_ref.rollout.gpu_memory_utilization`，但需要在报告中注明改动。

4. 结果不是 `58.6%`

   先确认使用的是 `checkpoints/early-experience-alfworld/il`，不是 `sr` 或 `iwm`；确认 `temperature=0.4`、`max_steps=50`、`ALFWORLD_EVAL_DATASET=eval_in_distribution`。如果仍有差异，保存 `eval.log`、`conda list`、`git -C external/verl-agent rev-parse HEAD` 作为可复查证据。

## 资料来源

- EarlyExperience README: https://github.com/OSU-NLP-Group/EarlyExperience
- EarlyExperience ALFWorld notes: https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/envs/alfworld/README.md
- IL checkpoint: https://huggingface.co/osunlp/early-experience-alfworld/tree/main/il
- IWM checkpoint: https://huggingface.co/osunlp/early-experience-alfworld/tree/main/iwm
- SR checkpoint: https://huggingface.co/osunlp/early-experience-alfworld/tree/main/sr
- ScienceWorld IL checkpoint: https://huggingface.co/osunlp/early-experience-scienceworld/tree/main/il
- WebShop IL checkpoint: https://huggingface.co/osunlp/early-experience-webshop/tree/main/il
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- `verl-agent` README: https://github.com/langfengQ/verl-agent
- `verl-agent` ALFWorld example: https://github.com/langfengQ/verl-agent/blob/master/examples/grpo_trainer/run_alfworld.sh
