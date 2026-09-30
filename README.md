# Early Experience：ALFWorld / ScienceWorld / WebShop 测评

本仓库在 `ssh lab` 的 `/data1/shn/agentwmv2` 中运行 Qwen2.5-7B-Instruct、Qwen3.8-27B 和各 benchmark 对应的 EarlyExperience IL checkpoint。这里只下载和评测，不从零训练。WebShop 使用下方 `webshop-index-v2` 修正流程；ScienceWorld 仍有待验证的上下文问题，历史 ALFWorld 混合引擎入口仅保留作参考。

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

## 环境安装（论文三环境，已在 `lab` 验证）

论文的环境实现来自 `langfengQ/verl-agent`（ALFWorld、WebShop）和 AgentGym（ScienceWorld）。在项目根目录执行：

```bash
git clone https://github.com/langfengQ/verl-agent.git external/verl-agent
git clone --recurse-submodules https://github.com/WooooDyy/AgentGym.git external/AgentGym

conda create -n ee-alfworld python=3.12 -y
conda create -n ee-scienceworld python=3.8 -y
conda create -n ee-webshop python=3.10 -y
```

所有运行命令均加 `PYTHONNOUSERSITE=1`，防止服务器既有用户级 Python 包污染新环境。

### ALFWorld

```bash
conda run -n ee-alfworld pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
conda run -n ee-alfworld pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld pyyaml
conda run -n ee-alfworld alfworld-download
```

`alfworld-download -f` 会无条件重拉约 2.4 GB 资源；已有完整 `~/.cache/alfworld` 时不要加 `-f`。验证：

```bash
cd ~/.cache/alfworld/alfworld
ALFWORLD_DATA=~/.cache/alfworld PYTHONNOUSERSITE=1 conda run -n ee-alfworld python -c \
'import yaml; from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv; c=yaml.safe_load(open("configs/base_config.yaml")); c["dataset"]["num_eval_games"]=1; e=AlfredTWEnv(c,"eval_in_distribution"); g=e.init_env(1); print(g.reset()[0][0][:80]); g.close()'
```

### ScienceWorld

需要 Java 8+。安装 AgentGym 的独立服务包，并增加论文仓库要求的 `/admissible_actions` 路由：该路由返回原生 `get_valid_action_object_combinations()`。

```bash
cd external/AgentGym/agentenv-sciworld
PYTHONNOUSERSITE=1 conda run -n ee-scienceworld pip install --ignore-installed scienceworld==1.2.3
PYTHONNOUSERSITE=1 conda run -n ee-scienceworld pip install -e .
PYTHONNOUSERSITE=1 conda run -n ee-scienceworld sciworld --host 0.0.0.0 --port 36001
```

本次在 `external/AgentGym` clean checkout 中已应用该端点补丁。验证方法：`POST /create`、`POST /reset`，再请求 `GET /admissible_actions?id=<id>`。

### WebShop

```bash
cd external/verl-agent/agent_system/environments/env_package/webshop/webshop
CONDA_ALWAYS_YES=true PYTHONNOUSERSITE=1 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  conda run -n ee-webshop bash ./setup.sh -d all
```

官方脚本会经 Google Drive 下载商品数据、下载 spaCy 模型并构建索引。若 lab 无法连 Google Drive，可从 Hugging Face 的 [`YWZBrandon/webshop-data`](https://huggingface.co/datasets/YWZBrandon/webshop-data/tree/main) 获取 `items_shuffle.json` 和 `items_ins_v2.json`；索引可复用本机已有兼容 WebShop cache，或重新构建。为兼容 `numpy==1.26.4`，本次验证固定了：`scipy==1.10.1`、`accelerate==0.34.2`，并从 HF 安装 `en_core_web_sm==3.7.1`。

```bash
PYTHONNOUSERSITE=1 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  conda run -n ee-webshop pip install --force-reinstall --no-deps scipy==1.10.1 accelerate==0.34.2
curl -fL -o /tmp/en_core_web_sm-3.7.1-py3-none-any.whl \
  https://hf-mirror.com/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
PYTHONNOUSERSITE=1 conda run -n ee-webshop pip install /tmp/en_core_web_sm-3.7.1-py3-none-any.whl
```

已验证 `WebAgentTextEnv(observation_mode="text", num_products=100)` 可 reset，并能执行 `search[laptop]`。

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

## WebShop 索引修复与三个模型重跑（webshop-index-v2）

2026-09-30 修正 `scripts/run_audited_paper_eval.py`：显式设置 `num_products=1000`，选择与 1000 件商品库匹配的 `search_engine/indexes_1k`，不再用百万商品索引。启动时逐条读取 1000 个索引文档，要求商品 ID 集合与商品库完全相同、无重复；数量或 ID 不一致直接报基础设施错误，不运行测评。商品/属性文件、索引文件 SHA256 和索引商品 ID SHA256 一并记录在清单中。

只改变搜索索引选择及校验，保持原先 seed 1000、500 个 goal 的顺序与内容、prompt、动作解析、temperature 0.4、4096 / 512 token 和 15 步上限不变。新版固定清单为 `data/paper_eval/audited/webshop_index_v2.json`，使用独立 protocol `webshop-index-v2`；旧版清单会被拒绝，不复用旧 WebShop 任务结果。该修复不代表已完全复现论文设置或保证达到论文分数。

验证包括：18 项离线回归；索引 1000 个 ID 全量一致；500 个 goal 与旧清单逐项哈希相等；20 个原先空结果的真实模型查询现在均返回商品；3 个测试任务完整购买后 `done=True, reward=1.0`。购买测试使用目标已知的测试夹具，只校验环境链路，**不是模型成绩，不进入测评结果**。三个模型的两任务、每任务两步冒烟均完成且无基础设施错误；EE-IL 和 Qwen3.8 已能搜索并点击真实商品。Qwen2.5 仍会输出不符合原生动作语法的内容，这是模型输出行为，不是本次索引故障；维持原有统一 prompt/解析，不做额外猜测修复。两步冒烟不评估任务成功率，正式重跑仍用完整 15 步。

```bash
ssh lab
cd /data1/shn/agentwmv2
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/test_paper_eval_protocol.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  /home/shn/miniconda3/envs/ee-webshop/bin/python scripts/validate_webshop_index.py

# 每模型 2 个任务、每任务 2 步：仅检查真实请求/搜索，不算正式成绩。
mkdir -p outputs/paper_eval/webshop_index_v2_smoke
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks webshop --limit 2 --smoke-steps 2 \
  --output /data1/shn/agentwmv2/outputs/paper_eval/webshop_index_v2_smoke \
  > outputs/paper_eval/webshop_index_v2_smoke/supervisor.log 2>&1 < /dev/null &

# 仅在冒烟三组均完成且 errors=0 后运行；不能与旧 WebShop runner 重复启动。
mkdir -p outputs/paper_eval/webshop_index_v2
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks webshop \
  --output /data1/shn/agentwmv2/outputs/paper_eval/webshop_index_v2 \
  --reference-output /data1/shn/agentwmv2/outputs/paper_eval/audited \
  > outputs/paper_eval/webshop_index_v2/supervisor.log 2>&1 < /dev/null &
cat outputs/paper_eval/webshop_index_v2/RESULTS.md
```

新输出目录 `outputs/paper_eval/webshop_index_v2/` 存三组独立新结果、`summary.json`、`audit.json` 和 `preflight.log`。其 `RESULTS.md` 将新 WebShop 结果与 `audited/summary.json` 中的 ALFWorld / ScienceWorld 状态合并展示；ScienceWorld 的已知上下文限制依然存在。旧 WebShop 数据保留在 `outputs/paper_eval/audited/webshop_*`，**均因错误索引无效，不再用于当前表**。

切换时只终止旧 supervisor 和已核实的旧 WebShop Qwen3.8 runner；未中断 ALFWorld / ScienceWorld 的在跑任务。新 supervisor 用 `--benchmarks alfworld scienceworld --adopt-pids outputs/paper_eval/audited/adopt_other_benchmarks.json` 核验原 runner 的 `--env/--output/--manifest/--model` 后接管；记录 `pre_webshop_index_v2_migration.json` 快照，继续更新 `audited/summary.json`。`--adopt-pids` 是迁移选项，不应盲目复用过期 PID 文件；日常恢复只需同配置重启，runner 会跳过已完成任务。

## 原九组后台测评（audited-v1，WebShop 已被新版取代）

### 2026-09-30 诊断：当前结果不能作为论文复现成绩

以下为修正前的诊断记录；旧实验现场已保留。WebShop 已按上方 `webshop-index-v2` 修正，ScienceWorld 尚未修改。下方九组启动命令仅作历史实现记录，不应原样用于当前重跑。

- **WebShop 已确认商品库与搜索索引不匹配。** 当前 runner 加载 `items_shuffle_1000.json` 的 1000 件商品，但 `num_products=None` 会选择 `search_engine/indexes`，本机该索引有 1,181,370 个文档。`get_top_n_product_from_keywords` 将不在当前商品库中的搜索命中过滤掉。EE-IL 的 500 个任务中，379 个全程没有搜到商品；用 `webshop_000` 的真实查询词读取索引，当前索引的前 50 个结果过滤后为 0，匹配的 `indexes_1k` 前 50 个结果全部在商品库中。三个模型的 WebShop 结果均受该配置影响，待统一商品库/索引后另建输出目录重跑；不能直接覆盖旧结果。
- **ScienceWorld 的历史上下文与 one-shot 尚未严格对齐。** 当前代码固定最多保留最近两轮，即使 token 预算充足也丢弃更早历史；已读取的作者公开训练样本包含 54 条连续 messages，而本次示范只截取其首个动作轮次，不能认定等同作者的 inference one-shot。EE-IL 中 145/200 个任务达到 100 步上限，轨迹可见重复拿起/聚焦/放下物体。历史截断是强疑点，尚需对照实验验证，不宣称它是全部差距的唯一原因。
- **区分论文和公开仓库的成绩，以及成功率和原始平均分。** 论文 v3 表 1 的 Qwen2.5-7B IL 成功率为 ScienceWorld 53.9%、WebShop 51.6%；当前 EE-IL 为 3.5% 和 16.6%。论文 WebShop 表 5 另列平均 score 62.1。当前运行无基础设施 errors 不代表评测协议或环境配置正确；通过现有 11 项离线回归测试也不能排除上述索引与 prompt 问题。

参考论文 [Agent Learning via Early Experience v3](https://arxiv.org/html/2510.08558v3) 附录 C.1/C.2/C.6、作者 [ScienceWorld 说明](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/envs/scienceworld/README.md)、[AgentGym](https://github.com/WooooDyy/AgentGym) 和本地引用的 Verl-Agent 源码。

本次是**按公开论文设置对齐的完整留出集测评，不是严格复现论文表格**。作者未公开全部 inference prompt、验证采样 task IDs、ScienceWorld horizon 和原始框架 revision；Qwen3.8-27B 也是用户新增 baseline，不是论文评测模型。不得将本次完整集合的成绩直接说成复现出论文数值，或把训练参数误当评测参数。

| 环境 | 固定集合 | prompt / response token 上限 | temperature | step 上限 | 指标 |
|---|---|---|---|---|---|
| ALFWorld | TextWorld `eval_in_distribution` / `valid_seen`，140 个可解任务，各一次 | 4096 / 1024 | 0.4 | 50，引用 Verl 配置 | `won` 成功率 |
| ScienceWorld | AgentGym 官方 AgentEval `sciworld_test.json`，200 个唯一 task IDs | 4096 / 1024 | 0.4 | 100，SDK 默认值；**论文未给出** | SDK 原始平均 score，另列 `done && score == 100` 完全成功率 |
| WebShop | 引用的 Verl wrapper 测试池 `range(500)`，seed 1000，固定 500 个 goal | 4096 / 512 | 0.4 | 15 | `done && reward == 1.0` 成功率，另存平均 reward |

WebShop 的 512 response / 15 steps 来自引用的 Verl example，不是论文 WebShop 段落明确给出的数字。产品文件显式使用 wrapper 默认 `items_shuffle_1000.json` / `items_ins_v2_1000.json`，synthetic goals、`human_goals=False`，与引用版本的小目录配置一致。`num_products=None` 不等于全量百万商品：默认文件本身只有 1000 件。不使用其他项目的 WebShop task JSON，也不把 AgentGym 的另一套 WebShop 划分混入 Verl 协议。

ScienceWorld 200 条来自 [AgentGym/AgentEval](https://huggingface.co/datasets/AgentGym/AgentEval)，revision `1f090d52ef01a83a889da02699e34a500e35a274`；本地保存 `data/paper_eval/sciworld_test.json`，清单固定 SDK 中的 taskName / variationIdx，reset 时逐项校验。不是 ScienceWorld SDK 自己枚举的 1171 个 test variations，也不是 200 个 task 类型。

ScienceWorld 进度分数通常为 0–100，但 SDK 在某些失败终止时返回 **-100**，本机正式运行已观察到。原始平均 score 按 AgentGym `BaseAgentEnvController.evaluate` 的 `rewards.mean()` 口径直接平均，**不裁剪负分或用部分得分冒充成功**；所以实时表的 score 可能为负。完整成功率是单独的百分比，两者不是同一个指标；不能将这个均值直接解释为论文成功率。

论文说 ScienceWorld 使用 one-shot，但未给出完整可直接运行的 inference conversation。本次固定采用作者已发布 `expert_sft.jsonl` 的**训练集第一条对话的首个示范轮次**，使用其原始 action-template introduction、assistant acknowledgement、示范 user/assistant，输出 `Thought:` / `Action:`。来源片段保存在 `data/paper_eval/sources/science_expert_sample.jsonl`。它是公开训练示范，不是从当前 test task 挑答案；但不能声称与作者实际 inference one-shot 逐字相同。保留任务目标，使用最近两轮历史；保留 connect/disconnect 模板和全部环境动作，不再删除电路动作类别。完整 admissible list 存在轨迹中，不作为超长扁平列表塞入 prompt。

ALFWorld / WebShop 从本地 Verl prompts 和 projection 文件运行加载，复用首步 no-history / 后续 history 模板、最近两步 pre-action observation 和记忆编号、原生动作提取；不再对 `[action]` 输出做额外猜测修复。WebShop 复用 observation 格式化和 >13000 字符时的 no-history 回退。所有请求由实际 checkpoint tokenizer 的 chat template 精确计数，整个 prompt 不得超过 4096 tokens；超长时逐步减少历史，不删除任务或 admissible actions，仍超限则记 infrastructure error，不假装任务失败。

本次通用固定设置：`temperature=0.4`、`top_p=1.0`、禁用 top-k（vLLM `-1` / Transformers `0`）、`repetition_penalty=1.0`、generation seed 0、environment seed 1000。27B Transformers API 不接受顶层 `top_k` / `repetition_penalty`，需在支持的 `generation_config` **JSON 字符串**中覆盖这两个值及 `do_sample=True`，保留 checkpoint 的 EOS / PAD 配置；该写法已真实请求验证，不能把 checkpoint 默认 top-k 20 悄悄混入正式设置。Qwen3.8 请求额外带 `chat_template_kwargs={"enable_thinking": false}`，防止输出只有隐藏推理而无可执行 action；这是新增 baseline 的明确选择。7B 使用 vLLM，27B 使用 Transformers 多卡 device-map 后端，因此推理后端不完全等同作者实现。

### GPU / Python 环境

| GPU | 服务 | 端口 |
|---|---|---|
| 0 | Qwen2.5-7B-Instruct，三个任务共享 | 8100 |
| 1 | EE ALFWorld IL | 8101 |
| 2 | EE ScienceWorld IL | 8103 |
| 3 | EE WebShop IL | 8104 |
| 4–7 | Qwen3.8-27B，三个任务共享 | 8102 |

环境 CPU runner 分别用 `ee-alfworld` / `ee-scienceworld` / `ee-webshop`，模型服务用 `verl-agent` / `ee-qwen38-direct`，均禁用用户 site-packages。7B 独立 inference 单卡可用，旧文档的“两卡”要求只针对同时加载 FSDP actor 和 rollout 的混合引擎，不是 ALFWorld 任务本身需要两卡。

27B 使用 torch 2.8.0+cu128 / Transformers 5.17.0。服务器有 `/usr/local/cuda-12.8/bin/nvcc`；加速依赖安装后需要重新加载模型：

```bash
ssh lab
cd /data1/shn/agentwmv2
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:/home/shn/miniconda3/envs/ee-qwen38-direct/bin:/usr/bin:/bin \
MAX_JOBS=4 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
/home/shn/miniconda3/envs/ee-qwen38-direct/bin/pip install \
  --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  causal-conv1d==1.7.0 flash-linear-attention==0.5.2
```

当前安装已通过，保留日志 `outputs/paper_eval/logs/qwen38_kernels_install.log`。ScienceWorld HTTP 环境服务需监听 `127.0.0.1:36001`；接口已按安装步骤提供 `/admissible_actions`。未运行时，用其专属环境启动：

```bash
nohup env PYTHONNOUSERSITE=1 \
  /home/shn/miniconda3/envs/ee-scienceworld/bin/sciworld \
  --host 127.0.0.1 --port 36001 \
  > outputs/paper_eval/logs/scienceworld_service.log 2>&1 < /dev/null &
```

### 启动、恢复与查看结果

服务脚本只在对应端口空闲时启动服务，不 kill 其他进程；已有 7B 服务必须通过模型名和权重路径检查。27B 已有服务需核对其启动日志路径。后台 supervisor 就绪检查通过后生成固定任务清单，同时运行九组 jobs；27B 三个客户端使用共享锁逐请求轮流使用模型，以避免其非批处理后端并发冲突。

```bash
ssh lab
cd /data1/shn/agentwmv2
bash scripts/start_paper_eval_services.sh
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/test_paper_eval_protocol.py

# 九组各执行一个真实任务的一步，仅验证接口，不用于正式成绩。
mkdir -p outputs/paper_eval/audited_smoke
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  --output /data1/shn/agentwmv2/outputs/paper_eval/audited_smoke \
  --limit 1 --smoke-steps 1 \
  > outputs/paper_eval/audited_smoke/supervisor.log 2>&1 < /dev/null &

# 冒烟九组均 completed、errors=0 后，启动完整正式集合。
mkdir -p outputs/paper_eval/audited
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  > outputs/paper_eval/audited/supervisor.log 2>&1 < /dev/null &

cat outputs/paper_eval/audited/RESULTS.md
tail -40 outputs/paper_eval/audited/supervisor.log
```

同一输出目录由锁防止重复启动；完整 config 一致时跳过已完成任务，未完成/异常任务可恢复。runner 对基础设施异常重试最多三次，supervisor 对崩溃 runner 额外重启最多三次，不无限等待日志行数。表每 15 秒更新，明确 `completed/total`、errors 和运行状态。基础设施 errors 不充当成功或任务失败，不完整集合只能是部分结果；只有分母完成且没有 errors 才能报告正式完整成绩。

输出：

- `outputs/paper_eval/audited/RESULTS.md`：三个模型 × 三个环境实时表。
- `summary.json`：机器可读进度、平均 score、成功率、errors、PID。
- `audit.json`：论文版本、配置、九组命令、源码/data SHA256、源仓库 commit、无法逐项复现的限制。
- `{benchmark}_{model}/config.json` / `tasks/*.json` / `steps.jsonl`：固定配置、每任务结果、逐步实际 prompt/output/action/raw score。
- `data/paper_eval/audited/{alfworld,scienceworld,webshop}.json`：固定任务与内容哈希。
- `data/paper_eval/sources/2510.08558v3.pdf` / `paper.txt`：核对使用的论文快照。

旧 `outputs/paper_eval/formal` 中的运行使用不同提示词、动作修复、ScienceWorld 成功阈值和 WebShop 奖励阈值，**不能作为本次正式成绩**。保留旧文件供审计，但新汇总完全排除它们。旧 `run_paper_agent_eval.py` / `summarize_paper_eval.py` 不是当前推荐入口；checkpoint 轮换脚本已替换成新 supervisor 入口。

正式启动时的 27B 顶层参数兼容错误与临时默认采样请求保存在 `outputs/paper_eval/audited_pre_sampling_fix/`，不混入正式三个 27B 输出。其他六组配置未改变，继续运行。

## 历史 ALFWorld 混合引擎评测（仅参考）

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
