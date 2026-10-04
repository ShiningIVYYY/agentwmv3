# Early Experience：ALFWorld / ScienceWorld / WebShop 测评

本仓库在 `ssh lab` 的 `/data1/shn/agentwmv2` 中运行 Qwen2.5-7B-Instruct、Qwen3.8-27B 和各 benchmark 对应的 EarlyExperience IL checkpoint。这里只下载和评测，不从零训练。WebShop 使用 `webshop-index-v2`；ScienceWorld 使用下方 `scienceworld-context-v2` 修正流程。历史 ALFWorld 混合引擎入口仅保留作参考。

## World model 与三轮动作反思（wm-reflection-v1）

支持 ALFWorld、ScienceWorld、WebShop，支持现有 Qwen2.5、Qwen3.8 和 EE-IL Agent。
每个真实环境步先生成候选动作，再预测其后果，让同一个 Agent 选择保留或替换。
最多反思 3 轮；保留时提前结束；第三轮替换后再预测最终动作一次，然后执行。
每步最多 4 次 Agent 请求（首次动作 + 3 次反思）和 4 次 WM 请求，只有最终动作计入原步数上限。

### 两种后端

- `oracle`：ALFWorld/WebShop 在独立环境中重放历史、试执行候选动作；ScienceWorld 采用下述实时 JVM 检查点。
  每次重建均校验初始及逐步 observation、reward/score、done；正式提交时再次逐字段核对预测。
  被拒绝的候选动作不会改变正式环境。重放或提交不一致会成为基础设施错误，不伪装成 100% 准确结果。
  该实现使用真实模拟器的反馈；重放任务出现真实随机转移或无法重建时明确报错；ScienceWorld 的实时检查点支持随机转移。
  ScienceWorld 的歧义菜单按完整标签稳定排序，并将 Agent 选择编号映射回每个实例的原始编号。
  两后端使用同一观察表示；`native_observation/native_action` 保留原始返回与实际执行命令。
  校验比较这个无损观察表示及原生分数/终止状态，不忽略真正的状态分歧；重复标签无法映射时明确报错。
  独立 JVM 的房间出口列表可能按不同顺序枚举同一组门。Oracle 校验仅对标准房间出口列表
  按完整门描述排序后比较，保留门名、开关状态和重复项，其他文本及 reward/score/done 仍严格核对。
  此规则只用于校验，Agent 和 checkpoint 接收的房间文本保持原始顺序。
  重放开销随已执行历史增长，长任务有额外 CPU/JVM 成本。
- `word2world`：使用 X1AOX1A 的三个官方 `WorldModel-{Alfworld,Sciworld,Webshop}-Qwen2.5-7B` checkpoint。
  在线采样固定 `temperature=0, top_p=1, max_tokens=512`，复用作者初始化环境描述和 action/observation 对话协议。
  达到官方输出上限时保留部分预测，prompt 明确标注预测不完整；日志及汇总记录截断次数。
  部署上下文上限为 32768 tokens，这是本地部署选择；输入和输出合计超限报错，不静默截掉 WM 历史。
  每次预测从相同的已执行历史分支，正式执行后只加入真实反馈（teacher forcing）。

反思 Agent 两种后端均只看到预测的下一观察，oracle 的额外 reward、score、隐藏环境信息不加入反思 prompt。
checkpoint 的私有初始化信息只进入 WM。ALFWorld 适配官方 `put`→`move` 动作前缀及 AVAILABLE ACTIONS 后缀。
ScienceWorld 的 checkpoint 和 baseline 正式执行沿用 HTTP 服务；Oracle 使用专属 SDK JVM。
Oracle 保存实际 JVM 堆、Python SDK、歧义映射、随机数和对象编号，试执行后完整回滚。
最终动作采用真实模拟器已产生的状态，只推进一次；提交时验证完整检查点。
这样避免跨 JVM 对象列表顺序和随机生物活动导致的重放不一致。
ALFWorld 的 WM runner 复用两份独占的原生引擎：正式环境与串行候选分支各一份。
关闭时 cleanup 并归还引擎，不提前 dlclose；库在进程生命周期内保持映射，避免原生回调指向已卸载代码。
最多保留两份库映射，在实验卷使用独立临时目录；每次复用仍重新载入同一任务并校验重放。
WebShop 的 oracle 隔离 session/browser 以及终止自动 reset。
这是一套接入现有评测环境的实验协议，不能直接称为 Word2World 论文完整复现。

### 准备 checkpoint 与源码

```bash
cd /data1/shn/agentwmv2
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/verl-agent/bin/python scripts/download_word2world.py
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/setup_word2world_source.py --source-repo /data1/shn/Word2World
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-scienceworld/bin/python scripts/build_science_snapshot.py
```

权重目录为 `checkpoints/word2world/{alfworld,scienceworld,webshop}`。
下载固定模型 revision、校验发布的 LFS SHA256，并保存 `download_manifest.json` 和 `provenance.json`；重跑可续传。
`--workers` 控制每个环境的并行文件数，默认 4；三个环境并行下载。
初始化源码固定 commit `e8dc240269fb222de3c242b09c70bfbf0b4ac1c9`，并保存文件哈希。
源码安装器不会执行作者会删除共享 ALFWorld cache 的下载脚本。
ScienceWorld 的作者 collector 还需要 `tqdm==4.67.1`（已安装在 lab 的 `ee-scienceworld` 环境）。

### 启动 WM 服务

先选择有足够空闲显存的 GPU；启动器不会停止已有服务。按环境顺序运行可减少同时加载的模型数。

```bash
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/verl-agent/bin/python \
  scripts/start_worldmodel_service.py --env alfworld --gpu 1
```

配置在 `configs/worldmodel/*.json`，默认端口 ALFWorld 8111、ScienceWorld 8112、WebShop 8113。
`--gpu` 可覆盖建议 GPU；vLLM 使用 TP=1、PP=1 和 prefix caching，显存预算 0.90 是本地部署选择。
日志和启动参数位于 `outputs/wm_reflection_setup/`。
使用 EE-IL Agent 时还需要对应 Agent 服务；根据空闲显卡按环境调度，避免与 WM 服务争用显存。

### Agent + WM + 反思

先保证所选 Agent、tokenizer（8105）和环境服务已经就绪。
ScienceWorld 使用现有 `127.0.0.1:36001` 服务。

```bash
# 三环境 oracle 冒烟，每环境一个任务、每任务两步。
bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks alfworld scienceworld webshop --models qwen25 \
  --wm-backend oracle --reflection-max-rounds 3 --limit 1 --smoke-steps 2 \
  --output /data1/shn/agentwmv2/outputs/wm_reflection/oracle_smoke

# ALFWorld checkpoint 冒烟，需要先启动 ALFWorld WM 服务。
bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks alfworld --models qwen25 \
  --wm-backend word2world --reflection-max-rounds 3 --limit 1 --smoke-steps 2 \
  --output /data1/shn/agentwmv2/outputs/wm_reflection/word2world_smoke
```

移除 `--limit` 和 `--smoke-steps` 可运行完整任务集，正式实验应使用独立目录。
`--models` 可选 `qwen25 qwen38 ee_il`；`--benchmarks` 可选三个环境。
`--wm-backend none` 或省略该参数沿用 baseline；已有 baseline 配置可以恢复。
反思模式默认输出目录为 `outputs/wm_reflection/<backend>`，不混入原有成绩。
需要不同配置时通过 `--wm-config-dir` 指定配置目录，直接 runner 则使用 `--wm-config`。
反思上限允许 1–3，默认 3，不能设为更大值。

反思输出必须包含 JSON `decision/action/reason`，prompt 见 `reflection/prompt.txt`。
无效模型输出消耗一轮，下一轮给出校验错误；达到上限时保留当前候选动作，不猜测动作、不额外增加轮数。
服务错误、输入上下文超限和 oracle 不一致记录为基础设施错误；实际模型失败仍按环境结果计分。
checkpoint 正常结束但输出为空时，明确标注下一观察未知，用当前状态和历史进行至多三轮反思；
不编造下一观察、不改采样配置，日志和汇总记录 wm_empty_predictions。

### 单独预测和轨迹重放

```bash
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/predict_next_observation.py --env alfworld --wm-backend oracle \
  --task-index 0 --action 'look' --compare
```

替换为 `--wm-backend word2world` 使用对应 checkpoint。
`--prefix-actions actions.json` 接受已执行动作的 JSON 列表，重建该历史之后再预测；支持 `--output result.json`。
`--compare` 只在隔离环境中执行动作，不改变正在运行的评测任务。

### 验证与结果

```bash
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python scripts/test_paper_eval_protocol.py
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python scripts/test_wm_reflection.py
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-scienceworld/bin/python \
  scripts/validate_wm_oracles.py --env scienceworld
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/validate_word2world.py --env alfworld --samples 2
```

oracle 验证需要使用对应 `ee-alfworld`、`ee-scienceworld`、`ee-webshop` 环境。
官方单步夹具验证使用作者公开数据和 `max_tokens=2000`，与在线 512 配置分开记录。
夹具、购买测试和短冒烟都不是正式 benchmark 成绩。

本次实现已验证：

- 原有 25 项协议检查、新增 25 项行为检查全部通过；新增检查在 Python 3.8/3.10/3.12 环境均通过。
- `oracle_smoke_final`、`word2world_smoke_final`：两后端各三个环境、每任务两步，共 12 个真实步，全部完成且基础设施错误为 0。
- oracle 的 6 个正式提交步均通过观察、分数和终止状态核对；ScienceWorld 另通过歧义菜单和编号选择的四步重放验证。
- checkpoint 冒烟最终预测观察完全匹配 5/6；WebShop 有 3 次部分预测，已明确标注，不能据此推断完整数据集准确率。
- 官方单步样本各 2 条：ALFWorld 2/2、ScienceWorld 2/2、WebShop 0/2 完全匹配，详情在 `outputs/wm_reflection_setup/*_official_fixtures.json`。

当前 lab 部署：GPU 0 为 Qwen2.5 Agent（8100），GPU 1/2/3 为三个 WM（8111/8112/8113）。
为此释放了已完成评测的 EE-IL 服务；GPU 4–7 的既有 Qwen3.8 任务持续运行。
ScienceWorld HTTP 服务的文件句柄软上限由 1024 调整为 16384，未重启服务；oracle 新分支使用独立 SDK，避免累积共享服务会话。

运行完整 Qwen2.5 + checkpoint 实验：

```bash
bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks alfworld scienceworld webshop --models qwen25 \
  --wm-backend word2world --reflection-max-rounds 3 \
  --output /data1/shn/agentwmv2/outputs/wm_reflection/word2world_qwen25
```

改成 `--wm-backend oracle` 并使用新的输出目录即可运行真实环境 oracle 对照。

### 多 Agent 的完整 WM 实验矩阵

```bash
bash scripts/start_wm_experiment_matrix.sh
```

统一入口为 `scripts/run_wm_experiment_matrix.py`，当前默认选择 Qwen2.5 和 EE-IL，输出 `outputs/wm_reflection/full_7b_v8/`。
v6 从 EE-IL 开始采用进程任务队列：Oracle 和 Word2World 每组各 4 个独立 worker，
共享 GPU 0 上的 Agent 和对应 checkpoint WM。Qwen2.5 保持串行；已完成结果带原始配置和源码哈希继承。
每个 worker 独立持有环境、ScienceWorld JVM、ALFWorld 原生引擎、临时目录和日志。
任务锁覆盖完整任务、重试和结果发布，完成后领取下一个任务；退出或崩溃后锁由内核释放，
已完成任务不会重复执行。worker 最多重启三次；每次非零退出记录在 `parallel_state.json` 的
`worker_failures` 和退出码中。只有所有结果齐全且所有 worker 最终正常退出，组实验才完成。
并发设置在 `experiment.json` 和每组 `config.json.execution` 中记录；采样参数和 seed 保持原样，
不同批处理条件下的模型输出不保证与串行逐字一致。可以通过 `--ee-workers 1` 使用新的串行实验目录。
`full_7b_v1` 至 `full_7b_v4` 保留为历史记录。v5 修复 PDDL 动态库提前卸载造成的
原生进程崩溃，复用两份独占的原生引擎；新增退出码和 runner_failures 监控。
`migration.json` 记录来源；仅继承已完成的任务，文件保持原样，原始配置和源码来源见
各组 `inherited_tasks.json`。十二组预检必须记录退出码 0；保留已确认正常退出的预检，缺少退出码的组合重新验证。
调度器等进程实际退出后才检查完成结果，覆盖结果写入与原生清理之间的竞态。
Agent 原始动作 prompt、采样、步数和 checkpoint 不变。
它从以下原始 `config.json/audit.json` 冻结所选 baseline 的参数和任务 ID：
ALFWorld 的 `outputs/paper_eval/audited/`、ScienceWorld 的 `scienceworld_context_v2/`、WebShop 的 `webshop_index_v2/`。
当前实验为 Qwen2.5、三个环境对应的 EE-IL checkpoint × 三环境 × 两后端，共 12 组、3360 个任务实例。
全量任务数/最大真实步数/Agent 输出预算保持 140/50/1024、200/100/1024、500/15/512；
保留原 prompt、checkpoint、采样、环境种子、Qwen3.8 thinking 设置和共享生成锁。只添加 WM 和最多三轮反思。

Qwen2.5 和 EE-IL 的 12 个组合先在 `preflight/` 验证，然后开始全量：Qwen2.5 每组一个任务、两步，
并发 EE-IL 每组四个任务、每任务两步，检查所有任务的预测/提交和所有 worker 的真实退出码。
GPU 0 按环境切换 EE-IL，GPU 1–3 保留三个 WM，前四张卡上的任务完成后恢复 Qwen2.5 服务。
当前不调度 Qwen3.8，也不等待它的 baseline。以后显式使用 `--models qwen25 qwen38 ee_il` 和新的输出目录可生成 18 组、5040 个任务实例；
只有选中 Qwen3.8 时才等待其三个 baseline 全部完成且其 runner 退出，再验证六个组合并启动全量。
执行期间不会停止 GPU 4–7 上的已有服务。GPU 0 切换前校验服务路径、模型名、端口、GPU 和活跃 runner，使用正常 SIGTERM。

每个 runner 最多启动三次；runner 内保留原有每任务三次尝试。已完成的任务恢复时跳过。
即使 runner 返回 0，缺失或错误任务也不会被判为完成；预检失败会阻止对应模型的全量运行。
每次启动和运行过程中核对冻结的源码、配置、任务文件及模型 provenance，配置改变时使用新实验目录。
再次运行启动脚本会复用活跃 manager；manager 异常退出后重跑相同命令会恢复其已记录的任务。

`experiment.json` 保存完整基线配置、所选模型矩阵和哈希；`runstate.json/events.jsonl/manager.log` 保存调度状态。
`summary.json` 显示每组进度、错误和已选中的模型依赖；`comparison.json` 保存与 baseline 相同任务的配对差值；
`RESULTS.md` 显示所选模型 × 三环境的 baseline/oracle/checkpoint 对照。
百分比是已完成任务的成功率，括号内为完成数量/总数；当前任务步数和状态见 progress.json。
errors 是任务异常数，runner_failures 是组 runner 非零退出次数，worker_failures 是其子 worker 非零退出次数
（包括 SIGSEGV）。并发组的 progress.json 列出每个 worker 的当前任务、步数和等待状态；
独立日志位于各组 workers/run_*/worker_*/ 下。
未完成的结果明确标为暂定，错误任务单独显示，不能把部分完成率当成最终 benchmark 分数。

```bash
# 只生成并校验矩阵，不启动模型或评测任务。
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-qwen38-direct/bin/python \
  scripts/run_wm_experiment_matrix.py --prepare-only

# 调度器行为检查。
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-qwen38-direct/bin/python \
  scripts/test_wm_experiment_matrix.py
```

每个任务保存完整 `steps` 和 `reflection_metrics`；逐步 `steps.jsonl` 包含候选、每轮预测/反思、最终动作和实际反馈。
`reflection_events.jsonl` 还保存中途失败前的请求/输出，方便诊断。
`summary.json` 汇总轮数、替换数、无效输出、预测匹配率、oracle 校验、调用数、token 和耗时；`RESULTS.md` 保留真实环境任务成绩。
`config.json` 固定完整功能设置、源码和模型 provenance；改动配置/源码后需要新目录，不能混用旧结果。

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

新输出目录 `outputs/paper_eval/webshop_index_v2/` 存三组独立新结果、`summary.json`、`audit.json` 和 `preflight.log`。该目录最初的 `RESULTS.md` 合并过旧 ScienceWorld 状态；SciWorld 迁移后，当前三环境完整表改看下方 `scienceworld_context_v2/RESULTS.md`。旧 WebShop 数据保留在 `outputs/paper_eval/audited/webshop_*`，**均因错误索引无效，不再用于当前表**。

切换时只终止旧 supervisor 和已核实的旧 WebShop Qwen3.8 runner；未中断 ALFWorld / ScienceWorld 的在跑任务。新 supervisor 用 `--benchmarks alfworld scienceworld --adopt-pids outputs/paper_eval/audited/adopt_other_benchmarks.json` 核验原 runner 的 `--env/--output/--manifest/--model` 后接管；记录 `pre_webshop_index_v2_migration.json` 快照，继续更新 `audited/summary.json`。`--adopt-pids` 是迁移选项，不应盲目复用过期 PID 文件；日常恢复只需同配置重启，runner 会跳过已完成任务。

## ScienceWorld 输入修正与三个模型重跑（scienceworld-context-v2）

修正旧评测器中已确认的三处输入差异：固定最近两轮历史、每一步重复拼接任务目标、将训练示范截成单个动作。新版统一用于三个模型：

- 使用同一条作者公开 `scienceworld/expert_sft.jsonl` 训练对话的全部 54 条 messages，包含 introduction、acknowledgement 和全部示范轮次；不从测试任务选示范，不使用 SDK gold path 或测试答案。该公开对话不被认定为终止成功的轨迹，也不是作者未公开的精确 inference one-shot。
- 采用 AgentGym 式交替 user/assistant 对话，任务目标只在当前任务第一条 user 中出现。完整历史能放入 4096 tokens 时全部保留；超预算时仅丢弃最早的完整 observation/assistant 轮次，并把目标保留在剩余对话第一条 user 中。示范、当前 observation、目标不做截字。仍超预算则报 infrastructure error。
- 不改 ScienceWorld 1.2.3、环境动作/物理/奖励、200 个任务及顺序、100 步上限、temperature 0.4、response 1024、采样参数及三个 checkpoint。100 步是 SDK 默认选择，非论文明确数字。不做动作猜测修复或根据成绩挑 prompt。

启动前运行 `validate_scienceworld_protocol.py`：从 HF 镜像读取官方 AgentEval revision `1f090d52ef01a83a889da02699e34a500e35a274`，逐项核对 200 个唯一 ID 及顺序；重新枚举 stock AgentGym 的排除集合/SDK flattened map，与旧清单逐项比较 taskName / variationIdx；用独立 HTTP 环境 ID reset 全部 200 个任务，对比运行服务与 SDK 的任务/变体/目标，以及旧 EE-IL 的目标文本。每个目标 SHA256 固定到新清单，runner 每次 reset 校验。也核对示范与发布训练文件第一条完全相同，记录训练数据 revision、SDK/JAR 哈希和 Java 版本。

本次核验已通过：200 个官方 ID/顺序、200 个实际 HTTP task/variation/goal、200 条旧 EE-IL 目标全部一致。178 个任务初始分为 0，22 个任务有原生非零部分进度；校验 SDK 初始分，不强制为零，不改奖励。模型 runner 启动时另校验 SDK 版本、JAR、官方任务文件和示范文件的 SHA256。当前 SDK 1.2.3、系统 Java 1.8.0_452；作者生成 65.4% 时的精确 SDK/JVM 未公开。25 项离线回归通过；三个真实 tokenizer 的长历史样例也通过：10 轮可全部保留，超预算样例保留 14–18 轮，均不超过 4096 tokens，目标只出现一次。验证报告为 `preflight.json` 和 `runtime_prompt_checks.json`。此前六个失败任务在独立 SDK 中使用环境自带标准解均能达 100 分，仅作为环境诊断，不是模型成绩，标准解不进入 prompt。

三个模型冒烟均已完成（每模型 2 个任务、每任务 4 步），共 24 次实际模型调用：`errors=0`，Thought/Action 格式均有效，服务 usage 与实际 tokenizer 计数一致；示范完整、当前任务目标只出现一次，四步中的第三轮历史正常保留。冒烟只检查链路，不用它估算成功率；正式重跑每模型 200 个任务、每任务最多 100 步，不复用冒烟结果。启动后的进度以新目录 `summary.json` 为准。

```bash
ssh lab
cd /data1/shn/agentwmv2
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-alfworld/bin/python \
  scripts/test_paper_eval_protocol.py
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-scienceworld/bin/python \
  scripts/validate_scienceworld_protocol.py

# 三模型各 2 个任务、每任务 4 步，仅检查接口和对话，不计正式成绩。
mkdir -p outputs/paper_eval/scienceworld_context_v2_smoke
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks scienceworld --limit 2 --smoke-steps 4 \
  --output /data1/shn/agentwmv2/outputs/paper_eval/scienceworld_context_v2_smoke \
  > outputs/paper_eval/scienceworld_context_v2_smoke/supervisor.log 2>&1 < /dev/null &

# 先确认旧 SciWorld runner/supervisor 已迁移，冒烟三组 completed 且 errors=0。
mkdir -p outputs/paper_eval/scienceworld_context_v2
nohup bash scripts/run_overnight_paper_benchmarks.sh \
  --benchmarks scienceworld \
  --output /data1/shn/agentwmv2/outputs/paper_eval/scienceworld_context_v2 \
  --reference-output /data1/shn/agentwmv2/outputs/paper_eval/audited \
  --reference-output /data1/shn/agentwmv2/outputs/paper_eval/webshop_index_v2 \
  > outputs/paper_eval/scienceworld_context_v2/supervisor.log 2>&1 < /dev/null &
cat outputs/paper_eval/scienceworld_context_v2/RESULTS.md
```

新清单 `data/paper_eval/audited/scienceworld_context_v2.json`，新结果 `outputs/paper_eval/scienceworld_context_v2/`，旧结果 `outputs/paper_eval/audited/scienceworld_*` 保留但不续跑、不混入新成绩。新目录 `RESULTS.md` 合并新版 SciWorld、新版 WebShop、既有 ALFWorld 状态；`summary.json` 只含新版 SciWorld 三组。协议不同会拒绝旧清单，恢复还校验示范哈希和完整 config。此次迁移只停止已核实的旧 SciWorld runner 与监管它的 supervisor；ALFWorld runner 通过 `--benchmarks alfworld --adopt-pids` 接管继续跑，WebShop 不受影响。迁移前 audit/summary/PID 信息另存旧目录 migration 子目录。不要复用过期 PID 文件。

任务集核对能证明与上述公开 AgentEval 的一致性，不能证明等同作者未公开的全部原始评测设置。官方 release 的 SciWorld IL 65.4% 是对照目标，不保证此次修正恢复该成绩。原始终态平均 score 和 `done && score==100` 成功率仍分别报告，不把部分得分、负分裁剪或最大中间分伪装成成功率。

## 原九组后台测评（audited-v1，SciWorld / WebShop 已被新版取代）

### 2026-09-30 诊断：当前结果不能作为论文复现成绩

以下为修正前的历史诊断记录；旧实验现场已保留。WebShop 已按上方 `webshop-index-v2` 修正，ScienceWorld 已按 `scienceworld-context-v2` 修正。下方旧 prompt 描述及九组启动命令仅作历史记录，不应原样用于当前重跑。

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

EE-IL 反思使用训练时的原生动作格式：ALFWorld/WebShop 的 `<think>` 与 `<action>`，ScienceWorld 的 `Thought:` 与 `Action:`。相同动作表示保留，不同动作经当前状态动作校验后表示替换。Qwen2.5 保持原 JSON 反思。初始 Agent 提示、采样、步数和 checkpoint 沿用基线；最多反思三轮。预检必须产生有效反思决定，不能仅凭环境执行和退出码通过。v7 保留已完成 Qwen2.5 结果，EE-IL 全部重新运行；v6 EE-IL 结果仅用于故障诊断。

发布的 WebShop IL 模型也会输出仅有 `<action>` 的完整原生动作块；此输出同样参与保留/替换判断，缺失的解释原样记为空，不补造反思文本。多个动作、截断动作、JSON/XML 混合和额外尾部文本均拒绝。

WebShop 反思先给出候选动作与 WM 预测，再以真实当前状态、目标、已执行历史和当前可选动作结束提示，避免该 IL checkpoint 机械复述末尾的候选动作。另有真实模型与真实 Oracle 的偏离目标搜索检查，验证它确实能返回替换动作。


EE-IL v8 uses `ee-conservative-v2` and `evidence-gated-repair-v1`. Reviews retain the original actor action unless current action availability, complete explicit failure feedback, or three consecutive executed repetitions with identical feedback establish a defect. Waiting and information actions are exempt from the repetition trigger. Proposals are predicted from the same committed real prefix and must yield complete nonfailure feedback different from the defective original before execution. One verified repair ends review; repeated rejected proposals stop without oscillating; at most three actual reviews and four actual WM calls occur. Fallback reuses the verified original prediction (logged as cached, excluded from WM call/token/time totals).

All environments end the reviewer turn with the actual current state and executed history; hypothetical previews precede that section. ScienceWorld retains the exact full 54-message baseline TRAIN dialogue, trimming only real history. If necessary only the displayed hypothetical preview is clipped and marked partial; if mandatory context still exceeds 4096 tokens the original action is retained with an explicit budget-skip log, without fabricating a reviewer call. Original actor output is byte-preserved on keep; accepted repairs use the actual native model completion rather than synthetic reviewer thoughts. SDK-native action aliases such as `drop` and combinations in the current SDK list are accepted.

Qwen2.5 completed results retain their audited legacy JSON review policy and are inherited as such; the changed EE-IL treatment is explicitly versioned and restarted from zero. Old v7 EE-IL results are archived for diagnosis.
# EE-IL rerun with copied AgentWM prompts

The new `agentwm-copied-v1` suite copies all templates into this repository.
ALFWorld uses AgentWM's JSON review, WebShop uses its conservative review, and
ScienceWorld uses critique followed by a separate guarded replan. See
[the v9 protocol and launch command](docs/ee_agentwm_prompts_v9.md).
