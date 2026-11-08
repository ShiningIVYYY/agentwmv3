# 真实环境 WM 下一状态预测修复报告

**已解决已定位的跨 JVM 状态漂移根因；新的官方1.3.0配置在本次真实受控、长历史和端到端验证中稳定返回真实下一状态。没有将有限验证宣称为完整隐藏状态等价。**

## A. 完成工作与根因

旧 ScienceWorld1.2.3 使用不稳定的对象集合遍历顺序，热传导按顺序原地更新数值；因此相同 Seed 和历史仍可能产生真实温度差。前次 object-tree mismatch 拒绝是正确的，不能忽略树或增加容差来“修好”。

本次采用官方1.3.0确定性修复及门动作引用修复，固定官方 Commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52` 和 JAR SHA256 `e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465`。没有自行重写物理更新、序列化JVM、修改已安装SDK、在正式环境试执行或回滚。原SDK1.2.3 Python客户端使用其公开 serverPath 参数调用固定官方1.3.0原生引擎，二者版本明确区分。

Actor/Reviewer仍共享冻结Qwen3.5-9B，Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；BF16/非思考/原始生成logits NLL路径不变。没有Qwen2.5在线调用、训练、经验库扩充、Gate或Reviewer规则调整、最终测试集执行、Phase6工作。

## B. 文件与接口

- `worldmodel/ground_truth_environment.py`：显式1.2.3/1.3.0引擎profile、固定官方JAR校验、逐前缀重放验证和候选下一状态指纹。旧默认库接口profile保留1.2.3，新的开发运行配置明确选择1.3.0。
- `scripts/run_phase57_replay.py`：开发CLI默认v2配置；引擎身份进入manifest；跨引擎校准复用被拒绝；KEEP后同时核对公共反馈与完整受检状态投影。
- `configs/evaluation/phase57_replay_dev_v2.json`：独立新引擎配置，旧v1与旧结果不覆盖。
- `scripts/prepare_scienceworld_replay.py`：固定官方Commit/校验和的约7.8MB原生JAR准备，实际下载和verify-only均通过。
- `scripts/validate_reset_replay.py`：增加显式引擎选择；`scripts/validate_replay_histories.py`：真实记录历史、重复worker及失败拒绝测试。
- `scripts/test_replay_profiles.py`：7项新测试；`scripts/audit_phase57_replay.py`增加版本、指纹、精确重放成本检查；`scripts/export_replay_fix_evidence.py`导出真实紧凑证据。
- README、Implementation Plan、Progress、模型/环境文档与旧报告入口更新；详细协议见 [REAL_ENV_WM_REPLAY_FIX.md](REAL_ENV_WM_REPLAY_FIX.md)。

`simulate_one_step(formal,draft_action)`仍仅模拟一次Draft。独立B从相同初始化条件重放全部已提交历史；每个前缀及当前状态检查通过后，才返回真实Next Observation/reward/done/score。私有树/RNG/指纹不进入Actor/Gate/Reviewer。B清理后必须证明A受检投影未变；A仅正常执行最终动作。失败时明确 unavailable、不给Reviewer反馈、不偷偷尝试其他WM。新profile没有放宽状态检查。

## C. 真实验证与自动化测试

| 验证组 | 结果 |
| --- | --- |
| 四开发实例、多variation、seed0/1/7、历史0/2/4/8 | **48/48** 重建及公共下一状态匹配；旧1.2.3同组为27/48通过、21拒绝 |
| 三个真实开发历史、seed0/1/7、历史0/1/4/8/12/16/19 | **63/63** 公共反馈与完整受检后状态指纹匹配 |
| 重复独立worker | **9/9** 后状态指纹一致 |
| 故意损坏历史证据的负例 | **9/9** 拒绝；没有候选转移或预测泄漏 |
| 新版本、旧Internal、Gate、真实tokenizer合约 | **98/98** 测试通过 |
| 独立推理环境 | pip check通过；未安装新依赖/改旧环境 |

两组环境验证均没有正式环境污染或错误接受。长历史输入来自先前真实Full开发轨迹，在1.3.0重新按固定命令执行；不是新Actor结果，也不声称新旧物理状态一致。63个检查点并非63个独立任务，不夸大样本独立性。

## D. 端到端结果与校准

新原生引擎、同一9B模型/Prompt/采样/经验快照，三个合法开发任务128/2417/3070、seed0，四策略共12个完整Episode或既定上限终止。H20、history2、prompt4096、response512、T=.4/p=1/k=0、native tick cap21，与原受限profile一致。不是最终200任务评测。

| 策略 | 三任务Task Score | 均值 | 成功率 | 正式动作 | 成功模拟/尝试 | LLM Tokens | Episode秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| never | -100,-100,17 | -61.00 | 0.0% | 44 | 0/0 | 20,775 | 94.40 |
| always | 6,60,8 | 24.67 | 0.0% | 60 | 60/60 | 71,492 | 737.70 |
| internal | -100,-100,17 | -61.00 | 0.0% | 44 | 7/7 | 25,053 | 131.32 |
| full | 40,100,25 | 55.00 | 33.3% | 51 | 13/13 | 32,774 | 195.31 |

**80/80 次真实WM尝试成功，重建失败0次，30次KEEP的正式转移与副本完整受检指纹一致。** 轨迹审计检查了版本、未来信息隔离、Prompt、预算、实际请求和成本，结果通过。实际副本中负分预测0次、done预测1次，如实保留，不用伪造结果补足分支。

重新采集6个开发任务的40个真实决策，无旧校准数据复用，缺失0。预设q=.75得到Local **0.574907227281014**、Global **0.333263408605042**，样本内无预算调用率25%；Pearson 0.875301、Spearman 0.918199。数值与旧9B试用阈值碰巧一致，因为这批实际可见决策输入/生成分数保持一致；这是重新运行的结果，不是跳过校准或复用4B阈值。预算仍独立固定Never0/Always20/Internal5/Full5，没有看成绩选参数。40步仍只是有限校准。

Reviewer实际改动作50次，公开反馈分类为 `{"accepted_feedback": 16, "unknown": 5, "explicit_rejection": 29}`；即时分数差为 `{"positive": 1, "zero": 48, "negative": 1}`。真实反馈正确，不代表修订一定有效或长期有益。格式错误10次；不存在严格后续Call/Skip反事实收益证明。具体真实案例见 [trace_examples.json](phase57_replay_fix/trace_examples.json)。

## E. 成本与计量范围

- 评测共279个模型请求、**150,094 Tokens**；计入Actor和Reviewer。整轮含新校准、parity/repeat/普通验证共322次生成、167,832 Tokens，角色明细见cost_scope.json。
- 额外真实WM：初始化查询80、历史重放693、候选一步80，合计 **853** 次额外转移查询；另有状态审计调用，计入模拟时延。
- 正式评测动作199、正式Episode初始化12次；校准40个动作与6次初始化另计。它们与副本查询、native ticks不是同一单位。
- 在线模拟总耗时 **208.81秒**；包括初始化、重放、验证、候选和清理。Episode秒排除正式JVM创建/销毁；内核启动tick到summary估计整轮 **1305.53秒**，包含模型加载、开发校准和正式JVM创建，排除summary后退出清理，属于有时钟tick分辨率的估计，不是假装精确整阶段墙钟。
- 单张RTX4090/GPU4；逐生成Torch峰值分配约17.87GiB，不等于连续采样的driver全峰值。Actor/Reviewer同一模型；无需额外WM模型驻留。在线NLL额外forward0，另有1次独立数值诊断forward。
- 新离线经验合成0；原经验构建成本保持历史记录。CPU基础设施验证成本单独见replay_validation.json，不能混进新评测收益。
- 无可靠货币/FLOP换算，不因WM LLM Tokens=0而声称总成本=0。模拟更可用会增加实际Reviewer调用，这一成本如实记录。

## F. 可复现命令与证据

```bash
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/prepare_scienceworld_replay.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_profiles.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_integration.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_internal_signals.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_gates.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_qwen35_tokenizer.py --model-dir checkpoints/qwen35_9b

# 已执行的真实运行；重跑时用新的output目录
CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-qwen35-clean/bin/python \
 scripts/run_phase57_replay.py --model-dir checkpoints/qwen35_9b \
 --profile configs/evaluation/phase57_replay_dev_v2.json --output outputs/replay_fix_130_e2e_v1

PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/audit_phase57_replay.py \
 --run outputs/replay_fix_130_e2e_v1 --output outputs/replay_fix_130_audit_v1
```

原始运行在lab `outputs/replay_fix_130_*`，紧凑真实证据在 `docs/phase57_replay_fix/`，包括原始文件SHA256、运行源码哈希、独立审计、校准记录、成本和案例。权重、JAR、源码克隆、大日志、缓存不进Git。新服务器使用固定下载脚本即可准备JAR；记录历史回归还需要lab保留的旧原始历史，不能声称它已随Git分发。

## G. 结论与停止

**真实环境副本可靠产生下一状态的已定位工程障碍已修复，并通过本次全部受检场景。** 尚未证明任意任务/任意隐藏状态都可重建，失败拒绝机制继续保留。Reviewer能力、经验覆盖、成功率和真正选择性调用收益是独立问题，本次没有按开发成绩改规则。

原生1.3.0具有物理/歧义解析变化，新旧分数不能当作只改变Gate的公平因果对照。方法有额外真实环境访问及私有状态验证权限，是Ground-Truth/Oracle-Like Lookahead，不是learned WM预测能力。Paper-Aligned协议和最终测试任务未修改；agentwmv2保持原样。本次修复完成后停止，不自动进入Phase6。
