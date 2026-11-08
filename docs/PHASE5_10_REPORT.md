# Phase5.10 — Reviewer Reliability Repair & Gate Utility Audit

**本阶段完成，等待用户确认；未进入Phase6。** 格式可靠性明显改善，但没有观察到任务收益改善，Full动作修订质量也没有稳定提升。不建议扩大现有方法的大规模评测。

## A. 完成工作与冻结范围

完成原始531次Review/53次格式错误、Full58次修订、13个外部新增调用的审计；保留Original，新增独立命名的 `compact-json-conservative-v2`。真实运行76个去重Fixed回归请求、2对512/768原版诊断（4调用）、32组三分支同状态单步控制，以及3个新Task ID×5条件的15个完整Episode或既定终止。

源基础 `327b46374e7489c91debf5ccb78ad30d7ca3ba07`；实验前注册/Reviewer版本 `a7736a14ad7774df94149dabaa18019136136589`；诊断器修复 `db67057a44c317fbadcea54fb95b1cf8e6f7bc0c` 和 `4680c296f1a1d076bd8d74d17ba844fa1e7933f9`。80个已完成模型请求始终复用，没有按得分重跑/修改模型输出。最终报告/分析Commit见交付说明与Git历史，不混同于数据生成版本。

Actor/Reviewer继续共享冻结的官方post-trained Qwen3.5-9B，Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，BF16/HF5.22.2/Torch2.8.0+cu128/Python3.12、官方nonthinking模板。Actor配置指纹 `cad3791bc197e7f95baa16837d072ab26b68b6f76cb12cf164e0ccfcaf71a7a6`；Internal保持`hf_qwen35_generation_raw_bf16_v1`、Premise/Consequence内容均值NLL的max及整段Draft均值NLL，生成原始logits/FP32归约，无额外forward。Reviewer原版512、新版192，仅每次Review独立上限，不改Actor512或NLL路径。

Native ScienceWorld1.3.0官方Commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`，Jar SHA `e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465`；SDK Python1.2.3客户端指向固定1.3.0 Jar，环境不改。Ground-Truth WM仍独立JVM Reset/Replay、仅Draft一步、最多一次Review，无正式环境预执行/回滚，未运行Qwen2.5。

Gate规则/Local .5749072272810143/Global .3332634086050418/预算0、20、5、5及Memory Snapshot v2（3 cards/54 train cases）逐字节冻结；本次只运行Never/Always/Full做Reviewer对照，Internal只作原轨迹审计。没有训练、Memory新增、阈值校准或Test调参。H20/History2/Prompt4096/Actor Response512/T.4/p1/k0/Native cap21/空simplifications/TaskSeed0均保持。Paper协议文件未修改；9B、结构化Prompt、Native1.3.0/Oracle-Like访问仍是已声明差异，不等同于普通learned WM或论文完全复现。

## B. 修改文件与关键实现

- `reflection/replay_review_v2.py`、`configs/reviewers/compact_json_conservative_v2.json`：三字段短JSON、枚举原因码、严格校验、保守Grounding提示、独立192上限；原 `reflection/replay_review.py`不变。
- `evaluation/independent_dev.py`：仅增加可选Reviewer回调，默认路径保留原函数；Actor/WM/Gate/环境执行顺序不变。Reviewer生成预算按版本记录，仍每步最多一次。旧Phase5.9完整源码重现请checkout327b463，不悄悄覆盖其冻结哈希。
- `scripts/prepare_reviewer_repair.py`、`configs/evaluation/phase510_reviewer_repair_v1.json`：在新输出前冻结诊断面板、任务ID与硬资源上限；按真实Prompt+Seed去重原53失败事件为44次新计算。
- `scripts/run_reviewer_repair.py`：单GPU持久共享模型、封存回归输出后作真实控制、严格来源/前缀/状态哈希、按版本识别旧日志、可审计地复用完整回执、最后15个Episode。预算不够会停止，不动态改方法。
- `scripts/analyze_reviewer_repair.py`：逐任务/版本、所有实际请求/成本、错误/Unknown/单步Utility分开；重算350步Gate及Full58逐例分类；没有拟合或调参。
- `scripts/export_reviewer_repair.py`、`scripts/export_reviewer_repair_traces.py`：76个紧凑真实输入/输出包、156个在线决策Trace、CSV/版本/成本/原始哈希。不提交权重/缓存/大规模原始日志。
- `scripts/test_reviewer_repair.py`（25）、`scripts/test_reviewer_repair_analysis.py`（7）：JSON/缺失/保守回退/Actor不变/历史Hash Schema/真实源预检/Unknown与分类测试。另8个旧Replay与12个独立评测合约实际回归通过，共52项唯一测试；不重复累加同一测试的多次执行。Mock只在单元测试，不进入科研结果。
- `.gitattributes`：控制CSV含真实多行Native反馈，按数据字段原样保留行末空白；仅该CSV允许此格式，其他空白检查保持。
- 三份本阶段报告、`docs/phase510/`、README、Implementation Plan、Progress更新；`agentwmv2`未修改。

核心变更同时包含输出格式、保守性提示和Token上限，不能把任何效果全部归因于某一因素。严格JSON校验不是grammar-constrained解码或Native动作合法性保证。解析失败始终明确INVALID+raw/error/fallback，不把失败强行当作合法KEEP。

## C. 数据与正式比较边界

现有197个dev实例的43个Goal组均已暴露；未使用Goal组数**0**。原12个独立Phase5.9任务只用于回归和Gate审计，没有再次当成独立泛化证据。另有20个未运行ID的已知Goal Variations，按预先SHA排名在三个大类各取一个，先于新输出固定：

| Task ID | Native Task | Variation | Rollout / Simulator Seed |
|---|---|---:|---|
| sciworld_2948 | find-living-thing | 12 | 0 / 0 |
| sciworld_715 | measure-melting-point-known-substance | 55 | 0 / 0 |
| sciworld_1619 | test-conductivity | 183 | 0 / 0 |

Task ID及(taskName, Variation)原生实例与旧调试/Phase5.9/训练/最终Test隔离；Goal与既有开发目标共享。所以这些是**新ID、已知Goal的回归/Variation验证**，不是新的独立Goal泛化评估。不按Full结果重选、不用最终200 Test、不开新Benchmark、不擅自重划train/dev/test。来源与限制：[task_isolation.json](phase510/task_isolation.json)。

每个任务比较Never一次（不调用Reviewer，标签original只是占位），Always Original/Fixed、Full Original/Fixed，共15。各策略相同Actor、Draft Prompt、种子、初始条件、步数、WM、Memory；任务间交替Original/Fixed顺序。5个条件初始受检状态与首个Actor Prompt/Response Token IDs完全相同，后续轨迹实际分别记录，不假定同状态。

## D. 实际测试结果

### 格式与修订质量

原531次Review中50次512长度截断且无Final Action，3次正常结束但额外空行；nonthinking正确。76个新回归输出全部可解析，原53失败事件映射的44唯一Prompt也为0失败。原版两次768诊断仍失败，证明无限加Token不能替代修复。

在线原版4/44格式失败（9.09%），Fixed0/47，下降9.09个百分点。Always4/36→0/37；Full0/8→0/10。因此不能宣称Full本次原格式有问题并被消除。更多细节见 [REVIEWER_FORMAT_AUDIT.md](REVIEWER_FORMAT_AUDIT.md)。

| 条件 | 实际改动作 | 接受 / 明确拒绝 / Unknown | 接受反馈/实际改动作 |
|---|---:|---:|---:|
| Always Original | 25 | 5 / 19 / 1 | 20.00% |
| Always Fixed | 23 | 11 / 12 / 0 | 47.83% |
| Full Original | 7 | 3 / 3 / 1 | 42.86% |
| Full Fixed | 7 | 3 / 4 / 0 | 42.86% |

“接受反馈率”是保守的执行观测指标，**不是语义正确率**；Unknown保留，原版Full那1个Unknown不能自动算False/有效。Always拒绝比例从19/25=76%降至12/23=52.17%，但仍有大量失败。Full3/7拒绝+1Unknown变为4/7拒绝，接受均3/7，没有改善。跨Always/Full合计接受反馈8/32→14/30，不能掩盖Full单项没有提升。

Phase5.9 Full58细分为28明确执行拒绝、11条可确认局部开门前提修复、2条实际目标决策有害、17条无法仅凭现有证据确定目标收益。原11Unknown中4条实际是接线点/工具使用错误，原24/11历史数字保留。没有“Draft已完成任务”的自然正例；不把普通可执行动作等同于足够好。[逐例分类](phase510/results/original_full58_classifications.json)。

### 完整Episode结果

每行3任务、只1颗Seed0，严格实例加权；没有把5个条件或Batch等权后冒充更多独立样本。没有Seed样本SD或统计显著性宣称。所有Score都取最后正式Native值，保留负分；Success=done且Score100，没有训练惩罚/奖励归一化。

| 条件 | Mean Task Score | SR | WM/Reviewer Calls | 格式失败/Review | 总Tokens | Episode总秒 | Native转移查询 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Never | -66.67 | 0.0% | 0 | 0/0 | 12,842 | 71.30 | 32 |
| Always Original | -66.67 | 0.0% | 36 | 4/36 | 42,052 | 451.74 | 366 |
| Always Fixed | -66.67 | 0.0% | 37 | 0/37 | 44,042 | 215.53 | 370 |
| Full Original | -66.67 | 0.0% | 8 | 0/8 | 19,052 | 125.95 | 87 |
| Full Fixed | -66.67 | 0.0% | 10 | 0/10 | 21,418 | 101.90 | 88 |

| Task ID | Never | Always Original | Always Fixed | Full Original | Full Fixed |
|---|---:|---:|---:|---:|---:|
| sciworld_2948 | -100 | -100 | -100 | -100 | -100 |
| sciworld_715 | -100 | -100 | -100 | -100 | -100 |
| sciworld_1619 | 0 | 0 | 0 | 0 | 0 |

六个同任务Original/Fixed Score差全部0，SR全部0%。**没有实质任务改善证据。** 不能把本组−66.67与Phase5.9的不同任务均值直接比较来归因于Reviewer。完整 [15行结果CSV](phase510/results/episodes.csv)、[配对差异](phase510/results/paired_episode_differences.json)、[156条真实在线决策](phase510/results/online_decision_traces.jsonl) 已导出。

### 单步与Gate

32个严格公共/私有受检投影相同的当前状态，分别真实执行Skip Draft、Original Final、Fixed Final。Original−Skip正/零/负1/29/2；Fixed−Skip0/32/0；Fixed−Original2/29/1。质量12面板刻意含原版已知正负案例，不能当成无偏Utility分布或长期任务收益。

Fixed避免了painting错误Focus（+117）与错误放箱（+160），但也把chair改couch后继续−100，失去一次原版避免立即失败的机会。其自身相对Draft即时分数均0；信息获取与后续进度仍可能不同，未测完整Continuation因果回报。详见 [GATE_UTILITY_AUDIT.md](GATE_UTILITY_AUDIT.md)。

13次低NLL外部新增调用所依赖的闭门经验全部适用，真实试执行确认13/13；修订10接受/2拒绝/1不变，即时分数差全0。原Full−Internal主要由2个Seed1任务的大幅下降主导。Local/Global高度相关但OR不重复发请求；全350步两项都高59、仅Local12、仅Global21、都不高258，没有按这些分布重新设阈值。

新实例2948首步Full直接Skip错误的`focus on air`（Local.540339/Global.319287均低于冻结阈值、External Unknown），正式Score−100。两版都未调用Reviewer，说明能力与调用漏检也是瓶颈，不能把所有问题归为输出格式。记录该漏检，不据此调Gate。

## E. 完整成本与工程有效性

新测量累计 **327模型调用 / 198,040 Tokens / 2,003 Native转移查询 / 5,107状态核验Bundles**：

- 在线15 Episode：156 Actor Calls、91 WM尝试/91成功Lookahead/91 Reviewer Calls、139,406 Tokens、943查询。
- 离线：76 Fixed请求+4原版长度诊断=80模型调用、58,634 Tokens；32组三分支=96个实际控制分支、1,020查询。
- 两个诊断器失败与真实Hash语义探针另40查询；已包含累计账本，已复用的Draft分支14查询和80请求只算一次。
- 新离线经验构建0，历史Memory构建不冒充免费；LLM Judge/Router/learned WM调用0。所有在线成本包含结构化Draft、Review、NLL、检索和真实模拟，副本查询不计正式Turn但计额外访问。

Always Fixed总Tokens44,042对42,052（+4.73%），Full Fixed21,418对19,052（+12.42%）；虽然输出短，新Prompt更长、实际调用和轨迹不同。Always总秒451.74→215.53，Full125.95→101.90，延迟下降而Token/查询并未全面下降，不能宣称总成本全面改善。查询Always366→370，Full87→88。各组Reviewer Tokens/Actor Tokens、输入/输出与全部计时见 [summary.json](phase510/results/summary.json)。

3次严格模型加载，累计77.85秒，前两次因诊断器安全停止而退出；从首个真实模型Run到最终完成墙钟6,729.23秒，最后进程1,243.03秒，包含CPU检查/恢复间隔，不是GPU Kernel Hours。RTX4090、峰值Torch已分配约17.77–17.86GiB；未连续测GPU功率/FLOPs，无法可靠换算货币。[成本范围](phase510/results/cost_scope.json)。

真实审计15/15 Episode、156步骤、91/91 Lookahead、29/29保留Draft后状态投影通过；初始状态与首Draft Token完全一致，Gate/Prompt/预算/因果时序/模型身份/Score/所有模型及环境回执与账本一致。135个冻结源码/配置/来源文件哈希复核通过。原始完整Trace位于lab输出目录，Git保存约2MB本次紧凑证据，不含权重/缓存/敏感配置。

### 必须保留的失败与修复

1. `outputs/phase510_reviewer_repair_v1.log`：启动器误把venv Python符号链接resolve到旧Conda，Qwen3.5类导入失败，0模型/0环境调用。改为venv原始绝对路径；未安装/修改旧环境。[记录](phase510/launch_incident.json)。
2. v2完整生成80请求后，新的诊断器误把会去除raw observation的state_fingerprint与历史WM `digest(full capture)`比较。安全停止，已花13查询；真实重放探针再花13查询，**完整legacy digest精确一致**，未忽略任何状态字段。修正比较合同并加测试。[证据](phase510/hash_semantics_probe.json)。
3. v3复用全部80请求，已完成Draft分支14查询，Original分支执行14查询后因旧Phase5.7缺少后来才新增的正式后指纹字段停止。新增来源版本预检：旧缺失明确标注，不伪造；Phase5.9字段仍强制要求。所有比较仍严格核对共同执行前完整哈希和Draft下一状态/公共转移。v4复用80请求及那条已验证Draft，未挑选结果/重跑模型。[记录](phase510/legacy_schema_repair.json)。

这两项是新诊断器的工程缺陷，不是ScienceWorld Replay失败或研究方法改进。Original/Fix Reviewer、Gate、Memory、阈值、Actor与Native WM均未在重试间修改。所有失败成本/代码版本/账本/原始日志保留；原3h截止与600Calls/650kTokens/6500Queries/25kBundles上限未重置。v2/v3使用GPU2，释放后他人使用；v4在GPU3完成，未抢占别人的任务。

## F. 复现命令与限制

使用现有`.venv-qwen35-clean/bin/python`路径，不能对解释器符号链接调用resolve；新目录才可运行，不重复启动当前已完成Run。

```bash
# CPU单元/契约，实际通过25+7+8+12=52项。
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_reviewer_repair.py
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_reviewer_repair_analysis.py
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_integration.py
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_independent_dev.py

# 已实际执行的原始完整计划；源码版本a7736a1。
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/run_reviewer_repair.py \
 --run outputs/phase510_reviewer_repair_v2 --model-dir checkpoints/qwen35_9b
# 已实际执行的两次可审计恢复：不重复76个Fixed/4个原版请求。
CUDA_VISIBLE_DEVICES=2 .venv-qwen35-clean/bin/python scripts/run_reviewer_repair.py \
 --run outputs/phase510_reviewer_repair_v3 --model-dir checkpoints/qwen35_9b --resume-offline-from outputs/phase510_reviewer_repair_v2
CUDA_VISIBLE_DEVICES=3 .venv-qwen35-clean/bin/python scripts/run_reviewer_repair.py \
 --run outputs/phase510_reviewer_repair_v4 --model-dir checkpoints/qwen35_9b --resume-offline-from outputs/phase510_reviewer_repair_v3

# 全部测量完成后才实际执行的CPU分析与紧凑导出。
.venv-qwen35-clean/bin/python scripts/analyze_reviewer_repair.py \
 --run outputs/phase510_reviewer_repair_v4 --output outputs/phase510_analysis_v1
.venv-qwen35-clean/bin/python scripts/export_reviewer_repair.py \
 --run outputs/phase510_reviewer_repair_v4 --analysis outputs/phase510_analysis_v1 --output docs/phase510/results
.venv-qwen35-clean/bin/python scripts/export_reviewer_repair_traces.py \
 --run outputs/phase510_reviewer_repair_v4 --output docs/phase510/results/online_decision_traces.jsonl
```

干净新运行使用当前修复后的CLI、全新目录且不带resume参数，无需人为制造失败。准备脚本拒绝覆盖冻结任务/面板；它依赖保留的历史完整源日志，因此新服务器运行完整诊断仍需这些源文件。Git中的76个 [case包](phase510/results/case_index.json)含真实公共输入/输出、历史Prefix/Native Spec和必要指纹，Audit-only未来结果与Reviewer输入严格分栏；能够审查和重建诊断证据，但不声称整个历史数据集已随Git分发或已在另一台物理服务器复现。

52项单元测试与真实评估都通过；未训练、未校准Gate、未运行最终200×8。207个已记录自有PID及4个自有后台会话检查无存活，两个早期失败JVM虽未单独记录PID，finally已等待退出且其会话无遗留。`agentwmv2`工作区干净，无他人进程被终止。[Cleanup](phase510/results/cleanup_audit.json)。

科学限制：单Seed/三新ID共享已暴露Goal、质量面板结果定向选择、成功很少、稀疏Score、有限状态投影而非完整JVM隐藏等价。真实模拟器有普通learned WM没有的信息访问权限；零WM LLM Tokens不等于零查询成本。多因素Reviewer变更不能识别格式单因子效果，原始方法和各版本证据完整保留。

## G. 六项验收结论与停止

1. **50次达上限是否根因？** 是这50次缺Final Action的直接截断原因；更深原因是自由Reason长、动作字段位于其后。另3次是空行，非thinking/EOS配置错；768诊断仍失败，不能仅扩大上限。
2. **格式失败下降多少？** 原失败事件回归53/53→0/53（44唯一输入），旧debug5/20→0/20；新在线对照4/44=9.09%→0/47。有限诊断不等于泛化零错误。
3. **无效修订减少？** Always明确拒绝19/25→12/23、接受5/25→11/23；Full接受仍3/7、拒绝3/7+1Unknown→4/7，**没有稳定全面改善**。JSON合法仍可目标错误。
4. **任务表现实质改善？** 没有：五条件Mean−66.67/SR0%，六个配对分数差全0。
5. **外部经验为何未带来正收益？** 13条闭门冲突判断可靠，但修订器不能稳定产生任务正确动作、局部修复不一定产生Score/长期净益，有限预算和后续轨迹改变也有影响。新Air案例还显示Gate会漏掉经验未覆盖的低NLL语义错误。没有严格长期Continuation，不能将单次修改当作正因果收益。
6. **能否进入Phase6？** 格式/记录工程修复通过有限验收；**现有证据不足，建议暂不扩大**。先经用户批准，把format-only与保守性提示分开做独立版本诊断，确认Reviewer是否能利用真实反馈持续改善决策，并另行规划未用于Memory/调试的新Goal开发验证。现有Goal已耗尽，不能继续把这批回归当独立，也不能改用最终Test调参。本阶段未实施这些建议、未改变Gate或增加替代动作在线模拟，完成后停止等待确认。
