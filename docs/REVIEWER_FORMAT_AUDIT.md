# Reviewer Format Audit — Phase5.10

状态：完成。原始 Reviewer 保留；`reflection/replay_review_v2.py` 是独立新版本。完整实验/费用见 [PHASE5_10_REPORT.md](PHASE5_10_REPORT.md)。

## 原始日志根因

审计 `outputs/phase59_independent_v2/episodes/**/steps.jsonl` 全部531个真实 Review：53次失败（9.98%）。

- **50次** `finish_reason=length`、completion_tokens=512、truncated=true，文本都缺少换行起始的 `Final Action:`。生成预算在自由文本 Reason 尚未结束时耗尽，是这50次无法解析的直接原因。
- **3次** `finish_reason=stop`、truncated=false，已有 Final Action，但其前多一个空行，严格三行正则拒绝。它们不是长度截断。
- 原协议是 Decision/Reason/Final Action 三行，**不是JSON协议**，因此原问题不能描述成53次JSON错误。所有已验证模拟最多一次 Review，没有输出重试/重复生成；不同策略的相同输入仍是各自合法的一次调用。
- 全部531个渲染 Prompt 均以官方 nonthinking 空且闭合的 `<think>…</think>` 前缀结束；生成文本中无 think 标记。实际调用 `enable_thinking=False`，固定EOS为248044/248046。非思考模式正确，但不能阻止模型在 Reason 字段写长篇自我反复推理。
- 长 Reason 的真实例子会反复争论工具不存在、指令无效、是否应该KEEP，并未输出行动字段。Parser只返回明确 INVALID/错误类型/fallback/raw日志；没有把失败算成有效KEEP。

证据：[53个原始失败记录](phase510/original_format_audit.json)。两个原版512 Token真实重现的response IDs与历史完全一致。随后仅作诊断、分别临时设768：一个到759 Tokens正常结束但仍因额外空行失败，另一个仍到768截断，两者均INVALID；首512 Token完全相同。**增加上限不是充分修复**；N=2也不能推广为所有长响应的表现。[实际输出](phase510/results/bounded_length_diagnostics.json)。768从未成为在线配置。

## 修订版协议和边界

版本 `compact-json-conservative-v2`；同一冻结Qwen3.5-9B对象、BF16、T=.4/p1/k0、History2、官方非思考模板。Actor仍为原512 Token配置/原Prompt/原NLL路径，只有Reviewer另设**192 Token**硬上限，不变更全局Actor config或model.generation_config。

只输出一个JSON对象，例如：

```json
{"decision":"KEEP","final_action":"look around","reason_code":"no_grounded_alternative"}
```

三个键全部必需且禁止额外键；拒绝重复JSON键、类型错误、多对象/Markdown/不完整JSON、控制字符/换行、空/过长动作。KEEP必须逐字保留Draft；Reason Code与Decision必须一致。允许JSON结构内部空白，不做任意“抢救”或从长文提取一个动作。格式合法只标记字段合法，`action_validity`和`semantic_utility`仍是Unknown。

Prompt删除自由文本Reason，使用有限原因码；明确零奖励不等于失败、试执行不是已提交状态、只有当前状态已出现的编号问题才可执行数字、没有可靠替代时KEEP、不能从名称推断物性/目标答案。以上是**格式、保守性提示与输出上限的联合版本调整**，不能将任何任务变化单独归因于JSON格式。

现有HF适配器没有暴露原生JSON Schema解码接口；本实现是JSON Schema **Prompt Guidance + 严格校验**，没有安装/声称grammar-constrained decoder，也没有切换推理后端。没有Native admissible-action Oracle、额外替代动作试执行、重复Review或模型训练。JSON可解析不保证动作语法/前提/目标正确。

Reviewer格式统计使用版本专属Parser；通用metadata中的Actor `decision_format_valid`不是Reviewer格式标记，不用于计算Reviewer失败率。

## 实际回归与在线结果

先固定20个旧debug去重请求（格式失败/明确拒绝/接受/不变各5），再固定Phase5.9全部53个失败事件对应的44个唯一Prompt+Seed；另12个Full修订质量请求定向覆盖已知问题。共76次真实修订版调用，**0/76格式错误**。44个失败Prompt的对应事件为0/53错误；这是对原失败样本的回归，不能声称总体错误率从9.98%变为0%。

| 回归面板 | 唯一请求 | 原版错误 → Fixed错误 | 平均输出Tokens | 平均总Tokens | 平均生成秒 |
|---|---:|---:|---:|---:|---:|
| old_debug | 20 | 5 → 0 | 194.45 → 20.20 | 839.30 → 768.40 | 8.761 → 0.959 |
| phase59_format_failure | 44 | 44 → 0 | 509.43 → 21.43 | 1064.77 → 676.68 | 22.966 → 0.995 |
| phase59_full_quality | 12 | 0 → 0 | 102.50 → 20.08 | 672.58 → 690.58 | 4.636 → 0.943 |

原版时间来自历史测量，硬件负载不完全配对；Token计数明确可比。新Prompt因增加保守性约束/结构字段而更长，输出变短不保证总Tokens下降。12个质量面板的平均总Tokens反而略升。原KEEP诊断例没有被修订版改动作；32个同状态控制中没有“Draft已成功完成整个任务”的自然正例，因此不能证明所有正确动作都能被保留。

随后锁定版本，在3个新ID/已暴露Goal的开发实例上运行15个Episode：原版4/44在线Review格式失败（9.09%），Fixed0/47（0%），下降9.09个百分点。分策略为Always4/36→0/37，Full0/8→0/10；Full原来就没有本次格式失败，不能声称修复让Full从高错误率下降。所有76+47个新JSON输出均解析成功，但样本有限且没有独立Goal泛化证据。

**Task Score没有改善：五个条件都−66.67，全部SR0%。** Fixed仍会提出`focus on couch`等目标错误动作。格式修复通过有限工程回归，任务能力和科学有效性未通过。
