# Gate Utility Audit — Phase5.10

状态：只做诊断；Gate、Local/Global阈值、Memory、Actor和Ground-Truth WM不变。旧Phase5.9的任务与结果保留，未根据本审计调整参数。当前仍不进入Phase6。

## External的13次额外调用

在Full同一已记录状态、同一实际Full剩余预算上重算Internal，13次Skip→Call。该影子决策不是另一条真实Internal轨迹；不同策略实际状态/后续调用不能视为相同。

全部13次来自 `exp_b5e8e19f18820e2efed9` 的 `closed_route_go`：当前观察明确存在到Draft目的地的关闭门，Draft却要直接go。可靠性supported依据四个真实训练案例，独立实例/Goal/episode支持、无反例，BM25分数不充当可靠性：

- phase3_rule_probes_v1/sciworld_1545/step-1
- phase3_rule_probes_v1/sciworld_1667/step-1
- phase3_train_v2/sciworld_3044/step-3
- phase3_train_v2/sciworld_325/step-4

来源与Evidence ID重新核验相符；所有13次真实试执行均返回`The door is not open.`，说明这些经验确实适用。Local/Global均低于冻结阈值.574907/.333263，属于生成较自信但通行前提错误的实际例子。下表Step为1-based；全部Draft为go命令，详细原文/支持来源见 [gate_audit.json](phase510/gate_audit.json)。

| Task ID | Seed | Step | Local / Global NLL | Reviewer最终动作 | 正式环境反馈 |
|---|---:|---:|---:|---|---|
| sciworld_1508 | 0 | 1 | 0.2697 / 0.1381 | open door to the hallway | The door is now open. |
| sciworld_1508 | 1 | 11 | 0.2385 / 0.1521 | go living room | No known action matches that input. |
| sciworld_2438 | 0 | 2 | 0.2011 / 0.1138 | open door to the hallway | The door is now open. |
| sciworld_2438 | 1 | 2 | 0.2237 / 0.1203 | open door to the hallway | The door is now open. |
| sciworld_2458 | 0 | 2 | 0.3444 / 0.2300 | open door to the hallway | The door is now open. |
| sciworld_2458 | 1 | 2 | 0.3516 / 0.1834 | open door to the hallway | The door is now open. |
| sciworld_2519 | 0 | 2 | 0.2944 / 0.1628 | look in large cupboard | The large cupboard isn't open, so you can't see inside. |
| sciworld_2519 | 0 | 8 | 0.2119 / 0.1256 | open door to the hallway | The door is now open. |
| sciworld_2519 | 0 | 11 | 0.2705 / 0.1629 | open door to the workshop | The door is now open. |
| sciworld_2519 | 1 | 2 | 0.2031 / 0.1284 | open door to the hallway | The door is now open. |
| sciworld_3068 | 1 | 1 | 0.4452 / 0.3278 | go living room | The door is not open. |
| sciworld_378 | 1 | 2 | 0.2337 / 0.1465 | open door to the hallway | The door is now open. |
| sciworld_378 | 1 | 5 | 0.5206 / 0.2579 | open door to the living room | The door is now open. |

12次改动作，其中10次环境接受、2次明确拒绝，另1次KEEP仍闭门失败。13次实际最终Score减去Draft试执行Score均0；开门可以修复即时前提，却不直接产生Task Score。没有执行相同状态/相同后续Actor和剩余预算的完整Continuation，**不能认定这13次的长期因果收益为0或为正**。

Full覆盖仅41/350=11.71%；Conflict14/350，其中1次同时有内部触发。Sufficiency无自然True，确认Skip/Memory Correction保持关闭。经验知道“go会失败”并不自动给出可执行的任务正确替代。Reviewer错误、Goal进度推断、稀疏评分和有限调用预算竞争都可能影响净收益；没有为增加触发率放宽经验条件。

## 性能差异与NLL重叠

Phase5.9 Full−Internal均分−8.54；最终Score只在Seed1的1508（+5）、378（−100）、2519（−110）不同，其他21个Task×Seed得分相同，轨迹未必相同。两个大幅下降主导差异。2519的致命替代是内部NLL也会触发的那次`use battery…`→错误放箱，不能把它简单归因于某张门经验。

全部350个Full执行前状态中：两项都高59、仅Local高12、仅Global高21、都不高258。Local/Global Pearson .9167、Spearman .9308，明显相关；但并非完全重复。按实际Full剩余预算的影子Call数Local51、Global58、Internal67、Full80；其中实际67个内部触发调用有42两者共同触发、9仅Local、16仅Global，额外13来自External。

OR每步最多一次WM，**没有两信号同时触发就重复请求的工程问题**。信号相关/重叠不能证明对应调用无收益；没有足够Utility标签判定这些重叠步可以安全删除。`phase510/gate_audit.json`的overlap只统计80个有Review的Full调用；[gate_utility_detail.json](phase510/results/gate_utility_detail.json)另给出完整350步统计，二者范围不同。

新增实例2948/variation12/seed0提供另一种漏检：Draft是`focus on air`，Premise/Expected Consequence错误声称空气是living thing。Local .540339、Global .319287均低于旧阈值，External全Unknown，Full返回no_active_trigger，第一步正式Score−100/Done。Original与Fixed在这条Full轨迹都没有Reviewer调用。该例说明问题不全在Reviewer：流畅文本的NLL不是事实错误概率，窄门经验也无法覆盖目标语义错误。阈值和经验仍保持冻结。

## 可靠的单步对照

预先固定20个旧debug状态与12个已暴露的Phase5.9质量诊断状态。每个状态在独立JVM中分别重放同一完整历史，执行Draft（Skip）、Original最终动作、Fixed最终动作。三个分支的完整受检执行前哈希一致，Draft真实转移/后状态指纹与来源完全匹配；Original公共反馈/reward/score/done/moves与来源匹配。Phase5.9来源还严格核对历史正式后指纹；20个旧debug来源没有该字段，明确标记缺失，不声称历史隐藏后状态全部相同。所有本次分支后指纹真实记录，不伪造缺失哈希。

所有76次Fixed输出在首次这些评估转移前已封存；后验分支反馈只用于评价，不传回Reviewer/Actor/Gate。完整结果：[32行控制CSV](phase510/results/controlled_one_step.csv)。这些是有限状态投影下的单步比较，不是完整JVM隐藏状态证明或长期回报估计。

| 面板 | Original−Skip 正/零/负 | Fixed−Skip 正/零/负 | Fixed−Original 正/零/负 |
|---|---:|---:|---:|
| 旧debug20状态 | 0/20/0 | 0/20/0 | 0/20/0 |
| Full质量12状态 | 1/9/2 | 0/12/0 | 2/9/1 |
| 合计32状态 | 1/29/2 | 0/32/0 | 2/29/1 |

12个质量样本刻意覆盖原版两个负向和一个正向Score案例，**不是无偏的Utility发生率估计**。

真实例子：

- 3003/seed0/step15：Draft `open painting` Score17；Original改`focus on painting`后−100；Fixed明确KEEP，仍不可打开但保留17，避免已知错误修订（+117相对Original）。
- 2519/seed1/step11：Draft `use battery on unknown substance J`反馈不确定如何使用、Score60；Original错误放yellow box后−100；Fixed改`look at battery`后60（+160相对Original），没有即时分数提升于Draft。
- 3003/seed1/step1：Draft `focus on chair`会−100；Original `go bathroom`虽然被拒绝但避免立即失败，Score0；Fixed改`focus on couch`同样−100（相对Original−100）。JSON格式完全正确，目标语义仍错。

原32个状态中20个旧debug不含Score增益案例，且没有已完成任务Draft正例，不能推出Fixed能保留所有正确动作。原版在质量面板4次改动已接受且非负奖励的Draft，Fixed仍有1次；“已接受且非负”也不等于已经足够完成任务，因此只记潜在不必要替代，不强行定性。

## Full的58次实际修订分类

[逐例分类](phase510/results/original_full58_classifications.json)保留Goal、当前Observation、Draft/Final、公共反馈、Score、原因与源记录定位。分类采用执行与目标意义两个轴，允许Unknown：

- 原保守分类：23接受、24明确拒绝、11Unknown。
- 原11Unknown中3条是电路连接未指定接触点、1条是不知道如何使用switch；这4条属于明确执行错误。其余7条为对象描述/查看反馈，不能直接当成任务正确。细分后28明确拒绝、23接受、7Unknown；历史Phase5.9数字不改写。
- 28拒绝中20是No known action（语法与对象引用原因无法仅凭此反馈分开），3接触点前提、1工具使用、4其他状态/访问前提。
- 11条满足严格的“闭门go→open door，真实门已打开”条件，可称**局部前提修订正确**，不能称任务已被正确解决。
- 2条接受了命令却导致即时Score下降至−100：painting Focus是目标不相关/错误，unknown substance放箱是目标选择错误。其余17条不能仅凭格式/接受反馈确定任务正确性。
- Full另有9/80格式失败，属于未产生可执行的替代决策，程序明确执行未改Draft并保留INVALID；它们不在58次实际修改内。
- 没有“原Draft已完成任务却被替换”的自然正例，不能把普通可执行/0分动作全判为“不需修订”。

没有据这些标签训练Router/Probe，修改Memory或选择Gate阈值。现有证据支持进一步定位能力瓶颈，不支持直接启动Phase6大规模实验。
