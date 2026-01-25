总体设计：把系统卖成 “Interactive Optimization Copilot”

不是“自动把电路调到最优”，而是：

Deterministic Search Engine（可信内核）

GridSearch（探索空间）

CornerSearch / CornerValidate（鲁棒性）

统一 metrics（UGBW/PM/Power）

统一 report（图表 + 解释 + 可追溯）

Interactive Intelligence Layer（智能层，LLM 可插拔）

LLM 负责：读 artifacts、提出 Plan/解释、建议下一步

用户可以在中途“暂停/交互/继续”，像和一个懂电路的同事一起看报告做决策

但 LLM 不直接改文件、不直接跑仿真，只能产出结构化 Plan/Patch，交给 registry 执行并落盘

这会让 paper 的故事更像“工具型系统”，而不是“我们串了个 pipeline”。

Step 列表：实现步骤（不按时间，按依赖关系）
Step 1 固化 3 个 Benchmark 的“可重复测试台”

目标：OTA/三段运放的 UGBW/PM/Power 不能每次测法不一样，否则 paper 结果会被质疑。

交付物：

benchmarks/ 目录（固定结构）

benchmarks/models/ptm_180.txt

benchmarks/rc/bench.sp

benchmarks/ota/dut.sp + bench.sp

benchmarks/opamp3/dut.sp + bench.sp

benchmarks/*/bench.json + spec.json

每个 benchmark 明确：

供电、负载、电路连接方式（unity-gain? open-loop?）

AC 分析频段与采样点

power 测量方式（通过供电源电流 I(VDD)）

验收：

baseline_noopt 对三个电路都稳定出 metrics（哪怕不满足目标）

Step 2 统一 Metrics 定义与“测量配方”

目标：UGBW/PM/Power 要变成可引用的定义，报告里要写清楚“怎么算”。

交付物（工程侧）：

metrics/ 中对 UGBW/PM/Power 的单一实现

sim_plan.json 或类似 artifact 里包含：

需要的仿真类型（AC + OP/DC）

输出节点（例如 out、vin、环路节点）

计算 PM 的方法（例如 unity-gain crossing 处的 phase）

交付物（报告侧）：

report.md 自动输出“Metric definitions”小节（固定模板）

验收：

单测：给定一份合成的 AC 数据，UGBW/PM 计算可重复

运行三电路 baseline，report.md 中的定义一致且清晰

这一步非常 paper friendly：审稿人最爱问“PM 怎么测的”。

Step 3 Grid Search 的 Range/Validity 机制“写死成可辩护规则”

目标：导师说的 “grid search needs to be trivial and valid”。你要能回答：范围从哪来？为什么这样扫？

设计要点（建议默认策略）：

优先用 ParamDef.lower/upper（如果有）

否则用 nominal × span_mul 推导

对 log scale：强制 lower > 0，否则 fallback linear 并记录 warning

对每个 param 输出：

nominal、lower、upper、levels 列表

是否被截断/跳过（原因写清楚）

交付物：

search/ranges.json（或 search/candidates.json 里附 ranges）

report.md 新增 “Search ranges & discretization” 小节

如果 budget 截断候选点：必须写明截断策略（字典序/seed shuffle）

验收：

任意一次 grid run 的 report 都能重建 candidates（可回放）

Step 4 Grid Search 报告升级：从“日志”变成“工程师读得懂的说明书”

目标：你导师要 “human readable”，你也提到希望能有“grid-like graph + highlight”。

建议最小图表组合（够写 paper）：

参数变化热力图（Grid Map）

行：候选点（TopK 或全部）

列：参数

值：log10(value/nominal) 或 (value-nominal)/nominal

高亮：best / pareto 点

单参响应曲线（Coordinate sweep plot）

每个参数一张：x=param value，y=score 或 y=UGBW/PM/Power（可选）

标注 best 值、以及导致失败的区间（如果有）

Trade-off 散点图（Power vs UGBW）

点：候选

高亮：Pareto front

用 marker 区分 pass/fail（PM 不达标等）

交付物：

plots/*.png（并嵌入 report.md）

report.md 在每张图后自动生成 2–4 行解释（模板 + 数据驱动）

验收：

一次 grid run 结束后 report.md 打开就能“看懂发生了什么”

不需要读 history jsonl 才能理解

Step 5 Robustness：把 CornerSearch 从“功能”变成“paper 观点”

目标：你们已经有 corner search，但 paper 要表达的是：我们做 worst-case 评估，并用它驱动选择。

建议你们把 M3 的结果进一步“结构化输出”，让它像论文表格那样可用：

对每个 candidate 输出：

nominal：UGBW/PM/Power

worst-case：UGBW/PM/Power（以及是哪一个 corner）

pass_rate

worst_score（聚合 score）

robust_losses（用来做 robust pareto）

交付物：

search/robust_topk.json

report.md 增加 “Nominal vs Worst-case” 表格

增加一张图：nominal vs worst-case 的对比散点（比如 UGBW_nominal vs UGBW_worst）

验收：

对 OTA/opamp3：能清楚指出 worst corner 是哪种（all_low / 某个 param_low）

Step 6 引入 “Spec 变化可引用机制”

你导师提到 “reference on how specs are changed”。你们一周内不一定要做到自动 spec 合成，但一定要做到：

spec 的版本与变化被记录

report 能展示 spec diff

设计建议：

新增 artifact：spec_trace.jsonl
每次 spec 变化都记录：

old_spec hash

new_spec hash

change reason（human/agent）

timestamp、run_id

并在 report 顶部增加一个小节：

“Spec used in this run”

“Spec changes (if any)”

验收：

人能解释：这次为什么 PM 从 60° 改到 55° 或权重怎么变

Step 7 把“交互式优化”做成 Session（核心智能化点）

这是你要避免“只是 pipeline”的关键一步：让系统能 中途停下来，让人/LLM 读报告再决定下一步。

我建议新增一个很克制的概念：

Session = ArtifactStore + Checkpoints + Continue Plan

每个阶段结束都写 checkpoint：

baseline checkpoint

grid checkpoint（含 topk/pareto/report）

robust checkpoint（含 worst-case）

用户/LLM 可以在 checkpoint 后做三类交互：

询问：为什么选这个点？哪个参数最敏感？

改配置继续：扩大某个 param range、改变 levels、改 top_k、改 robust corners 预算

选择候选进入 refine：从 Pareto 里选 2–3 个点做局部精修

交付物：

CLI（或脚本）支持：

eesizer session new ...

eesizer session step ...（跑一个阶段）

eesizer session continue --from <checkpoint> --edit <cfg/spec delta>

eesizer session inspect <run_dir>（打开 summary）

验收：

你能在 OTA 上跑：baseline -> grid ->（人工改范围）-> grid2 -> corner validate

每一步都落盘可回放

Step 8 LLM 作为“智能工具调用者”：Plan 提案而非执行

这一步把 LLM 放在最安全、也最容易写成论文亮点的位置。

新增一个或两个 LLM agent（可开关）：

8.1 ReportInterpreterAgent（读报告、解释 trade-off、生成文字/图表建议）

输入：

report.md + search/topk.json + search/pareto.json + robust_topk.json
输出（结构化）：

insights.json：

最敏感参数排名

UGBW/PM/Power 的 trade-off 解释（模板 + 数据）

推荐的下一步（比如扩大某参数范围、提高 PM 约束等）

可选：extra_plots_plan.json（要生成哪些额外图）

8.2 ToolPlannerAgent（提出下一轮 Plan）

输入：

insights + 当前 cfg/spec + 预算剩余
输出：

plan.json（只能调用 ToolRegistry 白名单动作）

action 示例：expand_range(param=...), refine_around(candidate=...), corner_validate(topk=...)

关键限制（写进 paper 的“guardrails”）：

LLM 不允许直接输出 netlist

只能输出 Plan（JSON schema 校验）

PlanExecutor 执行时有预算、参数白名单、动作白名单、seed 固定

prompt/response/plan 全部落盘

验收：

在一个 grid run 后，你让 LLM 根据 report 提出下一步 plan

你选择 accept 后继续跑，产物仍然可审计

这就是你说的“智能调用工具 / report 后交互协助产出工程师理解的文字/图”。

Step 9 Multi-result：用多策略产生多候选，再由报告层统一解释

你提到“multiagent 产出多结果再分析”。这是很好的“智能化点”，而且不需要并行也能做：

设计：

Orchestrator 生成多个子计划（顺序执行即可）：

Plan A：GridSearch（广）

Plan B：Greedy refine（深）

Plan C：CornerValidate（稳）

最后用 Aggregator 汇总三个 run 的：

best nominal

best worst-case

robust pareto

建议选择（工程解读）

交付物：

meta_report.md：跨策略对比报告

meta_summary.json

验收：

OTA 上能看到：grid 提供多样性，corner 提供鲁棒性筛选，greedy 提供局部改善

Step 10 Paper-ready 实验运行器与结果冻结

最后一步是让你能“发 paper”：可复现的批量运行与出图。

交付物：

tools/experiments/run_suite.py：读一个 suite yaml，跑 3 circuits × {grid, corner} × seeds

输出：

results.csv（每行一个 run：score、pass_rate、best UGBW/PM/Power、worst-case）

figures/*.png（paper 主图）

tables/*.tex 或 tables/*.md（论文表格）

验收：

一条命令能跑出 paper 所需图表（不手工拷贝）

这套设计如何“像论文，而不是像 pipeline”

你最终在 paper 里可以非常自然地写成：

我们提供 deterministic 搜索与鲁棒评估，作为可信内核

我们提供交互式 session，把优化过程拆成阶段性 checkpoints

LLM 作为“copilot”读报告、解释 trade-off、提出下一步 Plan（工具调用），并且所有决策可审计

这就把 LLM 变成“智能层”，而不是“我用 LLM 盲调电路”。
