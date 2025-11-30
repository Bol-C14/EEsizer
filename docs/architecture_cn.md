# EEsizer 核心架构与模块说明

本文概述 EEsizer 的核心闭环：Optimizer 按迭代驱动 Orchestrator 进行仿真测量，LLM Agent 负责分析与尺寸优化，生成新的网表，再由 Orchestrator 评估并回馈，形成仿真—分析—优化的闭环。

## 模块职责
- **Optimizer（eesizer.agents.optimizer）**：整体调度每轮流程，调用 Orchestrator 获取指标，构造并发送分析/优化请求给 LLM，应用网表补丁（`eesizer.netlist.patch`），记录迭代历史与报告（`eesizer.reporting`），根据 pass/收敛策略决定继续或终止。
- **Orchestrator（eesizer.agents.orchestrator / eesizer_core.agents.simple）**：封装仿真编排与评分。根据 tool_chain 构建控制卡（AC/DC/TRAN）、调用 NgSpiceSimulator，收集测量文件与波形，提取指标后返回 Optimizer；支持对多个网表变体评分选优。
- **NgSpiceSimulator 接口（eesizer.sim.ngspice / eesizer_core.simulation.NgSpiceRunner）**：负责调用 ngspice，收集 .measure / wrdata 输出与操作点检查；Mock 版本用于无依赖测试。
- **Netlist Builders（eesizer.sim.netlist_builders）**：为网表插入测量源与控制语句（如 AC sweep、瞬态窗口）。
- **Metrics Extractors / Oplog Parser（eesizer.analysis / eesizer_core.analysis.metrics & oplog）**：从仿真输出提取 AC/瞬态增益、带宽、相位裕度、功耗、输出摆幅、THD、CMRR、ICMR、偏移等；并解析 Vgs/Vth 裕量。
- **LLM Agent 接口（eesizer.llm）**：封装提示构造、API 调用与输出校验（`llm.validate`/`llm.schemas`）；包括 MockLLM，便于无真实 API 的闭环测试。
- **ScoringPolicy（eesizer_core.agents.scoring）**：独立的多指标评分/达标判定策略，使用几何平均结合增益/功耗归一化，供 SimpleSizingAgent 及未来多 Agent 复用。

## 数据与控制流
1) Optimizer 将当前网表交给 Orchestrator；Orchestrator 依据 tool_chain 构建 ControlDeck，触发 ngspice 仿真并提取指标。  
2) Optimizer 将指标与目标传给 LLM 进行分析，接收建议与具体参数改动。  
3) LLM 返回优化后的网表文本（或补丁）；Optimizer 应用并回写，再调用 Orchestrator 评估原/新网表、选优并记录历史。  
4) 若指标达标或达到停止条件，生成报告并退出；否则进入下一轮。

## 当前测量覆盖（核心实现于 `eesizer_core.analysis.metrics` 与 `eesizer_core.simulation`）
- 增益：`ac_gain_db`、`tran_gain_db`（AC/TRAN）
- 频率特性：`bandwidth_hz`、`unity_bandwidth_hz`/`unity_gain_frequency_hz`、`phase_margin_deg`
- 摆幅：`output_swing_max/min/pp` → `output_swing_v`
- 偏移/ICMR：`offset_v`、`icmr_min_v`、`icmr_max_v`、`icmr_range_v`
- 抑制/失真：`cmrr_db`、`thd_output_db` → `thd_db`
- 功耗：`power_mw`（由 supply 波形/measure 推导）
- 噪声：`noise_rms_v`、`noise_mv`（由输出波形标准差估计）

## 近期改进要点
- 已完成：在 ControlDeck 中增加相位导出、功耗与 THD 测量；新增指标聚合逻辑（功耗、相位裕度、ICMR 范围、unity 频率别名、噪声估计）；优化评分策略采用几何平均的多指标评分（增益↑、功耗↓），避免单指标偏置。  
- 待拓展：覆盖输出摆幅、CMRR、THD 之外的其他 README 指标（如噪声、失真阶次、输出摆幅多点扫描），以及更健壮的评分/收敛策略与错误回退。  
- 运行策略：默认使用 Mock ngspice（需显式设置 `EESIZER_ENABLE_REAL_NGSPICE=1` 才启用真机仿真），确保在缺失模型库时测试仍可通过。***
