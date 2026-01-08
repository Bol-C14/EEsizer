# EEsizer Notebook Mapping

Notebook 里的内容准备拆成三类；本阶段先把数据结构类的 Artifacts 显式列出，记录名称与含义，后续步骤再补充其他类别。

## Artifacts（数据结构）

- **AnalogNetlist**：完整的 SPICE 文本字符串，加上对应的 `filename` 记录。
- **SimContext**：仿真执行的上下文，包括 `ngspice` 路径、工作目录，以及所用 PDK/模型文件信息（目前在 notebook 里可能是隐式传入，这里先明确需要这些字段）。
- **SimOutputs**：一次仿真得到的输出集合。
  - **AcResult**：AC sweep 的输出文件名与数据。
  - **TranResult**：瞬态仿真输出。
  - **OpResult**：OP 分析结果，以及 vgs/vth 等检查。
- **MetricsVector**：结构化的指标向量，例如 `{gain, bandwidth, unity_bw, phase_margin, power, tr_gain, offset, thd, icmr, out_swing, cmrr, ...}`。
- **SizingSpec**：目标指标规格，等同于 notebook 里展开的 `target_values` JSON。
- **DesignPoint**：一次候选设计，包含 `{netlist, metrics, pass_flags, comments}`。
- **OptimizationHistory**：`DesignPoint` 的列表，加上每轮迭代的说明文本（当前以 `result_history.txt` 形式落盘）。

## Operators（干活的函数）

- **run_ngspice**：`AnalogNetlist` + `SimContext` → `SimOutputs`（AC/Tran/OP 结果文件与数据）。
- **ac_gain / bandwidth / unity_bandwidth / phase_margin**：基于 `SimOutputs`（AC）抽取指标。
- **stat_power / calculate_static_current**：基于 `SimOutputs`（OP/Tran）统计功耗或静态电流。
- **cmrr_tran / thd_input_range**：基于 `SimOutputs`（Tran）计算 CMRR、THD 等。
- **ICMR / out_swing / offset / tran_gain**：基于 `SimOutputs`（Tran/OP）计算输入共模范围、输出摆幅、失调、瞬态增益。
- **tool_calling**：组合型 operator，负责串起或编排上述计算。

## Strategy / Agent（决策逻辑）

- **optimization(tools, target_values, sim_netlist)**：主循环，对齐 SizingSpec → 维护 OptimizationHistory → 调用 llm_call 生成新 `AnalogNetlist` → 调用 Operators 计算 `MetricsVector` → 根据 pass/fail 决定是否继续迭代或收敛。
