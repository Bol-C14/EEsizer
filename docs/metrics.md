# Metrics Contracts

本节定义当前使用的核心指标：名称、单位、仿真条件与计算合同。代码中对应 `agent_test_gpt/metrics_contract.py` 的 `METRICS` 作为单一真理源。

## gain_db (dB)
- 仿真：AC
- 数据：`output_ac.dat`
- 定义：低频开环增益，取 AC 响应前 k=5 个采样的增益（dB）中位数。

## bw_3db_hz (Hz)
- 仿真：AC
- 数据：`output_ac.dat`
- 定义：相对低频增益下降 3 dB 的首次频率（线性插值）。

## ugbw_hz (Hz)
- 仿真：AC
- 数据：`output_ac.dat`
- 定义：增益与 0 dB 的首次交越频率（线性插值）。

## out_swing_v (V)
- 仿真：DC（带反馈的摆幅测试网表）
- 数据：`output_dc_ow.dat`
- 定义：闭环增益目标 `OUT_SWING_GAIN_TARGET` 下的线性输出摆幅；选取斜率满足 `(1 - OUT_SWING_GAIN_TOL) * gain_target` 的区间，返回该区间的 `max(out) - min(out)`。

> 参数 `OUT_SWING_GAIN_TARGET`、`OUT_SWING_GAIN_TOL` 可通过环境变量覆盖，默认 10 与 0.05。
