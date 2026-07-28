# 项目进展与实验归档

更新日期：2026-07-28。

本文档记录已经完成且仍可作为论文证据的实验、明确作废的运行，以及本地清理
决策。具体方法和指标定义以各阶段基线文档为准。

## 已完成的研究链路

| 阶段 | 已完成内容 | 可支持的结论 |
| --- | --- | --- |
| E0–E1 | 仓库、配置、测试体系和 GRID 数据管线；`s1` pilot 20 条、1,500 帧，失败 0 | 数据与代码分离的预处理闭环可复现 |
| E2 | LivePortrait 18 维嘴部运动提取、重建和 28 条件敏感性实验；20 条 pilot 共 560 条指标，失败 0 | 低维运动可驱动冻结重建器，扰动会产生可测质量变化 |
| E3 | 修正音视频时间对齐后完成三说话人实验；随后扩展到十说话人 800/100/100 身份隔离划分、三种子 GRU 和 200 条冻结重建 | 十说话人 test 上 GRU 同时优于 train mean 与 zero motion，RQ1 获得初步支持 |
| E4 | 固定 seed 43 分析预测残差；200 条样本产生 6,600 条运动指标和 4,600 条重建指标，失败 0 | 真实残差具有显著稀疏 oracle 上界，但这不是可部署选择器 |
| E5 | Sionna PHY 2.0.1 复数 AWGN；12 个残差 JSCC 模型、15,800 条 test 运动指标和 2,200 条视频指标，失败 0 | 0/5/10 dB 发送残差有效，训练范围外 -5 dB 会退化 |
| E6 | validation-only SNR 门控、12 个 hard Top-K scorer 和 24 模型 validation 消融 | 门控消除 -5 dB 退化；learned scorer 只在 K=2/4 有有限收益，K=6/8 负结果不能由 SNR 输入或速度损失解释 |
| E6.5 | 冻结 E5/E6 的通信代价与率—质量报告 | `C=1/2/3/4` 对应 74/148/222/296 复符号/段；所有 24 个可发送稀疏点均被同速率 dense residual JSCC 支配 |

## 规范化正式运行

这些目录均位于 Git 忽略的 `outputs/`，只在本机保留，不上传模型、媒体或逐样本
实验结果。

| 用途 | 正式运行目录 | 状态 |
| --- | --- | --- |
| LivePortrait 敏感性 | `outputs/motion_sensitivity/20260726T060450.218238Z` | 完成 |
| 修正后三说话人 E3 | `outputs/audio_to_motion/20260726T122634.907754Z` | 完成 |
| 十说话人 E3 | `outputs/audio_to_motion_ten_speaker/20260727T083555.175559Z` | 完成 |
| E4 残差 | `outputs/residual_baseline/20260727T161813.688662Z` | 完成 |
| E5 Sionna JSCC | `outputs/residual_jscc/20260728T060039.837712Z` | 完成 |
| E6 安全门控 | `outputs/channel_gate/20260728T093222.859499Z` | 完成 |
| E6 scorer | `outputs/residual_scorer/20260728T102637.647750Z` | 完成 |
| E6 validation 消融 | `outputs/residual_scorer_validation_ablation/20260728T112523.904356Z` | 完成且未访问 test |
| 通信代价报告 | `outputs/communication_report/20260728T120635.097427Z` | 完成，冻结源哈希已保存 |

## 作废实验与本地清理

首次 E3 运行
`outputs/audio_to_motion/20260726T084655.529497Z` 将长度不一的
`audio_25k` WAV 整体拉伸到三秒，破坏绝对时间对齐。该目录约 32 MB，已永久
删除；只能通过旧代码重跑恢复，但其方法和数值已经在
[音频到运动基线](AUDIO_MOTION_BASELINE.md) 中标记为无效，不应恢复或引用。

本轮还删除了：

- 仓库内 `.pytest_cache`、`.ruff_cache`、`.mypy_cache`、`__pycache__` 和
  `*.egg-info` 等可再生缓存；
- 未承载内容的 `docs/user_inputs/thesis_requirements.md` 空占位文件；
- 已确认不属于正式实验的临时/中断产物。

没有删除：

- 完整正式运行及其配置、指纹、汇总、逐样本指标和完成标记；
- 合法的空 `failures.jsonl`。这些文件表示管线实际执行且失败数为 0，是验收
  证据，不是垃圾空文件；
- 数据集、模型权重和 Python 环境。它们仍位于 Git 之外，但后续复现实验需要。

清理前仓库工作目录约 334 MB，清理后约 299 MB，主要回收来自无效 E3 运行与
可再生缓存。

## 当前论文证据与边界

目前可以陈述：

- 因果音频模型在十说话人身份隔离 test 上优于静态运动基线；
- 预测残差具有明显稀疏 oracle 价值；
- dense residual JSCC 在 0/5/10 dB 能改善运动和嘴部重建；
- validation-only 门控能在 -5 dB 安全回退；
- 当前 learned hard Top-K 不是稳定优于规则或 dense 编码的主要贡献。

目前不能陈述：

- 已获得真实 bit/s、完整协议吞吐量或数字压缩率；
- 稀疏选择降低了同一 `C` 下的复信道符号数；
- GRID 结果已经证明真实视频会议或跨数据集泛化；
- 当前 scorer 在所有预算优于原始幅度规则；
- 已计入音频链路、参考脸、索引、调制编码或协议头开销。

下一项必要实验是以同一 Sionna AWGN、同一 `C` 和同一模型规模，比较完整
18 维运动与预测残差 JSCC。现有 E3/E5/E6 结果全部冻结复用，只训练缺失的完整
运动对照。

