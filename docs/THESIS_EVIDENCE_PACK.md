# E8 冻结论文证据包

更新日期：2026-07-28。

## 1. 完成状态

E8 已完成。正式证据包位于：

```text
outputs/thesis_evidence/20260728T154216.719283Z
```

该目录只读取 E3–E7 的冻结正式产物，没有重新训练模型、重新选择超参数或重新
计算 test 预测。证据生成器对应 Git commit
`d2594c648c00c7be880a89e9c06779d9fb0cb1da`，源码 SHA-256 为
`74604a3638b6d74c3c78e46e7f5da97ee76823ee6d9fa1bd99f4914d485b2ceb`。

证据包已生成：

- 16 行统一主比较表；
- 1,600 行 E7 逐样本视频配对差值；
- 1,600 行先聚合噪声后的 E7 逐样本运动差值；
- 48 行运动 bootstrap 统计；
- 32 行视频 bootstrap 统计；
- 7 张统一风格图片；
- 3 个预先固定位置的定性样本引用；
- 中文结果章节自动草稿、源哈希、配置、环境和完成标记。

## 2. 统计协议

E7 运动评价中，每个 `C×SNR×样本` 有三个名义噪声种子。证据包先在同一样本
内部对三个噪声实现的残差方法与完整运动方法差值求均值，再把 100 条 test 样本
作为 100 个统计单位，执行 10,000 次固定种子配对 bootstrap。视频评价每个条件
只有共同的名义噪声种子 42，因此直接按相同样本、`C` 和 SNR 配对。

优势定义统一为：

```text
residual advantage = full-motion error - residual error
```

因此正值表示预测残差 JSCC 更好，负值表示完整运动 JSCC 更好。区间使用 95%
百分位法。16 个 `C×SNR` 区间是逐条件、描述性的点态区间，没有做多重比较
校正，不能把“不跨零”的数量解释为控制族错误率后的验证性显著性结论。

两个方法的名义噪声种子相同，但 validation 选出的模型种子在部分预算下不同，
所以不能声称两条链路使用了完全相同的逐元素噪声张量。

## 3. 核心结果

### 3.1 点估计

- 运动 L1：residual 在 16 个条件中胜出 14 个。
- 嘴部 NME：residual 在 16 个条件中胜出 13 个。
- 最大运动 L1 相对优势出现在 `C=2, SNR=-5 dB`，约为 19.38%。
- 高 SNR、大预算并非 residual 的稳定优势区；`C=2,10 dB` 的运动 L1 明确
  更支持 full-motion，`C=4,10 dB` 的嘴部 NME 明确更支持 full-motion。

### 3.2 逐样本置信区间

| 指标 | residual 区间胜出 | full-motion 区间胜出 | 跨零 |
| --- | ---: | ---: | ---: |
| 运动 L1 | 11/16 | 1/16 | 4/16 |
| 嘴部 NME | 12/16 | 1/16 | 3/16 |

运动 L1 的四个跨零条件是：

- `C=1,10 dB`；
- `C=2,5 dB`；
- `C=3,10 dB`；
- `C=4,10 dB`。

运动 L1 唯一明确支持 full-motion 的条件是 `C=2,10 dB`，平均差值为
`-0.00010053`，95% 区间为
`[-0.00014113, -0.00006244]`。

嘴部 NME 的三个跨零条件是：

- `C=2,5 dB`；
- `C=2,10 dB`；
- `C=4,5 dB`。

嘴部 NME 唯一明确支持 full-motion 的条件是 `C=4,10 dB`，平均差值为
`-0.00071538`，95% 区间为
`[-0.00131753, -0.00016333]`。

这些结果支持下面的有条件结论：

> 音频预测先验在受限视觉信道中减少了需要由信道恢复的不确定性，因此预测残差
> JSCC 的优势主要集中在低/中 SNR 和紧预算；当信道质量和预算充足时，直接传输
> 完整运动能够追平或局部超过残差方法。

它们不支持“预测残差在所有信道条件下都更优”。

## 4. 图、表和草稿索引

| 用途 | 文件 |
| --- | --- |
| 统一主表 | `main_comparison.csv` / `main_comparison.json` |
| E3 音频预测表 | `audio_baseline.csv` |
| E4 oracle 残差曲线数据 | `oracle_residual_curve.csv` |
| E6 scorer 摘要 | `scorer_summary.csv` |
| E7 运动逐样本差值 | `e7_motion_sample_differences.csv` |
| E7 运动区间 | `e7_motion_bootstrap.csv` |
| E7 视频逐样本差值 | `e7_video_sample_differences.csv` |
| E7 视频区间 | `e7_video_bootstrap.csv` |
| 图注、条件和边界 | `figure_manifest.json` |
| 方法学越界检查 | `methodological_checks.json` |
| 定性样本引用 | `qualitative_selection.json` |
| 自动章节草稿 | `results_chapter_draft.md` |
| 完整性与来源 | `summary.json`、`source_provenance.json`、`complete.json` |

7 张图位于 `figures/`：

1. 音频预测基线；
2. oracle 残差率—质量上界；
3. residual JSCC 运动 L1—SNR；
4. residual JSCC 嘴部 NME—SNR；
5. validation-only 低 SNR 安全门控；
6. residual 相对 full-motion 的匹配预算优势热图；
7. E7 运动 L1 逐样本配对 bootstrap 区间。

定性样本按 sample ID 字典序的零基位置 `0/50/99` 固定为
`s7_bbae6n`、`s7_bbwl8n` 和 `s7_bgim6p`。这是 first/middle/last 规则，不是按
结果或观感挑选。

## 5. 一键复现

在仓库根目录运行：

```bash
PYTHONPATH=src python scripts/eval/build_thesis_evidence.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/20260727T083555.175559Z \
  --e4-run-dir outputs/residual_baseline/20260727T161813.688662Z \
  --residual-jscc-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --gate-run-dir outputs/channel_gate/20260728T093222.859499Z \
  --scorer-run-dir outputs/residual_scorer/20260728T102637.647750Z \
  --scorer-ablation-run-dir \
    outputs/residual_scorer_validation_ablation/20260728T112523.904356Z \
  --communication-run-dir outputs/communication_report/20260728T120635.097427Z \
  --full-motion-run-dir outputs/full_motion_jscc/20260728T123819.525394Z \
  --comparison-run-dir outputs/full_motion_vs_residual/20260728T151133.243833Z
```

恢复已有正式目录时增加：

```bash
--run-dir outputs/thesis_evidence/20260728T154216.719283Z --resume
```

若配置、冻结源或生成器源码哈希变化，resume 会拒绝继续；完全匹配时不会改写
任何有效文件。

## 6. 论文边界

- 当前通信资源单位是复信道符号，不是 bit/s。
- 每段 75 帧、3 秒，其中首帧作为参考，74 帧参与传输。
- `C=1/2/3/4` 对应每段 `74/148/222/296` 个复信道符号。
- 未计入音频侧信息、参考脸或关键帧、量化、调制编码、协议头和同步开销。
- test 只有 GRID 的一个未见说话人 `s7`；不能外推为跨数据集或真实会议证明。
- LivePortrait 指标衡量冻结生成式重建，不等于真实视频纹理完全恢复。
- learned hard Top-K 不是最终主方法；同一 `C` 下它没有降低实际复符号数量。

## 7. 下一阶段

核心仿真 E0–E8 已闭环。下一阶段应转入 E9 论文成稿，而不是继续使用已经消费的
`s7` test 调参：

1. 将本证据包的 6 张正文图、1 张附录统计图和统一主表嵌入论文模板；
2. 完成方法、实验设置、结果、局限和复现附录的交叉引用；
3. 统一符号、指标方向、图注和“复信道符号”口径；
4. 建立答辩版 4 图精简叙事；
5. 仅在论文结构出现明确证据缺口时，再规划新的未触碰说话人或跨数据集实验。
